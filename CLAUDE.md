# matterix-eos-bridge

Bridges Matterix (Isaac Sim/Isaac Lab digital-twin simulation) into EOS (lab-automation
orchestration). See `README.md` for the architecture overview and the "using this from
your own EOS package" walkthrough — this file is operational knowledge for working in
this repo, not a design doc.

## Environment

- Everything Isaac-Sim-dependent must run from the `eos-isaaclab` conda env (cloned from
  `isaaclab`, with `eos` installed into it: `conda create --name eos-isaaclab --clone
  isaaclab && conda activate eos-isaaclab && pip install -e <eos repo>`). Ray's
  `runtime_env={"conda": ...}` does NOT relocate a process to a different Python
  installation — the raylet bakes in an absolute interpreter path at cluster boot — so
  `eos start` itself must already run from a Python with isaaclab/matterix installed, not
  EOS's plain `uv` venv.
- This repo is symlinked into `<eos repo>/user/matterix_bridge` and loaded as EOS's
  `user.matterix_bridge` namespace package.
- **On this machine, `conda activate` does not persist through Claude Code's sandboxed
  Bash tool** — each command's `PATH` gets reset, so `which python3` after `conda
  activate eos-isaaclab` still resolves to `/usr/bin/python3`. Invoke the env's
  interpreter by absolute path instead: `/home/lila/miniconda3/envs/eos-isaaclab/bin/python3`.

## Running the smoke test

From the `eos` repo root, in the `eos-isaaclab` env:
```
SKIP_STAGE0=1 /home/lila/miniconda3/envs/eos-isaaclab/bin/python3 user/matterix_bridge/smoke_test.py
```
`SKIP_STAGE0=1` skips the slowest baseline sanity stage once it's passed at least once.
Boots real Isaac Sim — expect ~1-2 minutes for the first environment build. All stages
should print `PASS`; a `FAIL` or an uncaught traceback means something regressed.

## This machine (`svart`, `152.2.130.61`)

- **Single GPU** (RTX 4080 SUPER). Only one Isaac Sim process can meaningfully run at a
  time. Check `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
  and `ps aux | grep -iE "isaac-sim|kit_"` before booting a new one — a second boot while
  one is already running will contend for the GPU, not fail cleanly.
- **Shared box**: other Claude Code sessions, other users (check `w`/`last -a`), and
  separately-started `eos start` processes can all be running concurrently. If something
  hangs, check for contention before assuming it's a code bug.
- **No sudo** in this session. `py-spy` is blocked by `ptrace_scope=1`. To get a stack
  trace from a hung Python process, register a `faulthandler` handler *in the script
  before running it* (`faulthandler.register(signal.SIGUSR1, all_threads=True)`), then
  `kill -SIGUSR1 <pid>` and read the dump from the process's own stdout/log — this works
  without ptrace since the dump runs inside the target process itself.

## Known hard constraints (all verified live, not theoretical)

- **A process can only build ONE Matterix environment for its whole lifetime.**
  `runtime._get_env()` raises `RuntimeError` immediately if a later call needs a
  different `task`/`num_envs`/`devices` than the one already built — it does NOT
  transparently rebuild. Root cause: constructing a second `isaaclab.sim.SimulationContext`
  in an already-booted process calls `.stop()` on the still-live first one from inside its
  own `__init__`, which calls `torch.cuda.set_device()` from an on-stop event handler —
  and that call hangs indefinitely. This is inside third-party `isaaclab`/`isaacsim` code,
  not fixable from this repo. Practical effect: one `MatterixBackend` actor / one
  `scope_id` = one scene for its whole life. A protocol whose DAG tasks need different
  Matterix scenes needs to split across more than one `scope_id`.
- **Import order matters for `matterix_assets`.** Whichever import is the *first* one
  ever to touch `matterix_assets` in a process determines whether it succeeds — importing
  `matterix_assets.robots` (or anything that transitively imports it) as the very first
  touch hits a circular import inside `matterix_assets/__init__.py` itself. Importing this
  repo's `scenes` package first happens to warm up the `matterix_assets`/`matterix`
  package graph safely. This is why `runtime.py`'s `_get_env()` calls
  `_discover_scene_modules()` before `_discover_device_registrations()`, even though
  devices are conceptually independent of scenes.
- **Isaac-Sim-dependent code can only run after `ensure_app_launched()`.** Nothing that
  imports `isaaclab`/`omni`/`matterix_assets` may run at module scope before
  `SimulationApp` exists (Kit's extension system requires it) — including in a throwaway
  diagnostic script. A raw script doing `from matterix_assets... import ...` before
  calling `ensure_app_launched()` (or before importing `user.matterix_bridge.scenes` to
  warm up the graph) will hit the circular import above.

## Architecture quick reference

- `common/protocol_registry.py` — EOS `(protocol_type, task_name)` string → Matterix
  `(gym_task_id, workflow_key)` string. Pre-boot safe (no isaaclab imports) — populated
  from each EOS package's `matterix_registrations.py`.
- `common/device_registry.py` — EOS `(lab_name, device_name)` → real Matterix asset
  config **class** (`DEVICE_TWINS`). Post-boot only. Populated from each EOS package's
  `matterix_devices.py`, deliberately not `scenes/__init__.py` (devices are independent
  of any particular scene). Validates every registration via `matterix_asset_types.py`'s
  `validate_asset_cfg()` before accepting it.
- `common/matterix_asset_types.py` — shared, registry-free utilities: `validate_asset_cfg()`
  and `discover_matterix_assets()` (introspects Matterix's own `robots`/`equipment`/
  `labware`/`infrastructure` packages for what's actually available to register). There
  used to be a separate name→class `ASSET_CATALOG` layer here; it was collapsed back into
  `DEVICE_TWINS` storing classes directly — with only ~11 assets in Matterix and one real
  device binding in this repo, the extra indirection wasn't earning its cost. Worth
  rebuilding only if/when a real name-based lookup consumer exists (e.g. a layout/scene
  loader).
- `common/runtime.py` — the actual Isaac Sim boot and environment lifecycle
  (`ensure_app_launched()`, `_get_env()`, `run_workflow()`).
- `common/matterix_backend.py` — the Ray actor (`MatterixBackend`) wrapping `runtime.py`
  so Isaac Sim's boot cost is paid once per `scope_id`, not once per call. Device drivers
  should call `run_matterix_workflow()` (the one-call helper) rather than
  `get_backend()`/`set_parameters()`/`run_workflow()` individually.
- `scenes/` — Gym environment definitions only (`gym.register()`), never device/catalog
  registration.
- A scene has exactly two asset containers (`matterix_base_env_cfg.py`):
  `articulated_assets` (robots) and `objects` (rigid + static objects — beakers, plates,
  tables — together in one dict). `devices=` substitution checks both.
- `resolve_device_twin()` carries over `pos`/`rot` and `semantics` from the original slot
  to the twin (both describe the slot's role in the scene), but lets the twin's own class
  defaults win for physically-intrinsic fields (usd_path, mass, scale, frames).

## Declarative scene schema (in progress)

Working toward a thin YAML/JSON schema (asset catalog entry, pos/rot, semantics preset +
params, workflow step sequence) that compiles to a scene's `*_env_cfg.py`, so authoring a
scene doesn't mean hand-writing `@configclass` Python — see the review discussion for the
full rationale (schema-validate before ever paying Isaac Sim's 1-2 min boot cost).

**`dump_catalog.py`** (repo root, run the same way as `smoke_test.py`) boots Isaac Sim
once and dumps every asset/action/semantics class Matterix currently ships — name, and
constructor field signature (required vs. default) — to `catalog.json`, checked in (not
gitignored) since it's a build artifact meant to be diffed like any other. Regenerate it
whenever Matterix's own asset/action/semantics classes change (e.g. a Matterix version
bump):

```
EOS_PATH=/home/lila/Documents/git/eos \
    /home/lila/miniconda3/envs/eos-isaaclab/bin/python3 dump_catalog.py
```

**`schema/`** is the schema + compiler itself — see `schema/__init__.py`'s docstring for
the package layout. Deliberately has no isaaclab/matterix/matterix_sm dependency at all;
it only reads `catalog.json`, so it validates and compiles in EOS's plain venv, no conda
env or Isaac Sim boot needed:

```
/home/lila/Documents/git/eos/.venv/bin/python3 -m schema.compile_scene scene_specs/beaker_pick.yaml
```

writes `scenes/<name>/<name>_env_cfg.py` + `__init__.py` — same shape and convention as a
hand-authored scene, but a new one still needs a manual one-line addition to
`scenes/__init__.py`'s own imports/`__all__` (same as adding any new `scenes/exp*/`) for
`_discover_scene_modules()` to pick it up; the CLI doesn't edit that file for you.

Scope is deliberately narrow right now (see `schema/models.py`'s docstring) — covers
exactly what `scene_specs/beaker_pick.yaml` needs (asset placement, one position-
randomization event kind, plain `pick_object`-shaped workflow steps), live-verified by
compiling it to `scenes/beaker_pick_generated/` and actually running its workflow in Isaac
Sim (`success: true`). NOT yet covered: semantics presets, global semantics, multi-agent
scenes, composite/bundled workflows — extend `models.py`/`compiler.py` together when the
next round-trip target (`exp3_heater_transfer`, the semantics-heavy scene) needs them.

## Known open gaps

- `get_backend()`'s conda-env relocation is a working stopgap, not the intended fix — the
  real fix (a separately joinable Ray worker node reporting a custom GPU resource) is
  blocked on EOS's own `ray.init()` not being externally joinable.
- Detached `MatterixBackend` actors on this single-GPU machine are reclaimed by a 15-minute
  idle timeout, not a real "run finished" signal (EOS gives devices no such signal) —
  mitigated, not solved.
- Matterix's `PickObjectCfg`/`PlaceObjectCfg` report success based on the robot reaching a
  target end-effector pose, not on whether the object was actually grasped or placed — a
  workflow can report `success=True` with nothing physically achieved. Not yet addressed.

## Conventions

- Every change to `common/` or `matterix_devices.py`/`scenes/` gets live-verified by
  actually booting Isaac Sim in `eos-isaaclab` and running the real code path — not just
  read/reasoned about — before being considered done. `smoke_test.py` should stay green;
  add a new stage for new registry/API surface rather than only testing it ad hoc.
- Commit messages document what was verified and how (not just what changed) — see `git
  log` for the pattern.
