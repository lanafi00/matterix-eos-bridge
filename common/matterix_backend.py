"""Shared Ray actor that owns the single live Matterix/Isaac Sim environment for one
protocol run (or campaign). Every sim-mode device driver gets or creates this SAME actor
via get_backend(scope_id) -- Isaac Sim boots once per scope_id, not once per device, and
every device acts on the one shared scene instead of a private, disconnected copy of it.

EOS protocol/task -> Matterix task/workflow resolution lives in protocol_registry.py;
EOS device -> Matterix twin resolution lives in device_registry.py (consumed internally by
run_workflow()'s `devices=` param). This actor is just a persistent process wrapping
run_workflow() (a sibling module, runtime.py -- all of this now lives in matterix_bridge,
not matterix-experiments, so there's no functional dependency on that sandbox repo) so
Isaac Sim's boot cost is paid once per scope_id, not once per call.

Parameter registry: a workflow's own config fields (e.g. TurnOnHeaterCfg's
target_temperature) are fixed at env_cfg authoring time -- there's no way for an EOS
task's dynamic parameter to reach Matterix without one. `set_parameter` stores
{field: value} overrides here, keyed by the resolved Matterix workflow key (not the EOS
task name -- see protocol_registry.py's docstring on why those aren't assumed equal);
`run_workflow` passes the current values for that workflow key as `workflow_overrides` on
every call. Because `env_cfg` is a live object cached across calls inside runtime.py
(workflow_value = env_cfg.workflows[workflow] is read fresh on every run_workflow() call,
not just at env build time), a `set_parameter` call takes effect on the very next
`run_workflow` call for that scope_id -- genuinely mid-run, no environment rebuild.
"""

import os
from typing import Any

import ray

from user.matterix_bridge.common.runtime import _eos_path


@ray.remote
class MatterixBackend:
    def __init__(self):
        self._params: dict[str, dict[str, Any]] = {}  # matterix workflow key -> {field: value}

    def set_parameter(self, protocol_type: str, eos_task_name: str, field: str, value: Any) -> None:
        """Update a workflow config field, effective on this scope's next run_workflow()
        call for that (protocol_type, eos_task_name) -- including mid-run."""
        from user.matterix_bridge.common.protocol_registry import resolve_matterix_call

        _, workflow = resolve_matterix_call(protocol_type, eos_task_name)
        self._params.setdefault(workflow, {})[field] = value

    def run_workflow(
        self,
        protocol_type: str,
        eos_task_name: str,
        devices: dict[str, tuple[str, str]] | None = None,
        **run_workflow_kwargs: Any,
    ) -> Any:
        """Resolve an EOS (protocol_type, eos_task_name) pair to a Matterix (task, workflow)
        pair and run it, applying any overrides registered via set_parameter for this
        workflow. `devices` (an EOS `{slot: (lab_name, device_name)}` mapping) and
        `run_workflow_kwargs` (num_envs, max_episodes, record_video, ... -- see
        runtime.run_workflow's real signature) pass straight through.
        """
        # Local imports: Isaac Sim is heavy and only available inside the isaaclab conda
        # env -- keep this actor (and its callers) importable without it.
        from user.matterix_bridge.common.protocol_registry import resolve_matterix_call
        from user.matterix_bridge.common.runtime import run_workflow

        task, workflow = resolve_matterix_call(protocol_type, eos_task_name)
        overrides = self._params.get(workflow)
        return run_workflow(
            task=task, workflow=workflow, devices=devices, workflow_overrides=overrides, **run_workflow_kwargs
        )


def get_backend(scope_id: str, conda_env: str | None = None, num_gpus: int = 1):
    """Get-or-create the one backend actor for this run/campaign.

    First call for a given scope_id creates the actor (and, on its first run_workflow
    call, boots Isaac Sim); every later call with the same scope_id attaches to the same
    live actor instead of creating a new one (get_if_exists=True short-circuits before
    `conda_env`/`num_gpus`/other constructor args are even looked at, so they only matter
    on that first call).

    conda_env: named for Ray's `runtime_env={"conda": ...}` plugin, but READ THE ROOT CAUSE
    BELOW -- this parameter does NOT actually make this actor conda-portable relative to the
    driver. It only matters (and only ever mattered) when `eos start`'s own process is
    already running inside a conda env with isaaclab installed, in which case this is a
    same-name no-op that Ray accepts.

    num_gpus: Ray actors get NO visible GPU by default (CUDA_VISIBLE_DEVICES is set to ""
    unless the actor requests num_gpus>0 in .options()) -- omitting this made Isaac Sim fail
    with `RuntimeError: No CUDA GPUs are available` even on a machine with a free GPU.
    VERIFIED: with num_gpus=1 requested here, the actor's CUDA_VISIBLE_DEVICES is correctly
    populated. Set to 0 only for a CPU-only smoke test of the actor plumbing itself
    (imports/registry resolution) without actually running Isaac Sim.

    VERIFIED with a live isaaclab install + GPU: a runtime_env-scoped actor boots with a
    genuinely fresh interpreter that does NOT inherit the driver's in-memory sys.path --
    `set_parameter`/`run_workflow`'s `from user.matterix_bridge...` imports failed with
    `ModuleNotFoundError: No module named 'user'` until PYTHONPATH was forwarded explicitly
    via `env_vars` below.

    ROOT CAUSE, RE-VERIFIED (this note went back and forth once before landing here -- an
    intermediate "CORRECTION" claiming the mechanism below was wrong turned out to itself be
    wrong; trust THIS version, reproduced cleanly and directly): Ray's per-actor
    `runtime_env={"conda": ...}` does NOT relocate this actor to a different Python
    installation than the one that called `ray.init()`. The raylet bakes an ABSOLUTE path to
    that Python (e.g. `.../eos/.venv/bin/python3`, confirmed by reading the live raylet
    process's own `--python_worker_command` argument) into its worker-launch command at
    startup; the "conda" plugin only wraps that fixed, already-resolved absolute path with a
    `conda activate isaaclab &&` shell prefix. That prefix correctly sets CONDA_PREFIX/PATH
    (confirmed via a live probe: an actor requesting `{"conda": "isaaclab"}` from a plain-venv
    `eos start` driver reported `CONDA_PREFIX=.../envs/isaaclab` -- looks right) but its
    `sys.executable` was STILL `.../eos/.venv/bin/python`, not isaaclab's -- an absolute path
    is immune to PATH changes, so the actor never actually gains access to isaaclab's
    site-packages. So a real `eos start` process (running from EOS's own uv-managed venv, per
    its README) reliably produces `ModuleNotFoundError: No module named 'isaaclab'` here,
    regardless of conda_env/PYTHONPATH -- reproduced twice, independently, with EOS's REST
    API end to end (task -> Heater actor -> this actor -> `ensure_app_launched()` -> `from
    isaaclab.app import AppLauncher`).

    A SEPARATE, real bug found and fixed along the way, worth keeping distinct from the
    above: `_discover_registrations()` in `protocol_registry.py` (and `_discover_scene_
    modules()` in `runtime.py`) used to import `eos.configuration.packages`, which pulls in
    EOS's full entity/pydantic model tree as a side effect (`LabDef` -> `bofire`, an EOS
    *optimizer* dependency, unrelated to package discovery) -- so even a correctly-relocated
    actor sitting in a bare `isaaclab` conda env (no `eos`, no `bofire` installed there) would
    fail with `ModuleNotFoundError: No module named 'bofire'` before ever reaching Isaac Sim.
    FIXED: both now use a small local directory-walk (`runtime.py`'s
    `_discover_user_package_dirs()`) instead of importing EOS's own discovery module -- see
    that function's docstring. VERIFIED: a bare `isaaclab` conda env (isaacsim/isaaclab/
    matterix/ray, no `eos`, no `bofire`) now runs scene discovery, device-twin resolution,
    protocol/task resolution, and a full `run_workflow()` call successfully end to end, none
    of which was true before this fix. This is real, standing progress -- it's just not
    sufficient on its own to fix the ROOT CAUSE above, since that's a completely separate
    problem (which Python the actor's process literally *is*, decided before any of this
    module's code runs at all).

    Also fixed alongside this: `conda_env`'s default used to be `os.environ.get(
    "CONDA_DEFAULT_ENV", "isaaclab")`, intended to make requesting the "same" env a no-op
    when the driver already has isaaclab. VERIFIED BROKEN on this machine: conda
    auto-activates "base" for every shell (a common conda install default), so
    CONDA_DEFAULT_ENV is *always* set -- to "base" -- regardless of whether isaaclab is
    anywhere on the driver's actual sys.path, so the hardcoded "isaaclab" fallback never
    actually triggered; a plain-venv `eos start` would request conda's irrelevant "base" env
    instead of "isaaclab". FIXED: check `importlib.util.find_spec("isaaclab") is not None`
    directly (locates the module via sys.path/finders, does not execute it -- safe pre-boot)
    instead of trusting an env var name. If the driver can already import isaaclab, skip
    "conda" in runtime_env entirely (the actor just inherits the driver's own environment,
    whatever it's named); only fall back to the literal string "isaaclab" when it can't.

    FIX, VERIFIED WORKING: `eos start` itself must run from a Python that already has
    isaaclab/matterix_sm installed, so the raylet's baked-in worker command is correct from
    the start and no runtime_env swap is needed at all (conda_env then just matches the
    driver's own env, a no-op). In this project: `conda create --name eos-isaaclab --clone
    isaaclab`, then `pip install -e <eos repo>` into that clone, then run `eos start` from
    it. Confirmed end to end via EOS's own REST API: `POST /api/protocols/` for
    heater_transfer_protocol reached COMPLETED, with output_resources.sample.meta.temperature
    matching the submitted target_temperature exactly. Cloning (rather than installing eos's
    web/DB/S3 stack directly into the isaaclab env other work may depend on) was a deliberate
    choice to keep isaaclab's own pins isolated -- `pip install -e .` still reported real
    conflicts against isaacsim/isaaclab's pins (numpy 2.x vs isaacsim-kernel's numpy==1.26.0
    pin, starlette 1.6.0 vs isaaclab's starlette==0.45.3 pin, several boto3/botocore/protobuf/
    packaging/cryptography version bumps) -- Isaac Sim still booted and ran correctly despite
    these in the clone, but that isn't guaranteed to hold on every isaaclab/eos version pair,
    so re-verify (at least an AppLauncher boot + matterix/matterix_assets/matterix_tasks
    import) after any future `eos`, `isaaclab`, or `isaacsim` upgrade on either side.

    REGRESSION FOUND AND FIXED IN THE CLONE, WORTH KNOWING ABOUT: after the above was
    verified working, a *later* run against the same eos-isaaclab clone started failing
    reliably (not flaky -- reproduced on two independent fresh actors) with
    `AttributeError: 'Loop' object has no attribute '_stopping'` deep inside Isaac Sim's
    own `omni.kit.async_engine`, during USD asset loading. Root cause: Ray installs
    uvloop's event loop policy process-wide for an actor, on its own, whenever uvloop is
    merely importable in that actor's environment -- confirmed live (a bare probe actor
    reports `asyncio.get_event_loop_policy()` as a uvloop policy before any of this
    module's code runs). `eos`'s own dependencies (litestar/uvicorn) pull in uvloop, so
    once `pip install -e <eos repo>` puts them in the same env as isaaclab, every
    `MatterixBackend` actor there inherits uvloop's policy -- and Isaac Sim's async_engine
    assumes a stock CPython event loop (accesses private attributes like `_stopping`/
    `_ready`/`_check_closed` uvloop's `Loop` doesn't have), so it crashes the moment it
    touches asyncio. This is a materially different risk than the version-pin conflicts
    listed above: it's runtime contamination between eos's dependency stack and Isaac
    Sim's internals sharing one process, not a static import-time incompatibility, so it
    can appear well after everything looked like it worked. FIXED in `runtime.py`'s
    `ensure_app_launched()`: reset the event loop policy to the stdlib default there,
    before `AppLauncher`/Isaac Sim's extension system ever gets a chance to touch asyncio.
    VERIFIED: 3/3 clean runs after the fix (0/2 before it) via EOS's REST API, including
    two back-to-back runs with the earlier actor explicitly killed in between.

    NOT YET DONE -- better long-term fix, noted for later: the clone above is a working
    stopgap, not the real fix, and it doesn't scale well -- every deployer has to rebuild
    that same custom hybrid conda env, re-verify it on every eos/isaaclab upgrade, AND
    (per the regression above) stay alert to this whole class of runtime-contamination
    bug, not just import-time version conflicts -- fixing the symptom each time it
    surfaces isn't the same as eliminating the risk. The root cause (Ray bakes in ONE
    Python at cluster boot) has a standard Ray answer that doesn't require touching eos's
    own environment at all: run a SEPARATE `ray start --address=<eos cluster address>
    --resources='{"isaaclab_gpu": 1}'` process from within the isaaclab conda env, joining
    eos's cluster as an extra worker node, then have this function request that resource
    (`options(resources={"isaaclab_gpu": 1}, ...)`) instead of `runtime_env={"conda":
    ...}`. That keeps `eos start` exactly as documented in EOS's README (plain uv venv, no
    isaaclab, no dependency conflicts, no shared process with Isaac Sim at all) -- setup
    for a new deployer becomes "start one extra process," not "rebuild a merged
    environment." Blocker: EOS's own `ray.init(...)` in orchestrator.py currently starts
    an in-process/embedded cluster, which isn't obviously reachable for an external node
    to join yet -- making it reachable is a small change to EOS's own startup code, not
    just this bridge, and hasn't been attempted.

    ALSO NOT YET DONE, separate issue: `get_backend()`'s actors are `lifetime="detached"`
    and each distinct `scope_id` creates its own, so on a single-GPU machine a completed
    (or even failed) run's actor permanently holds the only `num_gpus=1` claim -- a second
    protocol run with a different scope_id will hang indefinitely in `PENDING_CREATION`,
    not fail loudly, until the earlier actor is explicitly `ray.kill()`ed. VERIFIED live.
    No fix attempted yet; worth deciding whether idle detached actors should be torn down
    after some idle period, or whether `scope_id` should be coarser than one-per-protocol-
    run on constrained hardware.
    """
    runtime_env: dict[str, Any] = {}
    if conda_env is None:
        # Only relocate if the driver's OWN process genuinely can't already reach
        # isaaclab -- checked directly (can this process import isaaclab), not by reading
        # CONDA_DEFAULT_ENV's name: that env var is set to "base" in every shell on this
        # machine (conda auto-activates "base" on shell startup, a common conda install
        # default) regardless of whether isaaclab is anywhere on this process's sys.path,
        # so name-matching it never actually falls through to a real bare-isaaclab
        # relocation -- it always requested conda's irrelevant "base" env instead, which
        # has neither isaaclab nor eos. VERIFIED this was live and silent: `eos start`
        # from EOS's own plain venv has CONDA_DEFAULT_ENV=base yet no isaaclab on its
        # path, and the old `os.environ.get("CONDA_DEFAULT_ENV", "isaaclab")` logic would
        # have targeted "base" instead of "isaaclab" every time, on this machine.
        # find_spec() only locates the module (checks sys.path/finders), it does not
        # execute isaaclab/__init__.py -- safe to call before Isaac Sim has booted.
        import importlib.util

        if importlib.util.find_spec("isaaclab") is None:
            conda_env = "isaaclab"
        # else: the driver can already import isaaclab itself (e.g. `eos start` launched
        # from a conda env with isaaclab installed) -- leave conda_env unset and omit
        # "conda" from runtime_env entirely below, so the actor simply inherits the
        # driver's own environment instead of requesting a specific env by name (a no-op
        # relocation without having to know or guess that env's name).
    if conda_env is not None:
        runtime_env["conda"] = conda_env

    env_vars = {"PYTHONPATH": _eos_path()}
    if "DISPLAY" in os.environ:
        # Forwarded for the same reason as PYTHONPATH -- a runtime_env-scoped actor doesn't
        # inherit the driver's environment, so a headless=False run_workflow() call (opening
        # a real Isaac Sim window, e.g. for VNC-based verification) needs DISPLAY passed
        # through explicitly too, or it fails trying to open a display that isn't there.
        env_vars["DISPLAY"] = os.environ["DISPLAY"]
    runtime_env["env_vars"] = env_vars

    return MatterixBackend.options(
        name=f"matterix_backend.{scope_id}",
        get_if_exists=True,
        lifetime="detached",
        num_gpus=num_gpus,
        runtime_env=runtime_env,
    ).remote()
