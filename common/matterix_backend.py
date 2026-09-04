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


def _eos_path() -> str:
    """The sys.path entry that made `user.matterix_bridge` importable in *this* process --
    walked up from this file's own on-disk location (common/matterix_backend.py -> common
    -> matterix_bridge -> user -> EOS_PATH) instead of reading a separately-set env var, so
    it's correct regardless of how the caller's sys.path got EOS_PATH onto it (manual
    sys.path.insert, PYTHONPATH, EOS's own package loader, ...). Needed because a
    runtime_env-scoped actor (see get_backend() below) boots a genuinely fresh interpreter
    that does NOT inherit the driver's in-memory sys.path -- only real env vars survive
    that boundary, so this has to be forwarded explicitly as PYTHONPATH.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_backend(scope_id: str, conda_env: str = "isaaclab", num_gpus: int = 1):
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

    ROOT CAUSE, CONFIRMED (not just suspected) via a real `eos start` run: Ray's per-actor
    `runtime_env={"conda": ...}` CANNOT relocate this actor to a different Python
    installation than the one that called `ray.init()`. The raylet bakes an ABSOLUTE path to
    that Python (e.g. `.../eos/.venv/bin/python3`) into its `--python_worker_command` at
    startup; the "conda" plugin only wraps that fixed, already-resolved absolute path with a
    `conda activate isaaclab &&` shell prefix. That prefix correctly sets CONDA_PREFIX/PATH
    (which is why a naive check of those env vars looks right) but never changes *which
    binary actually gets exec'd* -- an absolute path is immune to PATH changes. So a real
    `eos start` process (running from EOS's own uv-managed venv, per its README) reliably
    produces `ModuleNotFoundError: No module named 'isaaclab'` here, regardless of
    conda_env/PYTHONPATH -- reproduced with EOS's REST API end to end (task -> Heater actor
    -> this actor -> `ensure_app_launched()` -> `from isaaclab.app import AppLauncher`).

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

    NOT YET DONE -- better long-term fix, noted for later: the clone above is a working
    stopgap, not the real fix, and it doesn't scale well -- every deployer has to rebuild
    that same custom hybrid conda env, and re-verify it on every eos/isaaclab upgrade. The
    root cause (Ray bakes in ONE Python at cluster boot) has a standard Ray answer that
    doesn't require touching eos's own environment at all: run a SEPARATE `ray start
    --address=<eos cluster address> --resources='{"isaaclab_gpu": 1}'` process from within
    the isaaclab conda env, joining eos's cluster as an extra worker node, then have this
    function request that resource (`options(resources={"isaaclab_gpu": 1}, ...)`) instead
    of `runtime_env={"conda": ...}`. That keeps `eos start` exactly as documented in EOS's
    README (plain uv venv, no isaaclab, no dependency conflicts) -- setup for a new deployer
    becomes "start one extra process," not "rebuild a merged environment." Blocker: EOS's
    own `ray.init(...)` in orchestrator.py currently starts an in-process/embedded cluster,
    which isn't obviously reachable for an external node to join yet -- making it reachable
    is a small change to EOS's own startup code, not just this bridge, and hasn't been
    attempted.
    """
    env_vars = {"PYTHONPATH": _eos_path()}
    if "DISPLAY" in os.environ:
        # Forwarded for the same reason as PYTHONPATH -- a runtime_env-scoped actor doesn't
        # inherit the driver's environment, so a headless=False run_workflow() call (opening
        # a real Isaac Sim window, e.g. for VNC-based verification) needs DISPLAY passed
        # through explicitly too, or it fails trying to open a display that isn't there.
        env_vars["DISPLAY"] = os.environ["DISPLAY"]

    return MatterixBackend.options(
        name=f"matterix_backend.{scope_id}",
        get_if_exists=True,
        lifetime="detached",
        num_gpus=num_gpus,
        runtime_env={"conda": conda_env, "env_vars": env_vars},
    ).remote()
