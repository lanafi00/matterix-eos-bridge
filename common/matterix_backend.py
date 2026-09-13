"""Shared Ray actor that owns the single live Matterix/Isaac Sim environment for one
protocol run (or campaign). Every sim-mode device driver gets or creates this SAME actor
via get_backend(scope_id) -- Isaac Sim boots once per scope_id, not once per device, and
every device acts on the one shared scene instead of a private, disconnected copy of it.

EOS protocol/task -> Matterix task/workflow resolution lives in protocol_registry.py;
EOS device -> Matterix twin resolution lives in device_registry.py (consumed internally by
run_workflow()'s `devices=` param). This actor is just a persistent process wrapping
run_workflow() (a sibling module, runtime.py) so Isaac Sim's boot cost is paid once per
scope_id, not once per call.

Parameter registry: a workflow's own config fields (e.g. TurnOnHeaterCfg's
target_temperature) are fixed at env_cfg authoring time -- there's no way for an EOS
task's dynamic parameter to reach Matterix without one. `set_parameters` stores
{field: value} overrides here, keyed by the resolved Matterix workflow key (not the EOS
task name -- see protocol_registry.py's docstring on why those aren't assumed equal);
`run_workflow` passes the current values for that workflow key as `workflow_overrides` on
every call. Because `env_cfg` is a live object cached across calls inside runtime.py
(workflow_value = env_cfg.workflows[workflow] is read fresh on every run_workflow() call,
not just at env build time), a `set_parameters` call takes effect on the very next
`run_workflow` call for that scope_id -- genuinely mid-run, no environment rebuild.

A device driver doesn't need to touch get_backend()/set_parameters()/run_workflow()
individually -- see `run_matterix_workflow()` at the bottom of this module, which wraps
that whole sequence into one call; put that (not the three calls it wraps) in your own
device.py. See this repo's README for the device.py template.
"""

import os
import time
from typing import Any

import ray

from user.matterix_bridge.common.runtime import _eos_path

# How long a MatterixBackend actor can go without a real call before get_backend()'s
# idle sweep (see _kill_idle_actors below) considers it abandoned and kills it. Well
# above any single workflow's expected runtime -- this is about reclaiming actors from
# *finished* runs, not interrupting an active one.
_IDLE_TIMEOUT_S = 15 * 60


@ray.remote
class MatterixBackend:
    def __init__(self):
        self._params: dict[str, dict[str, Any]] = {}  # matterix workflow key -> {field: value}
        self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last set_parameter()/run_workflow() call. Used by
        get_backend()'s idle sweep to decide whether this actor has been abandoned."""
        return time.monotonic() - self._last_activity

    def set_parameter(self, protocol_type: str, eos_task_name: str, field: str, value: Any) -> None:
        """Update a single workflow config field -- see `set_parameters()` for the real
        implementation and the effective-timing semantics (same here, just one field at a
        time). Kept for callers that only ever have one field to set (e.g. smoke_test.py);
        a device driver with more than one dynamic parameter should use `set_parameters()`
        (or the `run_matterix_workflow()` helper below, which calls it for you) instead of
        making one remote round-trip per field.
        """
        self.set_parameters(protocol_type, eos_task_name, {field: value})

    def set_parameters(self, protocol_type: str, eos_task_name: str, fields: dict[str, Any]) -> None:
        """Update multiple workflow config fields in one call, effective on this scope's
        next run_workflow() call for that (protocol_type, eos_task_name) -- including
        mid-run. One remote round-trip regardless of how many fields are in `fields`,
        unlike calling `set_parameter()` once per field.
        """
        self._last_activity = time.monotonic()
        from user.matterix_bridge.common.protocol_registry import resolve_matterix_call

        _, workflow = resolve_matterix_call(protocol_type, eos_task_name)
        self._params.setdefault(workflow, {}).update(fields)

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
        self._last_activity = time.monotonic()
        # Local imports: Isaac Sim is heavy and only available inside the isaaclab conda
        # env -- keep this actor (and its callers) importable without it.
        from user.matterix_bridge.common.protocol_registry import resolve_matterix_call
        from user.matterix_bridge.common.runtime import run_workflow

        task, workflow = resolve_matterix_call(protocol_type, eos_task_name)
        overrides = self._params.get(workflow)
        return run_workflow(
            task=task, workflow=workflow, devices=devices, workflow_overrides=overrides, **run_workflow_kwargs
        )


def _kill_idle_actors(exclude_name: str) -> None:
    """Kill every other `matterix_backend.*` actor that's been idle past _IDLE_TIMEOUT_S.

    Called from get_backend() on every invocation, not on a separate timer -- piggybacked
    on the one moment this actually matters: right before a NEW actor might need a GPU.
    `MatterixBackend` stays a plain synchronous actor (no asyncio/threading conversion)
    because Isaac Sim/PhysX/CUDA state inside it isn't safe to touch from a second thread
    -- a real background timer would need that conversion; this sweep doesn't.

    Best-effort: any failure here (an actor mid-shutdown, a transient RPC error, a
    genuinely busy actor we don't want to wait on) is swallowed so a sweep problem never
    blocks the real get_backend() call that triggered it.
    """
    try:
        from ray.util import list_named_actors

        names = [n for n in list_named_actors() if n.startswith("matterix_backend.") and n != exclude_name]
    except Exception:
        return

    for name in names:
        try:
            handle = ray.get_actor(name)
            # Short timeout: an actor mid-workflow is busy, not idle -- don't wait on it,
            # just skip it this sweep (it'll be checked again next time get_backend() runs).
            idle_for = ray.get(handle.idle_seconds.remote(), timeout=5)
            if idle_for > _IDLE_TIMEOUT_S:
                ray.kill(handle)
        except Exception:
            continue


def get_backend(scope_id: str, conda_env: str | None = None, num_gpus: int = 1):
    """Get-or-create the one backend actor for this run/campaign.

    First call for a given scope_id creates the actor (and, on its first run_workflow
    call, boots Isaac Sim); every later call with the same scope_id attaches to the same
    live actor instead of creating a new one (get_if_exists=True short-circuits before
    `conda_env`/`num_gpus`/other constructor args are even looked at, so they only matter
    on that first call).

    Ray's `runtime_env={"conda": ...}` does NOT relocate this actor to a different Python
    installation than the one that called `ray.init()` -- the raylet bakes an absolute
    path to that Python into its worker-launch command at startup, and "conda" just wraps
    that fixed path with a `conda activate <env> &&` shell prefix (sets CONDA_PREFIX/PATH,
    but `sys.executable` stays the original). So `eos start` itself must already run from
    a Python with isaaclab/matterix installed -- this project uses a dedicated
    `eos-isaaclab` conda env for that. `conda_env` defaults to `None`: if the driver can
    already import isaaclab, request no conda env at all (actor just inherits the
    driver's environment); otherwise fall back to `"isaaclab"`. Pass it explicitly only
    to override that choice.

    `num_gpus` must be >0 for the actor to get a visible GPU -- Ray actors get
    `CUDA_VISIBLE_DEVICES=""` by default. Set to 0 only for a CPU-only smoke test of the
    actor's plumbing (imports/registry resolution) without actually running Isaac Sim.

    `PYTHONPATH` and (when set) `DISPLAY` are forwarded explicitly via `env_vars` below --
    a runtime_env-scoped actor is a fresh interpreter that inherits neither the driver's
    sys.path nor its display.

    Actors are `lifetime="detached"`, one per `scope_id`, so on a single-GPU machine a
    finished (or failed) run's actor would otherwise permanently hold the only
    `num_gpus=1` claim. Fixed via an idle timeout, not a "run finished" signal (EOS gives
    devices no such signal) -- every call here first runs `_kill_idle_actors()`, which
    kills any OTHER `matterix_backend.*` actor idle past `_IDLE_TIMEOUT_S`. Piggybacked on
    the call path rather than a real background timer, since that would need converting
    this actor to async/threaded, unsafe for the Isaac Sim/PhysX/CUDA state it holds.

    TODO / known gaps:
    - Better long-term fix for the conda_env relocation problem above: run a separate
      `ray start --address=<eos cluster> --resources='{"isaaclab_gpu": 1}'` worker node
      from the isaaclab env, and request that resource here instead of `runtime_env=
      {"conda": ...}` -- keeps `eos start` on EOS's plain documented venv. Blocked on
      EOS's `ray.init()` using an embedded, not externally-joinable, cluster.
    """
    _kill_idle_actors(exclude_name=f"matterix_backend.{scope_id}")

    runtime_env: dict[str, Any] = {}
    if conda_env is None:
        # Check directly whether THIS process can import isaaclab, rather than trusting
        # CONDA_DEFAULT_ENV's name -- that var is "base" in every shell here (conda
        # auto-activates "base" on startup) regardless of whether isaaclab is actually on
        # sys.path, so name-matching it silently requested the wrong ("base") env every
        # time. find_spec() only locates the module, doesn't execute it -- safe pre-boot.
        import importlib.util

        if importlib.util.find_spec("isaaclab") is None:
            conda_env = "isaaclab"
        # else: driver can already import isaaclab -- leave conda_env unset so the actor
        # just inherits the driver's own environment (a no-op relocation).
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


def run_matterix_workflow(
    scope_id: str,
    protocol_type: str,
    eos_task_name: str,
    *,
    devices: dict[str, tuple[str, str]] | None = None,
    headless: bool = True,
    **fields: Any,
) -> Any:
    """Get-or-create this scope's MatterixBackend actor, push any dynamic parameters,
    run the workflow, and raise if it didn't succeed -- the one call a sim-mode device
    driver's method should make, instead of separately calling get_backend(),
    set_parameters(), run_workflow(), and checking result.success by hand. Every
    sim-mode device wants this same sequence (see devices/heater/device.py's heat_to()
    for the pattern this replaces); putting it here once means a new device.py is one
    call instead of five lines of Ray/actor boilerplate.

    scope_id: normally the calling task's protocol_run_name -- a device has no way to
    know which protocol run it's being called from on its own (see BaseTask), so the
    caller has to supply it. Every device in one protocol run should pass the SAME
    scope_id so they share one MatterixBackend actor/scene (see get_backend()'s
    docstring) -- this is a plain pass-through, not something this function derives.

    fields: the workflow's dynamic parameter overrides (e.g. target_temperature=350.0),
    pushed via ONE set_parameters() call regardless of how many there are. Omit entirely
    for a workflow with no dynamic parameters -- then no set_parameters() call is made
    at all, not even an empty one.

    devices/headless: passed straight through to run_workflow() (see runtime.run_workflow's
    real signature for the rest -- num_envs, record_video, etc. -- add
    **run_workflow_kwargs here if a device ever actually needs one of those).

    Raises RuntimeError if the workflow's action sequence didn't succeed on every env
    (result.success is False), with result.failure_detail in the message -- every device
    wants this check, so it lives here once instead of copy-pasted into every device.py.
    """
    backend = get_backend(scope_id)
    if fields:
        ray.get(backend.set_parameters.remote(protocol_type, eos_task_name, fields))
    result = ray.get(backend.run_workflow.remote(protocol_type, eos_task_name, devices=devices, headless=headless))
    if not result.success:
        raise RuntimeError(f"Failed to run workflow {protocol_type}/{eos_task_name}: {result.failure_detail}")
    return result
