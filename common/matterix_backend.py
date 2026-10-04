"""Shared Ray actor that owns the single live Matterix/Isaac Sim environment for one
protocol run (or campaign). Every sim-mode device driver gets or creates this SAME actor
via get_backend(scope_id) -- Isaac Sim boots once per scope_id, not once per device, and
every device acts on the one shared scene instead of a private, disconnected copy of it.

EOS lab/task -> Matterix task/workflow resolution lives in protocol_registry.py;
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
# idle sweep (see _kill_idle_actors below) considers it abandoned and kills it, when
# get_backend() is attaching to an EXISTING scope's actor. Creating a NEW scope's actor
# doesn't wait for this -- it kills every other backend actor that's idle right now (see
# get_backend()), since only one Isaac Sim can use this machine's GPU at a time anyway.
_IDLE_TIMEOUT_S = 15 * 60

# How long get_backend() waits for a newly created actor to be scheduled (i.e. granted its
# num_gpus claim) before giving up with a RuntimeError naming whoever holds the GPU,
# instead of handing back an actor whose every call would hang in PENDING_CREATION.
_GPU_WAIT_S = 30.0

# How long the sweep waits on another actor's idle_seconds() before calling it busy.
# MatterixBackend is a plain synchronous actor, so one mid-workflow can't answer at all.
_BUSY_PROBE_S = 5.0


def _resolve(lab: str, eos_task_name: str) -> tuple[str, str]:
    """Resolve an EOS task, as run in `lab`, to its Matterix (gym_task_id, workflow_key)
    pair -- see `protocol_registry.py`'s `resolve_matterix_call_by_lab()`.
    """
    from user.matterix_bridge.common.protocol_registry import resolve_matterix_call_by_lab

    return resolve_matterix_call_by_lab(lab, eos_task_name)


@ray.remote
class MatterixBackend:
    def __init__(self):
        self._params: dict[str, dict[str, Any]] = {}  # matterix workflow key -> {field: value}
        self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last set_parameter()/run_workflow() call finished. Used by
        get_backend()'s idle sweep. Never answers while a call is running (this is a
        synchronous actor) -- the sweep treats that as busy."""
        return time.monotonic() - self._last_activity

    def set_parameter(self, lab: str, eos_task_name: str, field: str, value: Any) -> None:
        """Update a single workflow config field -- see `set_parameters()` for the real
        implementation and the effective-timing semantics (same here, just one field at a
        time). Kept for callers that only ever have one field to set (e.g. smoke_test.py);
        a device driver with more than one dynamic parameter should use `set_parameters()`
        (or the `run_matterix_workflow()` helper below, which calls it for you) instead of
        making one remote round-trip per field.
        """
        self.set_parameters(lab, eos_task_name, {field: value})

    def set_parameters(self, lab: str, eos_task_name: str, fields: dict[str, Any]) -> None:
        """Update multiple workflow config fields in one call, effective on this scope's
        next run_workflow() call for that (lab, eos_task_name) -- including mid-run. One
        remote round-trip regardless of how many fields are in `fields`, unlike calling
        `set_parameter()` once per field.
        """
        self._last_activity = time.monotonic()
        _, workflow = _resolve(lab, eos_task_name)
        self._params.setdefault(workflow, {}).update(fields)

    def run_workflow(
        self,
        lab: str,
        eos_task_name: str,
        devices: dict[str, tuple[str, str]] | None = None,
        **run_workflow_kwargs: Any,
    ) -> Any:
        """Resolve an EOS (lab, eos_task_name) pair to a Matterix (task, workflow)
        pair and run it, applying any overrides registered via set_parameter for this
        workflow. `devices` (an EOS `{slot: (lab_name, device_name)}` mapping) and
        `run_workflow_kwargs` (num_envs, max_episodes, record_video, ... -- see
        runtime.run_workflow's real signature) pass straight through.

        Scene state persists across calls: each one continues from where the last left
        the scene (see runtime.run_workflow's `reset` param), not from a fresh reset.

        Every call against this SAME actor (i.e. every DAG task in one protocol run,
        sharing one scope_id) must resolve to the same `task`/`num_envs`/`devices` -- see
        `runtime._get_env()`'s docstring. Two EOS tasks in one protocol run that map to
        DIFFERENT Matterix gym tasks (different scene "shapes") will hit that RuntimeError
        the moment the second one calls this, because both share this one actor/process.
        Split such a protocol's tasks across more than one scope_id (a device driver can
        pass any string as `protocol_run_name`/scope_id -- it doesn't have to be EOS's
        `protocol_run_name` literally) if you need more than one scene per protocol run.
        """
        self._last_activity = time.monotonic()
        # Local import: Isaac Sim is heavy and only available inside the isaaclab conda
        # env -- keep this actor (and its callers) importable without it.
        from user.matterix_bridge.common.runtime import run_workflow

        task, workflow = _resolve(lab, eos_task_name)
        overrides = self._params.get(workflow)
        # reset=False: this actor's scene is the protocol run's one shared bench, so each
        # EOS task continues from where the previous one left it (e.g. place_beaker
        # starts with the beaker still in the gripper from pick_beaker) instead of a fresh
        # reset that would silently undo every earlier task. The env is reset once, when
        # first built for this scope.
        try:
            return run_workflow(
                task=task,
                workflow=workflow,
                devices=devices,
                workflow_overrides=overrides,
                reset=False,
                **run_workflow_kwargs,
            )
        finally:
            # Idle time counts from when the work ENDED, not when it started -- otherwise
            # an actor that just finished a 2-minute workflow already looks 2 minutes idle.
            self._last_activity = time.monotonic()


def _other_backends(exclude_name: str) -> list[tuple[str, Any, float | None]]:
    """`(name, handle, idle_seconds)` for every other `matterix_backend.*` actor in this Ray
    cluster, across all namespaces (a detached actor started from a different driver
    script lives in that script's namespace, not ours). `idle_seconds` is None when the
    actor didn't answer within _BUSY_PROBE_S -- i.e. it's busy mid-call (or wedged).

    Best-effort: an actor that can't be looked up at all (mid-shutdown, transient RPC
    error) is just left out.
    """
    try:
        from ray.util import list_named_actors

        entries = [
            e
            for e in list_named_actors(all_namespaces=True)
            if e["name"].startswith("matterix_backend.") and e["name"] != exclude_name
        ]
    except Exception:
        return []

    found = []
    for entry in entries:
        try:
            handle = ray.get_actor(entry["name"], namespace=entry["namespace"])
        except Exception:
            continue
        try:
            idle_for = ray.get(handle.idle_seconds.remote(), timeout=_BUSY_PROBE_S)
        except Exception:
            idle_for = None
        found.append((entry["name"], handle, idle_for))
    return found


def _kill_idle_actors(exclude_name: str, min_idle_s: float) -> None:
    """Kill every other `matterix_backend.*` actor that's been idle at least `min_idle_s`
    (0 = idle right now, i.e. not mid-call). A busy actor is never killed.

    Called from get_backend() rather than a background timer: `MatterixBackend` stays a
    plain synchronous actor (no asyncio/threading conversion) because Isaac Sim/PhysX/CUDA
    state inside it isn't safe to touch from a second thread -- a real timer would need
    that conversion; this sweep doesn't. Best-effort: a failed kill is swallowed so it
    never blocks the get_backend() call that triggered it (if that leaves the GPU held,
    get_backend()'s GPU wait reports it).
    """
    for _, handle, idle_for in _other_backends(exclude_name):
        if idle_for is not None and idle_for >= min_idle_s:
            try:
                ray.kill(handle)
            except Exception:
                pass


def _wait_for_gpu(handle: Any, name: str, gpu_wait_s: float) -> None:
    """Block until the just-created actor `handle` is scheduled -- its __init__ is trivial,
    so the first answer to idle_seconds() means Ray granted its num_gpus claim. Raise
    RuntimeError after `gpu_wait_s` instead, naming every other backend actor (busy, or
    idle and for how long), and kill the still-pending actor so it can't silently grab
    the GPU later and linger as an orphaned detached actor.
    """
    try:
        ray.get(handle.idle_seconds.remote(), timeout=gpu_wait_s)
        return
    except ray.exceptions.GetTimeoutError:
        pass

    # No answer can also mean the actor IS scheduled but busy: another caller in the same
    # run created it first (get_if_exists handed us theirs) and it's mid-workflow. Ray's
    # GCS actor table tells the two apart without needing the dashboard (VERIFIED: a busy
    # actor reads ALIVE, a GPU-starved one PENDING_CREATION) -- private API, hence the
    # try: if it ever breaks, fall through to the pending-actor path below.
    try:
        from ray._private import state as ray_state

        if ray_state.actors(actor_id=handle._actor_id.hex())["State"] == "ALIVE":
            return
    except Exception:
        pass

    holders = _other_backends(exclude_name=name)
    try:
        ray.kill(handle)
    except Exception:
        pass

    if holders:
        detail = "; ".join(
            f"{other!r} is busy (didn't answer within {_BUSY_PROBE_S:.0f}s -- likely mid-workflow)"
            if idle_for is None
            else f"{other!r} has been idle {idle_for:.0f}s but is still alive (killing it failed?)"
            for other, _, idle_for in holders
        )
        who = f"Other MatterixBackend actors: {detail}."
    else:
        who = (
            "No other MatterixBackend actor exists in this Ray cluster, so something else "
            "holds the GPU resource -- check `ray list actors --filter state=ALIVE`."
        )
    raise RuntimeError(
        f"{name!r} couldn't get a GPU within {gpu_wait_s:.0f}s (only one Isaac Sim can run "
        f"on this machine at a time). {who} Busy actors are never killed automatically -- "
        "wait for that run to finish, or `ray.kill()` its actor if it's stuck."
    )


def get_backend(
    scope_id: str, conda_env: str | None = None, num_gpus: int = 1, gpu_wait_s: float = _GPU_WAIT_S
):
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
    `num_gpus=1` claim -- EOS gives devices no "run finished" signal to release it on.
    So when this call is about to CREATE a new scope's actor (a new run starting), it
    first kills every OTHER `matterix_backend.*` actor that's idle right now -- only one
    Isaac Sim can use the GPU anyway. A busy actor (mid-call, so it can't answer within
    `_BUSY_PROBE_S`) is left alone. Attaching to an existing scope's actor only sweeps
    actors idle past `_IDLE_TIMEOUT_S`.

    Trade-off of "idle right now": a run that's merely between two of its tasks is idle
    too, so starting a second run concurrently kills the first one's actor (and its
    scene state); the first run's next task then fails or starts a fresh scene. On one
    GPU those two runs couldn't both proceed anyway.

    After creating a new actor (with `num_gpus > 0`), waits up to `gpu_wait_s` for Ray
    to schedule it. If the GPU is still held (e.g. by a busy actor), raises RuntimeError
    naming that actor instead of returning one whose calls would hang forever.

    TODO / known gaps:
    - Better long-term fix for the conda_env relocation problem above: run a separate
      `ray start --address=<eos cluster> --resources='{"isaaclab_gpu": 1}'` worker node
      from the isaaclab env, and request that resource here instead of `runtime_env=
      {"conda": ...}` -- keeps `eos start` on EOS's plain documented venv. Blocked on
      EOS's `ray.init()` using an embedded, not externally-joinable, cluster.
    """
    name = f"matterix_backend.{scope_id}"
    try:
        existing = ray.get_actor(name)
    except ValueError:
        existing = None
    if existing is not None:
        _kill_idle_actors(exclude_name=name, min_idle_s=_IDLE_TIMEOUT_S)
        return existing

    _kill_idle_actors(exclude_name=name, min_idle_s=0.0)

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

    handle = MatterixBackend.options(
        name=name,
        get_if_exists=True,  # another caller may have created it since the lookup above
        lifetime="detached",
        num_gpus=num_gpus,
        runtime_env=runtime_env,
    ).remote()
    if num_gpus > 0:
        _wait_for_gpu(handle, name, gpu_wait_s)
    return handle


def run_matterix_workflow(
    scope_id: str,
    eos_task_name: str,
    *,
    lab: str,
    devices: dict[str, tuple[str, str]] | None = None,
    headless: bool = True,
    **fields: Any,
) -> Any:
    """Get-or-create this scope's MatterixBackend actor, push any dynamic parameters,
    run the workflow, and raise if it didn't succeed -- the one call a sim-mode device
    driver's method should make, instead of separately calling get_backend(),
    set_parameters(), run_workflow(), and checking result.success by hand. Every
    sim-mode device wants this same sequence (see eos/user/beaker_lab's devices/arm/
    device.py for a real example); putting it here once means a new device.py is one
    call instead of five lines of Ray/actor boilerplate.

    scope_id: normally the calling task's protocol_run_name -- a device has no way to
    know which protocol run it's being called from on its own (see BaseTask), so the
    caller has to supply it. Every device in one protocol run should pass the SAME
    scope_id so they share one MatterixBackend actor/scene (see get_backend()'s
    docstring) -- this is a plain pass-through, not something this function derives.

    lab: the calling device's own lab -- pass `self.lab_name` (EOS's BaseDevice exposes
    it). Together with `eos_task_name` it picks the Matterix scene and workflow (see
    protocol_registry.py's `resolve_matterix_call_by_lab()`), so nothing about the
    calling protocol needs to be known or hardcoded.

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
        ray.get(backend.set_parameters.remote(lab, eos_task_name, fields))
    result = ray.get(backend.run_workflow.remote(lab, eos_task_name, devices=devices, headless=headless))
    if not result.success:
        raise RuntimeError(f"Failed to run workflow {eos_task_name!r}: {result.failure_detail}")
    return result
