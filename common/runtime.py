"""Programmatic entry point for running Matterix workflows.

Relocated here from matterix-experiments/runtime.py so matterix_bridge (and by extension
EOS) has no functional dependency on that sandbox repo -- see MatterixBackend.run_workflow
in matterix_backend.py, the actual EOS-facing caller. The reason this needs to be its own
module rather than "just call main() from a CLI script": Isaac Sim's `SimulationApp` must
be constructed exactly once per process, before any `isaaclab`/`omni` submodule is
imported (Kit's extension/runtime plugin system requires it) - see
`ensure_app_launched()` below. That boot is also expensive (~5-10s), so this module keeps
it - and the current gym environment - alive across calls instead of paying that cost on
every workflow invocation.

Typical usage from a long-lived worker process:

    from user.matterix_bridge.common.runtime import run_workflow

    result = run_workflow(
        task="Matterix-Experiment-Beaker-Pick-Franka-v1",
        workflow="pickup_beaker",
    )
    if result.success:
        ...

Calling `run_workflow()` again with a different `task`, `num_envs`, or `devices`
transparently tears down the current gym environment and builds the new one; calling it
again with the *same* task/num_envs/devices reuses the existing environment (just resets
it), so a sequence of workflow calls against one digital twin stays fast.

`devices` swaps which concrete device backs an agent slot (e.g. "robot") without
registering a new task - see `device_registry.py` (a sibling module in this same package)
and `run_workflow()`'s docstring.

Importable from outside EOS's own process (e.g. a raw script in the isaaclab conda env,
like this session's smoke test): put the eos repo root on sys.path first (matterix_bridge
is loaded as EOS's `user.matterix_bridge` namespace package, not a separately pip-installed
one), matching matterix_assets' own MATTERIX_PATH env var convention:

    import os, sys
    sys.path.insert(0, os.environ["EOS_PATH"])
    from user.matterix_bridge.common.runtime import run_workflow
"""

from __future__ import annotations

import dataclasses
import datetime
import os
from typing import Any

# --- Isaac Sim process-wide state -------------------------------------------------
# All of this must stay None until ensure_app_launched() runs, and nothing below may
# import isaaclab/omni/matterix at module scope - those imports only work *after*
# SimulationApp exists.
_app_launcher = None
_simulation_app = None
_current_task: str | None = None
_current_num_envs: int | None = None
_current_devices: dict[str, tuple[str, str]] | None = None
_current_env = None
_current_env_cfg = None


def ensure_app_launched(headless: bool = True, device: str = "cuda:0", enable_cameras: bool = False) -> None:
    """Launch Isaac Sim for this process, if it isn't already running.

    Safe to call on every `run_workflow()` invocation - a no-op after the first call.
    `headless`/`device`/`enable_cameras` only take effect on that first call; later
    calls with different values are silently ignored (Isaac Sim can't be reconfigured
    after boot without restarting the process).
    """
    global _app_launcher, _simulation_app
    if _simulation_app is not None:
        return

    from isaaclab.app import AppLauncher

    _app_launcher = AppLauncher(headless=headless, device=device, enable_cameras=enable_cameras)
    _simulation_app = _app_launcher.app


def is_app_launched() -> bool:
    return _simulation_app is not None


def register_app(app_launcher, simulation_app) -> None:
    """Hand this module an already-launched Isaac Sim app instead of it launching one.

    For callers (like the CLI script) that need the full `AppLauncher.add_app_launcher_args`
    surface (e.g. `--enable_cameras`, custom `--device`, livestreaming) rather than the
    reduced `headless`/`device`/`enable_cameras` kwargs `ensure_app_launched()` exposes.
    Once called, `run_workflow()`'s own `ensure_app_launched()` becomes a no-op.
    """
    global _app_launcher, _simulation_app
    _app_launcher = app_launcher
    _simulation_app = simulation_app


@dataclasses.dataclass
class WorkflowResult:
    """Outcome of one `run_workflow()` call."""

    task: str
    workflow: str
    num_envs: int
    episodes_run: int
    success: bool
    """True if the action sequence succeeded on every env in the final episode."""
    per_env_success: list[bool]
    video_path: str | None = None


def _get_env(
    task: str,
    num_envs: int,
    device: str,
    use_fabric: bool,
    render_mode: str | None,
    devices: dict[str, tuple[str, str]] | None,
):
    """Return the current gym env, (re)building it only if `task`/`num_envs`/`devices` changed."""
    global _current_task, _current_num_envs, _current_devices, _current_env, _current_env_cfg

    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import user.matterix_bridge.scenes  # noqa: F401  (registers Matterix-Experiment-* envs)
    from user.matterix_bridge.common.device_registry import resolve_device_twin

    try:
        import matterix_tasks  # noqa: F401  (registers upstream Matterix-Test-* envs)
    except ImportError:
        pass

    needs_new_env = (
        _current_env is None
        or _current_task != task
        or _current_num_envs != num_envs
        or _current_devices != devices
    )
    if needs_new_env:
        if _current_env is not None:
            _current_env.close()

        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs, use_fabric=use_fabric)

        for slot, (lab_name, device_name) in (devices or {}).items():
            if slot not in env_cfg.articulated_assets:
                raise ValueError(
                    f"devices[{slot!r}] has no matching slot in {task}'s articulated_assets "
                    f"(available: {list(env_cfg.articulated_assets)})."
                )
            env_cfg.articulated_assets[slot] = resolve_device_twin(
                lab_name, device_name, env_cfg.articulated_assets[slot]
            )

        env = gym.make(task, cfg=env_cfg, render_mode=render_mode).unwrapped
        env.reset()

        _current_task = task
        _current_num_envs = num_envs
        _current_devices = devices
        _current_env = env
        _current_env_cfg = env_cfg

    return _current_env, _current_env_cfg


def run_workflow(
    task: str,
    workflow: str,
    devices: dict[str, tuple[str, str]] | None = None,
    workflow_overrides: dict[str, Any] | None = None,
    num_envs: int = 1,
    max_episodes: int = 1,
    device: str = "cuda:0",
    use_fabric: bool = True,
    record_video: bool = False,
    video_dir: str = "out/videos",
    headless: bool = True,
    print_progress: bool = True,
) -> WorkflowResult:
    """Run one Matterix workflow to completion (or until `max_episodes`) and return the outcome.

    On the first call in a process this also boots Isaac Sim (see `ensure_app_launched`)
    and builds the requested task's gym environment; later calls reuse both unless
    `task`/`num_envs`/`devices` change.

    Args:
        task: A registered Gym env id, e.g. "Matterix-Experiment-Beaker-Pick-Franka-v1".
            Picks the scene "shape" - which objects, observations, and workflows exist -
            independent of which concrete device backs each agent slot (see `devices`).
        workflow: A key into that task's `env_cfg.workflows` dict, e.g. "pickup_beaker".
        devices: Optional override mapping an `articulated_assets` slot name (e.g.
            "robot") to a concrete `(lab_name, device_name)` pair, matching the shape of
            an EOS `ScheduledTask.devices` assignment. Each pair is resolved to a twin
            config via `matterix_bridge.common.device_registry.DEVICE_TWINS` (through the
            `device_registry.py` re-export shim in this package), replacing that slot's
            default config for this call - no new gym registration needed per device.
            Omit to use the task's default `articulated_assets` as-is.
        workflow_overrides: Optional {field: value} patch applied to `env_cfg.workflows
            [workflow]` (e.g. {"target_temperature": 350.0} for a TurnOnHeaterCfg) right
            before this call runs it. Unlike `devices`, this does NOT trigger an env
            rebuild - `env_cfg` is a live cached object read fresh on every call (see
            `workflow_value = env_cfg.workflows[workflow]` below), so an override takes
            effect on this call even mid-protocol-run, and persists for later calls until
            overwritten again. Omit to use the workflow's config as authored.
        num_envs: Number of parallel envs to simulate.
        max_episodes: Stop after this many episodes (each episode re-runs the same
            workflow from a fresh reset). Unlike the CLI script this defaults to 1,
            not "forever" - a function call returning is expected to mean "done".
        record_video: If True, save an mp4 of each episode to `video_dir`. Requires
            `headless=False` or Isaac Sim launched with `enable_cameras=True` on the
            *first* `run_workflow()` call in this process (camera rendering can't be
            turned on for an already-running app).
        headless: Only affects Isaac Sim's boot on the first call in this process.
        print_progress: Print the same per-episode/per-50-step status lines the CLI
            script prints. Set False for a quiet library call.

    Returns:
        A `WorkflowResult` with per-env success flags for the final episode run.

    Raises:
        ValueError: If `workflow` is not defined on `task`; if `devices` names a slot
            that doesn't exist on `task`; or if `workflow_overrides` is given for a
            composite (dict/list) workflow, or names a field the resolved workflow
            config doesn't have.
        KeyError: If `devices` names a `(lab_name, device_name)` pair with no entry in
            `DEVICE_TWINS`.
    """
    ensure_app_launched(headless=headless, device=device, enable_cameras=record_video)

    import torch

    from matterix_sm import StateMachine

    render_mode = "rgb_array" if record_video else None
    env, env_cfg = _get_env(task, num_envs, device, use_fabric, render_mode, devices)

    if not hasattr(env_cfg, "workflows") or not env_cfg.workflows:
        raise ValueError(f"No workflows defined for {task}!")
    if workflow not in env_cfg.workflows:
        available = list(env_cfg.workflows.keys())
        raise ValueError(f"Workflow '{workflow}' not found for {task}. Available workflows: {available}")

    workflow_value = env_cfg.workflows[workflow]

    if workflow_overrides:
        if isinstance(workflow_value, (dict, list)):
            raise ValueError(
                f"workflow_overrides given but '{workflow}' on {task} is a composite "
                f"workflow ({type(workflow_value).__name__}), not a single config object -- "
                "overrides only apply to atomic, single-config workflow entries."
            )
        for field, value in workflow_overrides.items():
            if not hasattr(workflow_value, field):
                raise ValueError(
                    f"workflow_overrides has unknown field '{field}' for workflow "
                    f"'{workflow}' ({type(workflow_value).__name__})."
                )
            setattr(workflow_value, field, value)

    if isinstance(workflow_value, dict):
        actions = workflow_value.get("actions", [])
    elif isinstance(workflow_value, list):
        actions = workflow_value
    else:
        actions = [workflow_value]

    sm = StateMachine(num_envs=env.num_envs, dt=env.step_dt, device=env.device)
    sm.set_action_sequence(actions)

    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    task_slug = task.replace("/", "_")
    workflow_slug = workflow.replace("/", "_")

    video_path = None
    episode_count = 0
    per_env_success = None

    while episode_count < max_episodes:
        with torch.inference_mode():
            obs, _ = env.reset()
            sm.reset()
            episode_count += 1
            step_count = 0

            if print_progress:
                print(f"\n{'=' * 80}\nEPISODE {episode_count}\n{'=' * 80}\n")

            if record_video:
                env.start_recording()

            while not (sm.action_sequence_success | sm.action_sequence_failure).all():
                action, semantic_actions = sm.step(obs)
                if action is not None:
                    action = action.to(env.device)
                else:
                    # A workflow with no agent actions at all (e.g. TurnOnHeaterCfg, a
                    # pure semantic/equipment action with no agent_assets) leaves
                    # matterix_sm's "hold current pose" fallback uninitialized -- it only
                    # scopes to agents referenced in THIS action sequence, not the
                    # scene's full action space (see Matterix's state_machine.py step()),
                    # so sm.step() legitimately returns None even though env.step() still
                    # needs a valid action tensor for whatever agents the scene has. We
                    # don't own Matterix, so rather than patch that scoping bug upstream,
                    # reuse the environment's own currently-cached action -- isaaclab's
                    # ActionManager keeps a correctly-shaped, zero-initialized-by-default
                    # buffer (env.action_manager.action) precisely for this "nothing new
                    # to apply" case. This is isaaclab's own standard state, not a value
                    # we're guessing at.
                    action = env.action_manager.action
                obs, _, terminated, truncated, _ = env.step(action, semantic_actions=semantic_actions)
                step_count += 1

                reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                if reset_ids.numel() > 0:
                    sm.reset_envs(reset_ids)

                if print_progress and step_count % 50 == 0:
                    sm.print_status(step=step_count, episode=episode_count)

            if print_progress:
                sm.print_status(step=step_count, episode=episode_count)

            per_env_success = sm.action_sequence_success.clone()

            if record_video:
                video_path = os.path.join(
                    video_dir, f"{task_slug}_{workflow_slug}_ep{episode_count}_{run_ts}.mp4"
                )
                env.save_video(video_path)
                if print_progress:
                    print(f"[INFO]: Video saved to {video_path}")

    return WorkflowResult(
        task=task,
        workflow=workflow,
        num_envs=num_envs,
        episodes_run=episode_count,
        success=bool(per_env_success.all().item()),
        per_env_success=per_env_success.tolist(),
        video_path=video_path,
    )


def shutdown() -> None:
    """Close the current env and Isaac Sim app. Call once at process exit - not between
    `run_workflow()` calls, which is exactly the cost this module exists to avoid."""
    global _current_env, _current_task, _current_num_envs, _current_devices, _simulation_app, _app_launcher
    if _current_env is not None:
        _current_env.close()
        _current_env = None
        _current_task = None
        _current_num_envs = None
        _current_devices = None
    if _simulation_app is not None:
        _simulation_app.close()
        _simulation_app = None
        _app_launcher = None
