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


def get_backend(scope_id: str, conda_env: str = "isaaclab"):
    """Get-or-create the one backend actor for this run/campaign.

    First call for a given scope_id creates the actor (and, on its first run_workflow
    call, boots Isaac Sim); every later call with the same scope_id attaches to the same
    live actor instead of creating a new one (get_if_exists=True short-circuits before
    `conda_env`/other constructor args are even looked at, so they only matter on that
    first call).

    conda_env: `eos start` launches Ray actors using EOS's own venv by default (no
    isaaclab/omni/matterix_sm there) -- this actor specifically needs to run inside the
    isaaclab conda env instead, via Ray's per-actor runtime_env override. UNVERIFIED as
    of this commit: I confirmed `ray.remote(runtime_env={"conda": ...})` is real,
    supported API (RuntimeEnv.__init__ accepts a conda env name), but haven't been able
    to actually exercise it end-to-end (no Isaac Sim in this dev environment) -- this is
    the first thing to check if get_backend()/run_workflow() fails with a
    ModuleNotFoundError on isaaclab/omni when called from an EOS task.
    """
    return MatterixBackend.options(
        name=f"matterix_backend.{scope_id}",
        get_if_exists=True,
        lifetime="detached",
        runtime_env={"conda": conda_env},
    ).remote()
