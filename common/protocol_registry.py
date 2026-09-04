"""Maps EOS protocol/task identities to the Matterix task/workflow that validates them.

Two registries, mirroring `device_registry.py`'s pattern:

- `PROTOCOL_TWINS` - EOS protocol type -> Matterix gym task id. Picks which scene
  "shape" (objects, observations, workflow definitions) backs a given protocol.
- `TASK_WORKFLOWS` - (EOS protocol type, EOS task name) -> Matterix workflow key.
  Picks which single `env_cfg.workflows` entry backs one EOS DAG task node.

Populated via `register_protocol()`/`register_task_workflow()` rather than edited as
literal dict entries, so a package other than this one (e.g. your own EOS package
declaring its own protocol) can register its bindings from its own code instead of
editing this file directly -- call these from your package's `__init__.py` (or
anywhere guaranteed to run before `resolve_matterix_call()` is first called for that
protocol/task).

These are explicit mappings rather than a naming convention (e.g. "assume the EOS task
name always matches the Matterix workflow key") on purpose: a rename on either side -
an EOS protocol definition or one of this repo's `workflows` dicts - should fail loudly
via `resolve_matterix_call()`'s KeyError, not silently resolve to the wrong workflow.

Each Matterix `workflows` dict has one atomic entry per EOS DAG task node (what an
EOS-facing caller dispatches one at a time), plus - where useful for manual/dev CLI
testing - bundled entries that chain several atomic steps in one call. Only the atomic
entries are meant to be reached through this registry.
"""

from __future__ import annotations

PROTOCOL_TWINS: dict[str, str] = {}
TASK_WORKFLOWS: dict[tuple[str, str], str] = {}


def register_protocol(protocol_type: str, gym_task_id: str, *, overwrite: bool = False) -> None:
    """Bind an EOS protocol type to the Matterix gym task id (scene) that backs it.

    Raises ValueError if `protocol_type` is already bound to a *different* gym task id
    and `overwrite` isn't set -- two packages silently fighting over the same protocol
    type should fail loudly, not resolve to whichever one happened to import last.
    Re-registering the same (protocol_type, gym_task_id) pair is always a no-op.
    """
    existing = PROTOCOL_TWINS.get(protocol_type)
    if existing is not None and existing != gym_task_id and not overwrite:
        raise ValueError(
            f"Protocol type {protocol_type!r} is already registered to gym task "
            f"{existing!r} (tried to register {gym_task_id!r}). Pass overwrite=True "
            "if this is intentional."
        )
    PROTOCOL_TWINS[protocol_type] = gym_task_id


def register_task_workflow(
    protocol_type: str, eos_task_name: str, workflow_key: str, *, overwrite: bool = False
) -> None:
    """Bind one EOS DAG task node to the Matterix workflow key that executes it.

    Same overwrite semantics as `register_protocol()`. `protocol_type` doesn't need to
    already be registered via `register_protocol()` at call time -- the two registries
    are independent dicts, only joined together inside `resolve_matterix_call()`.
    """
    key = (protocol_type, eos_task_name)
    existing = TASK_WORKFLOWS.get(key)
    if existing is not None and existing != workflow_key and not overwrite:
        raise ValueError(
            f"Task {key!r} is already registered to workflow {existing!r} (tried to "
            f"register {workflow_key!r}). Pass overwrite=True if this is intentional."
        )
    TASK_WORKFLOWS[key] = workflow_key


def resolve_matterix_call(protocol_type: str, eos_task_name: str) -> tuple[str, str]:
    """Resolve an EOS `(protocol_type, task_name)` pair to a Matterix `(task, workflow)` pair.

    Raises KeyError naming the offending side rather than silently mismatching, so a
    rename on either side of the EOS/Matterix boundary is caught immediately instead of
    resolving to the wrong (or a stale) workflow.
    """
    if protocol_type not in PROTOCOL_TWINS:
        raise KeyError(
            f"No Matterix task registered for EOS protocol {protocol_type!r}. Call "
            "register_protocol() for it (see matterix_bridge/common/protocol_registry.py)."
        )
    key = (protocol_type, eos_task_name)
    if key not in TASK_WORKFLOWS:
        raise KeyError(
            f"No Matterix workflow registered for EOS task {key!r}. Call "
            "register_task_workflow() for it (see matterix_bridge/common/protocol_registry.py)."
        )
    return PROTOCOL_TWINS[protocol_type], TASK_WORKFLOWS[key]


# This package's own example protocols/tasks, registered the same way an external
# package should register its own -- see this module's docstring.
register_protocol("beaker_pick_protocol", "Matterix-Experiment-Beaker-Pick-Franka-v1")
register_protocol("pick_and_place_protocol", "Matterix-Experiment-Pick-Place-Franka-v1")
register_protocol("heater_transfer_protocol", "Matterix-Experiment-Heater-Transfer-Franka-v1")
register_protocol("dual_arm_handoff_protocol", "Matterix-Experiment-Dual-Arm-Handoff-Franka-v1")

register_task_workflow("beaker_pick_protocol", "pick_beaker", "pickup_beaker")

register_task_workflow("pick_and_place_protocol", "pick_beaker", "pickup_beaker")
register_task_workflow("pick_and_place_protocol", "place_beaker", "place_beaker")

register_task_workflow("heater_transfer_protocol", "turn_on_heater", "turn_on_heater")
register_task_workflow("heater_transfer_protocol", "pick_beaker", "pick_beaker")
register_task_workflow("heater_transfer_protocol", "place_beaker", "place_beaker")
register_task_workflow("heater_transfer_protocol", "wait_for_heat_transfer", "wait_for_heat_transfer")
register_task_workflow("heater_transfer_protocol", "turn_off_heater", "turn_off_heater")

register_task_workflow("dual_arm_handoff_protocol", "robot_pick_beaker", "robot_pick_beaker")
register_task_workflow("dual_arm_handoff_protocol", "robot_place_on_station", "robot_place_on_station")
register_task_workflow("dual_arm_handoff_protocol", "wait_for_handoff", "wait_for_handoff")
register_task_workflow("dual_arm_handoff_protocol", "robot2_pick_beaker", "robot2_pick_beaker")
