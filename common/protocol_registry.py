"""Maps EOS protocol/task identities to the Matterix task/workflow that validates them.

Two dicts, mirroring `device_registry.py`'s pattern:

- `PROTOCOL_TWINS` - EOS protocol type -> Matterix gym task id. Picks which scene
  "shape" (objects, observations, workflow definitions) backs a given protocol.
- `TASK_WORKFLOWS` - (EOS protocol type, EOS task name) -> Matterix workflow key.
  Picks which single `env_cfg.workflows` entry backs one EOS DAG task node.

Both are explicit dicts rather than a naming convention (e.g. "assume the EOS task name
always matches the Matterix workflow key") on purpose: a rename on either side - an EOS
protocol definition or one of this repo's `workflows` dicts - should fail loudly via
`resolve_matterix_call()`'s KeyError, not silently resolve to the wrong workflow.

Each Matterix `workflows` dict has one atomic entry per EOS DAG task node (what an
EOS-facing caller dispatches one at a time), plus - where useful for manual/dev CLI
testing - bundled entries that chain several atomic steps in one call. Only the atomic
entries are meant to be reached through this registry.
"""

from __future__ import annotations

PROTOCOL_TWINS: dict[str, str] = {
    "beaker_pick_protocol": "Matterix-Experiment-Beaker-Pick-Franka-v1",
    "pick_and_place_protocol": "Matterix-Experiment-Pick-Place-Franka-v1",
    "heater_transfer_protocol": "Matterix-Experiment-Heater-Transfer-Franka-v1",
    "dual_arm_handoff_protocol": "Matterix-Experiment-Dual-Arm-Handoff-Franka-v1",
}

TASK_WORKFLOWS: dict[tuple[str, str], str] = {
    ("beaker_pick_protocol", "pick_beaker"): "pickup_beaker",

    ("pick_and_place_protocol", "pick_beaker"): "pickup_beaker",
    ("pick_and_place_protocol", "place_beaker"): "place_beaker",

    ("heater_transfer_protocol", "turn_on_heater"): "turn_on_heater",
    ("heater_transfer_protocol", "pick_beaker"): "pick_beaker",
    ("heater_transfer_protocol", "place_beaker"): "place_beaker",
    ("heater_transfer_protocol", "wait_for_heat_transfer"): "wait_for_heat_transfer",
    ("heater_transfer_protocol", "turn_off_heater"): "turn_off_heater",

    ("dual_arm_handoff_protocol", "robot_pick_beaker"): "robot_pick_beaker",
    ("dual_arm_handoff_protocol", "robot_place_on_station"): "robot_place_on_station",
    ("dual_arm_handoff_protocol", "wait_for_handoff"): "wait_for_handoff",
    ("dual_arm_handoff_protocol", "robot2_pick_beaker"): "robot2_pick_beaker",
}


def resolve_matterix_call(protocol_type: str, eos_task_name: str) -> tuple[str, str]:
    """Resolve an EOS `(protocol_type, task_name)` pair to a Matterix `(task, workflow)` pair.

    Raises KeyError naming the offending side rather than silently mismatching, so a
    rename on either side of the EOS/Matterix boundary is caught immediately instead of
    resolving to the wrong (or a stale) workflow.
    """
    if protocol_type not in PROTOCOL_TWINS:
        raise KeyError(
            f"No Matterix task registered for EOS protocol {protocol_type!r}. "
            f"Add it to PROTOCOL_TWINS in matterix_bridge/common/protocol_registry.py."
        )
    key = (protocol_type, eos_task_name)
    if key not in TASK_WORKFLOWS:
        raise KeyError(
            f"No Matterix workflow registered for EOS task {key!r}. "
            f"Add it to TASK_WORKFLOWS in matterix_bridge/common/protocol_registry.py."
        )
    return PROTOCOL_TWINS[protocol_type], TASK_WORKFLOWS[key]
