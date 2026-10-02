"""This package's own protocol/task -> Matterix bindings, registered the same way an
external EOS package should register its own -- see `common/protocol_registry.py`'s
module docstring for the convention this file follows (name, location, what's safe to
put here) and `common/protocol_registry.py`'s `_discover_registrations()` for how it
gets imported automatically.

Only plain strings here, deliberately -- see the docstring referenced above for why.
"""

from user.matterix_bridge.common.protocol_registry import register_protocol, register_task_workflow

register_protocol("beaker_pick_protocol", "Matterix-Experiment-Beaker-Pick-Franka-v1")
register_protocol("heater_transfer_protocol", "Matterix-Experiment-Heater-Transfer-Franka-v1")

# workflow_key is omitted below wherever it's the same string as the EOS task name (the
# common case) -- register_task_workflow() defaults it to eos_task_name. "pick_beaker" ->
# "pickup_beaker" is the one name that actually differs, so that one stays explicit.
register_task_workflow("beaker_pick_protocol", "pick_beaker", "pickup_beaker")

register_task_workflow("heater_transfer_protocol", "turn_on_heater")
register_task_workflow("heater_transfer_protocol", "pick_beaker")
register_task_workflow("heater_transfer_protocol", "place_beaker")
register_task_workflow("heater_transfer_protocol", "wait_for_heat_transfer")
register_task_workflow("heater_transfer_protocol", "turn_off_heater")
