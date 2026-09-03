"""Maps EOS device identities to the Matterix twin asset config that backs them.

Keyed exactly the way EOS names a device - `(lab_name, device_name)` - so a device
assignment coming out of an EOS `ScheduledTask.devices` resolves to a concrete twin
config with a single dict lookup. Adding a new physical unit's digital twin (or
retargeting an existing slot to a different unit) means adding one entry here - no
new gym registration, no new experiment file. See `run_workflow()`'s `devices=`
parameter in runtime.py.

Populate this with the actual (lab_name, device_name) pairs from your EOS lab
definitions (e.g. `user/<pkg>/labs/<lab_name>/lab.yml`) - the entry below is a
placeholder illustrating the shape, not a real device.
"""

from __future__ import annotations

from typing import Callable

from matterix_assets.matterix_articulation import MatterixArticulationCfg
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

DEVICE_TWINS: dict[tuple[str, str], Callable[..., MatterixArticulationCfg]] = {
    # ("lab_name", "device_name"): twin config class
    ("example_lab", "franka_01"): FRANKA_PANDA_HIGH_PD_IK_CFG,
}


def resolve_device_twin(
    lab_name: str, device_name: str, existing: MatterixArticulationCfg
) -> MatterixArticulationCfg:
    """Build the twin config for `(lab_name, device_name)`, keeping `existing`'s placement.

    `existing` is the slot's current config from the env cfg being overridden - its
    `pos`/`rot` describe where that slot sits in the scene (a property of the scene
    layout), not which physical device fills it, so they carry over across the swap.
    """
    key = (lab_name, device_name)
    if key not in DEVICE_TWINS:
        raise KeyError(
            f"No twin registered for device {key!r}. Add it to DEVICE_TWINS in "
            f"matterix_bridge/common/device_registry.py."
        )
    return DEVICE_TWINS[key](pos=existing.pos, rot=existing.rot)
