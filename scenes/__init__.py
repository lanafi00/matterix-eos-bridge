"""Matterix scene/environment definitions, relocated here from matterix-experiments so
matterix_bridge (and by extension EOS) has no functional dependency on that sandbox repo.

Each sub-package defines one self-contained environment (robots + objects + observations
+ workflows) and registers it as a Gym environment. Importing this package registers all
of them -- see common/runtime.py's `_discover_scene_modules()`, which imports this
package (and any other EOS package's own `scenes` subpackage) automatically.

Scenes, in increasing order of complexity:

    exp1_beaker_pick      - single Franka arm picks up a beaker (the "hello world" of Matterix).
    exp2_pick_and_place   - single Franka arm picks a beaker and places it on an IKA plate.
    exp3_heater_transfer  - adds the semantics engine: turn on a heater, observe heat transfer.
    exp4_dual_arm_handoff - two Franka arms hand a beaker to each other via a shared station.

Their env ids are what protocol_registry.py's PROTOCOL_TWINS maps EOS protocol types onto.

Catalog assets and device twins are NOT registered here -- see the package-root
`matterix_devices.py` instead (`common/runtime.py`'s `_discover_device_registrations()`,
called before this package's own discovery). Devices exist independently of any
particular scene; a scene's slots get filled by a device twin at workflow-run time, not
by this file importing one.
"""

from . import (
    beaker_pick_generated,
    exp1_beaker_pick,
    exp2_pick_and_place,
    exp3_heater_transfer,
    exp4_dual_arm_handoff,
    heater_transfer_generated,
)

__all__ = [
    "exp1_beaker_pick",
    "exp2_pick_and_place",
    "exp3_heater_transfer",
    "exp4_dual_arm_handoff",
    # schema/-compiled (see scene_specs/) -- CLAUDE.md
    "beaker_pick_generated",
    "heater_transfer_generated",
]
