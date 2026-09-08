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

This is also where device twins get registered (see device_registry.py's module
docstring for why this file specifically, not matterix_registrations.py or a
package-root __init__.py): this file is already auto-imported, by every package that
has one, after Isaac Sim has booted -- exactly what a twin config class needs, since
it comes from matterix_assets (isaaclab-dependent).
"""

from . import (
    exp1_beaker_pick,
    exp2_pick_and_place,
    exp3_heater_transfer,
    exp4_dual_arm_handoff,
)

__all__ = [
    "exp1_beaker_pick",
    "exp2_pick_and_place",
    "exp3_heater_transfer",
    "exp4_dual_arm_handoff",
]

# Imported down here, not at module top -- matterix_assets/matterix have their own
# internal import order that the scene submodules above already exercise correctly;
# importing matterix_assets.robots before them (tried first, reverted) hits a circular
# import (matterix_assets.equipment -> matterix.managers... -> matterix.envs ->
# matterix_assets again, still mid-init). Safe here since exp1-4 above already fully
# initialized both packages.
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG  # noqa: E402

from user.matterix_bridge.common.device_registry import register_device_twin  # noqa: E402

# This package's own example device, registered the same way an external package should
# register its own -- see device_registry.py's module docstring. Not a real device; it's
# a placeholder illustrating the shape.
register_device_twin("example_lab", "franka_01", FRANKA_PANDA_HIGH_PD_IK_CFG)
