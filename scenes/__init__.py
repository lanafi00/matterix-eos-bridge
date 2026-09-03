"""Matterix scene/environment definitions, relocated here from matterix-experiments so
matterix_bridge (and by extension EOS) has no functional dependency on that sandbox repo.

Each sub-package defines one self-contained environment (robots + objects + observations
+ workflows) and registers it as a Gym environment. Importing this package registers all
of them -- see common/runtime.py's `import user.matterix_bridge.scenes` (noqa: F401).

Scenes, in increasing order of complexity:

    exp1_beaker_pick      - single Franka arm picks up a beaker (the "hello world" of Matterix).
    exp2_pick_and_place   - single Franka arm picks a beaker and places it on an IKA plate.
    exp3_heater_transfer  - adds the semantics engine: turn on a heater, observe heat transfer.
    exp4_dual_arm_handoff - two Franka arms hand a beaker to each other via a shared station.

Their env ids are what protocol_registry.py's PROTOCOL_TWINS maps EOS protocol types onto.
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
