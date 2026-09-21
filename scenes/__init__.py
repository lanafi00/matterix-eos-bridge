"""Matterix scene/environment definitions, relocated here from matterix-experiments so
matterix_bridge (and by extension EOS) has no functional dependency on that sandbox repo.

Each sub-package defines one self-contained environment (robots + objects + observations
+ workflows) and registers it as a Gym environment. Importing this package registers all
of them -- see common/runtime.py's `_discover_scene_modules()`, which imports this
package (and any other EOS package's own `scenes` subpackage) automatically, AND walks
every scene subpackage found under it -- a new scene doesn't need to be added to the
imports below to be picked up (compiling one via schema/compile_scene.py, or copying an
exp*/ example, is enough on its own); exp1-4 stay explicit here only because
dump_catalog.py's own warm-up import (`import user.matterix_bridge.scenes`) needs AT
LEAST ONE scene submodule imported as a side effect to trigger matterix_assets' circular-
import fix, independent of `_discover_scene_modules()`'s own walk.

Scenes, in increasing order of complexity:

    exp1_beaker_pick      - single Franka arm picks up a beaker (the "hello world" of Matterix).
    exp2_pick_and_place   - single Franka arm picks a beaker and places it on an IKA plate.
    exp3_heater_transfer  - adds the semantics engine: turn on a heater, observe heat transfer.
    exp4_dual_arm_handoff - two Franka arms hand a beaker to each other via a shared station.

Plus three schema/-compiled scenes (scene_specs/*.yaml -> schema/compile_scene.py; see
CLAUDE.md) matching the first four's shapes -- beaker_pick_generated, heater_transfer_generated,
dual_arm_handoff -- not listed below; `_discover_scene_modules()`'s auto-walk finds them.

Their env ids are what protocol_registry.py's PROTOCOL_TWINS maps EOS protocol types onto.

Catalog assets and device twins are NOT registered here -- see the package-root
`matterix_devices.py` instead (`common/runtime.py`'s `_discover_device_registrations()`,
called before this package's own discovery). Devices exist independently of any
particular scene; a scene's slots get filled by a device twin at workflow-run time, not
by this file importing one.
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
