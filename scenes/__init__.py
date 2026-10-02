"""Matterix scene/environment definitions, relocated here from matterix-experiments so
matterix_bridge (and by extension EOS) has no functional dependency on that sandbox repo.

Each sub-package defines one self-contained environment (robots + objects + observations
+ workflows) and registers it as a Gym environment. Importing this package registers all
of them -- see common/runtime.py's `_discover_scene_modules()`, which imports this
package (and any other EOS package's own `scenes` subpackage) automatically, AND walks
every scene subpackage found under it -- a new scene doesn't need to be added to the
imports below to be picked up (compiling one via schema/compile_scene.py, or copying an
exp*/ example, is enough on its own); exp1/exp3 stay explicit here only because
dump_catalog.py's own warm-up import (`import user.matterix_bridge.scenes`) needs AT
LEAST ONE scene submodule imported as a side effect to trigger matterix_assets' circular-
import fix, independent of `_discover_scene_modules()`'s own walk.

Scenes, in increasing order of complexity -- genuinely exercised by smoke_test.py/
vnc_test.py (exp2_pick_and_place and exp4_dual_arm_handoff existed here too, as
schema-scope examples never actually booted by any test, exp4 additionally a known-broken
`matterix_sm` multi-agent limitation -- see CLAUDE.md's "Known open gaps"; removed):

    exp1_beaker_pick      - single Franka arm picks up a beaker (the "hello world" of Matterix).
    exp3_heater_transfer  - adds the semantics engine: turn on a heater, observe heat transfer.

(schema/compile_scene.py -- see CLAUDE.md's "Declarative scene schema" section -- can
compile a hand-authored scene_specs/*.yaml into this same directory; none are checked in
here, since a compiled scene is fully reproducible from its YAML in one command and
would otherwise just be a duplicate of whatever exp*/ example it matches. The three YAML
fixtures originally used to round-trip-verify the schema/compiler against exp1/exp3/exp4's
shapes were removed once `schema/from_eos.py` (an EOS package's own lab.yml, see
`eos/user/beaker_lab`) became the actively-used path -- CLAUDE.md still records what that
round-trip verified. Whatever gets compiled here is picked up automatically -- see
`_discover_scene_modules()`'s auto-walk, no listing needed below.)

Their env ids are what protocol_registry.py's PROTOCOL_TWINS maps EOS protocol types onto.

Catalog assets and device twins are NOT registered here -- see the package-root
`matterix_devices.py` instead (`common/runtime.py`'s `_discover_device_registrations()`,
called before this package's own discovery). Devices exist independently of any
particular scene; a scene's slots get filled by a device twin at workflow-run time, not
by this file importing one.
"""

from . import (
    exp1_beaker_pick,
    exp3_heater_transfer,
)

__all__ = [
    "exp1_beaker_pick",
    "exp3_heater_transfer",
]
