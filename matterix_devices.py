"""Registers this package's device twins.

Deliberately its own file, not folded into `scenes/__init__.py` -- devices/assets exist
independently of any particular scene: a scene's slots (e.g. "robot" in
`articulated_assets`) get filled by a device twin at workflow-run time (see
`device_registry.py`'s `resolve_device_twin()` and `runtime.py`'s `devices=` parameter on
`run_workflow()`), not the other way around. Keeping this separate means:

- A package can register its devices even before it has any custom scene of its own
  (reusing matterix_bridge's exp1-4 scenes as-is).
- A future layout/scene-description system can list and place devices without needing to
  import scene modules first.
- Editing which physical devices exist never means touching a scene file, and vice versa.

Auto-imported by `runtime.py`'s `_discover_device_registrations()`, once per process --
matches the naming convention of the pre-boot `matterix_registrations.py` (a plain file
at your package's root), but this one runs *after* Isaac Sim has booted, since
registering a device twin needs a real `matterix_assets` class (see
`matterix_asset_types.py`'s docstring for why). Runs after scenes are discovered, not
before, despite devices being conceptually independent of scenes -- see the comment at
that call site in runtime.py for the verified matterix_assets-internal circular import
this order avoids. You never need a custom scene yourself for this to work;
matterix_bridge's own exp1-4 scenes cover that requirement on every package's behalf.

This is where any package -- including one that isn't this one -- registers its own
device twins, binding an EOS (lab_name, device_name) directly to a real Matterix asset
config class. To see what classes are available to bind, call
`matterix_asset_types.discover_matterix_assets()` (post-boot) -- it introspects Matterix's
own robots/equipment/labware/infrastructure packages rather than requiring you to guess
or hunt for class names; it's a lookup tool, not something this file needs to call itself.

What follows is this package's own example: a placeholder Franka arm, not a real device.
"""

from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

from user.matterix_bridge.common.device_registry import register_device_twin

register_device_twin("example_lab", "franka_01", FRANKA_PANDA_HIGH_PD_IK_CFG)
