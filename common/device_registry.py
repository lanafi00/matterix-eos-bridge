"""Maps EOS device identities to the Matterix twin asset config that backs them.

Keyed exactly the way EOS names a device - `(lab_name, device_name)` - so a device
assignment coming out of an EOS `ScheduledTask.devices` resolves to a concrete twin
config with a single dict lookup. This is *twin substitution*: letting one already-
authored scene's actuated slot (e.g. "robot" in `articulated_assets`) be filled by
different concrete hardware depending on which lab/device is calling, without writing
a new scene per physical unit. Most new devices don't need this at all - only ones that
occupy an articulated-asset slot with more than one possible physical implementation
across labs/deployments. See `resolve_device_twin()`'s docstring and `run_workflow()`'s
`devices=` parameter in runtime.py.

Populated via `register_device_twin()` rather than edited as literal dict entries, so a
package other than this one can register its own lab's device -> twin bindings from its
own code instead of editing this file directly - call it from your package's top-level
`scenes/__init__.py` (the same file where you `gym.register()` your scene, and the same
file `protocol_registry.py`'s docstring points to for this). Not a package-root
`__init__.py` - EOS packages here are plain namespace packages with none of those (see
this repo's own `user/matterix_bridge`, or EOS's own `user/example`). `scenes/__init__.py`
is the right spot because it's auto-imported by runtime.py's `_discover_scene_modules()`
after Isaac Sim has booted, which twin config classes need (they come from
`matterix_assets`, an isaaclab-dependent package) - see `protocol_registry.py`'s
docstring for why that's also why this can't go in `matterix_registrations.py`.
"""

from __future__ import annotations

from typing import Callable

from matterix_assets.matterix_articulation import MatterixArticulationCfg

DEVICE_TWINS: dict[tuple[str, str], Callable[..., MatterixArticulationCfg]] = {}


def register_device_twin(
    lab_name: str,
    device_name: str,
    twin_cfg: Callable[..., MatterixArticulationCfg],
    *,
    overwrite: bool = False,
) -> None:
    """Bind an EOS `(lab_name, device_name)` device to the Matterix twin config class
    that represents it in an articulated-asset slot (see this module's docstring for
    when you do - and don't - need this).

    Raises ValueError if the pair is already bound to a *different* twin and
    `overwrite` isn't set. Re-registering the same (key, twin_cfg) pair is a no-op.
    """
    key = (lab_name, device_name)
    existing = DEVICE_TWINS.get(key)
    if existing is not None and existing is not twin_cfg and not overwrite:
        raise ValueError(
            f"Device {key!r} is already registered to twin {existing!r} (tried to "
            f"register {twin_cfg!r}). Pass overwrite=True if this is intentional."
        )
    DEVICE_TWINS[key] = twin_cfg


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
            f"No twin registered for device {key!r}. Call register_device_twin() for "
            "it (see matterix_bridge/common/device_registry.py)."
        )
    return DEVICE_TWINS[key](pos=existing.pos, rot=existing.rot)
