"""Maps EOS device identities to the Matterix catalog asset that backs them.

Keyed exactly the way EOS names a device - `(lab_name, device_name)` - so a device
assignment coming out of an EOS `ScheduledTask.devices` resolves to a concrete twin
config with a single dict lookup. This is *twin substitution*: letting one already-
authored scene's actuated slot (e.g. "robot" in `articulated_assets`) be filled by
different concrete hardware depending on which lab/device is calling, without writing
a new scene per physical unit. Most new devices don't need this at all - only ones that
occupy an articulated-asset slot with more than one possible physical implementation
across labs/deployments. See `resolve_device_twin()`'s docstring and `run_workflow()`'s
`devices=` parameter in runtime.py.

This registry stores plain `asset_catalog.py` catalog-path STRINGS, not Python classes --
the class itself, and the validation that it's a real USD-backed asset, live exactly once
in `asset_catalog.py`. A device binding here is just "this lab's this device -> that
catalog entry", so registering the same physical robot for five different labs doesn't
mean re-validating (or re-typo-ing) the same class five times.

Populated via `register_device_twin()` rather than edited as literal dict entries, so a
package other than this one can register its own lab's device -> twin bindings from its
own code instead of editing this file directly - call it from your package's root-level
`matterix_devices.py` (a plain file, sibling of `pyproject.toml`, `labs/`, `devices/`,
etc. - see this repo's own `matterix_devices.py` for the exact pattern). Deliberately
NOT `scenes/__init__.py`: a scene's slots get filled by a device twin at workflow-run
time (see `resolve_device_twin()` below and `run_workflow()`'s `devices=` parameter in
runtime.py) - the device list shouldn't be owned by whichever scene file happens to
import it, especially with a future layout/scene-description system in mind, where
devices are registered once and then placed into any number of scenes.
`matterix_devices.py` is auto-imported by runtime.py's `_discover_device_registrations()`
after Isaac Sim has booted (needed to resolve catalog entries, which go through
`asset_catalog.py`'s `matterix_assets` classes) - see that function's docstring for why
it's called after scene discovery specifically (a matterix_assets-internal circular
import, not a design choice here), and `protocol_registry.py`'s docstring for why this
can't go in the pre-boot
`matterix_registrations.py`.
"""

from __future__ import annotations

from user.matterix_bridge.common.asset_catalog import MatterixArticulationCfg, resolve_catalog_asset

DEVICE_TWINS: dict[tuple[str, str], str] = {}


def register_device_twin(
    lab_name: str,
    device_name: str,
    catalog_path: str,
    *,
    overwrite: bool = False,
) -> None:
    """Bind an EOS `(lab_name, device_name)` device to a catalog asset (see this module's
    docstring for when you do - and don't - need this, and `asset_catalog.py` for how to
    register the catalog entry itself first).

    Raises ValueError if the pair is already bound to a *different* catalog path and
    `overwrite` isn't set. Re-registering the same (key, catalog_path) pair is a no-op.

    Also raises KeyError (via `resolve_catalog_asset()`) if `catalog_path` isn't actually
    registered in the catalog -- fail loudly here, at device-registration time, rather
    than accepting a typo'd reference and only discovering it much later when a protocol
    run tries to use it.
    """
    resolve_catalog_asset(catalog_path)  # fail loudly now if catalog_path doesn't exist

    key = (lab_name, device_name)
    existing = DEVICE_TWINS.get(key)
    if existing is not None and existing != catalog_path and not overwrite:
        raise ValueError(
            f"Device {key!r} is already registered to catalog asset {existing!r} (tried "
            f"to register {catalog_path!r}). Pass overwrite=True if this is intentional."
        )
    DEVICE_TWINS[key] = catalog_path


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
    twin_cfg = resolve_catalog_asset(DEVICE_TWINS[key])
    return twin_cfg(pos=existing.pos, rot=existing.rot)
