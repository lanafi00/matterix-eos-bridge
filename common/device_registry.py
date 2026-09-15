"""Maps EOS device identities to the Matterix asset config class that backs them.

Keyed exactly the way EOS names a device - `(lab_name, device_name)` - so a device
assignment coming out of an EOS `ScheduledTask.devices` resolves to a concrete twin
config with a single dict lookup. This is *twin substitution*: letting one already-
authored scene's slot (e.g. "robot" in `articulated_assets`, or "beaker"/"ika_plate" in
`objects` - a scene's only two asset containers, see runtime.py's `_get_env()`) be filled
by different concrete hardware depending on which lab/device is calling, without writing
a new scene per physical unit. Not robot-only: this registry and `validate_asset_cfg()`
(matterix_asset_types.py) are already type-agnostic across all three Matterix asset base
types, and `_get_env()`'s substitution loop checks both `articulated_assets` and `objects`
for the slot - so the same registry covers a lab's specific hot plate or beaker, not just
its robot. Most new devices don't need this at all - only ones that occupy a scene slot
with more than one possible physical implementation across labs/deployments. See
`resolve_device_twin()`'s docstring and `run_workflow()`'s `devices=` parameter in
runtime.py.

This registry stores the real Matterix asset config CLASS directly, not an indirection
through a separate name->class catalog - there used to be one (`asset_catalog.py`), but
with only 11 assets total in Matterix and exactly one device binding in this repo, naming
assets independently of any device binding wasn't earning its cost, so it was collapsed
back to this single dict. `matterix_asset_types.py`'s `discover_matterix_assets()` is
still the right way to find what class to pass here (see its docstring) - it just returns
names for humans to read, not something this registry looks anything up by. A real
name-based catalog is worth rebuilding later, shaped by whatever a future layout/scene-
description system actually needs to look up by, not guessed at now.

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
after Isaac Sim has booted (needed to validate real `matterix_assets` classes, via
`matterix_asset_types.py`'s `validate_asset_cfg()`) - see that function's docstring for
why it's called after scene discovery specifically (a matterix_assets-internal circular
import, not a design choice here), and `protocol_registry.py`'s docstring for why this
can't go in the pre-boot `matterix_registrations.py`.
"""

from __future__ import annotations

from typing import Callable

from user.matterix_bridge.common.matterix_asset_types import MatterixAssetCfg, validate_asset_cfg

DEVICE_TWINS: dict[tuple[str, str], Callable[..., MatterixAssetCfg]] = {}


def register_device_twin(
    lab_name: str,
    device_name: str,
    cfg_class: Callable[..., MatterixAssetCfg],
    *,
    overwrite: bool = False,
) -> None:
    """Bind an EOS `(lab_name, device_name)` device directly to a Matterix asset config
    class (see this module's docstring for when you do - and don't - need this, and
    `matterix_asset_types.py`'s `discover_matterix_assets()` for finding available classes).

    Raises ValueError if the pair is already bound to a *different* class and `overwrite`
    isn't set. Re-registering the same (key, cfg_class) pair is a no-op.

    Also raises (TypeError/ValueError, via `validate_asset_cfg()`) if `cfg_class` doesn't
    actually produce a real, USD-backed Matterix asset -- fail loudly here, at
    registration time, rather than accepting anything callable and only discovering a
    broken class much later when a protocol run tries to use it.
    """
    validate_asset_cfg(cfg_class)

    key = (lab_name, device_name)
    existing = DEVICE_TWINS.get(key)
    if existing is not None and existing is not cfg_class and not overwrite:
        raise ValueError(
            f"Device {key!r} is already registered to {existing!r} (tried to register "
            f"{cfg_class!r}). Pass overwrite=True if this is intentional."
        )
    DEVICE_TWINS[key] = cfg_class


def resolve_device_twin(
    lab_name: str, device_name: str, existing: MatterixAssetCfg
) -> MatterixAssetCfg:
    """Build the twin config for `(lab_name, device_name)`, keeping `existing`'s placement
    AND scene role.

    `existing` is the slot's current config from the env cfg being overridden. Two things
    carry over from it to the twin: `pos`/`rot` (where this slot sits in the scene) and
    `semantics` (what simulated behavior this slot is expected to have - e.g. a heater's
    `HeaterCfg`, or a beaker's contact/heat-transfer semantics). Both describe the SLOT - a
    property of the scene, authored once by whoever built it - not the specific physical
    device filling it, so a twin swap shouldn't change them. Physical-device-intrinsic
    fields (usd_path/spawn, mass, scale, frames - e.g. grasp-point offsets, which depend on
    that specific object's own geometry) intentionally come from the twin's own class
    defaults instead, since a different physical device really can have a different
    mass/shape/grasp geometry - only `cfg_class(pos=..., rot=..., semantics=...)` is passed
    here, nothing else from `existing`.

    VERIFIED this matters, not just theoretical: this repo's own heater_transfer scene
    attaches `semantics=HeaterCfg(...)` directly on its "ika_plate" object (and
    `HeatTransferCfg(...)` on "robot" and "beaker") - substituting any of those slots
    without carrying `semantics` over would silently strip the exact behavior that makes
    "turn_on_heater" work, with no error raised anywhere - just an environment that builds
    fine and an observation manager that behaves as if nothing is happening.
    """
    key = (lab_name, device_name)
    if key not in DEVICE_TWINS:
        raise KeyError(
            f"No twin registered for device {key!r}. Call register_device_twin() for "
            "it (see matterix_bridge/common/device_registry.py)."
        )
    twin_cfg = DEVICE_TWINS[key]
    return twin_cfg(pos=existing.pos, rot=existing.rot, semantics=existing.semantics)
