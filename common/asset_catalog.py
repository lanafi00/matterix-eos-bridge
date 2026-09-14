"""Catalog of every Matterix asset (robot, labware, equipment, ...) available to any EOS
package, keyed by a plain string path (e.g. "robots/franka_panda_high_pd_ik",
"labware/beaker_500ml") -- not by which EOS device/lab happens to be using it right now.

This is the single source of truth for "what does this asset name actually mean."
`device_registry.py`'s `DEVICE_TWINS` builds on top of this (a lab device now references
a catalog entry by name, instead of each device registration separately importing and
naming the Python class itself) -- and a future layout/scene-description system (letting
someone place named catalog assets into a scene without writing Python) would too, for
the same reason: multiple different things might need to reference "this same physical
asset," and they should all point at one registration, not each carry their own copy of
the class mapping.

Populated via `register_catalog_asset()` the same way every other registry in this
package is populated -- call it at module level in your own package's root-level
`matterix_devices.py` (needs real `matterix_assets` classes, so only safe to do after
Isaac Sim has booted -- see `runtime.py`'s `_discover_device_registrations()`, which
auto-imports that file once Isaac Sim is running). Not `scenes/__init__.py`: which
assets/devices exist shouldn't be coupled to which scene file happens to import them --
see `device_registry.py`'s module docstring.
"""

from __future__ import annotations

from typing import Callable

from matterix_assets.matterix_articulation import MatterixArticulationCfg

ASSET_CATALOG: dict[str, Callable[..., MatterixArticulationCfg]] = {}


def validate_asset_cfg(cfg_class: Callable[..., MatterixArticulationCfg]) -> None:
    """Confirm `cfg_class` actually produces a real Matterix asset -- a
    `MatterixArticulationCfg` backed by a USD file that actually exists -- instead of
    accepting any callable at registration time and only discovering it's broken much
    later, when something tries to resolve and use it.

    Instantiates `cfg_class` with placeholder pos/rot to inspect the result -- safe to do
    here: this only ever runs from `register_catalog_asset()`, called from a package's
    `scenes/__init__.py` (itself only imported after Isaac Sim has booted), and building
    a `@configclass` instance is plain Python/dataclass construction with no GPU or
    simulation side effects -- the same thing `resolve_device_twin()` does for real later
    with the caller-supplied pos/rot.

    Existence is checked via `isaaclab.utils.assets.check_file_path()`, NOT a bare
    `os.path.isfile()` -- VERIFIED this actually matters: `matterix_assets`' own
    `FRANKA_PANDA_HIGH_PD_IK_CFG` (a real, working asset) has a `usd_path` that's a plain
    `https://` URL onto NVIDIA's Nucleus/CDN asset server, not a local file at all. A
    local-only check would reject that asset outright. `check_file_path()` is Isaac Lab's
    own utility for exactly this (local-or-remote) distinction; it returns 0 if the path
    doesn't exist anywhere, 1 if it's a real local file, 2 if it resolves on the Nucleus
    server (checked via a real `omni.client.stat()` call, so this does real I/O --
    registering a remote-backed asset costs a network round-trip, once, at registration
    time).

    Raises TypeError if `cfg_class(...)` can't be instantiated at all, or doesn't produce
    a `MatterixArticulationCfg`. Raises ValueError if it does but has no `spawn.usd_path`
    (not backed by a real Matterix USD asset at all), or that path resolves to neither a
    local file nor a Nucleus asset (a typo, or a moved/deleted asset) -- naming the exact
    problem, not just "invalid asset", so it's obvious from the error alone what to fix.
    """
    try:
        instance = cfg_class(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    except Exception as e:
        raise TypeError(f"{cfg_class!r} could not be instantiated as a catalog asset: {e}") from e

    if not isinstance(instance, MatterixArticulationCfg):
        raise TypeError(f"{cfg_class!r} does not produce a MatterixArticulationCfg (got {type(instance)!r}).")

    usd_path = getattr(instance.spawn, "usd_path", None)
    if usd_path is None:
        raise ValueError(
            f"{cfg_class!r} has no spawn.usd_path -- it isn't backed by a real Matterix "
            f"USD asset (spawn={instance.spawn!r})."
        )

    from isaaclab.utils.assets import check_file_path

    if check_file_path(usd_path) == 0:
        raise ValueError(f"{cfg_class!r}'s USD file does not exist locally or on the Nucleus server: {usd_path!r}")


def register_catalog_asset(
    catalog_path: str,
    cfg_class: Callable[..., MatterixArticulationCfg],
    *,
    overwrite: bool = False,
) -> None:
    """Register a Matterix asset config under a plain catalog path (e.g.
    "robots/franka_panda_high_pd_ik"), so it can be referenced by name from anywhere --
    a device twin binding today, a future layout/scene-description entry -- instead of
    every consumer importing and naming the Python class directly.

    Raises ValueError if `catalog_path` is already bound to a *different* class and
    `overwrite` isn't set -- two packages silently fighting over the same catalog path
    should fail loudly, not resolve to whichever one happened to import last.
    Re-registering the same (catalog_path, cfg_class) pair is always a no-op.

    Also raises (TypeError/ValueError, via `validate_asset_cfg()`) if `cfg_class` doesn't
    actually produce a real, USD-backed Matterix asset -- fail loudly here, at
    registration time, rather than accepting anything callable and only discovering a
    typo'd path or wrong class much later when something tries to use it.
    """
    validate_asset_cfg(cfg_class)

    existing = ASSET_CATALOG.get(catalog_path)
    if existing is not None and existing is not cfg_class and not overwrite:
        raise ValueError(
            f"Catalog path {catalog_path!r} is already registered to {existing!r} (tried "
            f"to register {cfg_class!r}). Pass overwrite=True if this is intentional."
        )
    ASSET_CATALOG[catalog_path] = cfg_class


def resolve_catalog_asset(catalog_path: str) -> Callable[..., MatterixArticulationCfg]:
    """Look up the Matterix asset config class registered under `catalog_path`.

    Raises KeyError, naming the missing path, if nothing is registered under it --
    fail loudly rather than let a typo'd reference resolve to nothing usable.
    """
    if catalog_path not in ASSET_CATALOG:
        raise KeyError(
            f"No asset registered under catalog path {catalog_path!r}. Call "
            "register_catalog_asset() for it (see matterix_bridge/common/asset_catalog.py)."
        )
    return ASSET_CATALOG[catalog_path]
