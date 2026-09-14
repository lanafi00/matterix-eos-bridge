"""Shared utilities for working with real Matterix asset config classes -- what counts as
one, whether a given class actually produces a working one, and what Matterix currently
ships. No registry lives here: `device_registry.py`'s `DEVICE_TWINS` stores the classes
directly (collapsed back from an earlier two-layer catalog-path-string design -- with
only 11 assets total in Matterix and exactly one device binding in this repo, naming
assets independently of any device binding wasn't earning its cost; a real name-based
catalog is worth rebuilding later, shaped by whatever a future layout/scene-description
system actually needs to look up by, not guessed at now).
"""

from __future__ import annotations

from typing import Union

from matterix_assets.matterix_articulation import MatterixArticulationCfg
from matterix_assets.matterix_rigid_object import MatterixRigidObjectCfg
from matterix_assets.matterix_static_object import MatterixStaticObjectCfg

# The three base config types every real Matterix asset is built on -- an articulated
# object (has joints/actuators, e.g. a robot arm), a rigid object (can be picked up/moved,
# e.g. a beaker), or a static object (fixed scenery, e.g. a table). All three share the
# same pos/rot constructor kwargs and end up with a `spawn.usd_path` after construction
# (robots set `spawn` directly; rigid/static objects build it in `__post_init__` from
# their own `usd_path` field) -- VERIFIED by reading matterix_articulation.py,
# matterix_rigid_object.py, and matterix_static_object.py directly, so `validate_asset_cfg`
# below can treat all three uniformly.
MatterixAssetCfg = Union[MatterixArticulationCfg, MatterixRigidObjectCfg, MatterixStaticObjectCfg]
_ASSET_BASE_TYPES = (MatterixArticulationCfg, MatterixRigidObjectCfg, MatterixStaticObjectCfg)


def validate_asset_cfg(cfg_class) -> None:
    """Confirm `cfg_class` actually produces a real Matterix asset -- an articulated,
    rigid, or static object config backed by a USD file that actually exists -- instead
    of accepting any callable and only discovering it's broken much later, when something
    tries to resolve and use it.

    Instantiates `cfg_class` with placeholder pos/rot to inspect the result -- safe to do
    here: this only ever runs from `device_registry.py`'s `register_device_twin()`,
    called from a package's `matterix_devices.py` (itself only imported after Isaac Sim
    has booted), and building a `@configclass` instance is plain Python/dataclass
    construction with no GPU or simulation side effects -- the same thing
    `resolve_device_twin()` does for real later with the caller-supplied pos/rot.

    Existence is checked via `isaaclab.utils.assets.check_file_path()`, NOT a bare
    `os.path.isfile()` -- VERIFIED this actually matters: `matterix_assets`' own
    `FRANKA_PANDA_HIGH_PD_IK_CFG` (a real, working asset) has a `usd_path` that's a plain
    `https://` URL onto NVIDIA's Nucleus/CDN asset server, not a local file at all. A
    local-only check would reject that asset outright. `check_file_path()` is Isaac Lab's
    own utility for exactly this (local-or-remote) distinction; it returns 0 if the path
    doesn't exist anywhere, 1 if it's a real local file, 2 if it resolves on the Nucleus
    server (checked via a real `omni.client.stat()` call, so this does real I/O --
    validating a remote-backed asset costs a network round-trip, once, at registration
    time).

    Raises TypeError if `cfg_class(...)` can't be instantiated at all, or doesn't produce
    one of the three Matterix asset base types. Raises ValueError if it does but has no
    `spawn.usd_path` (not backed by a real Matterix USD asset at all), or that path
    resolves to neither a local file nor a Nucleus asset (a typo, or a moved/deleted
    asset) -- naming the exact problem, not just "invalid asset", so it's obvious from the
    error alone what to fix.
    """
    try:
        instance = cfg_class(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    except Exception as e:
        raise TypeError(f"{cfg_class!r} could not be instantiated as a Matterix asset: {e}") from e

    if not isinstance(instance, _ASSET_BASE_TYPES):
        raise TypeError(
            f"{cfg_class!r} does not produce a Matterix asset config "
            f"(expected one of {[t.__name__ for t in _ASSET_BASE_TYPES]}, got {type(instance)!r})."
        )

    usd_path = getattr(instance.spawn, "usd_path", None)
    if usd_path is None:
        raise ValueError(
            f"{cfg_class!r} has no spawn.usd_path -- it isn't backed by a real Matterix "
            f"USD asset (spawn={instance.spawn!r})."
        )

    from isaaclab.utils.assets import check_file_path

    if check_file_path(usd_path) == 0:
        raise ValueError(f"{cfg_class!r}'s USD file does not exist locally or on the Nucleus server: {usd_path!r}")


# Matterix's own asset categories, one submodule each under `matterix_assets` -- see
# `discover_matterix_assets()` below. Not introspected from `matterix_assets` itself
# (e.g. `pkgutil.iter_modules`) because that package's top-level namespace star-imports
# every category into one flat set of names, losing which category each came from; the
# category is exactly what becomes each result's readable-name prefix.
_MATTERIX_ASSET_CATEGORIES = ("robots", "equipment", "labware", "infrastructure")


def _readable_name_from_class_name(class_name: str) -> str:
    """"FRANKA_PANDA_HIGH_PD_IK_CFG" -> "franka_panda_high_pd_ik"; "TABLE_SEATTLE_INST_Cfg"
    -> "table_seattle_inst". Strips a trailing _CFG/_Cfg/_cfg suffix (Matterix's own
    naming convention for these classes, VERIFIED against every class currently in
    matterix_assets/{robots,equipment,labware,infrastructure}), then lowercases.
    """
    import re

    return re.sub(r"_(CFG|Cfg|cfg)$", "", class_name).lower()


def discover_matterix_assets() -> dict[str, type]:
    """Introspect `matterix_assets` and return every real, concrete asset config class
    Matterix currently ships, keyed by a readable "category/name" string (e.g.
    "robots/franka_panda_high_pd_ik") for display purposes only -- this is a pure lookup
    tool for finding a class to pass to `register_device_twin()` directly, not a registry;
    nothing here gets stored anywhere.

    Answers "what's actually available" directly from Matterix's own source of truth
    instead of guessing class names or hand-maintaining a list that goes stale as Matterix
    adds assets -- walks each of `_MATTERIX_ASSET_CATEGORIES`' submodules
    (`matterix_assets.robots`, `.equipment`, `.labware`, `.infrastructure`) for classes
    that subclass one of the three Matterix asset base types (see `_ASSET_BASE_TYPES`
    above) but aren't one of those base types themselves.

    Two different real classes deriving the same readable name (a collision after
    lowercasing/suffix-stripping) is treated as a bug worth surfacing loudly, not silently
    picking one -- raises ValueError naming both classes.

    Must only be called after Isaac Sim has booted (this imports `matterix_assets`, which
    needs `isaaclab` already initialized).
    """
    import importlib
    import inspect

    discovered: dict[str, type] = {}
    for category in _MATTERIX_ASSET_CATEGORIES:
        submodule = importlib.import_module(f"matterix_assets.{category}")
        for attr_name in dir(submodule):
            obj = getattr(submodule, attr_name)
            if not inspect.isclass(obj):
                continue
            if obj in _ASSET_BASE_TYPES or not issubclass(obj, _ASSET_BASE_TYPES):
                continue

            readable_name = f"{category}/{_readable_name_from_class_name(attr_name)}"
            existing = discovered.get(readable_name)
            if existing is not None and existing is not obj:
                raise ValueError(
                    f"Both {existing!r} and {obj!r} derive the same readable name "
                    f"{readable_name!r} -- rename one, or handle the collision explicitly "
                    "instead of calling discover_matterix_assets() blindly."
                )
            discovered[readable_name] = obj

    return discovered
