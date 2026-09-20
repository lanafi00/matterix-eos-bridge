"""Offline catalog export: dump every asset/action/semantics class Matterix currently
ships, keyed by a readable name, with each one's constructor field signature, to a single
JSON file (default: catalog.json in this repo's root).

Why this exists: a future declarative scene schema (asset catalog entry, semantics preset
+ params, workflow step sequence -> compiles to a MatterixBaseEnvCfg subclass) needs to
validate a schema author's asset/action/semantics *names* and *required fields* without
booting Isaac Sim on every validation -- that boot costs 1-2 minutes (see this repo's
CLAUDE.md). This script pays that cost ONCE, offline, and writes the result to a file the
schema/compiler can load and check against instantly. Re-run it whenever Matterix's own
asset/action/semantics classes change (e.g. a Matterix version bump) to refresh catalog.json.

Run inside the eos-isaaclab conda env, same as smoke_test.py:

    EOS_PATH=/home/lila/Documents/git/eos \
        /home/lila/miniconda3/envs/eos-isaaclab/bin/python3 dump_catalog.py [-o OUTPUT]

Four vocabularies are covered, one per top-level catalog.json key:

- "assets": every real, USD-backed Matterix asset class (robots/equipment/labware/
  infrastructure) -- reuses common/matterix_asset_types.py's discover_matterix_assets(),
  already verified by smoke_test.py's Stage 6, rather than re-deriving this. Each entry
  also records which of the two env_cfg containers it belongs in ("articulated" ->
  articulated_assets, "object" -> objects) -- see runtime.py's _get_env() docstring for
  why a scene only has these two.
- "actions": every matterix_sm workflow-step config (PickObjectCfg, WaitCfg,
  TurnOnHeaterCfg, ...) -- walks matterix_sm's own __all__ (its public export list) and
  keeps subclasses of ActionBaseCfg, excluding the abstract base classes themselves
  (ActionBaseCfg/PrimitiveActionCfg/CompositionalActionCfg).
- "semantic_presets": the user-facing composed semantics (HeaterCfg, HeatTransferCfg) --
  walks matterix.managers.semantics.semantic_presets' __all__, keeping subclasses of
  SemanticPreset (excluding SemanticPreset itself).
- "primitive_semantics": the lower-level semantics scenes currently attach directly
  (IsInContactPhysicsCfg, AmbientAirHeatConvectionCfg -- see exp3/exp4) -- walks every
  module under primitive_semantics/ for its own `__semantics_cfg__` name list.

Field discovery (each entry's "fields") is a two-path affair -- see
_fields_from_dataclass()/_fields_from_init() for why one path isn't enough: "assets",
"actions", and "primitive_semantics" are real @configclass dataclasses (dataclasses.
fields(), correctly resolving isaaclab's own default_factory wrapping); "semantic_presets"
are plain hand-written classes (inspect.signature() on __init__ instead -- dataclasses.
fields() doesn't apply to them at all).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import inspect
import json
import os
import re
import sys

EOS_PATH = os.getenv("EOS_PATH", "/home/lila/Documents/git/eos")
if EOS_PATH not in sys.path:
    sys.path.insert(0, EOS_PATH)


def _camel_to_snake(name: str) -> str:
    """"PickObjectCfg" -> "pick_object"; "AmbientAirHeatConvectionCfg" -> "ambient_air_heat_convection".

    Matterix's action/semantics classes use CamelCase+"Cfg" (unlike matterix_assets' own
    ALL_CAPS_CFG instance constants, which matterix_asset_types.py's
    _readable_name_from_class_name() already handles separately) -- insert an underscore
    before each internal capital, lowercase, then strip a trailing "_cfg".
    """
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"_cfg$", "", s)


def _json_safe(value):
    """Best-effort JSON-safe conversion of a constructor default value.

    Most defaults here are plain floats/strings/bools/None or pos/rot-style tuples
    (-> list). A default that isn't representable in JSON (another cfg instance, a
    Callable, an enum) is stringified via repr() instead of dropped, so the catalog still
    records that a default exists even when it can't be round-tripped -- a schema/compiler
    consumer should treat a "default_repr"-only entry as "has a default, but you must
    still supply this field explicitly from the schema" rather than silently omitting it.
    """
    try:
        json.dumps(value)
        return {"default": value}
    except TypeError:
        if isinstance(value, tuple):
            try:
                json.dumps(list(value))
                return {"default": list(value)}
            except TypeError:
                pass
        return {"default_repr": repr(value)}


def _fields_from_dataclass(cls) -> dict:
    """Every dataclass field of `cls` (a real @configclass, e.g. asset/action/primitive-
    semantics configs), keyed by name.

    `dataclasses.fields()`, not `inspect.signature()`, is what makes this correct:
    isaaclab's configclass._process_mutable_types() wraps EVERY field's default (not just
    mutable ones -- VERIFIED by reading configclass.py directly) in
    `field(default_factory=...)`, so `inspect.signature(cls.__init__)` shows every
    parameter's default as the same unhelpful `_HAS_DEFAULT_FACTORY` sentinel (its own
    repr is the literal string "<factory>") instead of the real value. The factory itself
    (`_return_f()` in configclass.py) is just `lambda: deepcopy(original_value)` -- a pure,
    side-effect-free copy of whatever literal the field was declared with -- so calling it
    here to recover that real value is safe.

    A field declared with no default at all (`x: float = MISSING`, this codebase's
    "required, fill in a real value" convention -- see isaaclab.utils.configclass's
    `_add_annotation_types`, which raises exactly on a bare MISSING) still goes through the
    same default_factory wrapping, and the factory faithfully returns a MISSING right back
    -- but NOT the same `dataclasses.MISSING` *singleton*: `_return_f()`'s factory body is
    `deepcopy(original_value)`, and `copy.deepcopy()` on `_MISSING_TYPE` (which defines no
    `__deepcopy__`/`__reduce__` of its own) makes a distinct instance -- VERIFIED live: an
    `is dataclasses.MISSING` check silently failed here, so this checks the TYPE instead
    (`isinstance(resolved, type(dataclasses.MISSING))`), which survives that deepcopy.
    """
    fields = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            resolved = f.default
        elif f.default_factory is not dataclasses.MISSING:  # -- always true in practice, see docstring
            resolved = f.default_factory()
        else:
            resolved = dataclasses.MISSING

        entry: dict = {"annotation": str(f.type)}
        if isinstance(resolved, type(dataclasses.MISSING)):
            entry["required"] = True
        else:
            entry["required"] = False
            entry.update(_json_safe(resolved))
        fields[f.name] = entry
    return fields


def _fields_from_init(cls) -> dict:
    """Every __init__ parameter of `cls` (except self/*args/**kwargs), keyed by name.

    For a plain class with a hand-written __init__ (semantic_presets -- see this module's
    docstring; NOT a @configclass/dataclass, so _fields_from_dataclass() doesn't apply and
    doesn't hit that function's default_factory wrapping at all -- these classes' __init__
    parameter defaults are plain literals like `None`/`False`, correctly visible via
    inspect.signature() directly).

    A field with no default at all is marked `"required": true` with no "default"/
    "default_repr" key.
    """
    if dataclasses.is_dataclass(cls):
        return _fields_from_dataclass(cls)

    fields = {}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return fields

    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        entry: dict = {}
        if param.annotation is not inspect.Parameter.empty:
            entry["annotation"] = str(param.annotation)

        if param.default is inspect.Parameter.empty or param.default is dataclasses.MISSING:
            entry["required"] = True
        else:
            entry["required"] = False
            entry.update(_json_safe(param.default))

        fields[name] = entry

    return fields


def _class_path(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _discover_assets() -> dict:
    from matterix_assets.matterix_articulation import MatterixArticulationCfg
    from matterix_assets.matterix_rigid_object import MatterixRigidObjectCfg
    from matterix_assets.matterix_static_object import MatterixStaticObjectCfg

    from user.matterix_bridge.common.matterix_asset_types import discover_matterix_assets

    catalog = {}
    for readable_name, cls in discover_matterix_assets().items():
        if issubclass(cls, MatterixArticulationCfg):
            kind = "articulated"
        elif issubclass(cls, MatterixRigidObjectCfg):
            kind = "object"
        elif issubclass(cls, MatterixStaticObjectCfg):
            kind = "object"
        else:
            kind = "unknown"  # shouldn't happen -- discover_matterix_assets() only returns the 3 base types
        catalog[readable_name] = {
            "class": _class_path(cls),
            "kind": kind,
            "fields": _fields_from_init(cls),
        }
    return catalog


def _discover_actions() -> dict:
    import matterix_sm
    from matterix_sm import ActionBaseCfg, CompositionalActionCfg, PrimitiveActionCfg

    abstract_bases = {ActionBaseCfg, PrimitiveActionCfg, CompositionalActionCfg}
    catalog = {}
    for attr_name in matterix_sm.__all__:
        obj = getattr(matterix_sm, attr_name)
        if not inspect.isclass(obj) or obj in abstract_bases:
            continue
        if not issubclass(obj, ActionBaseCfg):
            continue
        readable_name = _camel_to_snake(obj.__name__)
        existing = catalog.get(readable_name)
        if existing is not None and existing["class"] != _class_path(obj):
            raise ValueError(
                f"Both {existing['class']} and {_class_path(obj)} derive the same readable "
                f"name {readable_name!r} -- rename one, or handle the collision explicitly."
            )
        catalog[readable_name] = {
            "class": _class_path(obj),
            "fields": _fields_from_init(obj),
        }
    return catalog


def _discover_semantic_presets() -> dict:
    from matterix.managers.semantics import semantic_presets
    from matterix.managers.semantics.semantic_presets import SemanticPreset

    catalog = {}
    for attr_name in semantic_presets.__all__:
        obj = getattr(semantic_presets, attr_name)
        if not inspect.isclass(obj) or obj is SemanticPreset:
            continue
        if not issubclass(obj, SemanticPreset):
            continue
        readable_name = _camel_to_snake(obj.__name__)
        catalog[readable_name] = {
            "class": _class_path(obj),
            "fields": _fields_from_init(obj),
        }
    return catalog


def _discover_primitive_semantics() -> dict:
    """Walk every module under primitive_semantics/ that declares its own `__semantics_cfg__`
    name list, and getattr each name directly.

    Deliberately NOT primitive_semantics.semantics_dict, despite that looking like the
    obvious name->class map to use here (matterix_assets' own discover_matterix_assets()
    uses exactly this kind of pattern for a different package) -- VERIFIED live that
    semantics_dict only ever contains the non-Cfg *runtime* classes (Temperature,
    HeatSource, IsInContactPhysics, ...), never their Cfg counterparts: both
    primitive_semantics/__init__.py and its heat_transfer/__init__.py build their dict
    from `__semantics__` (the runtime-class list) only, never from `__semantics_cfg__`
    (the Cfg-class list) -- a real asymmetry in Matterix's own module structure, not
    something to special-case here. `__semantics_cfg__` itself, present on every module
    that has one, is unaffected by this and is what this function actually reads.
    """
    import pkgutil

    from matterix.managers.semantics import primitive_semantics
    from matterix.managers.semantics.semantics_cfg import (
        SemanticPredicateCfg,
        SemanticsCfg,
        SemanticStateCfg,
        SemanticStateTransitionCfg,
    )

    abstract_bases = {SemanticsCfg, SemanticStateCfg, SemanticStateTransitionCfg, SemanticPredicateCfg}
    modules = [primitive_semantics]
    for module_info in pkgutil.walk_packages(primitive_semantics.__path__, prefix=primitive_semantics.__name__ + "."):
        modules.append(__import__(module_info.name, fromlist=["_"]))

    catalog = {}
    for module in modules:
        for name in getattr(module, "__semantics_cfg__", []):
            obj = getattr(module, name, None)
            if not inspect.isclass(obj) or obj in abstract_bases or not issubclass(obj, SemanticsCfg):
                continue
            readable_name = _camel_to_snake(obj.__name__)
            existing = catalog.get(readable_name)
            if existing is not None and existing["class"] != _class_path(obj):
                raise ValueError(
                    f"Both {existing['class']} and {_class_path(obj)} derive the same readable "
                    f"name {readable_name!r} -- rename one, or handle the collision explicitly."
                )
            catalog[readable_name] = {
                "class": _class_path(obj),
                "fields": _fields_from_init(obj),
            }
    return catalog


def build_catalog() -> dict:
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "assets": _discover_assets(),
        "actions": _discover_actions(),
        "semantic_presets": _discover_semantic_presets(),
        "primitive_semantics": _discover_primitive_semantics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-o", "--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")
    )
    args = parser.parse_args()

    from user.matterix_bridge.common.runtime import ensure_app_launched

    print("[dump_catalog] booting Isaac Sim (headless) -- ~1-2 min on first boot...")
    ensure_app_launched(headless=True)

    # Warm up matterix_assets' own import graph the same way runtime.py's _get_env() does
    # (_discover_scene_modules() before _discover_device_registrations()) -- whichever
    # import touches matterix_assets FIRST in this process determines whether it succeeds;
    # importing this repo's own scenes package first is what's verified to work. Importing
    # matterix_assets.* directly as this process's first touch (e.g. inside
    # _discover_assets() below) hits the same circular import CLAUDE.md documents for
    # matterix_devices.py. See runtime.py's _get_env() for the full verified explanation.
    import user.matterix_bridge.scenes  # noqa: F401

    catalog = build_catalog()

    with open(args.output, "w") as f:
        json.dump(catalog, f, indent=2, sort_keys=True)
        f.write("\n")

    counts = {key: len(value) for key, value in catalog.items() if isinstance(value, dict)}
    print(f"[dump_catalog] wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
