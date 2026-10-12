"""Package-local assets: labware/equipment an EOS package ships itself, as USD files plus
an `assets/assets.yml` describing each one -- so a lab's own vials, racks, holders etc. need
no change to Matterix (no new class in `matterix_assets`, no PR to its USD data repo).

Layout (inside the EOS package, next to `labs/` and `protocols/`):

    <package>/assets/assets.yml
    <package>/assets/usd/<whatever>.usda

`assets.yml` maps an asset name to its definition (see `PackageAssetDef` for every field):

    vial_20ml:
      kind: rigid
      usd: usd/vial_20ml.usda          # relative to assets/
      mass: 0.03
      frames:
        pre_grasp: [0.0, 0.0, 0.10]
        grasp: [0.0, 0.0, 0.045]
        post_grasp: [0.0, 0.0, 0.10]

A device/resource in `lab.yml` uses one by setting `meta.matterix_catalog` to
`<package>/<asset name>` (e.g. `ncsu_lab/vial_20ml`), exactly like a built-in catalog key
(`labware/beaker_500ml_inst`). `<package>` is the package's directory name. from_eos.py
loads this file, `models.py` validates package keys the same way as catalog.json keys
(kind, frames needed by pick/place), and `compiler.py` emits a small config class per
used asset straight into the generated scene file -- nothing has to be importable from
`matterix_assets`.

USD requirements (the spawner configures physics on the file, it doesn't create it -- see
Matterix's MatterixRigidObjectCfg/MatterixStaticObjectCfg `__post_init__`):
  - `rigid`: the default prim carries `PhysicsRigidBodyAPI` + `PhysicsMassAPI`, with
    collider geometry (`PhysicsCollisionAPI`) underneath it.
  - `static`: colliders only, no rigid body.
  - Z-up, metres, and no transform on the default prim itself -- isaaclab overwrites the
    spawned root's xform ops with the scene's `pos`/`rot`, so a rotation authored there
    (as some upstream Matterix USDs do to convert Y-up) is silently dropped.
  - Frame offsets are in the asset's own frame, relative to the default prim's origin.

Pure YAML + pydantic -- no isaaclab import, so this runs in EOS's plain venv like the rest
of schema/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ASSETS_DIRNAME = "assets"
ASSETS_FILENAME = "assets.yml"

# Built-in catalog.json key prefixes (matterix_assets' own category packages). A package
# named one of these would make `<package>/<asset>` keys ambiguous with built-in ones.
_RESERVED_NAMESPACES = {"robots", "equipment", "labware", "infrastructure"}


class PackageAssetDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["rigid", "static"]
    """`rigid` -> a MatterixRigidObjectCfg (simulated body; can be picked, carry frames,
    carry semantics like temperature/contact). `static` -> a MatterixStaticObjectCfg
    (immovable furniture; no frames -- isaaclab frame sensors need a rigid body)."""

    usd: str
    """Path to the USD file, relative to the package's `assets/` directory."""

    mass: float | None = Field(default=None, gt=0)
    """kg. Required for `rigid` (it sets the body's mass and is what heat-transfer
    semantics divide by); optional for `static` (semantics-only there)."""

    scale: tuple[float, float, float] | None = None

    frames: dict[str, tuple[float, float, float]] = Field(default_factory=dict)
    """Named offsets (m) in the asset's own frame. pick_object needs pre_grasp/grasp/
    post_grasp on what it picks (where the fingertip centre goes); place_object needs
    pre_place/place on its target (where the fingertip centre goes on release -- so a
    target's `place` height = resting height of the placed object's origin + that
    object's own `grasp` height)."""

    usd_abspath: Path | None = Field(default=None, exclude=True)
    """Resolved by `load_package_assets()`; not part of assets.yml."""

    @model_validator(mode="after")
    def _check_kind(self) -> "PackageAssetDef":
        if self.kind == "rigid" and self.mass is None:
            raise ValueError("kind 'rigid' needs a mass (kg)")
        if self.kind == "static" and self.frames:
            raise ValueError(
                "kind 'static' can't carry frames -- isaaclab frame sensors only work on rigid "
                "bodies. Make it 'rigid' (a heavy rigid body is fine for a place target)."
            )
        return self

    def catalog_entry(self) -> dict:
        """The same shape as a catalog.json "assets" entry, so models.py's validators
        treat package and built-in assets identically."""
        return {
            "kind": "object",
            "package_asset": True,
            "fields": {"frames": {"default": {name: list(offset) for name, offset in self.frames.items()}}},
        }


def load_package_assets(package_dir: Path) -> dict[str, PackageAssetDef]:
    """`{"<package>/<asset>": PackageAssetDef}` for `<package_dir>/assets/assets.yml`, or
    `{}` if the package has no such file. Raises ValueError naming the asset and the
    problem for anything malformed, including a USD path that doesn't exist."""
    package_dir = Path(package_dir).resolve()
    assets_dir = package_dir / ASSETS_DIRNAME
    assets_yml = assets_dir / ASSETS_FILENAME
    if not assets_yml.is_file():
        return {}

    namespace = package_dir.name
    if namespace in _RESERVED_NAMESPACES:
        raise ValueError(
            f"{assets_yml}: package directory name {namespace!r} collides with a built-in "
            f"catalog category ({sorted(_RESERVED_NAMESPACES)}) -- rename the package."
        )

    raw = yaml.safe_load(assets_yml.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{assets_yml}: expected a mapping of asset name -> definition")

    assets: dict[str, PackageAssetDef] = {}
    for name, body in raw.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"{assets_yml}: asset name {name!r} must be a valid identifier")
        try:
            asset = PackageAssetDef(**(body or {}))
        except (TypeError, ValidationError) as e:
            raise ValueError(f"{assets_yml}: asset {name!r}: {e}") from e
        usd_abspath = (assets_dir / asset.usd).resolve()
        if not usd_abspath.is_file():
            raise ValueError(f"{assets_yml}: asset {name!r}: usd file {usd_abspath} not found")
        asset.usd_abspath = usd_abspath
        assets[f"{namespace}/{name}"] = asset
    return assets
