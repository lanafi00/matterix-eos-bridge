"""CLI: generate an EOS package's Matterix scene(s) + matterix_registrations.py directly
from its own lab.yml + each relevant protocol's matterix_workflow.yml -- no bridge-owned
scene_specs/*.yaml needed. See README.md's "Using this from your own EOS package" for the
authoring convention this implements, and models.py's `WorkflowBindingSpec` docstring for
what a matterix_workflow.yml contains.

Placement convention (all per EOS package, alongside its own pyproject.toml):
  - `labs/<lab_name>/lab.yml` -- devices' and resources'/resource_types' `meta` dicts
    carry the sim-only fields this reads: `matterix_catalog` (required -- its absence is
    what marks a device/resource as having no sim twin at all, silently skipped), `pos`,
    `rot`, `mass`, `semantics`, `randomize_position`, `randomize_temperature`. Same
    fields AssetSpec already accepts; see its docstring for what each means.
  - `protocols/<protocol_type>/matterix_workflow.yml` -- one per protocol that has a sim
    twin (a protocol with none just has no such file, and is silently skipped). Parses
    as `WorkflowBindingSpec`.

Reuses EOS's own LabDef/ProtocolDef parsers (not a hand-rolled reimplementation), so this
stays in sync with whatever lab.yml/protocol.yml actually accept. Safe to do only because
this module is meant to run from EOS's own venv (same one `eos start` uses), same as
compile_scene.py -- it is NOT imported at runtime by runtime.py/protocol_registry.py,
which deliberately avoid eos.configuration.entities so they stay importable in a plain
isaaclab conda env with no `eos`/`bofire` installed at all (see protocol_registry.py's
`_discover_registrations()` docstring). from_eos.py is an offline, dev-time tool, exactly
like compile_scene.py -- it never runs inside a MatterixBackend actor.

Needs no isaaclab/matterix/Isaac Sim -- same "validate before the 1-2 minute Isaac Sim
boot" property as compile_scene.py.

Usage (from this repo's root, same convention as compile_scene.py):
    python -m schema.from_eos /path/to/eos/user/beaker_lab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from eos.configuration.entities.lab_def import LabDef
from eos.configuration.entities.protocol_def import ProtocolDef

from .compiler import compile_scene
from .models import AssetSpec, SceneSpec, WorkflowBindingSpec

_ARTICULATED = "articulated"
_OBJECT = "object"

# Fields AssetSpec accepts besides `kind`/`catalog`, read straight off a device/resource's
# merged `meta` dict when present.
_ASSET_META_FIELDS = ("pos", "rot", "mass", "semantics", "randomize_position", "randomize_temperature")


def _pascal(name: str) -> str:
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def _asset_from_meta(meta: dict[str, Any], *, kind: str, context: str) -> AssetSpec | None:
    """One AssetSpec from a device's or resource's merged `meta` dict, or None if it has
    no `matterix_catalog` -- most EOS devices/resources have no sim twin at all, and
    omitting the key (rather than requiring some other "skip this one" flag) is how a
    lab.yml author opts a device/resource out with zero extra syntax."""
    if "matterix_catalog" not in meta:
        return None
    kwargs: dict[str, Any] = {"kind": kind, "catalog": meta["matterix_catalog"]}
    for key in _ASSET_META_FIELDS:
        if key in meta:
            kwargs[key] = meta[key]
    try:
        return AssetSpec(**kwargs)
    except ValidationError as e:
        raise ValueError(f"{context}: invalid matterix meta -- {e}") from e


def _build_assets(lab: LabDef) -> dict[str, AssetSpec]:
    assets: dict[str, AssetSpec] = {}

    for device_name, device in lab.devices.items():
        asset = _asset_from_meta(device.meta, kind=_ARTICULATED, context=f"devices.{device_name}.meta")
        if asset is not None:
            assets[device_name] = asset

    for resource_name, resource in lab.resources.items():
        # Same merge order as eos/resources/resource_manager.py's own
        # `{**resource_type_meta, **lab_resource.meta}` -- a resource instance's own meta
        # (e.g. this beaker's pos) overrides its type's (e.g. every beaker's catalog key).
        resource_type = lab.resource_types.get(resource.type)
        merged_meta = {**(resource_type.meta if resource_type else {}), **resource.meta}
        asset = _asset_from_meta(merged_meta, kind=_OBJECT, context=f"resources.{resource_name}.meta")
        if asset is not None:
            assets[resource_name] = asset

    return assets


def _find_workflow_protocol_dirs(protocols_dir: Path, lab_name: str) -> list[Path]:
    """Every protocols/*/ directory whose protocol.yml lists `lab_name` under `labs:` AND
    has a sibling matterix_workflow.yml. A protocol with no sim twin just has no such
    file -- silently skipped, not an error."""
    if not protocols_dir.is_dir():
        return []
    found = []
    for protocol_yml in sorted(protocols_dir.glob("*/protocol.yml")):
        if not (protocol_yml.parent / "matterix_workflow.yml").is_file():
            continue
        protocol = ProtocolDef(**yaml.safe_load(protocol_yml.read_text()))
        if lab_name in protocol.labs:
            found.append(protocol_yml.parent)
    return found


def _load_workflow_binding(protocol_dir: Path) -> tuple[ProtocolDef, WorkflowBindingSpec]:
    protocol = ProtocolDef(**yaml.safe_load((protocol_dir / "protocol.yml").read_text()))
    binding = WorkflowBindingSpec(**yaml.safe_load((protocol_dir / "matterix_workflow.yml").read_text()))

    task_names = {t.name for t in protocol.tasks}
    unknown = sorted(set(binding.task_workflows) - task_names)
    if unknown:
        raise ValueError(
            f"{protocol_dir / 'matterix_workflow.yml'}: task_workflows names task(s) "
            f"{unknown} not declared in {protocol_dir / 'protocol.yml'}. "
            f"Known tasks: {sorted(task_names)}"
        )
    return protocol, binding


def compile_lab(
    lab_dir: Path, package_dir: Path, scenes_dir: Path
) -> tuple[SceneSpec, dict[str, str], list[tuple[str, str, str]]]:
    """Build + compile one lab's scene from its lab.yml, folding in every workflow
    binding contributed by the package's protocols that reference this lab. Returns the
    compiled spec plus the registration bindings the caller accumulates across labs:
    `{protocol_type: gym_id}` and `[(protocol_type, task_name, workflow_key), ...]`.
    """
    lab = LabDef(**yaml.safe_load((lab_dir / "lab.yml").read_text()))
    assets = _build_assets(lab)
    gym_id = f"Matterix-Lab-{_pascal(lab.name)}-v1"

    workflows: dict[str, Any] = {}
    bundles: dict[str, Any] = {}
    protocol_gym_ids: dict[str, str] = {}
    task_bindings: list[tuple[str, str, str]] = []

    for protocol_dir in _find_workflow_protocol_dirs(package_dir / "protocols", lab.name):
        protocol, binding = _load_workflow_binding(protocol_dir)

        for key, step in binding.workflows.items():
            if key in workflows and workflows[key] != step:
                raise ValueError(f"workflows key {key!r} redefined differently by {protocol_dir}")
            workflows[key] = step
        for key, steps in binding.bundles.items():
            if key in bundles and bundles[key] != steps:
                raise ValueError(f"bundles key {key!r} redefined differently by {protocol_dir}")
            bundles[key] = steps

        protocol_gym_ids[protocol.type] = gym_id
        for task_name, workflow_key in binding.task_workflows.items():
            task_bindings.append((protocol.type, task_name, workflow_key))

    spec = SceneSpec(name=lab.name, gym_id=gym_id, assets=assets, workflows=workflows, bundles=bundles)

    regen_hint = (
        f"Generated from {lab_dir / 'lab.yml'} (+ its protocols' matterix_workflow.yml) --\n"
        f"not a hand-authored scene_specs/*.yaml. Regenerate with:\n"
        f"    python -m schema.from_eos {package_dir}"
    )
    env_cfg_text, init_text = compile_scene(spec, regen_hint=regen_hint)
    out_dir = scenes_dir / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.name}_env_cfg.py").write_text(env_cfg_text)
    (out_dir / "__init__.py").write_text(init_text)

    return spec, protocol_gym_ids, task_bindings


def _render_registrations(protocol_gym_ids: dict[str, str], task_bindings: list[tuple[str, str, str]]) -> str:
    lines = [
        '"""Generated by `python -m schema.from_eos` -- DO NOT EDIT BY HAND.',
        "",
        "Regenerate after editing any labs/*/lab.yml or protocols/*/matterix_workflow.yml",
        "in this package (run from matterix_bridge's own repo root -- see schema/from_eos.py):",
        "    python -m schema.from_eos <this package's root>",
        '"""',
        "",
        "from user.matterix_bridge.common.protocol_registry import register_protocol, register_task_workflow",
        "",
    ]
    for protocol_type, gym_id in sorted(protocol_gym_ids.items()):
        lines.append(f"register_protocol({protocol_type!r}, {gym_id!r})")
    if task_bindings:
        lines.append("")
    for protocol_type, task_name, workflow_key in sorted(task_bindings):
        if workflow_key == task_name:
            lines.append(f"register_task_workflow({protocol_type!r}, {task_name!r})")
        else:
            lines.append(f"register_task_workflow({protocol_type!r}, {task_name!r}, {workflow_key!r})")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package_dir", help="Path to an EOS package root (sibling of its own pyproject.toml)")
    parser.add_argument("--scenes-dir", default=None, help="default: <package_dir>/scenes")
    parser.add_argument(
        "--registrations-out", default=None, help="default: <package_dir>/matterix_registrations.py"
    )
    args = parser.parse_args(argv)

    package_dir = Path(args.package_dir)
    scenes_dir = Path(args.scenes_dir) if args.scenes_dir else package_dir / "scenes"
    registrations_out = (
        Path(args.registrations_out) if args.registrations_out else package_dir / "matterix_registrations.py"
    )

    labs_dir = package_dir / "labs"
    lab_dirs = sorted(p.parent for p in labs_dir.glob("*/lab.yml")) if labs_dir.is_dir() else []
    if not lab_dirs:
        print(f"[from_eos] no labs/*/lab.yml found under {package_dir}", file=sys.stderr)
        return 1

    all_gym_ids: dict[str, str] = {}
    all_bindings: list[tuple[str, str, str]] = []

    try:
        for lab_dir in lab_dirs:
            spec, gym_ids, bindings = compile_lab(lab_dir, package_dir, scenes_dir)
            all_gym_ids.update(gym_ids)
            all_bindings.extend(bindings)
            print(f"[from_eos] wrote {scenes_dir / spec.name}/ ({spec.name}_env_cfg.py, __init__.py)")
    except (ValidationError, ValueError) as e:
        print(f"[from_eos] {e}", file=sys.stderr)
        return 1

    # Same "package's first scene ever" bootstrap as compile_scene.py -- see its
    # docstring for why this file's mere existence (not its contents) is the signal
    # _discover_scene_modules() gates on.
    parent_init = scenes_dir / "__init__.py"
    if not parent_init.is_file():
        parent_init.parent.mkdir(parents=True, exist_ok=True)
        parent_init.write_text(
            '"""Matterix scenes for this package -- generated by `python -m '
            'schema.from_eos` straight from this package\'s own labs/*/lab.yml, no '
            "hand-authored scene_specs/*.yaml needed. common/runtime.py's "
            '`_discover_scene_modules()` finds and imports each one automatically, no '
            'listing needed here.\n"""\n'
        )
        print(f"[from_eos] created {parent_init} (first scene in this package)")

    registrations_out.write_text(_render_registrations(all_gym_ids, all_bindings))
    print(f"[from_eos] wrote {registrations_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
