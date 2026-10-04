"""CLI: generate an EOS package's Matterix scene(s) + matterix_registrations.py directly
from its own lab.yml + protocol.yml files -- no bridge-owned scene_specs/*.yaml needed.
See README.md's "Using this from your own EOS package" for the authoring convention this
implements.

Placement convention (all per EOS package, alongside its own pyproject.toml):
  - `labs/<lab_name>/lab.yml` -- devices' and resources'/resource_types' `meta` dicts
    carry the sim-only fields this reads: `matterix_catalog` (required -- its absence is
    what marks a device/resource as having no sim twin at all, silently skipped), `pos`,
    `rot`, `mass`, `semantics`, `randomize_position`, `randomize_temperature`. Same
    fields AssetSpec already accepts; see its docstring for what each means.
  - `protocols/<protocol_type>/protocol.yml` -- each task with a sim twin carries a
    `matterix:` key, parsed as a `WorkflowStep` (`action` + `params`): the Matterix
    action that backs that task. A task without one has no sim twin and is silently
    skipped. The task's own name becomes its Matterix workflow key. `params` name asset
    slots (`agent_assets`, `object`, `target`) directly by their lab.yml device/resource
    name, not the task's own `devices:`/`resources:` role names.

    EOS has no official slot for this on a task (TaskDef has no `meta`, unlike lab.yml's
    devices/resources): it works because TaskDef/ProtocolDef are plain pydantic models
    that silently ignore unknown keys -- verified, and nothing else in EOS reads or
    rewrites protocol.yml's raw YAML in a way that would trip on it. Which is also why
    this reads `matterix:` from the raw YAML, not from the parsed ProtocolDef: pydantic
    drops it there. If EOS ever switches TaskDef to `extra="forbid"`, this breaks.

Registrations are keyed by lab, not protocol (`register_lab()`/
`register_lab_task_workflow()` -- see common/protocol_registry.py): one scene per lab,
and every protocol's `matterix:` tasks in that lab merged into its bindings. A device
driver then resolves with its own `self.lab_name`, never a hardcoded protocol type.

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
from .models import AssetSpec, SceneSpec, WorkflowStep

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


def _load_matterix_tasks(protocol_yml: Path) -> tuple[ProtocolDef, dict[str, WorkflowStep]]:
    """Parse one protocol.yml: the ProtocolDef (validated by EOS's own model) plus
    `{task_name: WorkflowStep}` for every task carrying a `matterix:` key -- read from the
    raw YAML, since ProtocolDef silently drops unknown keys (see module docstring)."""
    raw = yaml.safe_load(protocol_yml.read_text())
    protocol = ProtocolDef(**raw)
    steps: dict[str, WorkflowStep] = {}
    for task in raw.get("tasks", []):
        if "matterix" not in task:
            continue
        try:
            steps[task["name"]] = WorkflowStep(**task["matterix"])
        except (TypeError, ValidationError) as e:
            raise ValueError(f"{protocol_yml}: task {task['name']!r}: invalid matterix block -- {e}") from e
    return protocol, steps


def compile_lab(lab_dir: Path, package_dir: Path, scenes_dir: Path) -> tuple[SceneSpec, dict[str, str]]:
    """Build + compile one lab's scene from its lab.yml, folding in every workflow
    binding contributed by the package's protocols that reference this lab. Returns the
    compiled spec plus this lab's `{task_name: workflow_key}` bindings, merged across
    those protocols (always identity -- a task's workflow key is its own name).

    Raises ValueError if two protocols in this lab give the same EOS task name different
    `matterix:` blocks -- registrations are keyed by (lab, task name), so that would be
    genuinely ambiguous at runtime.
    """
    lab = LabDef(**yaml.safe_load((lab_dir / "lab.yml").read_text()))
    assets = _build_assets(lab)
    gym_id = f"Matterix-Lab-{_pascal(lab.name)}-v1"

    workflows: dict[str, WorkflowStep] = {}
    sources: dict[str, Path] = {}

    for protocol_yml in sorted((package_dir / "protocols").glob("*/protocol.yml")):
        protocol, steps = _load_matterix_tasks(protocol_yml)
        if lab.name not in protocol.labs:
            continue
        for task_name, step in steps.items():
            if task_name in workflows and workflows[task_name] != step:
                raise ValueError(
                    f"task {task_name!r} in lab {lab.name!r} has a different matterix block in "
                    f"{protocol_yml} than in {sources[task_name]} -- registrations are keyed "
                    "by (lab, task name), so both protocols must agree"
                )
            workflows[task_name] = step
            sources[task_name] = protocol_yml

    spec = SceneSpec(name=lab.name, gym_id=gym_id, assets=assets, workflows=workflows)

    regen_hint = (
        f"Generated from {lab_dir / 'lab.yml'} (+ its protocols' matterix: task blocks) --\n"
        f"not a hand-authored scene_specs/*.yaml. Regenerate with:\n"
        f"    python -m schema.from_eos {package_dir}"
    )
    env_cfg_text, init_text = compile_scene(spec, regen_hint=regen_hint)
    out_dir = scenes_dir / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.name}_env_cfg.py").write_text(env_cfg_text)
    (out_dir / "__init__.py").write_text(init_text)

    return spec, {task_name: task_name for task_name in workflows}


def _render_registrations(labs: list[tuple[str, str, dict[str, str]]]) -> str:
    """`labs` is `[(lab_name, gym_id, {task_name: workflow_key}), ...]`."""
    lines = [
        '"""Generated by `python -m schema.from_eos` -- DO NOT EDIT BY HAND.',
        "",
        "Regenerate after editing any labs/*/lab.yml or a protocols/*/protocol.yml matterix: block",
        "in this package (run from matterix_bridge's own repo root -- see schema/from_eos.py):",
        "    python -m schema.from_eos <this package's root>",
        '"""',
        "",
        "from user.matterix_bridge.common.protocol_registry import register_lab, register_lab_task_workflow",
    ]
    for lab_name, gym_id, task_workflows in sorted(labs, key=lambda lab: lab[0]):
        lines.append("")
        lines.append(f"register_lab({lab_name!r}, {gym_id!r})")
        for task_name, workflow_key in sorted(task_workflows.items()):
            if workflow_key == task_name:
                lines.append(f"register_lab_task_workflow({lab_name!r}, {task_name!r})")
            else:
                lines.append(f"register_lab_task_workflow({lab_name!r}, {task_name!r}, {workflow_key!r})")
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

    lab_bindings: list[tuple[str, str, dict[str, str]]] = []

    try:
        for lab_dir in lab_dirs:
            spec, task_workflows = compile_lab(lab_dir, package_dir, scenes_dir)
            lab_bindings.append((spec.name, spec.gym_id, task_workflows))
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

    registrations_out.write_text(_render_registrations(lab_bindings))
    print(f"[from_eos] wrote {registrations_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
