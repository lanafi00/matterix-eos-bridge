"""CLI: validate a scene_specs/*.yaml file against models.py + catalog.json, compile it,
and write the result into scenes/<name>/. Needs no isaaclab/matterix/Isaac Sim -- runs in
EOS's plain venv (or any Python with pydantic/jinja2/pyyaml installed), which is the whole
point (see schema/__init__.py's docstring): catching an authoring mistake here costs
milliseconds, not the 1-2 minute Isaac Sim boot a mistake would otherwise cost.

Usage:
    python -m user.matterix_bridge.schema.compile_scene scene_specs/beaker_pick.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from .compiler import compile_scene
from .models import SceneSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec_path", help="Path to a scene_specs/*.yaml file")
    parser.add_argument(
        "--scenes-dir",
        default=str(_REPO_ROOT / "scenes"),
        help="Directory to write <name>/__init__.py and <name>/<name>_env_cfg.py into (default: scenes/)",
    )
    args = parser.parse_args(argv)

    with open(args.spec_path) as f:
        raw = yaml.safe_load(f)

    try:
        spec = SceneSpec(**raw)
    except ValidationError as e:
        print(f"[compile_scene] {args.spec_path} failed validation:\n{e}", file=sys.stderr)
        return 1

    env_cfg_text, init_text = compile_scene(spec)

    scenes_dir = Path(args.scenes_dir)
    out_dir = scenes_dir / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.name}_env_cfg.py").write_text(env_cfg_text)
    (out_dir / "__init__.py").write_text(init_text)

    # common/runtime.py's _discover_scene_modules() auto-walks every scene subpackage
    # under scenes/ -- no per-scene import needed -- but it still gates on scenes/
    # __init__.py itself existing (the "this package has scenes at all" signal), so a
    # package compiling its very first scene needs that one file created once. Never
    # touched again after this -- a second scene in the same package just lands in its
    # own new subdirectory, found the same way.
    parent_init = scenes_dir / "__init__.py"
    if not parent_init.is_file():
        parent_init.parent.mkdir(parents=True, exist_ok=True)
        parent_init.write_text(
            '"""Matterix scenes for this package -- each subpackage compiled via '
            "matterix_bridge's schema/compile_scene.py registers itself on import; "
            'common/runtime.py\'s `_discover_scene_modules()` finds and imports each one '
            'automatically, no listing needed here.\n"""\n'
        )
        print(f"[compile_scene] created {parent_init} (first scene in this package)")

    print(f"[compile_scene] wrote {out_dir}/ ({spec.name}_env_cfg.py, __init__.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
