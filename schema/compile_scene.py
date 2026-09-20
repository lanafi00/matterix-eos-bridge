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

    out_dir = Path(args.scenes_dir) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.name}_env_cfg.py").write_text(env_cfg_text)
    (out_dir / "__init__.py").write_text(init_text)

    print(f"[compile_scene] wrote {out_dir}/ ({spec.name}_env_cfg.py, __init__.py)")
    print(
        "[compile_scene] NOTE: this new scene directory needs to be added to scenes/__init__.py's "
        "own imports/__all__ (same as every other scenes/exp*/) for _discover_scene_modules() "
        "to pick it up -- this CLI does not edit that file for you."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
