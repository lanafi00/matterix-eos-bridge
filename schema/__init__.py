"""Declarative Matterix scene schema + compiler.

Lets a scene (asset catalog entries, pos/rot, workflow step sequence) be authored as a
small YAML file instead of hand-written `@configclass` Python -- see scene_specs/ for
examples and CLAUDE.md for the design rationale. Nothing in this package imports
isaaclab/matterix/matterix_sm -- it only reads catalog.json (see dump_catalog.py at the
repo root) and Python's own dataclasses/pydantic, so a scene spec can be validated and
compiled into real Python on a plain machine, no conda env or Isaac Sim boot required.
The generated `*_env_cfg.py` file is what actually imports isaaclab/matterix, exactly like
a hand-written scene -- that import only happens later, when Isaac Sim boots and
`_discover_scene_modules()` (common/runtime.py) imports it.

    catalog.py     - loads catalog.json, the offline record of what Matterix ships
                      (see dump_catalog.py) that validation checks names/fields against.
    models.py       - the schema itself: SceneSpec/AssetSpec/WorkflowStep (pydantic).
    robot_metadata.py - the small set of facts that are properties of a robot catalog
                         entry, not the scene (gripper joint names, action-space constant)
                         -- see that module's docstring for why this exists and its
                         current, deliberately narrow scope.
    compiler.py     - SceneSpec -> rendered `*_env_cfg.py` + `__init__.py` text.
    compile_scene.py - CLI: yaml path -> validate -> render -> write into scenes/<name>/.
"""
