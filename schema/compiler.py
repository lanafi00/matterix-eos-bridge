"""Compiles a validated SceneSpec (models.py) into real Python source text: a
`*_env_cfg.py` matching this repo's hand-authored scenes/exp*/ convention, plus the
two-line `__init__.py` that registers it as a Gym env (see scenes/__init__.py's own
docstring for that convention).

Design choice, deliberate: the Jinja2 template (templates/env_cfg.py.jinja2) only does
plain string substitution -- every loop/conditional/formatting decision (which imports are
needed, how a tuple/kwarg renders as valid Python, indentation) happens here in Python
first, using `repr()` for anything that needs to come out as a valid Python literal. Doing
that logic in Jinja2 directly (nested `{% for %}`/`{% if %}` building up indentation by
hand) is exactly the kind of thing that silently produces subtly-wrong whitespace; building
each already-correct multi-line block as a plain string in Python, then substituting whole
blocks into the template, means the "is this valid Python" question only has one place to
get wrong, and it's a place with normal Python tooling (not template whitespace rules).

Only what exp1_beaker_pick needs is implemented (see models.py's docstring for the same
scope note) -- pick_object-shaped workflow steps specifically, since that's the only
action this repo's schema round-trip has actually exercised. A different action's
observation-derivation needs (e.g. place_object's pre_place/place frames) aren't
implemented yet; `_render_rigid_objects_group()` below only knows about pick targets.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from .catalog import load_catalog
from .models import SceneSpec
from .robot_metadata import get_robot_metadata

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_INDENT2 = "        "  # two levels: class body -> nested class/dict body
_INDENT1 = "    "  # one level: class body


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _class_name_for(spec: SceneSpec) -> str:
    return _snake_to_pascal(spec.name) + "EnvCfg"


def _render_kwarg(key: str, value) -> str:
    """`key=repr(value)`, except `action_space_info`, whose value is already a bare
    identifier string (e.g. "FRANKA_IK_ACTION_SPACE") this function must NOT quote --
    see compile_scene()'s injection of it into a workflow step's kwargs below."""
    if key == "action_space_info":
        return f"{key}={value}"
    return f"{key}={value!r}"


class CompileError(ValueError):
    pass


def _asset_import(catalog_key: str, class_name: str) -> str:
    category = catalog_key.split("/", 1)[0]
    return f"from matterix_assets.{category} import {class_name}"


def _render_asset_entry(slot: str, asset, class_name: str) -> str:
    kwargs = [f"pos={asset.pos!r}"]
    if asset.rot is not None:
        kwargs.append(f"rot={asset.rot!r}")
    if asset.mass is not None:
        kwargs.append(f"mass={asset.mass!r}")
    return f'{_INDENT2}"{slot}": {class_name}({", ".join(kwargs)}),'


def _render_event_cfg_body(spec: SceneSpec) -> str:
    lines = [f'{_INDENT1}reset_scene_to_default = EventTerm(func=isaaclab_mdp.reset_scene_to_default, mode="reset")']
    for slot, asset in spec.assets.items():
        r = asset.randomize_position
        if r is None:
            continue
        lines.append(
            f"{_INDENT1}randomize_{slot}_position = EventTerm(\n"
            f"{_INDENT1}    func=isaaclab_mdp.reset_root_state_uniform,\n"
            f'{_INDENT1}    mode="reset",\n'
            f"{_INDENT1}    params={{\n"
            f'{_INDENT1}        "pose_range": {{"x": {r.x!r}, "y": {r.y!r}, "z": {r.z!r}}},\n'
            f'{_INDENT1}        "velocity_range": {{}},\n'
            f'{_INDENT1}        "asset_cfg": SceneEntityCfg("{slot}"),\n'
            f"{_INDENT1}    }},\n"
            f"{_INDENT1})"
        )
    return "\n".join(lines)


def _render_articulations_group(articulated_slots: list[str], action_space_const: str) -> str:
    if not articulated_slots:
        return f"{_INDENT2}pass"
    return "\n".join(
        f'{_INDENT2}locals().update(robot_obs_terms("{slot}", {action_space_const}))' for slot in articulated_slots
    )


def _render_rigid_objects_group(pick_targets: list[str]) -> str:
    if not pick_targets:
        return f"{_INDENT2}pass"
    lines = []
    for slot in pick_targets:
        lines.append(
            f'{_INDENT2}{slot}__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={{"asset_name": "{slot}"}})'
        )
        lines.append(
            f'{_INDENT2}{slot}__object_world_quat = ObsTerm(func=mdp.object_world_quat, params={{"asset_name": "{slot}"}})'
        )
        for frame in ("pre_grasp", "grasp", "post_grasp"):
            lines.append(
                f"{_INDENT2}{slot}__{frame}_frame = ObsTerm(\n"
                f"{_INDENT2}    func=mdp.frame_world_pose, "
                f'params={{"asset_name": "{slot}", "frame_name": "{frame}"}}\n'
                f"{_INDENT2})"
            )
    return "\n".join(lines)


def compile_scene(spec: SceneSpec) -> tuple[str, str]:
    """Render `spec` into (env_cfg_py_text, init_py_text). Raises CompileError naming the
    exact problem for anything this function itself detects (e.g. an unknown robot in
    robot_metadata.py) -- spec is assumed already validated (see models.py); this function
    does not re-check catalog names/fields.
    """
    catalog = load_catalog()
    class_name = _class_name_for(spec)

    articulated_slots = [s for s, a in spec.assets.items() if a.kind == "articulated"]
    object_slots = [s for s, a in spec.assets.items() if a.kind == "object"]

    # One robot's metadata drives gripper_joint_names/action_space_info for the whole
    # scene -- MatterixBaseEnvCfg.gripper_joint_names is itself a single scene-level list,
    # not per-robot, so this isn't a limitation this compiler introduces. See
    # robot_metadata.py's docstring for why this table exists and its narrow scope.
    if not articulated_slots:
        raise CompileError("SceneSpec has no articulated (robot) asset -- nothing to drive a workflow.")
    primary_robot_meta = get_robot_metadata(spec.assets[articulated_slots[0]].catalog)

    pick_targets = [
        step.params["object"]
        for step in spec.workflows.values()
        if step.action == "pick_object" and "object" in step.params
    ]

    asset_imports: dict[str, str] = {}  # catalog_key -> import line, de-duplicated
    asset_class_names: dict[str, str] = {}  # slot -> class name
    for slot, asset in spec.assets.items():
        entry = catalog.assets[asset.catalog]
        cls_name = entry["class"].rsplit(".", 1)[-1]
        asset_class_names[slot] = cls_name
        asset_imports[asset.catalog] = _asset_import(asset.catalog, cls_name)

    action_imports: dict[str, str] = {}  # step name -> class name, and the import line set
    workflow_lines = []
    for wf_name, step in spec.workflows.items():
        entry = catalog.actions[step.action]
        cls_name = entry["class"].rsplit(".", 1)[-1]
        action_imports[cls_name] = f"from matterix_sm import {cls_name}"

        kwargs = dict(step.params)
        if "action_space_info" in entry["fields"] and "agent_assets" in kwargs:
            kwargs["action_space_info"] = primary_robot_meta.action_space_const
        rendered_kwargs = ", ".join(_render_kwarg(k, v) for k, v in kwargs.items())
        workflow_lines.append(f'{_INDENT2}"{wf_name}": {cls_name}({rendered_kwargs}),')

    import_lines = [
        "from matterix.envs import MatterixBaseEnvCfg, mdp",
        "from matterix.managers import EventManagerCfg",
        *sorted(asset_imports.values()),
    ]
    if articulated_slots:
        import_lines.append("from user.matterix_bridge.scenes.common import robot_obs_terms")
    import_lines += sorted(action_imports.values())
    import_lines.append(f"from matterix_sm.robot_action_spaces import {primary_robot_meta.action_space_const}")
    import_lines += [
        "",
        "import isaaclab.envs.mdp as isaaclab_mdp",
        "from isaaclab.managers import EventTermCfg as EventTerm",
        "from isaaclab.managers import ObservationGroupCfg as ObsGroup",
        "from isaaclab.managers import ObservationTermCfg as ObsTerm",
        "from isaaclab.managers import SceneEntityCfg",
        "from isaaclab.utils import configclass",
    ]

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )

    env_cfg_text = env.get_template("env_cfg.py.jinja2").render(
        scene_name=spec.name,
        imports="\n".join(import_lines),
        event_cfg_body=_render_event_cfg_body(spec),
        articulations_group_body=_render_articulations_group(articulated_slots, primary_robot_meta.action_space_const),
        rigid_objects_group_body=_render_rigid_objects_group(pick_targets),
        class_name=class_name,
        env_spacing=repr(spec.env_spacing),
        episode_length_line=(
            f"{_INDENT1}episode_length_s = {spec.episode_length_s!r}" if spec.episode_length_s is not None else ""
        ),
        gripper_joint_names=repr(primary_robot_meta.gripper_joint_names),
        objects_body="\n".join(
            _render_asset_entry(s, spec.assets[s], asset_class_names[s]) for s in object_slots
        ),
        articulated_assets_body="\n".join(
            _render_asset_entry(s, spec.assets[s], asset_class_names[s]) for s in articulated_slots
        ),
        workflows_body="\n".join(workflow_lines),
    )

    init_text = env.get_template("scene_init.py.jinja2").render(
        gym_id=spec.gym_id,
        module_name=f"{spec.name}_env_cfg",
        class_name=class_name,
    )

    return env_cfg_text, init_text
