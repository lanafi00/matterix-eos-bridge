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

Covers what exp1_beaker_pick, exp3_heater_transfer, and exp4_dual_arm_handoff need (see
models.py's docstring for the same scope note): pick_object/place_object-shaped workflow
steps, per-asset and global semantics, position/temperature randomization, bundled
(composite) workflows built from refs to atomic entries, and multi-agent scenes (more than
one articulated asset, each step's action_space_info resolved from ITS OWN agent_assets,
not one scene-wide "primary" robot -- see `_robot_meta_by_slot()`/`_resolve_step_kwargs()`).
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from .catalog import Catalog, load_catalog
from .models import AssetSpec, BundleStep, SceneSpec, SemanticsSpec, WorkflowStep
from .robot_metadata import RobotMetadata, get_robot_metadata

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_INDENT1 = "    "  # one level: class body
_INDENT2 = "        "  # two levels: class body -> nested class/dict body
_INDENT3 = "            "  # three levels: dict body -> list value -> list item


class CompileError(ValueError):
    pass


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _class_name_for(spec: SceneSpec) -> str:
    return _snake_to_pascal(spec.name) + "EnvCfg"


def _kwargs_const_name(wf_name: str) -> str:
    return f"_{wf_name.upper()}_KWARGS"


def _render_kwarg(key: str, value) -> str:
    """`key=repr(value)`, except `action_space_info`, whose value is already a bare
    identifier string (e.g. "FRANKA_IK_ACTION_SPACE") this function must NOT quote --
    see `_resolve_step_kwargs()` below, which injects it."""
    if key == "action_space_info":
        return f"{key}={value}"
    return f"{key}={value!r}"


def _asset_import(catalog_key: str, class_name: str) -> str:
    category = catalog_key.split("/", 1)[0]
    return f"from matterix_assets.{category} import {class_name}"


def _semantics_class_and_import(catalog: Catalog, preset_key: str) -> tuple[str, str]:
    """(class_name, import_line) for a semantics catalog key -- checks semantic_presets
    first, then primitive_semantics (same lookup order as models.py's own validator).
    Imports from the catalog's own full module path (not a shallower category re-export,
    unlike asset imports) -- VERIFIED both forms are valid Python for any of these
    classes, but the full path needs no guessing about which shallower alias exists,
    unlike matterix_assets' category packages which discover_matterix_assets() guarantees
    re-export everything found."""
    entry = catalog.semantic_presets.get(preset_key) or catalog.primitive_semantics.get(preset_key)
    assert entry is not None, f"{preset_key!r} should already be validated against the catalog"
    module, _, class_name = entry["class"].rpartition(".")
    return class_name, f"from {module} import {class_name}"


def _render_semantics_value(
    semantics: list[SemanticsSpec], catalog: Catalog, imports: dict[str, str], *, bare_if_single_preset: bool = False
) -> str:
    """`[Class1(kw=v, ...), Class2(...)]`, or (see `bare_if_single_preset`) a single
    unwrapped `Class(...)` call. Populates `imports` (catalog_key -> import line) as a
    side effect.

    `bare_if_single_preset`: pass True for a per-ASSET `semantics=` kwarg (never for the
    env-level `semantics = [...]` class attribute -- SemanticManager iterates
    `env.cfg.semantics` as a plain list unconditionally, with no bare-preset case at all;
    see semantic_manager.py's "environment level semantics" block).

    Why this matters, VERIFIED live (not a style choice): a list containing an embedded
    `SemanticPreset` instance (e.g. `semantics=[HeatTransferCfg(...)]`, a single preset
    still wrapped in a list) is only flattened into real primitives by
    MatterixRigidObjectCfg's and MatterixArticulationCfg's own `__post_init__` --
    MatterixStaticObjectCfg's does NOT have that list-flattening branch, only a bare-
    preset check (`isinstance(self.semantics, SemanticPreset)`) -- a real, narrow
    inconsistency in Matterix's own three asset base types, not something to work around
    by editing Matterix itself. Reproduced live: `TABLE_SEATTLE_INST_Cfg(semantics=
    [HeatTransferCfg(...)])` crashed SemanticManager with `AttributeError: 'HeatTransferCfg'
    object has no attribute 'type'` (it never got expanded into primitives); the bare form
    `semantics=HeatTransferCfg(...)` is handled by all three base types' `__post_init__`
    correctly. catalog.json doesn't currently distinguish "rigid" from "static" within its
    "object" kind (see dump_catalog.py), so this compiler can't look up which base type an
    asset actually is -- using the bare form whenever there's exactly one preset-only entry
    is safe across all three regardless. KNOWN GAP: an asset needing 2+ semantics entries
    where one is a preset AND the underlying class happens to be
    MatterixStaticObjectCfg-based would still hit this -- not reachable by either scene
    this schema currently compiles (only beaker needs 2+ entries, and it's a
    MatterixRigidObjectCfg), so not fixed here; extend catalog.json with the real
    rigid/static distinction if/when a scene actually needs it.
    """
    if bare_if_single_preset and len(semantics) == 1 and semantics[0].preset in catalog.semantic_presets:
        sem = semantics[0]
        class_name, import_line = _semantics_class_and_import(catalog, sem.preset)
        imports[sem.preset] = import_line
        rendered_kwargs = ", ".join(_render_kwarg(k, v) for k, v in sem.params.items())
        return f"{class_name}({rendered_kwargs})"

    parts = []
    for sem in semantics:
        class_name, import_line = _semantics_class_and_import(catalog, sem.preset)
        imports[sem.preset] = import_line
        rendered_kwargs = ", ".join(_render_kwarg(k, v) for k, v in sem.params.items())
        parts.append(f"{class_name}({rendered_kwargs})")
    return "[" + ", ".join(parts) + "]"


def _render_asset_entry(slot: str, asset: AssetSpec, class_name: str, catalog: Catalog, imports: dict[str, str]) -> str:
    kwargs = [f"pos={asset.pos!r}"]
    if asset.rot is not None:
        kwargs.append(f"rot={asset.rot!r}")
    if asset.mass is not None:
        kwargs.append(f"mass={asset.mass!r}")
    if asset.semantics:
        rendered = _render_semantics_value(asset.semantics, catalog, imports, bare_if_single_preset=True)
        kwargs.append(f"semantics={rendered}")
    return f'{_INDENT2}"{slot}": {class_name}({", ".join(kwargs)}),'


def _render_event_cfg_body(spec: SceneSpec) -> str:
    lines = [f'{_INDENT1}reset_scene_to_default = EventTerm(func=isaaclab_mdp.reset_scene_to_default, mode="reset")']
    for slot, asset in spec.assets.items():
        if asset.randomize_position is not None:
            r = asset.randomize_position
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
        if asset.randomize_temperature is not None:
            t = asset.randomize_temperature
            # matterix.envs.mdp (imported plain as `mdp`), NOT isaaclab_mdp -- temperature
            # is a semantics-engine (Matterix) concept, not a base isaaclab one. VERIFIED
            # against exp3_heater_transfer's own __post_init__, which uses exactly this
            # (bare `mdp.randomize_temperature`) alongside `isaaclab_mdp.*` for position.
            lines.append(
                f"{_INDENT1}randomize_{slot}_temperature = EventTerm(\n"
                f"{_INDENT1}    func=mdp.randomize_temperature,\n"
                f'{_INDENT1}    mode="reset",\n'
                f"{_INDENT1}    params={{"
                f'"asset_name": "{slot}", "min_temp": {t.min_temp!r}, "max_temp": {t.max_temp!r}}},\n'
                f"{_INDENT1})"
            )
    return "\n".join(lines)


def _render_articulations_group(articulated_slots: list[str], robot_meta_by_slot: dict[str, RobotMetadata]) -> str:
    if not articulated_slots:
        return f"{_INDENT2}pass"
    return "\n".join(
        f'{_INDENT2}locals().update(robot_obs_terms("{slot}", {robot_meta_by_slot[slot].action_space_const}))'
        for slot in articulated_slots
    )


def _render_rigid_objects_group(pick_targets: list[str], place_targets: list[str]) -> str:
    if not pick_targets and not place_targets:
        return ""
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
    for slot in place_targets:
        # Place targets only get their frame poses (pre_place/place), not object_world_pos/
        # quat -- VERIFIED against exp3_heater_transfer's ika_plate (a place target, never
        # picked up, so its own world pose isn't part of the SM-driven grasp/place loop).
        for frame in ("pre_place", "place"):
            lines.append(
                f"{_INDENT2}{slot}__{frame}_frame = ObsTerm(\n"
                f"{_INDENT2}    func=mdp.frame_world_pose, "
                f'params={{"asset_name": "{slot}", "frame_name": "{frame}"}}\n'
                f"{_INDENT2})"
            )
    return "\n".join(lines)


# catalog key -> (PolicyCfg term-name suffix, mdp function name). "heater" maps to TWO
# rows (temperature AND is_heater_on) -- a plain list of pairs, not a dict, since a preset
# key can drive more than one derived term. VERIFIED against exp3_heater_transfer's own
# PolicyCfg: every {slot}_temperature/{slot}_is_in_contact/{slot}_is_heater_on term there
# traces back to exactly one of these rules.
_POLICY_TERM_RULES: list[tuple[str, str, str]] = [
    ("heat_transfer", "temperature", "object_temperature"),
    ("heater", "temperature", "object_temperature"),
    ("is_in_contact_physics", "is_in_contact", "object_is_in_contact"),
    ("heater", "is_heater_on", "object_is_heater_on"),
]


def _policy_terms_for_asset(asset: AssetSpec) -> list[tuple[str, str]]:
    preset_keys = {sem.preset for sem in asset.semantics}
    seen: set[str] = set()
    terms: list[tuple[str, str]] = []
    for preset_key, suffix, mdp_func in _POLICY_TERM_RULES:
        if preset_key in preset_keys and suffix not in seen:
            seen.add(suffix)
            terms.append((suffix, mdp_func))
    return terms


def _render_policy_group(spec: SceneSpec, articulated_slots: list[str]) -> str:
    """Empty string if no asset has any semantics at all -- matching exp1_beaker_pick's
    total absence of a PolicyCfg group (see compile_scene()'s conditional use of this).

    One `ee_pos_{slot}` term per articulated asset (not just a single "ee_pos_robot"),
    named after its own slot -- for a single-robot scene this produces exactly
    "ee_pos_robot", matching exp3_heater_transfer's own PolicyCfg naming. NOT yet
    live-verified for a multi-robot scene that ALSO has semantics (exp4_dual_arm_handoff
    has no semantics at all, so this specific generalization isn't exercised by either
    scene this schema currently round-trips) -- the single-robot case it's proven against
    is unaffected either way.
    """
    lines = [
        f'{_INDENT2}ee_pos_{slot} = ObsTerm(func=mdp.ee_env_pos, params={{"asset_name": "{slot}"}})'
        for slot in articulated_slots
    ]
    n_ee_lines = len(lines)
    for slot, asset in spec.assets.items():
        for suffix, mdp_func in _policy_terms_for_asset(asset):
            lines.append(f'{_INDENT2}{slot}_{suffix} = ObsTerm(func=mdp.{mdp_func}, params={{"asset_name": "{slot}"}})')
    if len(lines) == n_ee_lines:
        return ""  # no asset has any semantics -- omit PolicyCfg entirely, ee_pos-only isn't useful alone
    return "\n".join(lines)


def _resolve_step_kwargs(
    action: str, params: dict, catalog: Catalog, robot_meta_by_slot: dict[str, RobotMetadata], action_imports: dict[str, str]
) -> tuple[str, str]:
    """(class_name, rendered_kwargs_string) for one action+params pair. Injects
    action_space_info the same way for every step (atomic, bundle-inline, or the shared
    dict a bundle `ref` reuses) -- one place for this rule, not copy-pasted per caller.

    Resolved from THIS STEP's own `agent_assets` (the first named slot, if it's a list),
    not one scene-wide "primary" robot -- a multi-agent scene (e.g. exp4_dual_arm_handoff's
    "robot"/"robot2") needs each step to drive whichever robot IT names, which can differ
    step to step. models.py's `_validate_step` already guarantees `agent_assets` (when
    present) names a declared, articulated asset slot, so the lookup here can't miss.
    """
    entry = catalog.actions[action]
    class_name = entry["class"].rsplit(".", 1)[-1]
    action_imports[class_name] = f"from matterix_sm import {class_name}"

    kwargs = dict(params)
    if "action_space_info" in entry["fields"] and "agent_assets" in kwargs:
        agent_assets = kwargs["agent_assets"]
        first_agent = agent_assets[0] if isinstance(agent_assets, list) else agent_assets
        kwargs["action_space_info"] = robot_meta_by_slot[first_agent].action_space_const
    rendered = ", ".join(_render_kwarg(k, v) for k, v in kwargs.items())
    return class_name, rendered


def _collect_steps(spec: SceneSpec) -> list[WorkflowStep | BundleStep]:
    """Every atomic workflow step, plus every INLINE (non-ref) bundle step -- what
    observation derivation (pick/place targets) scans, since a future scene could
    introduce a pick_object/place_object usage only inside a bundle, not as its own
    atomic entry. A `ref` step contributes nothing new here; it already resolves to an
    atomic entry this function also visits directly.
    """
    steps: list[WorkflowStep | BundleStep] = list(spec.workflows.values())
    for bundle_steps in spec.bundles.values():
        steps.extend(s for s in bundle_steps if s.ref is None)
    return steps


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

    if not articulated_slots:
        raise CompileError("SceneSpec has no articulated (robot) asset -- nothing to drive a workflow.")

    # Each robot's OWN metadata drives its own action_space_info (see
    # _resolve_step_kwargs()) and ArticulationsGroup entry (see
    # _render_articulations_group()) -- a multi-agent scene's robots can be different
    # models. gripper_joint_names is the one exception: MatterixBaseEnvCfg.
    # gripper_joint_names is a single SCENE-level list, not per-robot (an upstream Matterix
    # limitation, not something this compiler can route around -- see robot_metadata.py's
    # docstring), so every robot in the scene must resolve to the SAME value; fail loudly
    # with a clear message rather than silently picking one if they don't.
    robot_meta_by_slot = {slot: get_robot_metadata(spec.assets[slot].catalog) for slot in articulated_slots}
    gripper_joint_names_options = {tuple(m.gripper_joint_names) for m in robot_meta_by_slot.values()}
    if len(gripper_joint_names_options) > 1:
        raise CompileError(
            "Scene has articulated assets with different gripper_joint_names "
            f"({ {slot: robot_meta_by_slot[slot].gripper_joint_names for slot in articulated_slots} }) -- "
            "MatterixBaseEnvCfg.gripper_joint_names is a single scene-level list, so one scene "
            "can't mix robots with different grippers. Split into more than one scene."
        )
    gripper_joint_names = list(next(iter(gripper_joint_names_options)))

    all_steps = _collect_steps(spec)
    # dict.fromkeys(), not a set: preserves first-seen order (cosmetic, for stable/
    # readable output) while still deduping -- the SAME object/target named by more than
    # one step (e.g. an atomic entry and a bundle-only variant both picking "beaker",
    # VERIFIED to happen in scene_specs/heater_transfer.yaml's "pick_beaker"/
    # "observe_heating") would otherwise emit the same ObsTerm class attribute twice.
    pick_targets = list(
        dict.fromkeys(s.params["object"] for s in all_steps if s.action == "pick_object" and "object" in s.params)
    )
    place_targets = list(
        dict.fromkeys(s.params["target"] for s in all_steps if s.action == "place_object" and "target" in s.params)
    )

    semantics_imports: dict[str, str] = {}  # catalog key -> import line

    asset_imports: dict[str, str] = {}  # catalog_key -> import line, de-duplicated
    asset_class_names: dict[str, str] = {}  # slot -> class name
    for slot, asset in spec.assets.items():
        entry = catalog.assets[asset.catalog]
        cls_name = entry["class"].rsplit(".", 1)[-1]
        asset_class_names[slot] = cls_name
        asset_imports[asset.catalog] = _asset_import(asset.catalog, cls_name)

    action_imports: dict[str, str] = {}

    # Atomic entries referenced by at least one bundle get a module-level `_X_KWARGS`
    # constant, shared verbatim between the atomic dict entry and every bundle that refs
    # it -- see BundleStep's docstring for why this matters (workflow_overrides mutates a
    # workflow's Cfg instance in place; two entries sharing one instance, not two
    # separately-typed-but-equal ones, is what keeps an override from silently failing to
    # reach the bundle, or leaking into it unexpectedly). Entries no bundle refs keep the
    # plain inline-kwargs form (unchanged from a scene with no bundles at all).
    refd_names = {bstep.ref for steps in spec.bundles.values() for bstep in steps if bstep.ref is not None}

    const_lines: list[str] = []
    workflow_lines: list[str] = []
    for wf_name, step in spec.workflows.items():
        cls_name, rendered_kwargs = _resolve_step_kwargs(
            step.action, step.params, catalog, robot_meta_by_slot, action_imports
        )
        if wf_name in refd_names:
            const_name = _kwargs_const_name(wf_name)
            const_lines.append(f"{const_name} = dict({rendered_kwargs})")
            workflow_lines.append(f'{_INDENT2}"{wf_name}": {cls_name}(**{const_name}),')
        else:
            workflow_lines.append(f'{_INDENT2}"{wf_name}": {cls_name}({rendered_kwargs}),')

    for bundle_name, bsteps in spec.bundles.items():
        item_lines = []
        for bstep in bsteps:
            if bstep.ref is not None:
                atomic = spec.workflows[bstep.ref]
                cls_name, _ = _resolve_step_kwargs(
                    atomic.action, atomic.params, catalog, robot_meta_by_slot, action_imports
                )
                item_lines.append(f"{_INDENT3}{cls_name}(**{_kwargs_const_name(bstep.ref)}),")
            else:
                cls_name, rendered_kwargs = _resolve_step_kwargs(
                    bstep.action, bstep.params, catalog, robot_meta_by_slot, action_imports
                )
                item_lines.append(f"{_INDENT3}{cls_name}({rendered_kwargs}),")
        workflow_lines.append(f'{_INDENT2}"{bundle_name}": [\n' + "\n".join(item_lines) + f"\n{_INDENT2}],")

    objects_body = "\n".join(
        _render_asset_entry(s, spec.assets[s], asset_class_names[s], catalog, semantics_imports) for s in object_slots
    )
    articulated_assets_body = "\n".join(
        _render_asset_entry(s, spec.assets[s], asset_class_names[s], catalog, semantics_imports)
        for s in articulated_slots
    )
    global_semantics_line = ""
    if spec.global_semantics:
        rendered = _render_semantics_value(spec.global_semantics, catalog, semantics_imports)
        global_semantics_line = f"{_INDENT1}semantics = {rendered}"

    policy_group_body = _render_policy_group(spec, articulated_slots)
    rigid_objects_group_body = _render_rigid_objects_group(pick_targets, place_targets)

    import_lines = [
        "from matterix.envs import MatterixBaseEnvCfg, mdp",
        "from matterix.managers import EventManagerCfg",
        *sorted(semantics_imports.values()),
        *sorted(asset_imports.values()),
    ]
    if articulated_slots:
        import_lines.append("from user.matterix_bridge.scenes.common import robot_obs_terms")
    import_lines += sorted(action_imports.values())
    # Every distinct action-space constant any robot in the scene needs -- each robot's
    # own ArticulationsGroup entry (_render_articulations_group()) always needs its own
    # constant regardless of whether any workflow step happens to reference that robot.
    action_space_consts = sorted({m.action_space_const for m in robot_meta_by_slot.values()})
    import_lines.append(f"from matterix_sm.robot_action_spaces import {', '.join(action_space_consts)}")
    import_lines += [
        "",
        "import isaaclab.envs.mdp as isaaclab_mdp",
        "from isaaclab.managers import EventTermCfg as EventTerm",
        "from isaaclab.managers import ObservationGroupCfg as ObsGroup",
        "from isaaclab.managers import ObservationTermCfg as ObsTerm",
        "from isaaclab.managers import SceneEntityCfg",
        "from isaaclab.utils import configclass",
    ]
    if const_lines:
        import_lines += ["", *const_lines]

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
        has_policy_group=bool(policy_group_body),
        policy_group_body=policy_group_body,
        articulations_group_body=_render_articulations_group(articulated_slots, robot_meta_by_slot),
        rigid_objects_group_body=rigid_objects_group_body,
        class_name=class_name,
        env_spacing=repr(spec.env_spacing),
        episode_length_line=(
            f"{_INDENT1}episode_length_s = {spec.episode_length_s!r}" if spec.episode_length_s is not None else ""
        ),
        gripper_joint_names=repr(gripper_joint_names),
        objects_body=objects_body,
        articulated_assets_body=articulated_assets_body,
        global_semantics_line=global_semantics_line,
        workflows_body="\n".join(workflow_lines),
    )

    init_text = env.get_template("scene_init.py.jinja2").render(
        gym_id=spec.gym_id,
        module_name=f"{spec.name}_env_cfg",
        class_name=class_name,
    )

    return env_cfg_text, init_text
