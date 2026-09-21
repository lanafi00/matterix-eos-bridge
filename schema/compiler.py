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
not one scene-wide "primary" robot -- see `_resolve_step_kwargs()`).

`compile_scene()` at the bottom is the one entry point; everything else is a small,
focused step it calls in sequence, sharing state through `_CompileContext` (catalog,
per-slot robot metadata, per-slot asset class names, and the accumulated set of extra
import lines every renderer contributes to) rather than each function taking its own
handful of loose parameters.
"""

from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass
class _CompileContext:
    """Shared state every rendering step below reads from or contributes to, instead of
    each function taking its own handful of the same few loose parameters
    (catalog/robot_meta_by_slot/imports showed up in almost every function's signature
    before this existed).

    Three separate import sets, not one merged set: `_render_import_lines()` emits them
    as three separate sorted blocks (semantics, then assets, then actions -- with the
    `robot_obs_terms` import between the asset and action blocks) to match this file's own
    established output shape exactly. Each is a plain `set[str]` of full `from x import y`
    lines -- dropped the dict-keyed-by-source-key shape a couple of these used before,
    since nothing ever looked a specific entry back up by key, only ever dumped the whole
    collection, sorted, at the end.
    """

    spec: SceneSpec
    catalog: Catalog
    articulated_slots: list[str]
    object_slots: list[str]
    robot_meta_by_slot: dict[str, RobotMetadata]
    asset_class_names: dict[str, str]
    semantics_imports: set[str] = dataclasses.field(default_factory=set)
    asset_imports: set[str] = dataclasses.field(default_factory=set)
    action_imports: set[str] = dataclasses.field(default_factory=set)


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
    semantics: list[SemanticsSpec], ctx: _CompileContext, *, bare_if_single_preset: bool = False
) -> str:
    """`[Class1(kw=v, ...), Class2(...)]`, or (see `bare_if_single_preset`) a single
    unwrapped `Class(...)` call. Adds each class's import to `ctx.semantics_imports` as a
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
    if bare_if_single_preset and len(semantics) == 1 and semantics[0].preset in ctx.catalog.semantic_presets:
        sem = semantics[0]
        class_name, import_line = _semantics_class_and_import(ctx.catalog, sem.preset)
        ctx.semantics_imports.add(import_line)
        rendered_kwargs = ", ".join(_render_kwarg(k, v) for k, v in sem.params.items())
        return f"{class_name}({rendered_kwargs})"

    parts = []
    for sem in semantics:
        class_name, import_line = _semantics_class_and_import(ctx.catalog, sem.preset)
        ctx.semantics_imports.add(import_line)
        rendered_kwargs = ", ".join(_render_kwarg(k, v) for k, v in sem.params.items())
        parts.append(f"{class_name}({rendered_kwargs})")
    return "[" + ", ".join(parts) + "]"


def _render_asset_entry(slot: str, asset: AssetSpec, ctx: _CompileContext) -> str:
    kwargs = [f"pos={asset.pos!r}"]
    if asset.rot is not None:
        kwargs.append(f"rot={asset.rot!r}")
    if asset.mass is not None:
        kwargs.append(f"mass={asset.mass!r}")
    if asset.semantics:
        rendered = _render_semantics_value(asset.semantics, ctx, bare_if_single_preset=True)
        kwargs.append(f"semantics={rendered}")
    class_name = ctx.asset_class_names[slot]
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


def _render_articulations_group(ctx: _CompileContext) -> str:
    if not ctx.articulated_slots:
        return f"{_INDENT2}pass"
    return "\n".join(
        f'{_INDENT2}locals().update(robot_obs_terms("{slot}", {ctx.robot_meta_by_slot[slot].action_space_const}))'
        for slot in ctx.articulated_slots
    )


def _render_frame_obs_terms(slot: str, frames: tuple[str, ...]) -> list[str]:
    return [
        f"{_INDENT2}{slot}__{frame}_frame = ObsTerm(\n"
        f"{_INDENT2}    func=mdp.frame_world_pose, "
        f'params={{"asset_name": "{slot}", "frame_name": "{frame}"}}\n'
        f"{_INDENT2})"
        for frame in frames
    ]


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
        lines += _render_frame_obs_terms(slot, ("pre_grasp", "grasp", "post_grasp"))
    for slot in place_targets:
        # Place targets only get their frame poses (pre_place/place), not object_world_pos/
        # quat -- VERIFIED against exp3_heater_transfer's ika_plate (a place target, never
        # picked up, so its own world pose isn't part of the SM-driven grasp/place loop).
        lines += _render_frame_obs_terms(slot, ("pre_place", "place"))
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


def _render_policy_group(ctx: _CompileContext) -> str:
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
        for slot in ctx.articulated_slots
    ]
    n_ee_lines = len(lines)
    for slot, asset in ctx.spec.assets.items():
        for suffix, mdp_func in _policy_terms_for_asset(asset):
            lines.append(f'{_INDENT2}{slot}_{suffix} = ObsTerm(func=mdp.{mdp_func}, params={{"asset_name": "{slot}"}})')
    if len(lines) == n_ee_lines:
        return ""  # no asset has any semantics -- omit PolicyCfg entirely, ee_pos-only isn't useful alone
    return "\n".join(lines)


def _resolve_step_kwargs(action: str, params: dict, ctx: _CompileContext) -> tuple[str, str]:
    """(class_name, rendered_kwargs_string) for one action+params pair. Injects
    action_space_info the same way for every step (atomic, bundle-inline, or the shared
    dict a bundle `ref` reuses) -- one place for this rule, not copy-pasted per caller.

    Resolved from THIS STEP's own `agent_assets` (the first named slot, if it's a list),
    not one scene-wide "primary" robot -- a multi-agent scene (e.g. exp4_dual_arm_handoff's
    "robot"/"robot2") needs each step to drive whichever robot IT names, which can differ
    step to step. models.py's `_validate_step` already guarantees `agent_assets` (when
    present) names a declared, articulated asset slot, so the lookup here can't miss.
    """
    entry = ctx.catalog.actions[action]
    class_name = entry["class"].rsplit(".", 1)[-1]
    ctx.action_imports.add(f"from matterix_sm import {class_name}")

    kwargs = dict(params)
    if "action_space_info" in entry["fields"] and "agent_assets" in kwargs:
        agent_assets = kwargs["agent_assets"]
        first_agent = agent_assets[0] if isinstance(agent_assets, list) else agent_assets
        kwargs["action_space_info"] = ctx.robot_meta_by_slot[first_agent].action_space_const
    rendered = ", ".join(_render_kwarg(k, v) for k, v in kwargs.items())
    return class_name, rendered


def _render_step_call(action: str, params: dict, ctx: _CompileContext, *, const_name: str | None) -> tuple[str, str | None]:
    """The single `Cls(...)` text for one action+params pair, plus (only when `const_name`
    is given) the `const_name = dict(...)` line it should be paired with.

    `const_name` given: renders `Cls(**const_name)` and returns the matching `const_name =
    dict(kwargs)` line as well, for the caller to place once at module level -- used for
    an atomic entry a bundle `ref`s (see BundleStep's docstring for why the atomic dict
    entry and every bundle that refs it must share, not duplicate, the same kwargs).
    `const_name` omitted (None): renders `Cls(kwargs)` inline, second return value None --
    used for a plain atomic entry, a bundle-inline step, and (again, discarding the
    duplicate const line) a bundle step that `ref`s an atomic entry which already emitted
    its own const line when `spec.workflows` was processed.
    """
    cls_name, rendered_kwargs = _resolve_step_kwargs(action, params, ctx)
    if const_name is not None:
        return f"{cls_name}(**{const_name})", f"{const_name} = dict({rendered_kwargs})"
    return f"{cls_name}({rendered_kwargs})", None


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


def _build_context(spec: SceneSpec, catalog: Catalog) -> _CompileContext:
    articulated_slots = [s for s, a in spec.assets.items() if a.kind == "articulated"]
    object_slots = [s for s, a in spec.assets.items() if a.kind == "object"]
    if not articulated_slots:
        raise CompileError("SceneSpec has no articulated (robot) asset -- nothing to drive a workflow.")

    # Each robot's OWN metadata drives its own action_space_info (_resolve_step_kwargs())
    # and ArticulationsGroup entry (_render_articulations_group()) -- a multi-agent
    # scene's robots can be different models.
    robot_meta_by_slot = {slot: get_robot_metadata(spec.assets[slot].catalog) for slot in articulated_slots}

    asset_class_names: dict[str, str] = {}
    ctx = _CompileContext(
        spec=spec,
        catalog=catalog,
        articulated_slots=articulated_slots,
        object_slots=object_slots,
        robot_meta_by_slot=robot_meta_by_slot,
        asset_class_names=asset_class_names,
    )
    for slot, asset in spec.assets.items():
        cls_name = catalog.assets[asset.catalog]["class"].rsplit(".", 1)[-1]
        asset_class_names[slot] = cls_name
        ctx.asset_imports.add(_asset_import(asset.catalog, cls_name))
    return ctx


def _resolve_gripper_joint_names(ctx: _CompileContext) -> list[str]:
    """MatterixBaseEnvCfg.gripper_joint_names is a single SCENE-level list, not per-robot
    (an upstream Matterix limitation, not something this compiler can route around -- see
    robot_metadata.py's docstring), so every robot in the scene must resolve to the SAME
    value; fail loudly with a clear message rather than silently picking one if they don't.
    """
    options = {tuple(m.gripper_joint_names) for m in ctx.robot_meta_by_slot.values()}
    if len(options) > 1:
        by_slot = {slot: ctx.robot_meta_by_slot[slot].gripper_joint_names for slot in ctx.articulated_slots}
        raise CompileError(
            f"Scene has articulated assets with different gripper_joint_names ({by_slot}) -- "
            "MatterixBaseEnvCfg.gripper_joint_names is a single scene-level list, so one scene "
            "can't mix robots with different grippers. Split into more than one scene."
        )
    return list(next(iter(options)))


def _pick_and_place_targets(spec: SceneSpec) -> tuple[list[str], list[str]]:
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
    return pick_targets, place_targets


def _render_workflows_body(spec: SceneSpec, ctx: _CompileContext) -> tuple[str, list[str]]:
    """(workflows_dict_body, const_lines) -- the full `workflows = {...}` dict contents
    (atomic entries and bundles together, bundles as list literals -- see BundleStep's
    docstring for why both land in the same dict) plus the module-level `_X_KWARGS`
    constants any bundle `ref` needs.

    Atomic entries referenced by at least one bundle get a module-level `_X_KWARGS`
    constant, shared verbatim between the atomic dict entry and every bundle that refs
    it -- see BundleStep's docstring for why this matters (workflow_overrides mutates a
    workflow's Cfg instance in place; two entries sharing one instance, not two
    separately-typed-but-equal ones, is what keeps an override from silently failing to
    reach the bundle, or leaking into it unexpectedly). Entries no bundle refs keep the
    plain inline-kwargs form (unchanged from a scene with no bundles at all).
    """
    refd_names = {bstep.ref for steps in spec.bundles.values() for bstep in steps if bstep.ref is not None}

    const_lines: list[str] = []
    workflow_lines: list[str] = []
    for wf_name, step in spec.workflows.items():
        const_name = _kwargs_const_name(wf_name) if wf_name in refd_names else None
        call_text, const_line = _render_step_call(step.action, step.params, ctx, const_name=const_name)
        if const_line is not None:
            const_lines.append(const_line)
        workflow_lines.append(f'{_INDENT2}"{wf_name}": {call_text},')

    for bundle_name, bsteps in spec.bundles.items():
        item_lines = []
        for bstep in bsteps:
            if bstep.ref is not None:
                atomic = spec.workflows[bstep.ref]
                call_text, _ = _render_step_call(
                    atomic.action, atomic.params, ctx, const_name=_kwargs_const_name(bstep.ref)
                )
            else:
                call_text, _ = _render_step_call(bstep.action, bstep.params, ctx, const_name=None)
            item_lines.append(f"{_INDENT3}{call_text},")
        workflow_lines.append(f'{_INDENT2}"{bundle_name}": [\n' + "\n".join(item_lines) + f"\n{_INDENT2}],")

    return "\n".join(workflow_lines), const_lines


def _render_import_lines(ctx: _CompileContext, const_lines: list[str]) -> list[str]:
    lines = [
        "from matterix.envs import MatterixBaseEnvCfg, mdp",
        "from matterix.managers import EventManagerCfg",
        *sorted(ctx.semantics_imports),
        *sorted(ctx.asset_imports),
    ]
    if ctx.articulated_slots:
        lines.append("from user.matterix_bridge.scenes.common import robot_obs_terms")
    lines += sorted(ctx.action_imports)
    # Every distinct action-space constant any robot in the scene needs -- each robot's
    # own ArticulationsGroup entry (_render_articulations_group()) always needs its own
    # constant regardless of whether any workflow step happens to reference that robot.
    action_space_consts = sorted({m.action_space_const for m in ctx.robot_meta_by_slot.values()})
    lines.append(f"from matterix_sm.robot_action_spaces import {', '.join(action_space_consts)}")
    lines += [
        "",
        "import isaaclab.envs.mdp as isaaclab_mdp",
        "from isaaclab.managers import EventTermCfg as EventTerm",
        "from isaaclab.managers import ObservationGroupCfg as ObsGroup",
        "from isaaclab.managers import ObservationTermCfg as ObsTerm",
        "from isaaclab.managers import SceneEntityCfg",
        "from isaaclab.utils import configclass",
    ]
    if const_lines:
        lines += ["", *const_lines]
    return lines


def compile_scene(spec: SceneSpec) -> tuple[str, str]:
    """Render `spec` into (env_cfg_py_text, init_py_text). Raises CompileError naming the
    exact problem for anything this function itself detects (e.g. an unknown robot in
    robot_metadata.py, or two robots with different gripper_joint_names) -- spec is
    assumed already validated (see models.py); this function does not re-check catalog
    names/fields.
    """
    catalog = load_catalog()
    ctx = _build_context(spec, catalog)
    gripper_joint_names = _resolve_gripper_joint_names(ctx)

    # Order matters here: several calls below add to ctx.semantics_imports/action_imports
    # as a side effect, and those are only read once, by _render_import_lines(), last.
    workflows_body, const_lines = _render_workflows_body(spec, ctx)

    pick_targets, place_targets = _pick_and_place_targets(spec)
    objects_body = "\n".join(_render_asset_entry(s, spec.assets[s], ctx) for s in ctx.object_slots)
    articulated_assets_body = "\n".join(_render_asset_entry(s, spec.assets[s], ctx) for s in ctx.articulated_slots)

    global_semantics_line = ""
    if spec.global_semantics:
        rendered = _render_semantics_value(spec.global_semantics, ctx)
        global_semantics_line = f"{_INDENT1}semantics = {rendered}"

    policy_group_body = _render_policy_group(ctx)
    rigid_objects_group_body = _render_rigid_objects_group(pick_targets, place_targets)
    articulations_group_body = _render_articulations_group(ctx)

    import_lines = _render_import_lines(ctx, const_lines)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )

    class_name = _class_name_for(spec)
    env_cfg_text = env.get_template("env_cfg.py.jinja2").render(
        scene_name=spec.name,
        imports="\n".join(import_lines),
        event_cfg_body=_render_event_cfg_body(spec),
        has_policy_group=bool(policy_group_body),
        policy_group_body=policy_group_body,
        articulations_group_body=articulations_group_body,
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
        workflows_body=workflows_body,
    )

    init_text = env.get_template("scene_init.py.jinja2").render(
        gym_id=spec.gym_id,
        module_name=f"{spec.name}_env_cfg",
        class_name=class_name,
    )

    return env_cfg_text, init_text
