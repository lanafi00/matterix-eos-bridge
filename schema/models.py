"""The scene schema itself: SceneSpec (top level) / AssetSpec / WorkflowStep /
SemanticsSpec / BundleStep / PositionRandomization / TemperatureRandomization.

This is what a scene_specs/*.yaml file parses into, and what a drag-and-drop UI or an
LLM's structured output should target (`SceneSpec.model_json_schema()` gives you the JSON
Schema for that) -- see schema/__init__.py's docstring and CLAUDE.md for the full
rationale.

Scope (v3 -- see CLAUDE.md's rollout plan): covers what exp1_beaker_pick,
exp3_heater_transfer, and exp4_dual_arm_handoff all need -- asset placement, position/
temperature randomization, plain and composite (bundled) workflow steps, per-asset and
global semantics, and multi-agent scenes (more than one articulated asset). `agent_assets`
is validated to actually name an articulated asset (not just any declared slot) -- see
`_validate_step` below -- since compiler.py resolves each step's action_space_info from
THAT step's own agent_assets, not one scene-wide "primary" robot.

Validation happens in two places, deliberately: pydantic's own field types catch
structural mistakes (wrong type, missing required key) for free; the `_validate_catalog`
model_validator below catches everything that needs catalog.json (see catalog.py) --
unknown asset/action/semantics names, an asset's `kind` not matching what the catalog
says, an action or semantics param naming a field that doesn't exist, a workflow step's
agent_assets/object/target naming an asset slot this scene never declared, or a bundle
step's `ref` naming something other than a declared atomic workflow. Neither tier boots
Isaac Sim.

One deliberate gap, not a bug (unchanged from v1): field-name validation for action/
semantics params, not catalog-reported "required" enforcement -- see the v1 docstring this
replaced (`git log -p` on this file) for the verified false-positive (PickObjectCfg.
sub_actions) that makes catalog "required" an unreliable signal for compositional actions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .catalog import Catalog, load_catalog


class PositionRandomization(BaseModel):
    """Per-axis (min, max) uniform position offset, re-sampled on every env reset --
    compiles to one `isaaclab.envs.mdp.reset_root_state_uniform` EventTerm (see
    compiler.py). Matches exp1's/exp3's `randomize_*_position` EventTerm pattern exactly;
    both currently leave z fixed, hence the (0.0, 0.0) default rather than making z
    required."""

    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    z: tuple[float, float] = (0.0, 0.0)


class TemperatureRandomization(BaseModel):
    """(min, max) initial temperature (Kelvin), re-sampled on every env reset -- compiles
    to one `matterix.envs.mdp.randomize_temperature` EventTerm (see compiler.py). Note this
    is Matterix's OWN mdp namespace, not isaaclab's -- temperature is a semantics-engine
    concept, not a base isaaclab one, so it lives in a different function namespace than
    position randomization (`isaaclab.envs.mdp.reset_root_state_uniform`)."""

    min_temp: float
    max_temp: float


class SemanticsSpec(BaseModel):
    preset: str
    """A key into catalog.json's "semantic_presets" (composite, e.g. "heater",
    "heat_transfer") OR "primitive_semantics" (lower-level, e.g. "is_in_contact_physics",
    "ambient_air_heat_convection") -- see dump_catalog.py. Whichever category matches is
    resolved by `SceneSpec._validate_catalog` below; a name present in neither is an
    error."""

    params: dict[str, Any] = Field(default_factory=dict)


class AssetSpec(BaseModel):
    kind: Literal["articulated", "object"]
    """Which of MatterixBaseEnvCfg's two asset containers this slot goes in -- "articulated"
    -> articulated_assets, "object" -> objects (rigid + static together; see runtime.py's
    _get_env() docstring for why there are only these two)."""

    catalog: str
    """A key into catalog.json's "assets" (e.g. "robots/franka_panda_high_pd_ik") -- see
    dump_catalog.py. Checked against the catalog by `SceneSpec._validate_catalog` below."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] | None = None
    """None (the common case) omits `rot=` from the generated constructor call entirely,
    letting the asset class's own default apply -- matches every hand-authored scene's own
    convention of only passing `rot=` when it genuinely differs from the class default."""
    mass: float | None = None

    semantics: list[SemanticsSpec] = Field(default_factory=list)
    """Rendered as a Python list literal regardless of length (even a single entry) --
    VERIFIED against exp3_heater_transfer that Matterix accepts a list mixing primitive
    semantics (e.g. IsInContactPhysicsCfg) and composite presets (e.g. HeatTransferCfg)
    together, so there's no need for compiler.py to special-case "exactly one preset,
    render unwrapped" the way MatterixAssetCfg's own `semantics: list[SemanticCfg] |
    SemanticPreset` type hint might suggest -- always-a-list is simpler and covers every
    case this schema needs. Any asset with a semantics entry whose catalog key is
    "heat_transfer"/"heater" gets a derived `{slot}_temperature` PolicyCfg observation
    term; "is_in_contact_physics" gets `{slot}_is_in_contact`; "heater" additionally gets
    `{slot}_is_heater_on` -- see compiler.py's `_policy_terms_for_asset()`, matching
    exp3_heater_transfer's own PolicyCfg exactly."""

    randomize_position: PositionRandomization | None = None
    randomize_temperature: TemperatureRandomization | None = None


class WorkflowStep(BaseModel):
    action: str
    """A key into catalog.json's "actions" (e.g. "pick_object") -- see dump_catalog.py."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Constructor kwargs for the resolved action class, EXCEPT `action_space_info` --
    compiler.py injects that itself, derived from whichever articulated asset this step's
    `agent_assets` names (see robot_metadata.py) -- a schema author names a robot, not an
    action-space constant that's really a property of that robot."""


class BundleStep(BaseModel):
    """One step inside a bundle's ordered list (see SceneSpec.bundles) -- either a
    reference to an already-declared atomic `workflows` entry (`ref`), or a fresh,
    bundle-only step (`action`/`params`, same shape as WorkflowStep). Exactly one form,
    never both -- see `_check_shape` below.

    Compiling a `ref` step reuses the SAME kwargs the referenced atomic entry uses (as a
    module-level constant compiler.py emits once, shared between the atomic dict entry and
    every bundle that refs it) rather than a second, independently-typed copy -- this is
    exactly the hand-fix already applied to exp3_heater_transfer/exp4_dual_arm_handoff
    (see that commit): two copies of the same literal params can silently drift, and
    `workflow_overrides` (runtime.run_workflow) mutates a workflow's Cfg instance in place,
    keyed by name, so two entries built from the SAME instance (not two equal-valued ones)
    is what keeps an override on the atomic entry from leaking into the bundle, or vice
    versa. See compiler.py's `compile_scene()` for where these constants get built.
    """

    ref: str | None = None
    action: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_shape(self) -> "BundleStep":
        if (self.ref is None) == (self.action is None):
            raise ValueError("BundleStep must set exactly one of 'ref' or 'action', not both or neither")
        if self.ref is not None and self.params:
            raise ValueError("BundleStep with 'ref' set takes no 'params' -- it reuses the referenced step's own")
        return self


class SceneSpec(BaseModel):
    name: str
    """Used for the generated directory name (scenes/<name>/) and the env cfg class name."""

    gym_id: str
    """The Gym env id workflows/protocol_registry.py's PROTOCOL_TWINS will map an EOS
    protocol type onto -- e.g. "Matterix-Experiment-Beaker-Pick-Generated-v1"."""

    env_spacing: float = 5.0
    episode_length_s: float | None = None

    assets: dict[str, AssetSpec]
    workflows: dict[str, WorkflowStep]
    """Atomic entries -- one per EOS DAG task node, what protocol_registry.py's
    TASK_WORKFLOWS is meant to resolve to (see common/protocol_registry.py). This schema
    doesn't register those bindings itself -- that's an EOS package's own
    matterix_registrations.py concern (see README.md), out of scope here."""

    bundles: dict[str, list[BundleStep]] = Field(default_factory=dict)
    """Composite, dev-CLI-only workflow sequences (matching exp3's "pickup_and_place"/
    "observe_heating" pattern) -- never reached through protocol_registry.py. Compiled into
    the SAME `workflows` dict as the atomic entries (that's genuinely where Matterix looks
    for them -- see runtime.run_workflow's `workflow` param), just as a list value instead
    of a single Cfg call."""

    global_semantics: list[SemanticsSpec] = Field(default_factory=list)
    """Environment-level semantics (e.g. "ambient_air_heat_convection") -- rendered as the
    env cfg class's own `semantics = [...]` attribute, distinct from any asset's own
    per-object `semantics=` constructor kwarg (AssetSpec.semantics)."""

    @staticmethod
    def _validate_semantics_list(
        semantics: list[SemanticsSpec], catalog: Catalog, context: str, errors: list[str]
    ) -> None:
        for i, sem in enumerate(semantics):
            entry = catalog.semantic_presets.get(sem.preset) or catalog.primitive_semantics.get(sem.preset)
            if entry is None:
                errors.append(
                    f"{context}[{i}].preset: {sem.preset!r} not in catalog.json's "
                    f"semantic_presets or primitive_semantics. Known: "
                    f"{sorted(set(catalog.semantic_presets) | set(catalog.primitive_semantics))}"
                )
                continue
            known_fields = set(entry["fields"])
            for field_name in sem.params:
                if field_name not in known_fields:
                    errors.append(
                        f"{context}[{i}].params has unknown field {field_name!r} for preset "
                        f"{sem.preset!r}. Known fields: {sorted(known_fields)}"
                    )

    def _validate_step(
        self, action: str, params: dict[str, Any], catalog: Catalog, context: str, errors: list[str]
    ) -> None:
        entry = catalog.actions.get(action)
        if entry is None:
            errors.append(f"{context}.action: {action!r} not in catalog.json's actions. Known: {sorted(catalog.actions)}")
            return

        known_fields = set(entry["fields"])
        if "action_space_info" in params:
            errors.append(
                f"{context}.params sets 'action_space_info' explicitly -- the compiler "
                "derives this from the step's agent_assets robot; remove it."
            )
        for field_name in params:
            if field_name != "action_space_info" and field_name not in known_fields:
                errors.append(
                    f"{context}.params has unknown field {field_name!r} for action "
                    f"{action!r}. Known fields: {sorted(known_fields)}"
                )

        for ref_field in ("agent_assets", "object", "target"):
            value = params.get(ref_field)
            if value is None:
                continue
            refs = value if isinstance(value, list) else [value]
            for ref in refs:
                if ref not in self.assets:
                    errors.append(
                        f"{context}.params.{ref_field} names {ref!r}, which isn't a "
                        f"declared asset slot. Known: {sorted(self.assets)}"
                    )
                elif ref_field == "agent_assets" and self.assets[ref].kind != "articulated":
                    # compiler.py's _resolve_step_kwargs() looks up robot_metadata by
                    # this exact name to inject action_space_info -- a non-articulated
                    # asset has no robot metadata at all, so this would otherwise surface
                    # as a KeyError deep in the compiler instead of a clear schema error.
                    errors.append(
                        f"{context}.params.agent_assets names {ref!r}, which is "
                        f"kind={self.assets[ref].kind!r}, not 'articulated' -- only a "
                        "robot (articulated asset) can be an agent."
                    )

    @model_validator(mode="after")
    def _validate_catalog(self) -> "SceneSpec":
        catalog = load_catalog()
        errors: list[str] = []

        for slot, asset in self.assets.items():
            entry = catalog.assets.get(asset.catalog)
            if entry is None:
                errors.append(
                    f"assets.{slot}.catalog: {asset.catalog!r} not in catalog.json's "
                    f"assets. Known: {sorted(catalog.assets)}"
                )
                continue
            if entry["kind"] != asset.kind:
                errors.append(
                    f"assets.{slot}.kind: {asset.kind!r} but catalog.json says "
                    f"{asset.catalog!r} is {entry['kind']!r}"
                )
            self._validate_semantics_list(asset.semantics, catalog, f"assets.{slot}.semantics", errors)

        self._validate_semantics_list(self.global_semantics, catalog, "global_semantics", errors)

        for wf_name, step in self.workflows.items():
            self._validate_step(step.action, step.params, catalog, f"workflows.{wf_name}", errors)

        for bundle_name, steps in self.bundles.items():
            for i, bstep in enumerate(steps):
                context = f"bundles.{bundle_name}[{i}]"
                if bstep.ref is not None:
                    if bstep.ref not in self.workflows:
                        errors.append(
                            f"{context}.ref: {bstep.ref!r} isn't a declared atomic "
                            f"workflows entry. Known: {sorted(self.workflows)}"
                        )
                else:
                    self._validate_step(bstep.action, bstep.params, catalog, context, errors)

        if errors:
            raise ValueError("\n  - " + "\n  - ".join(errors))
        return self
