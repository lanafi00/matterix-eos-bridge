"""The scene schema itself: SceneSpec (top level) / AssetSpec / WorkflowStep / PositionRandomization.

This is what a scene_specs/*.yaml file parses into, and what a drag-and-drop UI or an
LLM's structured output should target (`SceneSpec.model_json_schema()` gives you the JSON
Schema for that) -- see schema/__init__.py's docstring and CLAUDE.md for the full
rationale.

Scope, deliberately narrow (v1 -- see CLAUDE.md's rollout plan): covers exactly what
exp1_beaker_pick needs -- asset placement, one position-randomization event kind, and
plain (non-composite) workflow steps. NOT yet covered: semantics presets, global
semantics, multi-agent scenes, composite/bundled workflows. Extend this file (and
compiler.py alongside it) when the next round-trip target (exp3, the semantics-heavy
scene) needs them -- don't pre-build fields nothing exercises yet.

Validation happens in two places, deliberately: pydantic's own field types catch
structural mistakes (wrong type, missing required key) for free; the `_validate_catalog`
model_validator below catches everything that needs catalog.json (see catalog.py) --
unknown asset/action names, an asset's `kind` not matching what the catalog says, an
action param naming a field that doesn't exist, or a workflow step's agent_assets/object/
target naming an asset slot this scene never declared. Neither tier boots Isaac Sim.

One deliberate gap, not a bug: `_validate_catalog` checks that a workflow step's param
names are real fields on the resolved action class, but does NOT enforce catalog.json's
per-field "required" flag. Reason, verified by reading matterix_sm source: PickObjectCfg's
`sub_actions` field has no dataclass default (so catalog.json marks it "required": true)
but is unconditionally populated by PickObjectCfg.__post_init__ -- no real caller ever
passes it, so enforcing "required" here would reject the exact usage pattern every hand-
authored scene relies on. Catalog-reported "required" is a lower bound on what
compositional actions need, not a ready-to-enforce contract; the field-name-exists check
(which has no such false positive) is what's actually enforced today.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .catalog import load_catalog


class PositionRandomization(BaseModel):
    """Per-axis (min, max) uniform position offset, re-sampled on every env reset --
    compiles to one `isaaclab.envs.mdp.reset_root_state_uniform` EventTerm (see
    compiler.py). Matches exp1's/exp3's `randomize_*_position` EventTerm pattern exactly;
    both currently leave z fixed, hence the (0.0, 0.0) default rather than making z
    required."""

    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    z: tuple[float, float] = (0.0, 0.0)


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

    randomize_position: PositionRandomization | None = None


class WorkflowStep(BaseModel):
    action: str
    """A key into catalog.json's "actions" (e.g. "pick_object") -- see dump_catalog.py."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Constructor kwargs for the resolved action class, EXCEPT `action_space_info` --
    compiler.py injects that itself, derived from whichever articulated asset this step's
    `agent_assets` names (see robot_metadata.py) -- a schema author names a robot, not an
    action-space constant that's really a property of that robot."""


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

        for wf_name, step in self.workflows.items():
            entry = catalog.actions.get(step.action)
            if entry is None:
                errors.append(
                    f"workflows.{wf_name}.action: {step.action!r} not in catalog.json's "
                    f"actions. Known: {sorted(catalog.actions)}"
                )
                continue

            known_fields = set(entry["fields"])
            if "action_space_info" in step.params:
                errors.append(
                    f"workflows.{wf_name}.params sets 'action_space_info' explicitly -- "
                    "the compiler derives this from the step's agent_assets robot; remove it."
                )
            for field_name in step.params:
                if field_name != "action_space_info" and field_name not in known_fields:
                    errors.append(
                        f"workflows.{wf_name}.params has unknown field {field_name!r} for "
                        f"action {step.action!r}. Known fields: {sorted(known_fields)}"
                    )

            for ref_field in ("agent_assets", "object", "target"):
                value = step.params.get(ref_field)
                if value is None:
                    continue
                refs = value if isinstance(value, list) else [value]
                for ref in refs:
                    if ref not in self.assets:
                        errors.append(
                            f"workflows.{wf_name}.params.{ref_field} names {ref!r}, which "
                            f"isn't a declared asset slot. Known: {sorted(self.assets)}"
                        )

        if errors:
            raise ValueError("\n  - " + "\n  - ".join(errors))
        return self
