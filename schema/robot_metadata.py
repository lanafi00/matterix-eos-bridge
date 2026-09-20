"""Facts about a robot catalog entry that a scene schema author should never have to
restate, because they're properties of the ROBOT, not the scene: which joints are its
gripper, and which `ActionSpaceInfo` constant (matterix_sm.robot_action_spaces) its
workflow steps need to drive it.

Every hand-authored scene in this repo (exp1-4) hardcodes both as scene-level config
(`gripper_joint_names = [...]` on the env cfg class; `action_space_info=FRANKA_IK_ACTION_SPACE`
on every single workflow step) even though all four scenes use the exact same Franka IK
robot and so restate the exact same two values every time -- exactly the kind of
authoring-time duplication a schema should derive instead of asking for.

This table is a deliberate, narrow stopgap, not a general solution: Matterix's own catalog
(catalog.json, via dump_catalog.py) has no field for "this robot's gripper joints" or "this
robot's action space" -- those aren't constructor kwargs on the asset config class itself,
so `discover_matterix_assets()` can't see them. Until Matterix exposes this on the asset
class itself (or some other real source of truth), compiler.py raises a clear error for any
articulated asset catalog key not listed here, rather than guessing -- see
`get_robot_metadata()`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotMetadata:
    gripper_joint_names: list[str]
    action_space_const: str
    """Name of the constant to import from `matterix_sm.robot_action_spaces`, e.g.
    "FRANKA_IK_ACTION_SPACE" -- not the value itself, since this module never imports
    matterix_sm (see this package's own __init__.py docstring for why)."""


_ROBOT_METADATA: dict[str, RobotMetadata] = {
    "robots/franka_panda_high_pd_ik": RobotMetadata(
        gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        action_space_const="FRANKA_IK_ACTION_SPACE",
    ),
}
"""Keyed by the same "category/name" catalog key catalog.json uses (see
matterix_asset_types.discover_matterix_assets()). Only the one robot every existing scene
in this repo actually uses is listed -- add an entry here (and verify it live, same as any
other change to what a scene's generated code does) before referencing a different
articulated asset from a scene spec."""


class UnknownRobotError(KeyError):
    pass


def get_robot_metadata(catalog_key: str) -> RobotMetadata:
    try:
        return _ROBOT_METADATA[catalog_key]
    except KeyError:
        raise UnknownRobotError(
            f"No gripper/action-space metadata for articulated asset {catalog_key!r}. "
            f"Known robots: {sorted(_ROBOT_METADATA)}. Add an entry to "
            "schema/robot_metadata.py (and verify it live) before using this asset as an "
            "articulated_assets slot in a scene spec."
        ) from None
