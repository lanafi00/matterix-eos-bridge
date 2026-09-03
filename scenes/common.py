"""Shared building blocks for matterix_experiments environment configs.

Currently just one: the per-robot `ArticulationsGroup` observation terms every experiment
needs so `StateMachine.step()` can drive that agent (see `_initialize_action_dict_for_agent`
in matterix_sm's `state_machine.py`). Which fields are required depends on whether the
robot's action space is task-space (IK) or joint-space - `robot_obs_terms()` derives that
from the same `ActionSpaceInfo` object each workflow action already takes as
`action_space_info=`, so a robot's observation block can't drift out of sync with its action
space the way a hand-copied block could.
"""

from matterix.envs import mdp
from matterix_sm.robot_action_spaces import ActionSpaceInfo

from isaaclab.managers import ObservationTermCfg as ObsTerm


def robot_obs_terms(asset_name: str, action_space_info: ActionSpaceInfo) -> dict[str, ObsTerm]:
    """Observation terms the StateMachine needs to drive `asset_name` under `action_space_info`.

    Task-space (IK) agents need root/EE world pose so the state machine can transform the EE
    pose into the robot's base frame; joint-space agents need `joint_pos` instead. Gripper is
    common to both. Return value is meant to be splice into an `ObservationGroupCfg` subclass's
    body via `locals().update(...)` - see any experiment's `ArticulationsGroup` for the pattern.
    """
    if action_space_info.joint_indices is not None:
        terms = {
            f"{asset_name}__joint_pos": ObsTerm(func=mdp.joint_pos, params={"asset_name": asset_name}),
        }
    else:
        terms = {
            f"{asset_name}__root_world_pos": ObsTerm(func=mdp.root_world_pos, params={"asset_name": asset_name}),
            f"{asset_name}__root_world_quat": ObsTerm(func=mdp.root_world_quat, params={"asset_name": asset_name}),
            f"{asset_name}__ee_world_pos": ObsTerm(func=mdp.ee_world_pos, params={"asset_name": asset_name}),
            f"{asset_name}__ee_world_quat": ObsTerm(func=mdp.ee_world_quat, params={"asset_name": asset_name}),
        }
    terms[f"{asset_name}__gripper_pos"] = ObsTerm(func=mdp.gripper_pos, params={"asset_name": asset_name})
    return terms
