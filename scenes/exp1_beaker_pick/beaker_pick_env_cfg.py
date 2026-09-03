"""Experiment 1: the "hello world" of Matterix.

One Franka arm, one beaker, one table, one workflow: pick up the beaker.

This is the smallest possible Matterix pipeline: an env config wires together assets
(robot/objects), an observation spec, and a workflow (a sequence of state-machine actions).
Everything else (physics, the semantics engine, RL wrappers) builds on this same shape.
"""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
from user.matterix_bridge.scenes.common import robot_obs_terms
from matterix_sm import PickObjectCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class EventCfg(EventManagerCfg):
    """Reset the scene, then randomize where the beaker spawns each episode."""

    reset_scene_to_default = EventTerm(
        func=isaaclab_mdp.reset_scene_to_default,
        mode="reset",
    )

    randomize_beaker_position = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("beaker"),
        },
    )


@configclass
class ObservationManagerCfg:
    """The minimum observations needed to close the loop: robot state + beaker pose/frames."""

    @configclass
    class ArticulationsGroup(ObsGroup):
        locals().update(robot_obs_terms("robot", FRANKA_IK_ACTION_SPACE))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RigidObjectsGroup(ObsGroup):
        beaker__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "beaker"})
        beaker__object_world_quat = ObsTerm(func=mdp.object_world_quat, params={"asset_name": "beaker"})
        beaker__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "pre_grasp"}
        )
        beaker__grasp_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "grasp"})
        beaker__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "post_grasp"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


@configclass
class BeakerPickEnvCfg(MatterixBaseEnvCfg):
    """Franka + beaker + table. Workflow: pick up the beaker."""

    env_spacing = 10.0
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(pos=(0.6, 0.05, 0.05)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0, 0)),
    }

    observations = ObservationManagerCfg()
    events = EventCfg()

    record_path = "datasets/dataset.hdf5"

    workflows = {
        "pickup_beaker": PickObjectCfg(
            description="Pick up the beaker",
            agent_assets="robot",
            object="beaker",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
    }
