"""Experiment 2: chaining compositional actions.

One Franka arm, one beaker, one IKA plate (used here purely as a placement target - no
heater semantics yet, see exp3), one table. Workflow: pick up the beaker, then place it
on the plate.

This introduces PlaceObjectCfg alongside PickObjectCfg, and shows that a `workflow` is just
a list of compositional actions run back-to-back by the state machine. PlaceObjectCfg needs
its `target` object to define `pre_place`/`place` frames (see matterix_assets equipment
configs) - a plain table has no such frames, which is why the plate is the target here.
"""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg
from matterix_assets.equipment.ika_plate import IKA_PLATE_INST_CFG
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
from user.matterix_bridge.scenes.common import robot_obs_terms
from matterix_sm import PickObjectCfg, PlaceObjectCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class EventCfg(EventManagerCfg):
    reset_scene_to_default = EventTerm(
        func=isaaclab_mdp.reset_scene_to_default,
        mode="reset",
    )

    randomize_beaker_position = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("beaker"),
        },
    )


@configclass
class ObservationManagerCfg:
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

        ika_plate__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "ika_plate"})
        ika_plate__pre_place_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "ika_plate", "frame_name": "pre_place"}
        )
        ika_plate__place_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "ika_plate", "frame_name": "place"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


@configclass
class PickAndPlaceEnvCfg(MatterixBaseEnvCfg):
    """Franka + beaker + IKA plate + table. Workflow: pick beaker, place it on the plate."""

    env_spacing = 10.0
    episode_length_s = 45.0
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(pos=(0.6, 0.05, 0.05)),
        "ika_plate": IKA_PLATE_INST_CFG(pos=(0.4, -0.3, 0.12), rot=(0, 0, 0, 1)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0, 0)),
    }

    observations = ObservationManagerCfg()
    events = EventCfg()

    record_path = "datasets/dataset.hdf5"

    workflows = {
        # Atomic, one entry per EOS DAG task node - what an EOS-facing caller dispatches
        # one at a time (see matterix_bridge/common/protocol_registry.py).
        "pickup_beaker": PickObjectCfg(
            description="Pick up the beaker",
            agent_assets="robot",
            object="beaker",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        "place_beaker": PlaceObjectCfg(
            description="Place the beaker on the IKA plate",
            agent_assets="robot",
            target="ika_plate",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        # Bundled convenience workflow for manual/dev CLI use - runs both atomic steps
        # back to back in one call.
        "pick_and_place": [
            PickObjectCfg(
                description="Pick up the beaker",
                agent_assets="robot",
                object="beaker",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            PlaceObjectCfg(
                description="Place the beaker on the IKA plate",
                agent_assets="robot",
                target="ika_plate",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
        ],
    }
