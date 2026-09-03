"""Experiment 4: multi-agent choreography.

Two Franka arms share one scene. `robot` starts near a beaker on a table; `robot2` sits
close enough to reach a shared IKA-plate "handoff station" but nowhere near the table.
The workflow has `robot` pick the beaker up and place it on the station, then `robot2`
picks it up from there - a minimal two-agent hand-off built entirely out of the same
PickObjectCfg/PlaceObjectCfg building blocks used in exp1/exp2, just assigned to
different `agent_assets`.

This is the piece of the pipeline that matters once a lab workflow needs more than one
manipulator (e.g. one arm prepping labware while another runs an assay).
"""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers import EventManagerCfg
from matterix_assets.equipment.ika_plate import IKA_PLATE_INST_CFG
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
from user.matterix_bridge.scenes.common import robot_obs_terms
from matterix_sm import PickObjectCfg, PlaceObjectCfg, WaitCfg
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
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("beaker"),
        },
    )


@configclass
class ObservationManagerCfg:
    @configclass
    class ArticulationsGroup(ObsGroup):
        # Required for every agent that gets commanded, not just display - see robot_obs_terms()'s
        # docstring for what the StateMachine needs these for.
        locals().update(robot_obs_terms("robot", FRANKA_IK_ACTION_SPACE))
        locals().update(robot_obs_terms("robot2", FRANKA_IK_ACTION_SPACE))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RigidObjectsGroup(ObsGroup):
        beaker__object_world_pos = ObsTerm(func=mdp.object_world_pos, params={"asset_name": "beaker"})
        beaker__pre_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "pre_grasp"}
        )
        beaker__grasp_frame = ObsTerm(func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "grasp"})
        beaker__post_grasp_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "beaker", "frame_name": "post_grasp"}
        )
        station__pre_place_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "station", "frame_name": "pre_place"}
        )
        station__place_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "station", "frame_name": "place"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    articulations: ArticulationsGroup = ArticulationsGroup()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()


@configclass
class DualArmHandoffEnvCfg(MatterixBaseEnvCfg):
    """Two Franka arms + one beaker + a shared hand-off station + one table."""

    env_spacing = 10.0
    episode_length_s = 60.0
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(pos=(0.6, 0.05, 0.05)),
        "table": TABLE_SEATTLE_INST_Cfg(pos=(0.5, 0, 0)),
        # Reuse the IKA plate purely as a static hand-off surface (frames only, heater unused).
        "station": IKA_PLATE_INST_CFG(pos=(0.5, 0.4, 0.12), rot=(0, 0, 0, 1)),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0.0, 0.0)),
        "robot2": FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(0.0, 0.8, 0.0)),
    }

    observations = ObservationManagerCfg()
    events = EventCfg()

    record_path = "datasets/dataset.hdf5"

    workflows = {
        # Atomic, one entry per EOS DAG task node - what an EOS-facing caller dispatches
        # one at a time (see matterix_bridge/common/protocol_registry.py).
        "robot_pick_beaker": PickObjectCfg(
            description="robot picks up the beaker from the table",
            agent_assets="robot",
            object="beaker",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        "robot_place_on_station": PlaceObjectCfg(
            description="robot places the beaker on the shared station",
            agent_assets="robot",
            target="station",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        "wait_for_handoff": WaitCfg(duration=1.0),
        "robot2_pick_beaker": PickObjectCfg(
            description="robot2 picks up the beaker from the station",
            agent_assets="robot2",
            object="beaker",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        # Bundled convenience workflow for manual/dev CLI use - runs all four atomic
        # steps back to back in one call.
        "handoff": [
            PickObjectCfg(
                description="robot picks up the beaker from the table",
                agent_assets="robot",
                object="beaker",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            PlaceObjectCfg(
                description="robot places the beaker on the shared station",
                agent_assets="robot",
                target="station",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            WaitCfg(duration=1.0),
            PickObjectCfg(
                description="robot2 picks up the beaker from the station",
                agent_assets="robot2",
                object="beaker",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
        ],
    }
