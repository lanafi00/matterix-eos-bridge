"""Experiment 3: the semantics engine.

Same physical setup as exp2 (Franka + beaker + IKA plate + table), but now the plate is a
real heater and every object carries thermal semantics (`HeatTransferCfg`/`HeaterCfg`), plus
a global ambient-air convection term. This is the part of Matterix that goes beyond rigid-body
physics: continuous state (temperature) and logical device state (heater on/off) evolve
alongside the physics sim and are exposed as observations just like poses are.

Two workflows:
    "observe_heating"     - just wait and watch heat flow from the (pre-heated) robot into
                             the beaker while it's held - no manipulation needed to see
                             the semantics engine working.
    "pickup_and_place"    - the full loop: turn the heater on, pick up the beaker, place it
                             on the plate, wait for heat transfer, then turn the heater off.

`TurnOnHeaterCfg` is a *semantic* action (see matterix_sm.semantic_actions) - unlike Pick/Place
it doesn't move a robot, it flips a logical device state that the semantics engine then acts on.
"""

from matterix.envs import MatterixBaseEnvCfg, mdp
from matterix.managers.semantics.primitive_semantics import IsInContactPhysicsCfg
from matterix.managers.semantics.primitive_semantics.heat_transfer import AmbientAirHeatConvectionCfg
from matterix.managers.semantics.semantic_presets import HeaterCfg, HeatTransferCfg
from matterix_assets.equipment.ika_plate import IKA_PLATE_INST_CFG
from matterix_assets.infrastructure.tables import TABLE_SEATTLE_INST_Cfg
from matterix_assets.labware.beakers import BEAKER_500ML_INST_CFG
from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
from user.matterix_bridge.scenes.common import robot_obs_terms
from matterix_sm import PickObjectCfg, PlaceObjectCfg, TurnOnHeaterCfg, WaitCfg
from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class ObservationManagerCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Flat, RL-friendly view: end-effector position + every temperature/state in the scene."""

        ee_pos_robot = ObsTerm(func=mdp.ee_env_pos, params={"asset_name": "robot"})
        beaker_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "beaker"})
        beaker_is_in_contact = ObsTerm(func=mdp.object_is_in_contact, params={"asset_name": "beaker"})
        table_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "table"})
        robot_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "robot"})
        ika_plate_temperature = ObsTerm(func=mdp.object_temperature, params={"asset_name": "ika_plate"})
        ika_plate_is_heater_on = ObsTerm(func=mdp.object_is_heater_on, params={"asset_name": "ika_plate"})

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
        ika_plate__pre_place_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "ika_plate", "frame_name": "pre_place"}
        )
        ika_plate__place_frame = ObsTerm(
            func=mdp.frame_world_pose, params={"asset_name": "ika_plate", "frame_name": "place"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ArticulationsGroup(ObsGroup):
        # These fields (not `ee_env_pos` in PolicyCfg above) are what the StateMachine reads to
        # drive the robot - required for the workflow to run, independent of what PolicyCfg
        # exposes to an RL policy. See robot_obs_terms()'s docstring for which fields and why.
        locals().update(robot_obs_terms("robot", FRANKA_IK_ACTION_SPACE))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rigid_objects: RigidObjectsGroup = RigidObjectsGroup()
    articulations: ArticulationsGroup = ArticulationsGroup()


@configclass
class HeaterTransferEnvCfg(MatterixBaseEnvCfg):
    """Franka + beaker + IKA heater + table, with thermal semantics active on every object."""

    env_spacing = 5.0
    episode_length_s = 100.0
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    objects = {
        "beaker": BEAKER_500ML_INST_CFG(
            pos=(0.6, 0.05, 0.05),
            semantics=[
                # setup_scene() prepends the per-env Articulations_/RigidObjects_ prefix automatically.
                IsInContactPhysicsCfg(filter_prim_paths_expr=["ika_plate"], verbose=True),
                HeatTransferCfg(temp=298.15, C=7920.0, A=0.1, K=200, verbose=True),
            ],
        ),
        "ika_plate": IKA_PLATE_INST_CFG(
            pos=(0.4, -0.3, 0.12),
            rot=(0, 0, 0, 1),
            semantics=HeaterCfg(
                temp=350.15,
                C=7920.0,
                A=0.5,
                heater_on=False,
                target_temperature=298.15,
                K_heater=0.1,
                verbose=True,
            ),
        ),
        "table": TABLE_SEATTLE_INST_Cfg(
            pos=(0.5, 0, 0),
            mass=50.0,
            semantics=HeatTransferCfg(temp=323.15, C=7920.0, A=1.1, verbose=True),
        ),
    }

    articulated_assets = {
        "robot": FRANKA_PANDA_HIGH_PD_IK_CFG(
            pos=(0.0, 0, 0),
            mass=20.0,
            semantics=HeatTransferCfg(temp=350.15, C=7920.0, A=0.5, verbose=True),
        ),
    }

    observations = ObservationManagerCfg()

    # Global (environment-level) semantics: ambient air convection cools/heats every object.
    semantics = [AmbientAirHeatConvectionCfg(verbose=True)]

    workflows = {
        # Atomic, one entry per EOS DAG task node - what an EOS-facing caller dispatches
        # one at a time (see matterix_bridge/common/protocol_registry.py).
        "turn_on_heater": TurnOnHeaterCfg(
            asset_name="ika_plate",
            value=True,
            target_temperature=373.15,
        ),
        "pick_beaker": PickObjectCfg(
            description="Pick up the beaker",
            agent_assets="robot",
            object="beaker",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        "place_beaker": PlaceObjectCfg(
            description="Place the beaker on top of the IKA plate",
            agent_assets="robot",
            target="ika_plate",
            action_space_info=FRANKA_IK_ACTION_SPACE,
        ),
        "wait_for_heat_transfer": WaitCfg(duration=10.0),
        "turn_off_heater": TurnOnHeaterCfg(
            asset_name="ika_plate",
            value=False,
        ),
        # Bundled convenience workflows for manual/dev CLI use.
        "observe_heating": [
            PickObjectCfg(
                description="Pick up the (pre-heated) beaker and hold it - watch its temperature rise",
                agent_assets="robot",
                object="beaker",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            WaitCfg(duration=5.0),
        ],
        "pickup_and_place": [
            WaitCfg(duration=2.0),  # baseline: observe ambient/plate heat transfer before touching anything
            TurnOnHeaterCfg(
                asset_name="ika_plate",
                value=True,
                target_temperature=373.15,
            ),
            PickObjectCfg(
                description="Pick up the beaker",
                agent_assets="robot",
                object="beaker",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            PlaceObjectCfg(
                description="Place the beaker on top of the IKA plate",
                agent_assets="robot",
                target="ika_plate",
                action_space_info=FRANKA_IK_ACTION_SPACE,
            ),
            WaitCfg(duration=10.0),  # watch heat flow from the now-hot plate into the beaker
            TurnOnHeaterCfg(
                asset_name="ika_plate",
                value=False,
            ),
            WaitCfg(duration=5.0),
        ],
    }

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_table_temp = EventTerm(
            func=mdp.randomize_temperature,
            mode="reset",
            params={"asset_name": "beaker", "min_temp": 293.15, "max_temp": 323.15},  # 20-50 C
        )
        self.events.randomize_robot_temp = EventTerm(
            func=mdp.randomize_temperature,
            mode="reset",
            params={"asset_name": "robot", "min_temp": 340.15, "max_temp": 360.15},  # 67-87 C
        )
        self.events.randomize_beaker_pos = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("beaker"),
            },
        )
