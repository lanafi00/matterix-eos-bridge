import gymnasium as gym

from . import dual_arm_handoff_env_cfg

gym.register(
    id="Matterix-Experiment-Dual-Arm-Handoff-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": dual_arm_handoff_env_cfg.DualArmHandoffEnvCfg,
    },
    disable_env_checker=True,
)
