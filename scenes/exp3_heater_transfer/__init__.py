import gymnasium as gym

from . import heater_transfer_env_cfg

gym.register(
    id="Matterix-Experiment-Heater-Transfer-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": heater_transfer_env_cfg.HeaterTransferEnvCfg,
    },
    disable_env_checker=True,
)
