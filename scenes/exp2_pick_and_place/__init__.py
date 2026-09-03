import gymnasium as gym

from . import pick_and_place_env_cfg

gym.register(
    id="Matterix-Experiment-Pick-Place-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": pick_and_place_env_cfg.PickAndPlaceEnvCfg,
    },
    disable_env_checker=True,
)
