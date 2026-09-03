import gymnasium as gym

from . import beaker_pick_env_cfg

gym.register(
    id="Matterix-Experiment-Beaker-Pick-Franka-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={
        "env_cfg_entry_point": beaker_pick_env_cfg.BeakerPickEnvCfg,
    },
    disable_env_checker=True,
)
