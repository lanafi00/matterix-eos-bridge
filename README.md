# matterix_bridge

This package is designed to integrate Matterix digital twin generation into the Experimental Orchestration Framework, allowing for the virtual commissioning of laboratory protocols before they are deployed on hardware. 

## Overview
Broad level overview of how matterix_bridge interacts with EOS: 

1. An EOS protocol runs a task (e.g. "turn on heater").
2. The task calls a device driver (e.g. Heater.heat_to(...)).
3. If that device is configured in the EOS with `backend: sim` in `<your_package>/labs/<lab_name>/lab.yml`, the driver reaches into this bridge instead of real hardware.
4. The bridge translates the EOS device identifiers (task name, device) into Matterix device identifiers (gym task id, workflow key, twin config) and runs the simulation.
5. Isaac Sim/Matterix executes the corresponding robot/physics/thermal workflow and returns a result (e.g. a temperature), which flows back into the EOS resource.

Example workflow, as included in this package:

1. `protocols/heater_transfer_protocol/protocol.yml` defines task turn_on_heater on the heater device, with a target temperature.
2. `tasks/turn_on_heater/task.py` runs, calls `devices["heater"].heat_to(...)`.
3. `devices/heater/device.py` sees `backend == "sim"` (set in `labs/heater_transfer_lab/lab.yml`), so it calls `run_matterix_workflow()` (`common/matterix_backend.py`) with the target temperature as a dynamic parameter -- which gets/creates the shared MatterixBackend Ray actor for this protocol run (so all tasks in a protocol share the same physics environment), pushes the parameter, and runs the workflow.
4. `run_matterix_workflow()` resolves `"turn_on_heater"` via `common/protocol_registry.py`'s `resolve_matterix_call_by_task_name()` into a Matterix gym task id + workflow key ("Matterix-Experiment-Heater-Transfer-Franka-v1", "turn_on_heater") -- from the task name alone, no protocol type needed, since that task name is only ever registered under one protocol.
5. `common/runtime.py` boots Isaac Sim (once), builds/reuses the gym environment defined in `scenes/exp3_heater_transfer/`, and runs that one workflow (a TurnOnHeaterCfg semantic action) to completion.
6. The result (e.g. sample temperature) flows back up through the actor to the device driver to the task, and is stored on the EOS Resource.

## Installing this into an EOS deployment

1. Get the Matterix packages (`matterix`, `matterix_assets`, `matterix_sm`, `matterix_tasks`) installed in editable mode into a conda env that already has Isaac Lab -- see [Matterix's own README](https://github.com/ac-rad/Matterix) for the exact steps.

2. Make sure `eos` is installed in that *same* env, not EOS's plain `uv` venv. `eos start` has to run from a Python that already has both `eos` and isaaclab/matterix. This project uses a dedicated conda env for that:

   ```bash
   conda create --name eos-isaaclab --clone isaaclab
   conda activate eos-isaaclab
   pip install -e <path to eos repo>
   ```

3. Symlink this repo into `<eos repo>/user/matterix_bridge`:

   ```bash
   ln -s /path/to/matterix-eos-bridge <eos repo>/user/matterix_bridge
   ```

4. `eos start`, from the conda env in step 2.

## Using this from your own EOS package

You don't need to touch this repo. Your package just imports it.

1. **Scene.** Copy the closest example under `scenes/`. This is what decides the gym id and workflow key names you'll use in step 2. Needs an `__init__.py`:

   ```python
   import gymnasium as gym
   from . import my_scene_env_cfg

   gym.register(
       id="Matterix-Experiment-My-Scene-v1",
       entry_point="matterix.envs:MatterixBaseEnv",
       kwargs={"env_cfg_entry_point": my_scene_env_cfg.MySceneEnvCfg},
       disable_env_checker=True,
   )
   ```

2. **Register the protocol/task -> Matterix bindings**, in `<your_package>/matterix_registrations.py` (plain strings only), using the gym id and workflow keys from step 1:

   ```python
   from user.matterix_bridge.common.protocol_registry import register_protocol, register_task_workflow

   register_protocol("my_protocol", "Matterix-Experiment-My-Scene-v1")
   register_task_workflow("my_protocol", "my_task")  # workflow_key defaults to the task name -- pass it explicitly only if it differs
   ```

3. **Device driver.** In `<your_package>/devices/<type>/device.py`:

   ```python
   from eos.devices.base_device import BaseDevice
   from user.matterix_bridge.common.matterix_backend import run_matterix_workflow

   class DeviceName(BaseDevice):
       def device_action(self, sample, target_value, protocol_run_name, eos_task_name, headless=True):
           if self._backend_mode == "sim":
               run_matterix_workflow(
                   protocol_run_name, eos_task_name,
                   headless=headless, target_value=target_value,
               )
   ```

4. **Register a device twin, only if your device fills an articulated-asset slot more than one physical robot could occupy.** Most devices skip this. Goes in the same `scenes/__init__.py` from step 1 -- two steps: register the asset in the catalog once (by a plain string path), then bind your specific lab device to that catalog entry.

   ```python
   from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
   from user.matterix_bridge.common.asset_catalog import register_catalog_asset
   from user.matterix_bridge.common.device_registry import register_device_twin

   register_catalog_asset("robots/franka_panda_high_pd_ik", FRANKA_PANDA_HIGH_PD_IK_CFG)
   register_device_twin("my_lab", "my_arm", "robots/franka_panda_high_pd_ik")
   ```

   The catalog is the single source of truth for "what does this asset name mean" -- register each real asset once, then any number of devices (across any number of labs) can reference it by its catalog path instead of each importing and naming the Python class themselves.

Your package doesn't need `eos` installed wherever Isaac Sim runs, only isaaclab/matterix. `eos start` itself needs both in the same env.

To watch a run instead of headless: add `headless: eos_dynamic` to the task's parameters in `protocol.yml`, then pass `"headless": false` on submission.
