# matterix_bridge

This package is designed to integrate Matterix digital twin generation into the Experimental Orchestration Framework, allowing for the virtual commissioning of laboratory protocols before they are deployed on hardware. 

## Overview
Broad level overview of how matterix_bridge interacts with EOS: 

1. An EOS protocol runs a task (e.g. "turn on heater").
2. The task calls a device driver (e.g. Heater.heat_to(...)).
3. If that device is configured in the EOS with `backend: sim` in `<your_package>/labs/<lab_name>/lab.yml`, the driver reaches into this bridge instead of real hardware.
4. The bridge translates the EOS device identifiers (protocol type, task name, device) into Matterix device identifiers (gym task id, workflow key, twin config) and runs the simulation.
5. Isaac Sim/Matterix executes the corresponding robot/physics/thermal workflow and returns a result (e.g. a temperature), which flows back into the EOS resource.

Example workflow, as included in this package:

1. `protocols/heater_transfer_protocol/protocol.yml` defines task turn_on_heater on the heater device, with a target temperature.
2. `tasks/turn_on_heater/task.py` runs, calls `devices["heater"].heat_to(...)`.
3. `devices/heater/device.py` sees `backend == "sim"` (set in `labs/heater_transfer_lab/lab.yml`), so it:
   - Gets the shared MatterixBackend Ray actor for this protocol run (`common/matterix_backend.py`), used to ensure that all tasks in a protocol share the same physics environment.
   - Sets the target temperature as a live parameter override.
   - Tells the backend to run the workflow.
4. `common/matterix_backend.py` resolves ("heater_transfer_protocol", "turn_on_heater") via `common/protocol_registry.py` into a Matterix gym task id + workflow key ("Matterix-Experiment-Heater-Transfer-Franka-v1", "turn_on_heater").
5. `common/runtime.py` boots Isaac Sim (once), builds/reuses the gym environment defined in `scenes/exp3_heater_transfer/`, and runs that one workflow (a TurnOnHeaterCfg semantic action) to completion.
6. The result (e.g. sample temperature) flows back up through the actor to the device driver to the task, and is stored on the EOS Resource.

This example is included in this package for ease of testing. Within the EOS package, to declare a protocol as able to be simulated:

1. For any unregistered devices, include this code within `devices/<your_device_type>/device.py`, swapping out DeviceName and device_action as necessary:

`from user.matterix_bridge.common.matterix_backend 
import get_backend
import ray
from eos.devices.base_device import BaseDevice
from user.matterix_bridge.common.matterix_backend import get_backend

class DeviceName(BaseDevice):
    def device_action(self, sample, target_rpm, protocol_run_name, protocol_type, eos_task_name, headless=True):
        if self._backend_mode == "sim":
            backend = get_backend(protocol_run_name)
            ray.get(backend.set_parameter.remote(protocol_type, eos_task_name, "target_rpm", target_rpm))
            ray.get(backend.run_workflow.remote(protocol_type, eos_task_name, headless=headless))`

            
2. (Optional) If your scene has an actuated slot (e.g. a robot) that should be fillable by different physical hardware depending on which lab/device calls it, add an entry to DEVICE_TWINS in device_registry.py. Most new devices (anything that just triggers an existing or new workflow, rather than swapping which robot backs a scene)  don't need this at all.  
3. Modify an existing scene in `scenes/` to swap out assets, change parameters, or add/re-sequence task nodes. If needed, add a new scene to `scenes/`, selecting whichever example is closest to your intended goal and using it as a template. Then add an `__init__.py` next to it, following this template:
```import gymnasium as gym
from . import my_scene_env_cfg

gym.register(
    id="Matterix-Experiment-My-Scene-v1",
    entry_point="matterix.envs:MatterixBaseEnv",
    kwargs={"env_cfg_entry_point": my_scene_env_cfg.MySceneEnvCfg},
    disable_env_checker=True,
)
```
After creating `scenes/<your_scene>/`, add it to the import list in `scenes/__init__.py`.
4. Add an entry to PROTOCOL_TWINS/TASK_WORKFLOWS dictionary inside `matterix-eos-bridge/common/protocol_registry.py`. PROTOCOL_TWINS maps one protocol type to one scene.   


