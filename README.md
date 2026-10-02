# matterix_bridge

This package is designed to integrate Matterix digital twin generation into the Experimental Orchestration Framework, allowing for the virtual commissioning of laboratory protocols before they are deployed on hardware. 

## Overview
Broad level overview of how matterix_bridge interacts with EOS: 

1. An EOS protocol runs a task (e.g. "turn on heater").
2. The task calls a device driver (e.g. Heater.heat_to(...)).
3. If that device is configured in the EOS with `backend: sim` in `<your_package>/labs/<lab_name>/lab.yml`, the driver reaches into this bridge instead of real hardware.
4. The bridge translates the EOS device identifiers (task name, device) into Matterix device identifiers (gym task id, workflow key, twin config) and runs the simulation.
5. Isaac Sim/Matterix executes the corresponding robot/physics/thermal workflow and returns a result (e.g. a temperature), which flows back into the EOS resource.

Example workflow (illustrative -- this package itself ships only the Matterix side of
this; see `eos/user/beaker_lab` for a complete, runnable EOS-side package following this
exact pattern end to end):

1. Your EOS package's `protocol.yml` defines task `turn_on_heater` on the heater device, with a target temperature.
2. Your package's `tasks/turn_on_heater/task.py` runs, calls `devices["heater"].heat_to(...)`.
3. Your package's `devices/heater/device.py` sees `backend == "sim"` (set in your `lab.yml`), so it calls `run_matterix_workflow()` (`common/matterix_backend.py`) with the target temperature as a dynamic parameter -- which gets/creates the shared MatterixBackend Ray actor for this protocol run (so all tasks in a protocol share the same physics environment), pushes the parameter, and runs the workflow.
4. `run_matterix_workflow()` resolves `"turn_on_heater"` via `common/protocol_registry.py`'s `resolve_matterix_call_by_task_name()` into a Matterix gym task id + workflow key ("Matterix-Experiment-Heater-Transfer-Franka-v1", "turn_on_heater") -- from the task name alone, no protocol type needed, since that task name is only ever registered under one protocol. This repo's own `matterix_registrations.py` registers that exact binding (against `heater_transfer_protocol`) for use by `smoke_test.py`/`vnc_test.py`, which exercise this same path directly without going through EOS at all.
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

You don't need to touch this repo. Your package just imports it. Two ways to get a scene
+ registrations; pick one, then both converge on the same device driver step.

### Option A: straight from your EOS package's own files (recommended)

No scene-authoring file of any kind -- positions and catalog keys live in `lab.yml`
itself, which you're writing anyway. See `eos/user/beaker_lab` for a real package using
this end to end, and `schema/from_eos.py`'s own docstring / `schema/models.py`'s
`WorkflowBindingSpec` for the full schema this reads.

1. **Positions and catalog keys go on your devices/resources' `meta` dict in
   `<your_package>/labs/<lab_name>/lab.yml`** -- `meta` is EOS's own unvalidated metadata
   bag (unlike `init_parameters`, nothing in EOS cross-checks it), so real-mode `eos
   start` never looks at these keys:

   ```yaml
   devices:
     my_arm:
       init_parameters:
         backend: sim
       meta:
         matterix_catalog: robots/franka_panda_high_pd_ik
         pos: [0.0, 0.0, 0.0]

   resource_types:
     beaker:
       meta:
         matterix_catalog: labware/beaker_500ml_inst

   resources:
     beaker_1:
       type: beaker
       meta:
         pos: [0.6, 0.05, 0.05]
         randomize_position:
           x: [-0.1, 0.1]
   ```

   Pure scene furniture with no real device/resource behind it (a table nothing ever
   allocates) still gets declared as a `resource_type`/`resource` pair -- it just never
   appears in any `protocol.yml` task's `resources:` block.

2. **The one thing that can't come from `lab.yml`: which Matterix action backs each EOS
   task.** Declare it in `<your_package>/protocols/<protocol_type>/matterix_workflow.yml`
   (a sibling of that protocol's own `protocol.yml`) -- `agent_assets`/`object`/`target`
   name assets directly by their `lab.yml` device/resource name:

   ```yaml
   task_workflows:
     my_task: my_workflow

   workflows:
     my_workflow:
       action: pick_object
       params:
         agent_assets: my_arm
         object: beaker_1
   ```

3. **Generate the scene + `matterix_registrations.py` together:**

   ```bash
   python -m schema.from_eos <your_package>
   ```

   Run from this repo's root. No conda env or Isaac Sim boot needed -- reads your
   `lab.yml`/`matterix_workflow.yml`, validates against `catalog.json`, writes
   `<your_package>/scenes/<lab_name>/` AND `<your_package>/matterix_registrations.py` for
   you (regenerate both by rerunning this after editing either file -- don't hand-edit
   either output).

   A device whose scene slot only one real class could ever fill needs no separate device
   twin registration either -- `matterix_catalog` in `lab.yml` already pins the concrete
   class. Skip `matterix_devices.py`/`register_device_twin()` (the "Register a device
   twin" step below) entirely unless more than one physical implementation could occupy
   that same slot.

### Option B: a scene decoupled from any one lab.yml

For a scene not tied 1:1 to a single EOS lab (e.g. shared across labs, or authored before
the EOS package exists), or anything `from_eos.py`'s scope doesn't cover yet:

1. **Scene.** This decides the gym id and workflow key names you'll use in step 2. Two ways to make one:

   - **YAML.** Write a scene spec, no Python needed -- `gym_id` is a required top-level field, same job as the Python path's `id=` kwarg below:

     ```yaml
     name: my_scene
     gym_id: Matterix-Experiment-My-Scene-v1
     assets:
       my_arm: {kind: articulated, catalog: robots/franka_panda_high_pd_ik}
       beaker_1: {kind: object, catalog: labware/beaker_500ml_inst}
     workflows:
       my_workflow:
         action: pick_object
         params: {agent_assets: my_arm, object: beaker_1}  # see schema/models.py for every field
     ```

     See `schema/models.py` for the full schema (every field has its own docstring) -- no bundled example file ships in this repo (the ones that did were round-trip fixtures for the schema itself, removed once `eos/user/beaker_lab` moved to Option A; see CLAUDE.md).

     ```bash
     python -m schema.compile_scene <your_package>/scene_specs/my_scene.yaml \
         --scenes-dir <your_package>/scenes
     ```

     Run from this repo's root. No conda env or Isaac Sim boot needed -- just validates against `catalog.json` and writes the Python for you. Covers asset placement, semantics, randomization, multi-robot scenes, and plain/composite workflows.

   - **Hand-written Python**, for anything the YAML schema doesn't cover yet. Copy the closest example under `scenes/`. Needs an `__init__.py`:

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

   Nothing else to wire up -- your package's `scenes/__init__.py` doesn't need to list this scene. `_discover_scene_modules()` finds every scene subpackage on its own; the YAML path even creates `scenes/__init__.py` for you if this is your package's first scene.

2. **Register the protocol/task -> Matterix bindings**, in `<your_package>/matterix_registrations.py` (plain strings only), using the gym id and workflow keys from step 1:

   ```python
   from user.matterix_bridge.common.protocol_registry import register_protocol, register_task_workflow

   register_protocol("my_protocol", "Matterix-Experiment-My-Scene-v1")
   register_task_workflow("my_protocol", "my_task")  # workflow_key defaults to the task name -- pass it explicitly only if it differs
   ```

### Both options continue with

**Device driver.** In `<your_package>/devices/<type>/device.py`:

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

**Register a device twin, only if your device fills a scene slot (a robot in `articulated_assets`, or a non-robot asset like a hot plate or beaker in `objects`) more than one physical implementation could occupy.** Most devices skip this (and Option A's `matterix_catalog` in `lab.yml` already covers the common case). Goes in a `matterix_devices.py` at your package's root (a sibling of `pyproject.toml`, `labs/`, `devices/`, etc. -- not `scenes/__init__.py`: devices exist independently of any particular scene, and a scene's slots get filled by a device twin at workflow-run time, not the other way around). Bind your lab device directly to the real Matterix asset class:

   ```python
   from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG
   from user.matterix_bridge.common.device_registry import register_device_twin

   register_device_twin("my_lab", "my_arm", FRANKA_PANDA_HIGH_PD_IK_CFG)
   ```

   Not sure what classes are available? `user.matterix_bridge.common.matterix_asset_types.discover_matterix_assets()` (call after Isaac Sim has booted) introspects Matterix's own robots/equipment/labware/infrastructure packages and lists every real asset class it ships, so you don't have to guess or hunt for names. `matterix_devices.py` is auto-imported once Isaac Sim has booted, the same way `matterix_registrations.py` is auto-imported before it boots -- you never call `register_device_twin()` yourself from outside this file, it just needs to exist.

Your package doesn't need `eos` installed wherever Isaac Sim runs, only isaaclab/matterix. `eos start` itself needs both in the same env.

To watch a run instead of headless: add `headless: eos_dynamic` to the task's parameters in `protocol.yml`, then pass `"headless": false` on submission.
