# matterix_bridge

This package is designed to integrate Matterix digital twin generation into the Experimental Orchestration Framework, allowing for the virtual commissioning of laboratory protocols before they are deployed on hardware. 

## Overview
Example workflow (`eos/user/beaker_lab`, a complete, runnable EOS package following this pattern end to end):

**Once, at authoring time:**

1. `labs/beaker_lab/lab.yml` declares the lab as usual, plus sim-only `meta` on each device/resource: the `arm` device is `robots/franka_panda_high_pd_ik`, `beaker_1` is `labware/beaker_500ml_inst` at `[0.6, 0.05, 0.05]` with randomized x/y, and `table_1` is `infrastructure/table_seattle_inst`. EOS never validates `meta`, so the same file still works for real runs.
2. Each protocol that should have a sim twin gets a `matterix_workflow.yml` next to its `protocol.yml`, saying which Matterix action backs each EOS task: `pick_beaker` → `pick_object` (agent `arm`, object `beaker_1`) and `place_beaker` → `place_object` (target `table_1`). A protocol without this file just has no sim twin.
3. `python -m schema.from_eos <eos>/user/beaker_lab` validates both against `catalog.json` (no Isaac Sim needed) and writes `scenes/beaker_lab/` (gym id `Matterix-Lab-BeakerLab-v1`, one scene per lab) plus `matterix_registrations.py`, binding the lab to that scene and each of its tasks (merged across every protocol with a `matterix_workflow.yml` in that lab) to a workflow.

**Every protocol run:**

4. `beaker_lab_pick_and_place_protocol` runs task `pick_beaker` with `backend: sim`. Its `task.py` calls `devices["arm"].pick_beaker(beaker, protocol_run_name, task_name, backend=...)`.
5. `devices/arm/device.py` sees `backend == "sim"` and calls `run_matterix_workflow(..., lab=self.lab_name)` (`common/matterix_backend.py`). That gets or creates the shared `MatterixBackend` Ray actor for this protocol run, so every task in the run shares one physics scene and Isaac Sim boots only once.
6. The bridge resolves (`beaker_lab`, `pick_beaker`) via `common/protocol_registry.py` to (`Matterix-Lab-BeakerLab-v1`, `pickup_beaker`). Keyed by the device's own lab, so the calling protocol never needs to be known or hardcoded.
7. `common/runtime.py` boots Isaac Sim on the first call, builds the `beaker_lab` scene, and runs the pick workflow to completion. `place_beaker` then runs in the same scene.
8. A `WorkflowResult` (success, per-env success, failure detail) comes back to the driver. `run_matterix_workflow()` raises if the workflow failed; otherwise the driver updates the EOS resource (here `beaker.meta["picked"]`) exactly as its real-hardware path would.

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

2. **Register the lab/task -> Matterix bindings**, in `<your_package>/matterix_registrations.py` (plain strings only), using the gym id and workflow keys from step 1:

   ```python
   from user.matterix_bridge.common.protocol_registry import register_lab, register_lab_task_workflow

   register_lab("my_lab", "Matterix-Experiment-My-Scene-v1")
   register_lab_task_workflow("my_lab", "my_task")  # workflow_key defaults to the task name -- pass it explicitly only if it differs
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
                   protocol_run_name, eos_task_name, lab=self.lab_name,
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
