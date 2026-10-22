# matterix_bridge

Bridges Matterix digital-twin simulation into EOS, for virtual commissioning of lab protocols before running on real hardware.

## Install

1. Install `matterix`, `matterix_assets`, `matterix_sm`, `matterix_tasks` (editable) into a conda env that has Isaac Lab.
2. Install `eos` into that same env.
3. Symlink this repo into `<eos repo>/user/matterix_bridge`.
4. `eos start` from that env.

## Use from an EOS package

- Generate a scene per lab: `python -m schema.from_eos <your_package>`. Devices resolve it by naming convention (`schema/naming.py`), nothing to register.
- Device drivers call `run_matterix_workflow(...)` from `common/matterix_backend.py` when `backend == "sim"`.

See `schema/models.py` for the full schema.
# matterix_bridge

Bridges Matterix digital-twin simulation into EOS, for virtual commissioning of lab protocols before running on real hardware.

## Install

1. Install `matterix`, `matterix_assets`, `matterix_sm`, `matterix_tasks` (editable) into a conda env that has Isaac Lab.
2. Install `eos` into that same env.
3. Symlink this repo into `<eos repo>/user/matterix_bridge`, so that it can be discovered by EOS's discover_packages().
4. `eos start` from that env.

## Set up a lab to be simulated

1. Within `eos/user/<package>/labs/<lab_name>/lab.yml`, add simulation metadata for each device under its `meta` field:

- `matterix_catalog`: which Matterix model to use (see the `assets` entries in this repo's `catalog.json`)
- `pos`: position in meters `[x, y, z]`, with z up, relative to the scene origin (the robot sits at `[0, 0, 0]`)
- `semantics` (optional): non-physical behavior such as heating. Use the `heater` preset for things that make heat and `heat_transfer` for things that warm up or cool down. Valid presets and their parameters are listed under `semantic_presets` in `catalog.json`.

For example:

    hotplate:
      ...
      meta:
        matterix_catalog: equipment/ika_plate_inst
        pos: [0.4, -0.3, 0.12]
        semantics:
          - preset: heater
            params: {temp: 298.15, C: 500.0, A: 0.05, heater_on: false, target_temperature: 298.15, K_heater: 0.2}

2. Within `eos/user/<package>/protocols/<protocol_name>/protocol.yml`, add simulation metadata for each task under its `matterix` field. Valid actions are listed in `catalog.json`, under "actions", along with each one's parameters and defaults. `from_eos.py` turns each block into a workflow in the generated scene, named after the task.

For example:

    tasks:
      - name: pick_vial_from_hotplate
        type: Pick Vial
        desc: Take the vial off the hotplate
        ...
        matterix:
          action: pick_object
          params:
            description: Pick the vial off the hotplate
            agent_assets: arm
            object: vial_1
        dependencies: [place_vial_on_hotplate]

Asset names (`agent_assets`, `object`, `target`) must match device/resource names in `lab.yml`, and `agent_assets` must be a robot.

3. Within `eos/user/<package>/tasks/<task_name>/task.yml`, add `backend` and `headless` to its `input_parameters` to toggle sim vs real between runs.

For example:

    backend:
      type: choice
      choices: [sim, real]
      value: sim
      desc: >
        "sim" runs this call on the Matterix digital twin; "real" calls the hardware driver
        (a stub until one is wired up).
    headless:
      type: bool
      value: true
      desc: Run Isaac Sim without opening its window.

4. In each task's `eos/user/<package>/tasks/<task_name>/task.py`, pass these four extra arguments to the device method you call:

- `protocol_run_name=self._protocol_run_name`
- `eos_task_name=self._task_name`
- `backend=parameters["backend"]`
- `headless=parameters["headless"]`

The device uses them to run the right simulated workflow. You can bundle them into a dict and pass `**dict` to save typing.

For instance:

    class HeatHold(BaseTask):
        async def _execute(
            self,
            devices: BaseTask.DevicesType,
            parameters: BaseTask.ParametersType,
            resources: BaseTask.ResourcesType,
        ) -> BaseTask.OutputType:
            run = {
                "protocol_run_name": self._protocol_run_name,
                "eos_task_name": self._task_name,
                "backend": parameters["backend"],
                "headless": parameters["headless"],
            }
            temperature_c = devices["hotplate"].hold(
                resources["vial"], parameters["duration_s"], parameters["min_temp_c"], **run
            )
            return {"temperature_c": temperature_c}, resources, None

5. Within `eos/user/<package>/devices/<device_name>/device.py`, add `from user.matterix_bridge.common.matterix_backend import run_matterix_workflow`. Then add an `if backend == "sim":` block to every method that will be simulated, calling `run_matterix_workflow(...)`.

For example:

    def pick(
            self,
            item: Resource,
            protocol_run_name: str,
            eos_task_name: str,
            backend: str = "sim",
            headless: bool = True,
        ) -> Resource:
            if backend == "sim":
                run_matterix_workflow(protocol_run_name, eos_task_name, lab=self.lab_name, headless=headless)
            else:
                pass # TODO: self._client.pick(...)

            item.meta["location"] = "gripper"
            return item

To set an action's parameter at runtime instead of in `protocol.yml`, pass it as a keyword argument to `run_matterix_workflow`, e.g. `target_temperature=350.0`.

## Steps to run from an EOS package

1. Follow the steps above to edit `lab.yml`, `protocol.yml`, `task.yml`, `task.py`, and `device.py`.
2. From this repo's root, generate a scene + registrations: `python -m schema.from_eos <path to your package>`. Rerun this after any edit to `lab.yml` or a `matterix:` block.
3. Run `eos start` from the conda env that has Isaac Lab. Restart it after editing `task.py`, `task.yml`, or `device.py`.
4. Submit the protocol, whether through the CLI or the EOS web UI.

See `schema/models.py` for the full schema.

## Adding custom labware

If you want to simulate a piece of labware that is not included in `catalog.json`, you can add your own 3D model. Files go in `eos/user/<package>/assets/assets.yml` and `eos/user/<package>/assets/usd/<model>.usda`.

Describe each model in `assets.yml`:

    vial_20ml:
      kind: rigid                  # rigid = can be moved/picked; static = fixed in place
      usd: usd/vial_20ml.usda      # relative to assets/
      mass: 0.03                   # kg, required for rigid
      frames:                      # gripper targets in meters, relative to the model's origin
        pre_grasp: [0.0, 0.0, 0.10]
        grasp: [0.0, 0.0, 0.045]
        post_grasp: [0.0, 0.0, 0.10]

Then use it in `lab.yml` with `matterix_catalog: <package>/<asset>` (e.g. `ncsu_lab/vial_20ml`).

- Objects that get picked need `pre_grasp`, `grasp`, and `post_grasp` frames. Objects that things get placed onto need `pre_place` and `place`.
- Only `rigid` objects can have frames, so make place targets `rigid`.
- USD files must be Z-up, in meters, with physics already set up. See `schema/package_assets.py` for details.