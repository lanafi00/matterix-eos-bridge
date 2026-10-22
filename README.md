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
