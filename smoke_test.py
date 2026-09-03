"""Smoke test for the EOS<->Matterix bridge, run inside the isaaclab conda env.

Run over VNC with:
    conda activate isaaclab
    export EOS_PATH=/home/lila/Documents/git/eos
    python matterix_bridge_smoke_test.py

Everything (runtime.py, the registries, the scene definitions) now lives in
matterix_bridge (matterix-eos-bridge/{common,scenes}/) -- matterix-experiments no longer
has any functional role, this script doesn't touch it at all.

Staged on purpose -- each stage is independent so a failure tells you exactly what's
broken instead of one opaque crash. Stops at the first failing stage by default; comment
out the `sys.exit(1)` calls if you want it to keep going and report everything.
"""

import os
import sys

EOS_PATH = os.getenv("EOS_PATH", "/home/lila/Documents/git/eos")
if EOS_PATH not in sys.path:
    sys.path.insert(0, EOS_PATH)


def stage(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


def ok(msg):
    print(f"  PASS: {msg}")


def fail(msg, exc=None):
    print(f"  FAIL: {msg}")
    if exc is not None:
        import traceback

        traceback.print_exc()
    sys.exit(1)


# --- Stage 0: baseline sanity -- is Matterix/Isaac Sim itself working at all.
# Uses exp1_beaker_pick since that's the one the matterix-experiments README said was most
# validated -- isolates "is Matterix working" from "does the bridge's own logic work"
# (stages 1+ exercise bridge-specific code: registries, workflow_overrides, MatterixBackend).
stage("Stage 0: baseline run_workflow() sanity (exp1_beaker_pick)")
try:
    from user.matterix_bridge.common.runtime import run_workflow

    result = run_workflow(
        task="Matterix-Experiment-Beaker-Pick-Franka-v1",
        workflow="pickup_beaker",
        num_envs=1,
        max_episodes=1,
        headless=True,
        print_progress=False,
    )
    ok(f"run_workflow() completed: success={result.success}")
except Exception as e:
    fail("baseline run_workflow() call failed -- fix this before testing the bridge", e)


# --- Stage 1: protocol_registry.py resolves correctly.
stage("Stage 1: protocol_registry.resolve_matterix_call")
try:
    from user.matterix_bridge.common.protocol_registry import resolve_matterix_call

    task, workflow = resolve_matterix_call("heater_transfer_protocol", "turn_on_heater")
    expected = ("Matterix-Experiment-Heater-Transfer-Franka-v1", "turn_on_heater")
    assert (task, workflow) == expected, f"got {(task, workflow)}, expected {expected}"
    ok(f"resolved to {(task, workflow)}")

    try:
        resolve_matterix_call("nonexistent_protocol", "x")
        fail("expected KeyError for unregistered protocol_type, got none")
    except KeyError:
        ok("KeyError correctly raised for unregistered protocol_type")
except Exception as e:
    fail("protocol_registry import/resolution failed", e)


# --- Stage 2: the workflow_overrides mechanism -- the actual point of this session's work.
stage("Stage 2: workflow_overrides actually reaches TurnOnHeaterCfg.target_temperature")
try:
    import user.matterix_bridge.common.runtime as rt

    target_kelvin = 300.0  # deliberately not 373.15 (the authored default) so a no-op would be caught
    result = run_workflow(
        task="Matterix-Experiment-Heater-Transfer-Franka-v1",
        workflow="turn_on_heater",
        workflow_overrides={"target_temperature": target_kelvin},
        num_envs=1,
        max_episodes=1,
        headless=True,
        print_progress=False,
    )
    ok(f"run_workflow() with workflow_overrides completed: success={result.success}")

    # Reaching into runtime's private cached env_cfg -- fine for a diagnostic smoke test,
    # not something to do in real device code.
    actual = rt._current_env_cfg.workflows["turn_on_heater"].target_temperature
    assert actual == target_kelvin, f"override did not take effect: env_cfg has {actual}, expected {target_kelvin}"
    ok(f"env_cfg.workflows['turn_on_heater'].target_temperature == {actual} (override took effect)")
except Exception as e:
    fail("workflow_overrides did not work as expected", e)


# --- Stage 3: workflow_overrides error paths.
stage("Stage 3: workflow_overrides error paths")
try:
    try:
        run_workflow(
            task="Matterix-Experiment-Heater-Transfer-Franka-v1",
            workflow="turn_on_heater",
            workflow_overrides={"not_a_real_field": 1},
            headless=True,
            print_progress=False,
        )
        fail("expected ValueError for unknown override field, got none")
    except ValueError:
        ok("ValueError correctly raised for unknown workflow_overrides field")
except Exception as e:
    fail("error-path check itself failed", e)


# --- Stage 4: device_registry.py resolves correctly (direct sibling import now, no shim).
stage("Stage 4: device_registry.resolve_device_twin")
try:
    from user.matterix_bridge.common.device_registry import resolve_device_twin
    from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

    existing = FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(1.0, 2.0, 3.0))
    twin = resolve_device_twin("example_lab", "franka_01", existing)
    assert twin.pos == existing.pos, "twin lost the existing slot's position"
    ok(f"resolved twin for ('example_lab', 'franka_01'), position preserved: {twin.pos}")

    try:
        resolve_device_twin("no_such_lab", "no_such_device", existing)
        fail("expected KeyError for unregistered device, got none")
    except KeyError:
        ok("KeyError correctly raised for unregistered (lab_name, device_name)")
except Exception as e:
    fail("device_registry resolution failed", e)


# --- Stage 5: the MatterixBackend Ray actor, end to end.
stage("Stage 5: MatterixBackend actor (get_backend, set_parameter, run_workflow)")
try:
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from user.matterix_bridge.common.matterix_backend import get_backend

    backend = get_backend("smoke_test_scope")
    ray.get(
        backend.set_parameter.remote("heater_transfer_protocol", "turn_on_heater", "target_temperature", 310.0)
    )
    result = ray.get(backend.run_workflow.remote("heater_transfer_protocol", "turn_on_heater"))
    ok(f"MatterixBackend.run_workflow() completed via Ray: success={result.success}")

    backend2 = get_backend("smoke_test_scope")
    assert backend == backend2, "get_backend did not return the SAME actor for the same scope_id"
    ok("get_backend() correctly returned the same actor for a repeated scope_id (get_if_exists working)")
    # NOTE: this only confirms reuse *within* this one script/process's default Ray
    # namespace. Ray warns that a detached actor created in an anonymous namespace isn't
    # reliably reconnectable from a SEPARATE process/script without ray.init(namespace=...)
    # matching -- worth setting an explicit namespace before this is used for real, so a
    # later EOS task in the same protocol run (a different Ray worker) reconnects correctly.
except Exception as e:
    fail("MatterixBackend actor test failed", e)


print(f"\n{'=' * 70}\nALL STAGES PASSED\n{'=' * 70}")
