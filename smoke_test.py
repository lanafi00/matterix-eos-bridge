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

# Set SKIP_STAGE0=1 to jump straight to Stage 1 once Stage 0 has already passed once --
# cuts a slow debug loop (Isaac Sim boot + full exp1 workflow) down to just the stage
# that's actually being debugged.
SKIP_STAGE0 = os.getenv("SKIP_STAGE0") == "1"


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
from user.matterix_bridge.common.runtime import run_workflow

if SKIP_STAGE0:
    print("\nSKIPPING Stage 0 (SKIP_STAGE0=1) -- assuming it already passed once.")
else:
    stage("Stage 0: baseline run_workflow() sanity (exp1_beaker_pick)")
    try:
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


# --- Stage 2: resolve_matterix_call_by_task_name -- the no-protocol_type-needed path
# devices actually use now (see run_matterix_workflow()/Heater.heat_to()). Covers both
# the common unambiguous case and the genuinely-ambiguous case, which should fail loudly
# naming every colliding protocol rather than silently picking one.
stage("Stage 2: protocol_registry.resolve_matterix_call_by_task_name")
try:
    from user.matterix_bridge.common.protocol_registry import resolve_matterix_call_by_task_name

    task, workflow = resolve_matterix_call_by_task_name("turn_on_heater")
    expected = ("Matterix-Experiment-Heater-Transfer-Franka-v1", "turn_on_heater")
    assert (task, workflow) == expected, f"got {(task, workflow)}, expected {expected}"
    ok(f"resolved 'turn_on_heater' unambiguously to {(task, workflow)}, no protocol_type needed")

    try:
        resolve_matterix_call_by_task_name("pick_beaker")
        fail("expected ValueError for ambiguous task name 'pick_beaker', got none")
    except ValueError as e:
        # "pick_beaker" is genuinely registered under beaker_pick_protocol,
        # pick_and_place_protocol, AND heater_transfer_protocol with two different
        # results -- confirm the error actually names the collision, not just that
        # *a* ValueError happened.
        msg = str(e)
        assert "beaker_pick_protocol" in msg and "heater_transfer_protocol" in msg, (
            f"ValueError didn't name the colliding protocols: {msg}"
        )
        ok("ValueError correctly raised for ambiguous 'pick_beaker', naming the colliding protocols")

    try:
        resolve_matterix_call_by_task_name("no_such_task_at_all")
        fail("expected KeyError for a task name registered under no protocol, got none")
    except KeyError:
        ok("KeyError correctly raised for a task name registered under no protocol")
except Exception as e:
    fail("resolve_matterix_call_by_task_name failed", e)


# --- Stage 3: registry introspection helpers force discovery instead of silently showing
# an empty dict to anyone who inspects PROTOCOL_TWINS/TASK_WORKFLOWS before the first real
# resolution has triggered _discover_registrations().
stage("Stage 3: get_protocol_twins()/get_task_workflows() force discovery")
try:
    from user.matterix_bridge.common.protocol_registry import get_protocol_twins, get_task_workflows

    twins = get_protocol_twins()
    workflows = get_task_workflows()
    assert "heater_transfer_protocol" in twins, f"expected heater_transfer_protocol in {twins}"
    assert ("heater_transfer_protocol", "turn_on_heater") in workflows, (
        f"expected ('heater_transfer_protocol', 'turn_on_heater') in {workflows}"
    )
    ok(
        f"get_protocol_twins() returned {len(twins)} protocol(s), "
        f"get_task_workflows() returned {len(workflows)} task(s)"
    )
except Exception as e:
    fail("registry introspection helpers failed", e)


# --- Stage 4: the workflow_overrides mechanism -- the actual point of an earlier session's work.
stage("Stage 4: workflow_overrides actually reaches TurnOnHeaterCfg.target_temperature")
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
stage("Stage 5: workflow_overrides error paths")
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


# --- Stage 6: matterix_asset_types.py -- pure introspection/validation utilities,
# no registry (that layer was collapsed back into device_registry.py -- see its module
# docstring). Covers discover_matterix_assets() finding every real asset category, and
# validate_asset_cfg() accepting a real class / rejecting a fake one directly.
stage("Stage 6: matterix_asset_types.discover_matterix_assets / validate_asset_cfg")
try:
    from user.matterix_bridge.common.matterix_asset_types import (
        discover_matterix_assets,
        validate_asset_cfg,
    )
    from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

    discovered = discover_matterix_assets()
    categories_found = {name.split("/", 1)[0] for name in discovered}
    assert categories_found == {"robots", "equipment", "labware", "infrastructure"}, (
        f"expected all 4 asset categories, got {categories_found}"
    )
    assert discovered.get("robots/franka_panda_high_pd_ik") is FRANKA_PANDA_HIGH_PD_IK_CFG, (
        "discover_matterix_assets() didn't find the known Franka IK arm under the expected name"
    )
    ok(f"discover_matterix_assets() found {len(discovered)} real assets across all 4 categories")

    validate_asset_cfg(FRANKA_PANDA_HIGH_PD_IK_CFG)  # raises on failure
    ok("validate_asset_cfg() accepted a real, known-good asset class")

    class NotARealAsset:
        def __init__(self, pos=None, rot=None):
            pass

    try:
        validate_asset_cfg(NotARealAsset)
        fail("expected TypeError for a class producing no Matterix asset base type, got none")
    except TypeError:
        ok("TypeError correctly raised validating a class producing no Matterix asset base type")
except Exception as e:
    fail("matterix_asset_types discovery/validation failed", e)


# --- Stage 7: device_registry.py -- DEVICE_TWINS stores classes directly (no catalog
# indirection). Covers resolve_device_twin() resolving a real binding, a KeyError for an
# unregistered device, and register_device_twin()'s own validation/conflict checks.
stage("Stage 7: device_registry.register_device_twin / resolve_device_twin")
try:
    from user.matterix_bridge.common.device_registry import register_device_twin, resolve_device_twin
    from matterix_assets.robots import FRANKA_PANDA_HIGH_PD_IK_CFG

    existing = FRANKA_PANDA_HIGH_PD_IK_CFG(pos=(1.0, 2.0, 3.0))
    twin = resolve_device_twin("example_lab", "franka_01", existing)
    assert twin.pos == existing.pos, "twin lost the existing slot's position"
    ok(f"resolved twin for ('example_lab', 'franka_01'), position preserved: {twin.pos}")

    class NotARealAsset:
        def __init__(self, pos=None, rot=None):
            pass

    try:
        register_device_twin("some_lab", "some_device", NotARealAsset)
        fail("expected TypeError registering a device against a non-Matterix-asset class, got none")
    except TypeError:
        ok("TypeError correctly raised registering a device against a non-Matterix-asset class")

    # A distinct but still-valid callable, not NotARealAsset -- validate_asset_cfg() runs
    # before the conflict check, so an invalid class would raise TypeError there and never
    # actually exercise the conflict path this is meant to test.
    def _a_different_but_valid_asset(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)):
        return FRANKA_PANDA_HIGH_PD_IK_CFG(pos=pos, rot=rot)

    try:
        register_device_twin("example_lab", "franka_01", _a_different_but_valid_asset)
        fail("expected ValueError for re-registering an existing device with a different class, got none")
    except ValueError:
        ok("ValueError correctly raised for a conflicting re-registration without overwrite=True")

    try:
        resolve_device_twin("no_such_lab", "no_such_device", existing)
        fail("expected KeyError for unregistered device, got none")
    except KeyError:
        ok("KeyError correctly raised for unregistered (lab_name, device_name)")
except Exception as e:
    fail("device_registry resolution failed", e)


# --- Stage 8: the MatterixBackend Ray actor, end to end.
stage("Stage 8: MatterixBackend actor (get_backend, set_parameter, run_workflow)")
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


# --- Stage 8: set_parameters() (plural, batched) and run_matterix_workflow() (the
# get_backend/set_parameters/run_workflow/check-success wrapper device.py actually calls)
# -- neither was exercised above, and both are what a real device driver uses now, with
# protocol_type omitted (None), the same no-protocol_type-needed path Stage 2 covers.
# Reuses Stage 7's "smoke_test_scope" (not a new scope_id) -- MatterixBackend actors are
# lifetime="detached" and each distinct scope_id claims its own num_gpus=1, so a second
# scope here would compete with Stage 7's still-alive actor for this machine's one GPU
# and hang in PENDING_CREATION rather than fail loudly (the idle-timeout sweep only frees
# actors idle 15+ minutes, nowhere near this script's runtime).
stage("Stage 9: set_parameters() batch setter and run_matterix_workflow() helper")
try:
    from user.matterix_bridge.common.matterix_backend import run_matterix_workflow

    # Two fields in one call -- confirms set_parameters() (not just set_parameter())
    # actually gets exercised, and that protocol_type=None resolves correctly through
    # the real Ray actor, not just the bare protocol_registry functions Stage 2 tested.
    result = run_matterix_workflow(
        "smoke_test_scope",
        "turn_on_heater",
        headless=True,
        target_temperature=320.0,
    )
    ok(f"run_matterix_workflow() completed with protocol_type omitted: success={result.success}")

    try:
        run_matterix_workflow("smoke_test_scope", "no_such_task_at_all", headless=True)
        fail("expected KeyError for an unregistered task name, got none")
    except KeyError:
        ok("KeyError correctly propagated through run_matterix_workflow() for an unregistered task name")
except Exception as e:
    fail("run_matterix_workflow()/set_parameters() test failed", e)


print(f"\n{'=' * 70}\nALL STAGES PASSED\n{'=' * 70}")
