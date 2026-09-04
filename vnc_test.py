"""VNC-visible verification for the EOS<->Matterix bridge.

Unlike smoke_test.py (headless, for CI/automated checks), every stage here runs with
headless=False so you can actually watch it on a real display -- from a terminal INSIDE
the VNC desktop session (so DISPLAY is already set correctly for you):

    conda activate isaaclab
    export EOS_PATH=/home/lila/Documents/git/eos
    cd /home/lila/Documents/git/matterix-eos-bridge
    python vnc_test.py            # runs all stages in sequence
    python vnc_test.py 1          # runs only stage 1
    python vnc_test.py 2 3        # runs stages 2 and 3

Stages, in increasing order of how much of the real stack they exercise:

    1. Sanity render   -- exp1_beaker_pick, direct run_workflow() call. Confirms Isaac Sim
                           itself renders on this display at all.
    2. Heater-transfer -- exp3_heater_transfer's "pickup_and_place" bundled workflow,
                           direct run_workflow() call. Watch the semantics engine: heater
                           turns on, beaker gets picked up and placed on the plate, and its
                           temperature should visibly climb in the printed observations.
    3. Real device path -- get_backend() -> MatterixBackend Ray actor -> run_workflow(),
                            the *exact* code path Heater.heat_to() takes from a real EOS
                            task. Exercises the PYTHONPATH/num_gpus/DISPLAY/record_path
                            fixes made to matterix_backend.py and runtime.py.

Isaac Sim boots once and stays alive across stages run in the same invocation (that's the
whole point of runtime.py's design) -- so `python vnc_test.py` (all stages) only pays the
boot cost once. Stage 3 is a separate Ray actor process, so it boots its own, second Isaac
Sim instance the first time it runs.
"""

import os
import sys

EOS_PATH = os.getenv("EOS_PATH", "/home/lila/Documents/git/eos")
if EOS_PATH not in sys.path:
    sys.path.insert(0, EOS_PATH)

os.environ.setdefault("DISPLAY", ":2")


def stage(name: str) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


def ok(msg: str) -> None:
    print(f"  PASS: {msg}")


def run_stage_1() -> None:
    stage("Stage 1: sanity render (exp1_beaker_pick, direct run_workflow())")
    from user.matterix_bridge.common.runtime import run_workflow

    result = run_workflow(
        task="Matterix-Experiment-Beaker-Pick-Franka-v1",
        workflow="pickup_beaker",
        num_envs=1,
        max_episodes=1,
        headless=False,
        print_progress=True,
    )
    print(f"\nRESULT: {result}")
    assert result.success, "workflow did not succeed"
    ok("Stage 1 -- Isaac Sim rendered and the beaker pick succeeded")


def run_stage_2() -> None:
    stage("Stage 2: heater-transfer bundled workflow (exp3, direct run_workflow())")
    from user.matterix_bridge.common.runtime import run_workflow

    result = run_workflow(
        task="Matterix-Experiment-Heater-Transfer-Franka-v1",
        workflow="pickup_and_place",
        num_envs=1,
        max_episodes=1,
        headless=False,
        print_progress=True,
    )
    print(f"\nRESULT: {result}")
    assert result.success, "workflow did not succeed"
    ok("Stage 2 -- heater-transfer pickup_and_place succeeded")


def run_stage_3() -> None:
    stage("Stage 3: real device path (get_backend() -> MatterixBackend actor -> run_workflow())")
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from user.matterix_bridge.common.matterix_backend import get_backend

    backend = get_backend("vnc_test_scope")
    ray.get(
        backend.set_parameter.remote("heater_transfer_protocol", "turn_on_heater", "target_temperature", 363.15)
    )
    result = ray.get(backend.run_workflow.remote("heater_transfer_protocol", "turn_on_heater", headless=False))
    print(f"\nRESULT: {result}")
    assert result.success, "workflow did not succeed"
    ok("Stage 3 -- MatterixBackend actor path succeeded (this is the real EOS-triggered path)")


STAGES = {"1": run_stage_1, "2": run_stage_2, "3": run_stage_3}

if __name__ == "__main__":
    requested = sys.argv[1:] or list(STAGES.keys())
    unknown = [key for key in requested if key not in STAGES]
    if unknown:
        print(f"Unknown stage(s) {unknown}. Choose from: {list(STAGES.keys())}")
        sys.exit(1)

    for key in requested:
        STAGES[key]()

    print(f"\n{'=' * 70}\nALL REQUESTED STAGES PASSED\n{'=' * 70}")
