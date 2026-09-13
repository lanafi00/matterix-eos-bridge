"""Maps EOS protocol/task identities to the Matterix task/workflow that validates them.

Two registries, mirroring `device_registry.py`'s pattern:

- `PROTOCOL_TWINS` - EOS protocol type -> Matterix gym task id. Picks which scene
  "shape" (objects, observations, workflow definitions) backs a given protocol.
- `TASK_WORKFLOWS` - (EOS protocol type, EOS task name) -> Matterix workflow key.
  Picks which single `env_cfg.workflows` entry backs one EOS DAG task node.

Populated via `register_protocol()`/`register_task_workflow()` rather than edited as
literal dict entries, so a package other than this one (e.g. your own EOS package
declaring its own protocol) can register its bindings from its own code instead of
editing this file directly. Concretely: put those calls at module level in a file named
`matterix_registrations.py` at your package's root (a sibling of its `pyproject.toml`,
`labs/`, `devices/`, etc. -- see this repo's own `matterix_registrations.py` for the
exact pattern). `_discover_registrations()` below imports every loaded EOS package's
`matterix_registrations.py` automatically, the first time `resolve_matterix_call()` is
called in this process, so you never call `register_protocol()` yourself by hand --
you only need the file to exist with those calls in it.

Keep `matterix_registrations.py` import-safe without isaaclab/matterix_assets
installed or Isaac Sim running -- only reference plain strings here (protocol types,
gym task ids, task/workflow names), never a twin config class. That's because
protocol/task resolution can happen before Isaac Sim boots (e.g. inside
`MatterixBackend.set_parameter()`, which never touches runtime.py at all), so this
discovery is deliberately separate from -- and runs earlier than -- runtime.py's
`_discover_scene_modules()`, which imports each package's `scenes/` subpackage (that
one is fine importing isaaclab-dependent code, since it only ever runs after Isaac Sim
has booted). Device twins, which DO need isaaclab-dependent classes, are registered
from `scenes/__init__.py` instead, for exactly this reason.

These are explicit mappings rather than a naming convention (e.g. "assume the EOS task
name always matches the Matterix workflow key") on purpose: a rename on either side -
an EOS protocol definition or one of this repo's `workflows` dicts - should fail loudly
via `resolve_matterix_call()`'s KeyError, not silently resolve to the wrong workflow.

Each Matterix `workflows` dict has one atomic entry per EOS DAG task node (what an
EOS-facing caller dispatches one at a time), plus - where useful for manual/dev CLI
testing - bundled entries that chain several atomic steps in one call. Only the atomic
entries are meant to be reached through this registry.
"""

from __future__ import annotations

PROTOCOL_TWINS: dict[str, str] = {}
TASK_WORKFLOWS: dict[tuple[str, str], str] = {}

_registrations_discovered = False


def register_protocol(protocol_type: str, gym_task_id: str, *, overwrite: bool = False) -> None:
    """Bind an EOS protocol type to the Matterix gym task id (scene) that backs it.

    Raises ValueError if `protocol_type` is already bound to a *different* gym task id
    and `overwrite` isn't set -- two packages silently fighting over the same protocol
    type should fail loudly, not resolve to whichever one happened to import last.
    Re-registering the same (protocol_type, gym_task_id) pair is always a no-op.
    """
    existing = PROTOCOL_TWINS.get(protocol_type)
    if existing is not None and existing != gym_task_id and not overwrite:
        raise ValueError(
            f"Protocol type {protocol_type!r} is already registered to gym task "
            f"{existing!r} (tried to register {gym_task_id!r}). Pass overwrite=True "
            "if this is intentional."
        )
    PROTOCOL_TWINS[protocol_type] = gym_task_id


def register_task_workflow(
    protocol_type: str, eos_task_name: str, workflow_key: str | None = None, *, overwrite: bool = False
) -> None:
    """Bind one EOS DAG task node to the Matterix workflow key that executes it.

    workflow_key defaults to eos_task_name when omitted -- the common case (this repo's
    own matterix_registrations.py needs it explicit for only 1 of its 9 registrations;
    the rest use the same name on both sides). This is a convenience default the caller
    opts into by leaving the argument out, not a silent naming-convention fallback: the
    module docstring's "why explicit mappings" reasoning still holds -- if you never call
    register_task_workflow() for a given task at all, resolve_matterix_call() still
    raises KeyError rather than guessing. Pass workflow_key explicitly only when the
    Matterix workflow key genuinely differs from the EOS task name.

    Same overwrite semantics as `register_protocol()`. `protocol_type` doesn't need to
    already be registered via `register_protocol()` at call time -- the two registries
    are independent dicts, only joined together inside `resolve_matterix_call()`.
    """
    if workflow_key is None:
        workflow_key = eos_task_name
    key = (protocol_type, eos_task_name)
    existing = TASK_WORKFLOWS.get(key)
    if existing is not None and existing != workflow_key and not overwrite:
        raise ValueError(
            f"Task {key!r} is already registered to workflow {existing!r} (tried to "
            f"register {workflow_key!r}). Pass overwrite=True if this is intentional."
        )
    TASK_WORKFLOWS[key] = workflow_key


def _discover_registrations() -> None:
    """Import every EOS package's `matterix_registrations.py`, if it has one -- once per
    process. See this module's docstring for the convention this implements.

    Uses `runtime.py`'s `_discover_user_package_dirs()`, NOT EOS's own package discovery
    (`eos.configuration.packages.discover_packages()`) -- deliberately, because this runs
    inside `MatterixBackend`'s actor (see `matterix_backend.py`), which `get_backend()`
    can relocate via `runtime_env={"conda": ...}` to a conda env that has isaaclab/
    matterix but not `eos` itself. Importing `eos.configuration.packages` there pulls in
    EOS's full entity/pydantic model tree (`LabDef` -> `bofire`) as a side effect and
    fails with `ModuleNotFoundError: No module named 'bofire'` before this function's own
    logic ever runs -- see `_discover_user_package_dirs()`'s docstring for the verified
    repro. Same reason `_discover_scene_modules()` in runtime.py avoids it too.
    """
    global _registrations_discovered
    if _registrations_discovered:
        return
    _registrations_discovered = True

    import importlib
    from pathlib import Path

    from user.matterix_bridge.common.runtime import _discover_user_package_dirs, _eos_path

    user_dir = Path(_eos_path()) / "user"
    for package_dir in _discover_user_package_dirs(user_dir):
        if (package_dir / "matterix_registrations.py").is_file():
            importlib.import_module(f"user.{package_dir.name}.matterix_registrations")


def get_protocol_twins() -> dict[str, str]:
    """The current `PROTOCOL_TWINS` mapping, forcing registration discovery first.

    Use this (not the bare `PROTOCOL_TWINS` module attribute) if you want to inspect
    what's registered -- `PROTOCOL_TWINS` only gets populated lazily, the first time
    `resolve_matterix_call()`/`resolve_matterix_call_by_task_name()` runs in this
    process, so reading it directly before that can silently show an empty dict with no
    hint why. Returns the live dict, not a copy -- treat it as read-only.
    """
    _discover_registrations()
    return PROTOCOL_TWINS


def get_task_workflows() -> dict[tuple[str, str], str]:
    """The current `TASK_WORKFLOWS` mapping, forcing registration discovery first -- see
    `get_protocol_twins()`'s docstring for why this exists instead of reading
    `TASK_WORKFLOWS` directly. Returns the live dict, not a copy -- treat it as read-only.
    """
    _discover_registrations()
    return TASK_WORKFLOWS


def resolve_matterix_call(protocol_type: str, eos_task_name: str) -> tuple[str, str]:
    """Resolve an EOS `(protocol_type, task_name)` pair to a Matterix `(task, workflow)` pair.

    Raises KeyError naming the offending side rather than silently mismatching, so a
    rename on either side of the EOS/Matterix boundary is caught immediately instead of
    resolving to the wrong (or a stale) workflow.

    Most callers don't actually need to supply `protocol_type` -- see
    `resolve_matterix_call_by_task_name()`, which resolves from `eos_task_name` alone
    whenever that's unambiguous (the common case) and only requires disambiguation when
    it genuinely isn't. Use this function directly only when you already know
    `protocol_type` and want the registry to double-check it (e.g. inside
    `MatterixBackend`, which already has it because `run_matterix_workflow()`'s caller
    supplied it).
    """
    _discover_registrations()
    if protocol_type not in PROTOCOL_TWINS:
        raise KeyError(
            f"No Matterix task registered for EOS protocol {protocol_type!r}. Call "
            "register_protocol() for it (see matterix_bridge/common/protocol_registry.py)."
        )
    key = (protocol_type, eos_task_name)
    if key not in TASK_WORKFLOWS:
        raise KeyError(
            f"No Matterix workflow registered for EOS task {key!r}. Call "
            "register_task_workflow() for it (see matterix_bridge/common/protocol_registry.py)."
        )
    return PROTOCOL_TWINS[protocol_type], TASK_WORKFLOWS[key]


def resolve_matterix_call_by_task_name(eos_task_name: str) -> tuple[str, str]:
    """Resolve a Matterix `(task, workflow)` pair from JUST an EOS task name, without
    needing its `protocol_type`.

    This exists because `protocol_type` isn't actually something a device driver can get
    for free: `BaseTask` only exposes `protocol_run_name` (a run instance id) and
    `task_name` to a task, not the protocol's own type string, so getting `protocol_type`
    into a device call has historically meant threading it through as a redundant,
    hand-maintained task parameter (e.g. a `matterix_protocol_type` value in `task.yml`,
    restating what `protocol.yml`'s own `type:` field already says) -- copy a
    protocol.yml+task.yml pair to start a new protocol, forget to update that one nested
    value, and it silently resolves against the wrong (or a stale) protocol's
    registrations. Most EOS task names are unique across every registered protocol, so
    there's usually no real ambiguity to resolve in the first place -- this function
    finds that out for you instead of asking the caller to know a fact they don't have.

    Raises KeyError if `eos_task_name` isn't registered under any protocol at all.
    Raises ValueError, naming every colliding protocol_type, if `eos_task_name` IS
    registered under two or more protocols that resolve to DIFFERENT `(gym_task_id,
    workflow_key)` pairs -- genuinely ambiguous without knowing which protocol is
    calling. That's the one case where you still need `resolve_matterix_call(
    protocol_type, eos_task_name)` and a real `protocol_type` for that specific task --
    everything else can drop it.
    """
    _discover_registrations()
    candidates: dict[tuple[str, str], list[str]] = {}
    for (protocol_type, task_name), workflow_key in TASK_WORKFLOWS.items():
        if task_name != eos_task_name:
            continue
        gym_task_id = PROTOCOL_TWINS.get(protocol_type)
        if gym_task_id is None:
            continue  # registered task, but its protocol_type was never registered -- not a candidate.
        candidates.setdefault((gym_task_id, workflow_key), []).append(protocol_type)

    if not candidates:
        raise KeyError(
            f"No Matterix workflow registered for EOS task {eos_task_name!r} under any "
            "protocol. Call register_task_workflow() for it (see "
            "matterix_bridge/common/protocol_registry.py)."
        )
    if len(candidates) > 1:
        detail = "; ".join(f"{result!r} via protocol(s) {ptypes}" for result, ptypes in candidates.items())
        raise ValueError(
            f"EOS task {eos_task_name!r} resolves ambiguously across protocols: {detail}. "
            "Call resolve_matterix_call(protocol_type, eos_task_name) explicitly for this "
            "task instead, with a real protocol_type."
        )
    return next(iter(candidates))
