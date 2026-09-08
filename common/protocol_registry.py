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

    Reuses EOS's own package discovery (the same mechanism EOS itself uses to find
    labs/devices/protocols/tasks, and that runtime.py's `_discover_scene_modules()`
    also reuses) rather than a separate, matterix_bridge-specific plugin system.
    """
    global _registrations_discovered
    if _registrations_discovered:
        return
    _registrations_discovered = True

    import importlib
    from pathlib import Path

    from eos.configuration.packages import discover_packages

    from user.matterix_bridge.common.runtime import _eos_path

    user_dir = Path(_eos_path()) / "user"
    for package in discover_packages(user_dir).values():
        if (package.path / "matterix_registrations.py").is_file():
            importlib.import_module(f"user.{package.name}.matterix_registrations")


def resolve_matterix_call(protocol_type: str, eos_task_name: str) -> tuple[str, str]:
    """Resolve an EOS `(protocol_type, task_name)` pair to a Matterix `(task, workflow)` pair.

    Raises KeyError naming the offending side rather than silently mismatching, so a
    rename on either side of the EOS/Matterix boundary is caught immediately instead of
    resolving to the wrong (or a stale) workflow.
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
