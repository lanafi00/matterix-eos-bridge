"""Maps EOS device identities to the Matterix twin asset config that backs them.

Keyed exactly the way EOS names a device - `(lab_name, device_name)` - so a device
assignment coming out of an EOS `ScheduledTask.devices` resolves to a concrete twin
config with a single dict lookup. This is *twin substitution*: letting one already-
authored scene's actuated slot (e.g. "robot" in `articulated_assets`) be filled by
different concrete hardware depending on which lab/device is calling, without writing
a new scene per physical unit. Most new devices don't need this at all - only ones that
occupy an articulated-asset slot with more than one possible physical implementation
across labs/deployments. See `resolve_device_twin()`'s docstring and `run_workflow()`'s
`devices=` parameter in runtime.py.

Populated via `register_device_twin()` rather than edited as literal dict entries, so a
package other than this one can register its own lab's device -> twin bindings from its
own code instead of editing this file directly - call it from your package's top-level
`scenes/__init__.py` (the same file where you `gym.register()` your scene, and the same
file `protocol_registry.py`'s docstring points to for this). Not a package-root
`__init__.py` - EOS packages here are plain namespace packages with none of those (see
this repo's own `user/matterix_bridge`, or EOS's own `user/example`). `scenes/__init__.py`
is the right spot because it's auto-imported by runtime.py's `_discover_scene_modules()`
after Isaac Sim has booted, which twin config classes need (they come from
`matterix_assets`, an isaaclab-dependent package) - see `protocol_registry.py`'s
docstring for why that's also why this can't go in `matterix_registrations.py`.
"""

from __future__ import annotations

from typing import Callable

from matterix_assets.matterix_articulation import MatterixArticulationCfg

DEVICE_TWINS: dict[tuple[str, str], Callable[..., MatterixArticulationCfg]] = {}


def validate_device_twin(twin_cfg: Callable[..., MatterixArticulationCfg]) -> None:
    """Confirm `twin_cfg` actually produces a real Matterix asset -- a
    `MatterixArticulationCfg` backed by a USD file that actually exists -- instead of
    accepting any callable at registration time and only discovering it's broken much
    later, when some protocol run first calls `resolve_device_twin()` on it.

    Instantiates `twin_cfg` with placeholder pos/rot to inspect the result -- safe to do
    here: this only ever runs from `register_device_twin()`, called from a package's
    `scenes/__init__.py` (itself only imported after Isaac Sim has booted, per this
    module's own docstring), and building a `@configclass` instance is plain Python/
    dataclass construction with no GPU or simulation side effects -- the same thing
    `resolve_device_twin()` does for real later with the caller-supplied pos/rot.

    Existence is checked via `isaaclab.utils.assets.check_file_path()`, NOT a bare
    `os.path.isfile()` -- VERIFIED this actually matters: `matterix_assets`' own
    `FRANKA_PANDA_HIGH_PD_IK_CFG` (a real, working, already-registered twin) has a
    `usd_path` that's a plain `https://` URL onto NVIDIA's Nucleus/CDN asset server, not
    a local file at all. A local-only check would have rejected that twin outright and
    broken `scenes/__init__.py`'s own module-level registration of it -- caught live
    before landing. `check_file_path()` is Isaac Lab's own utility for exactly this
    (local-or-remote) distinction; it returns 0 if the path doesn't exist anywhere, 1 if
    it's a real local file, 2 if it resolves on the Nucleus server (checked via a real
    `omni.client.stat()` call, so this does real I/O -- registering a remote-backed twin
    costs a network round-trip, once, at registration time).

    Raises TypeError if `twin_cfg(...)` can't be instantiated at all, or doesn't produce
    a `MatterixArticulationCfg`. Raises ValueError if it does but has no `spawn.usd_path`
    (not backed by a real Matterix USD asset at all), or that path resolves to neither a
    local file nor a Nucleus asset (a typo, or a moved/deleted asset) -- naming the exact
    problem, not just "invalid twin", so it's obvious from the error alone what to fix.
    """
    try:
        instance = twin_cfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    except Exception as e:
        raise TypeError(f"{twin_cfg!r} could not be instantiated as a device twin: {e}") from e

    if not isinstance(instance, MatterixArticulationCfg):
        raise TypeError(f"{twin_cfg!r} does not produce a MatterixArticulationCfg (got {type(instance)!r}).")

    usd_path = getattr(instance.spawn, "usd_path", None)
    if usd_path is None:
        raise ValueError(
            f"{twin_cfg!r} has no spawn.usd_path -- it isn't backed by a real Matterix "
            f"USD asset (spawn={instance.spawn!r})."
        )

    from isaaclab.utils.assets import check_file_path

    if check_file_path(usd_path) == 0:
        raise ValueError(f"{twin_cfg!r}'s USD file does not exist locally or on the Nucleus server: {usd_path!r}")


def register_device_twin(
    lab_name: str,
    device_name: str,
    twin_cfg: Callable[..., MatterixArticulationCfg],
    *,
    overwrite: bool = False,
) -> None:
    """Bind an EOS `(lab_name, device_name)` device to the Matterix twin config class
    that represents it in an articulated-asset slot (see this module's docstring for
    when you do - and don't - need this).

    Raises ValueError if the pair is already bound to a *different* twin and
    `overwrite` isn't set. Re-registering the same (key, twin_cfg) pair is a no-op.

    Also raises (TypeError/ValueError, via `validate_device_twin()`) if `twin_cfg`
    doesn't actually produce a real, USD-backed Matterix asset -- fail loudly here, at
    registration time, rather than accepting anything callable and only discovering a
    typo'd path or wrong class much later when a protocol run tries to use it.
    """
    validate_device_twin(twin_cfg)

    key = (lab_name, device_name)
    existing = DEVICE_TWINS.get(key)
    if existing is not None and existing is not twin_cfg and not overwrite:
        raise ValueError(
            f"Device {key!r} is already registered to twin {existing!r} (tried to "
            f"register {twin_cfg!r}). Pass overwrite=True if this is intentional."
        )
    DEVICE_TWINS[key] = twin_cfg


def resolve_device_twin(
    lab_name: str, device_name: str, existing: MatterixArticulationCfg
) -> MatterixArticulationCfg:
    """Build the twin config for `(lab_name, device_name)`, keeping `existing`'s placement.

    `existing` is the slot's current config from the env cfg being overridden - its
    `pos`/`rot` describe where that slot sits in the scene (a property of the scene
    layout), not which physical device fills it, so they carry over across the swap.
    """
    key = (lab_name, device_name)
    if key not in DEVICE_TWINS:
        raise KeyError(
            f"No twin registered for device {key!r}. Call register_device_twin() for "
            "it (see matterix_bridge/common/device_registry.py)."
        )
    return DEVICE_TWINS[key](pos=existing.pos, rot=existing.rot)
