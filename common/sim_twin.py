"""`@sim_twin`: the sim/real switch for a device method, so a device.py never has to
write its own `if backend == "sim": run_matterix_workflow(...)` branch.

    from user.matterix_bridge.common.sim_twin import sim_twin

    class Arm(BaseDevice):
        @sim_twin(returns="beaker")
        def pick_beaker(self, beaker, protocol_run_name, eos_task_name, backend="sim", headless=True):
            self._client.send_command("pick", {})   # real hardware only
            return beaker

The decorated method's body is the REAL-hardware path. On a call with
`backend="real"` it runs unchanged. On a call with `backend="sim"` the body is skipped
entirely and the method instead runs `run_matterix_workflow(protocol_run_name,
eos_task_name, lab=self.lab_name, headless=headless)` -- the same call beaker_lab's arm
device made by hand before this existed, so the Matterix side (resolution by lab +
task name, shared per-run MatterixBackend actor, raising on a failed workflow) is
unchanged. Skipping the body in sim mode (rather than running it after the workflow)
is deliberate: a body that talks to hardware unguarded must never run during a
simulated call.

The method must take `protocol_run_name`, `eos_task_name` and `backend` parameters
(checked once, at class-definition time, so a typo fails on import rather than mid-
protocol); `headless` is optional (default True). Any other `backend` value than "sim"
or "real" raises ValueError instead of silently picking one -- a misspelt "Sim" must not
fall through to real hardware.

    returns: name of one of the method's parameters to return as-is on a sim call (e.g.
        "beaker" for a method that takes and returns a Resource, the usual EOS device
        shape). Omit for a method that returns nothing; a sim call then returns None.
    fields: names of the method's parameters to forward to the Matterix workflow as
        dynamic overrides (e.g. fields=("target_temperature",) -- see
        run_matterix_workflow()'s `fields`). Names must match the workflow config's own
        field names.

Bookkeeping that should happen in BOTH modes (e.g. `beaker.meta["picked"] = True`)
doesn't belong in the decorated body, since that only runs for real hardware -- put it
in the calling task, or in an undecorated wrapper method around the decorated one.

Works on both plain and `async def` methods; for an async method the (blocking,
Ray-backed) sim call runs in a worker thread via asyncio.to_thread so it doesn't stall
the event loop.

`run_matterix_workflow` is imported lazily, on the first sim call, so importing this
module (and decorating a device class) needs neither ray nor Isaac Sim -- a device
module stays importable wherever EOS loads it.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable

_BACKENDS = ("sim", "real")
_REQUIRED_PARAMS = ("protocol_run_name", "eos_task_name", "backend")


def sim_twin(
    func: Callable | None = None, *, returns: str | None = None, fields: tuple[str, ...] = ()
) -> Callable:
    """See this module's docstring. Usable bare (`@sim_twin`) or with options
    (`@sim_twin(returns="beaker", fields=("target_temperature",))`)."""

    def decorate(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        missing = [name for name in _REQUIRED_PARAMS if name not in sig.parameters]
        if missing:
            raise TypeError(
                f"@sim_twin on {fn.__qualname__}: method must take parameter(s) {missing} -- "
                "protocol_run_name/eos_task_name come from the calling task "
                "(self._protocol_run_name, self._task_name on BaseTask), backend is 'sim' or 'real'."
            )
        named = ([returns] if returns else []) + list(fields)
        unknown = [name for name in named if name not in sig.parameters]
        if unknown:
            raise TypeError(f"@sim_twin on {fn.__qualname__}: returns/fields name(s) {unknown} aren't parameters of it")

        def bind(args: tuple, kwargs: dict) -> dict[str, Any]:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            backend = bound.arguments["backend"]
            if backend not in _BACKENDS:
                raise ValueError(f"{fn.__qualname__}: backend must be one of {_BACKENDS}, got {backend!r}")
            return bound.arguments

        def run_sim(self: Any, arguments: dict[str, Any]) -> Any:
            from user.matterix_bridge.common.matterix_backend import run_matterix_workflow

            run_matterix_workflow(
                arguments["protocol_run_name"],
                arguments["eos_task_name"],
                lab=self.lab_name,
                headless=arguments.get("headless", True),
                **{name: arguments[name] for name in fields},
            )
            return arguments[returns] if returns else None

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(self, *args, **kwargs):
                arguments = bind((self, *args), kwargs)
                if arguments["backend"] == "sim":
                    return await asyncio.to_thread(run_sim, self, arguments)
                return await fn(self, *args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            arguments = bind((self, *args), kwargs)
            if arguments["backend"] == "sim":
                return run_sim(self, arguments)
            return fn(self, *args, **kwargs)

        return wrapper

    return decorate(func) if func is not None else decorate
