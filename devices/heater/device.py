from typing import Any

import ray

from eos.devices.base_device import BaseDevice
from eos.resources.entities.resource import Resource

from user.matterix_bridge.common.matterix_backend import get_backend


class Heater(BaseDevice):
    async def _initialize(self, init_parameters: dict[str, Any]) -> None:
        self._backend_mode = init_parameters.get("backend", "real")
        if self._backend_mode != "sim":
            # TODO: replace with the real heater's client/driver.
            self._port = int(init_parameters["port"])
            self._client = None

    async def _cleanup(self) -> None:
        if self._backend_mode == "real" and self._client is not None:
            pass  # TODO: self._client.close_connection()

    async def _report(self) -> dict[str, Any]:
        return {"backend": self._backend_mode}

    def heat_to(
        self,
        sample: Resource,
        target_temperature: float,
        protocol_run_name: str,
        protocol_type: str,
        eos_task_name: str,
        headless: bool = True,
    ) -> Resource:
        """protocol_run_name/protocol_type/eos_task_name must come from the calling task
        (self._protocol_run_name, self._task_name on BaseTask) -- a device has no way to
        know which protocol run or DAG task it's being called from on its own.

        protocol_type specifically isn't exposed by BaseTask today (only
        protocol_run_name, an instance id, and task_name are) -- the calling task.py needs
        to supply it explicitly, e.g. hardcoded as a parameter in this task's protocol.yml,
        since the protocol author already knows their own protocol's type at authoring time.

        headless only takes effect on the first run_workflow() call in this scope's
        MatterixBackend process (Isaac Sim can't be reconfigured after boot).
        """
        if self._backend_mode == "sim":
            backend = get_backend(protocol_run_name)
            # TurnOnHeaterCfg.target_temperature is in Kelvin; this device's target_temperature
            # is authored in Celsius (matches heat_sample/task.yml's `unit: celsius`) -- convert,
            # or e.g. 75 would be read as 75K (~-198C) instead of 348.15K.
            target_kelvin = target_temperature + 273.15
            ray.get(backend.set_parameter.remote(protocol_type, eos_task_name, "target_temperature", target_kelvin))
            result = ray.get(backend.run_workflow.remote(protocol_type, eos_task_name, headless=headless))
            if not result.success:
                raise RuntimeError(f"Failed to run workflow {protocol_type}/{eos_task_name}: {result.failure_detail}")
        else:
            pass  # TODO: self._client.send_command("heat", {"target_temperature": target_temperature})

        sample.meta["temperature"] = target_temperature
        return sample
