from typing import Any

from eos.devices.base_device import BaseDevice
from eos.resources.entities.resource import Resource

from user.matterix_bridge.common.matterix_backend import run_matterix_workflow


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
        eos_task_name: str,
        headless: bool = True,
    ) -> Resource:
        """protocol_run_name/eos_task_name must come from the calling task
        (self._protocol_run_name, self._task_name on BaseTask) -- a device has no way to
        know which protocol run or DAG task it's being called from on its own. No
        protocol_type needed: run_matterix_workflow() resolves the Matterix workflow from
        eos_task_name alone (see its docstring for when a task name is ambiguous enough
        to need one anyway -- not the case here, "turn_on_heater" is only ever registered
        under heater_transfer_protocol).

        headless only takes effect on the first run_workflow() call in this scope's
        MatterixBackend process (Isaac Sim can't be reconfigured after boot).
        """
        if self._backend_mode == "sim":
            # TurnOnHeaterCfg.target_temperature is in Kelvin; this device's target_temperature
            # is authored in Celsius (matches heat_sample/task.yml's `unit: celsius`) -- convert,
            # or e.g. 75 would be read as 75K (~-198C) instead of 348.15K.
            target_kelvin = target_temperature + 273.15
            run_matterix_workflow(
                protocol_run_name,
                eos_task_name,
                headless=headless,
                target_temperature=target_kelvin,
            )
        else:
            pass  # TODO: self._client.send_command("heat", {"target_temperature": target_temperature})

        sample.meta["temperature"] = target_temperature
        return sample
