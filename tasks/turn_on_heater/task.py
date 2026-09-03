from eos.tasks.base_task import BaseTask


class TurnOnHeater(BaseTask):
    async def _execute(
        self,
        devices: BaseTask.DevicesType,
        parameters: BaseTask.ParametersType,
        resources: BaseTask.ResourcesType,
    ) -> BaseTask.OutputType:
        heater = devices["heater"]

        resources["sample"] = heater.heat_to(
            resources["sample"],
            parameters["target_temperature"],
            protocol_run_name=self._protocol_run_name,
            protocol_type=parameters["matterix_protocol_type"],
            eos_task_name=self._task_name,
        )

        return None, resources, None
