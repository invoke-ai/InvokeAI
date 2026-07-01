import math
from abc import ABC, abstractmethod
from typing import List, Union

import torch

from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput


class BaseControlNetExtension(ABC):
    def __init__(
        self,
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
    ):
        self._weight = weight
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent

    def _get_weight(self, timestep_index: int, total_num_timesteps: int) -> float:
        first_step = math.floor(self._begin_step_percent * total_num_timesteps)
        last_step = math.ceil(self._end_step_percent * total_num_timesteps)

        if timestep_index < first_step or timestep_index > last_step:
            return 0.0

        if isinstance(self._weight, list):
            return self._weight[timestep_index]

        return self._weight

    @abstractmethod
    def run_controlnet(
        self,
        timestep_index: int,
        total_num_timesteps: int,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        y: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> ControlNetFluxOutput: ...
