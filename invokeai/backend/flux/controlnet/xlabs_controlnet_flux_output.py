from dataclasses import dataclass

import torch


@dataclass
class XLabsControlNetFluxOutput:
    controlnet_double_block_residuals: list[torch.Tensor] | None

    def apply_weight(self, weight: float):
        if self.controlnet_double_block_residuals is not None:
            for i in range(len(self.controlnet_double_block_residuals)):
                self.controlnet_double_block_residuals[i] = self.controlnet_double_block_residuals[i] * weight
