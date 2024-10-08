from dataclasses import dataclass

import torch


@dataclass
class InstantXControlNetFluxOutput:
    controlnet_block_samples: list[torch.Tensor] | None
    controlnet_single_block_samples: list[torch.Tensor] | None

    def apply_weight(self, weight: float):
        if self.controlnet_block_samples is not None:
            for i in range(len(self.controlnet_block_samples)):
                self.controlnet_block_samples[i] = self.controlnet_block_samples[i] * weight
        if self.controlnet_single_block_samples is not None:
            for i in range(len(self.controlnet_single_block_samples)):
                self.controlnet_single_block_samples[i] = self.controlnet_single_block_samples[i] * weight
