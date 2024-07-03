import torch
from .base import ExtensionBase, modifier
from ..denoise_context import DenoiseContext


class RescaleCFGExt(ExtensionBase):
    def __init__(self, guidance_rescale_multiplier: float, priority: int):
        super().__init__(priority=priority)
        self.guidance_rescale_multiplier = guidance_rescale_multiplier

    @staticmethod
    def _rescale_cfg(total_noise_pred: torch.Tensor, pos_noise_pred: torch.Tensor, multiplier: float=0.7):
        """Implementation of Algorithm 2 from https://arxiv.org/pdf/2305.08891.pdf."""
        ro_pos = torch.std(pos_noise_pred, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(total_noise_pred, dim=(1, 2, 3), keepdim=True)

        x_rescaled = total_noise_pred * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * total_noise_pred
        return x_final

    @modifier("modify_noise_prediction")
    def rescale_noise_pred(self, ctx: DenoiseContext):
        if self.guidance_rescale_multiplier > 0:
            ctx.noise_pred = self._rescale_cfg(
                ctx.noise_pred,
                ctx.positive_noise_pred,
                self.guidance_rescale_multiplier,
            )
