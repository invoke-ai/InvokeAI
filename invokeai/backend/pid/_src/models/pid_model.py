# PID (PixelDiT SR) model — inference subset.
#
# At inference the only thing this class adds on top of PixelDiTModel is the
# frozen VAE (`vae_encoder`) used by `encode_lq_latent`. The training-time
# degradation pipeline, LoRA injection, LPIPS loss, and training/validation
# steps have all been removed.

from __future__ import annotations

import logging
from typing import Any

import attrs
import torch
from torch import Tensor

from invokeai.backend.pid._ext.imaginaire.lazy_config import instantiate as lazy_instantiate
from invokeai.backend.pid._ext.imaginaire.utils import misc
from invokeai.backend.pid._src.models.pixeldit_model import PixelDiTModel, PixelDiTModelConfig

logger = logging.getLogger(__name__)


@attrs.define(slots=False)
class PidModelConfig(PixelDiTModelConfig):
    # "image" = LQ image only, "latent" = LQ latent only, "image_latent" = both.
    lq_condition_type: str = "latent"

    # Frozen VAE config for encoding LQ images to latent.
    tokenizer: Any = None

    # VAE latent channels (must match tokenizer.latent_ch).
    state_ch: int = 16

    # Fixed prompt override (training convenience kept here so checkpoints that set
    # use_fixed_prompt=True still load).
    use_fixed_prompt: bool = False
    fixed_positive_prompt: str = ""


class PidModel(PixelDiTModel):
    """PID (PixelDiT SR) inference model (frozen VAE + LQ-conditioned student)."""

    def __init__(self, config: PidModelConfig):
        super().__init__(config)

        if config.tokenizer is not None:
            with misc.timer("PidModel: load_vae"):
                from invokeai.backend.pid._src.tokenizers.base_vae import BaseVAE

                self.vae_encoder: BaseVAE = lazy_instantiate(config.tokenizer)
                if config.state_ch > 0:
                    assert self.vae_encoder.latent_ch == config.state_ch, (
                        f"latent_ch {self.vae_encoder.latent_ch} != state_ch {config.state_ch}"
                    )
        else:
            self.vae_encoder = None
            logger.warning("No VAE configured — LQ latent encoding disabled.")

    @torch.no_grad()
    def encode_lq_latent(self, lq_image: Tensor) -> Tensor:
        """Encode an LQ image through the frozen VAE.

        Args:
            lq_image: [B, C, H_lq, W_lq] in [-1, 1].

        Returns:
            LQ latent [B, z_dim, zH, zW].
        """
        if lq_image.ndim == 4:
            lq_image = lq_image.unsqueeze(2)
        latent = self.vae_encoder.encode(lq_image)
        if latent.ndim == 5:
            latent = latent[:, :, 0, :, :]
        return latent
