from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase
from invokeai.backend.stable_diffusion.hidiffusion_utils import hidiffusion_patch
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


class HiDiffusionExt(ExtensionBase):
    def __init__(
        self,
        name_or_path: Optional[str],
        apply_raunet: bool = True,
        apply_window_attn: bool = True,
    ):
        super().__init__()
        self._name_or_path = name_or_path
        self._apply_raunet = apply_raunet
        self._apply_window_attn = apply_window_attn

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, original_weights: OriginalWeightsStorage):
        with hidiffusion_patch(
            unet,
            name_or_path=self._name_or_path,
            apply_raunet=self._apply_raunet,
            apply_window_attn=self._apply_window_attn,
        ):
            yield None
