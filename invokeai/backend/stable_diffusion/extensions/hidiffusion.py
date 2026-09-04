from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
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
        t1_ratio: Optional[float] = None,
        t2_ratio: Optional[float] = None,
        generator: torch.Generator | None = None,
        has_controlnet: bool = False,
        is_controlnet_text_to_image: bool = False,
        is_inpainting_task: bool | None = None,
        use_aggressive_raunet: bool | None = None,
        denoising_start: float = 0.0,
        denoising_end: float = 1.0,
    ):
        super().__init__()
        self._name_or_path = name_or_path
        self._apply_raunet = apply_raunet
        self._apply_window_attn = apply_window_attn
        self._has_controlnet = has_controlnet
        self._is_controlnet_text_to_image = is_controlnet_text_to_image
        self._is_inpainting_task = is_inpainting_task
        self._use_aggressive_raunet = use_aggressive_raunet
        self._denoising_start = denoising_start
        self._denoising_end = denoising_end
        self._t1_ratio = t1_ratio
        self._t2_ratio = t2_ratio
        self._generator = generator

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, original_weights: OriginalWeightsStorage):
        with hidiffusion_patch(
            unet,
            name_or_path=self._name_or_path,
            apply_raunet=self._apply_raunet,
            apply_window_attn=self._apply_window_attn,
            has_controlnet=self._has_controlnet,
            is_controlnet_text_to_image=self._is_controlnet_text_to_image,
            t1_ratio=self._t1_ratio,
            t2_ratio=self._t2_ratio,
            generator=self._generator,
            is_inpainting_task=self._is_inpainting_task,
            use_aggressive_raunet=self._use_aggressive_raunet,
            denoising_start=self._denoising_start,
            denoising_end=self._denoising_end,
        ):
            yield None
