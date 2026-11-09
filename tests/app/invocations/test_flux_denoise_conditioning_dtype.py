"""The FLUX Fill and Control LoRA image conditioning reaches the transformer in its dtype, not the VAE's.

The VAE encodes in its own dtype: float32 under `precise_image_storage`, or when `precision: float32` is configured
with a standalone VAE. The denoiser concatenates this conditioning onto its bfloat16 image tokens, so it casts it at
the source, as the Kontext and InstantX ControlNet extensions do.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from invokeai.app.invocations.flux.flux_denoise import FluxDenoiseInvocation

SIDE = 64


def _context() -> MagicMock:
    context = MagicMock()
    context.images.get_pil.side_effect = lambda *args, **kwargs: Image.new("RGB", (SIDE, SIDE), (200, 100, 50))
    context.tensors.load.return_value = torch.zeros(1, SIDE, SIDE, dtype=torch.bool)
    return context


def _encode(**_kwargs) -> torch.Tensor:
    return torch.randn(1, 16, SIDE // 8, SIDE // 8, dtype=torch.float32)


@pytest.mark.parametrize("vae_dtype", [torch.float32, torch.bfloat16])
def test_the_fill_conditioning_is_in_the_transformers_dtype_whatever_the_vae_ran_in(vae_dtype: torch.dtype) -> None:
    node = FluxDenoiseInvocation.model_construct(
        width=SIDE,
        height=SIDE,
        guidance=30.0,
        denoise_mask=None,
        control_lora=None,
        controlnet_vae=SimpleNamespace(vae="vae"),
        fill_conditioning=SimpleNamespace(
            image=SimpleNamespace(image_name="image"), mask=SimpleNamespace(tensor_name="mask")
        ),
    )

    with patch(
        "invokeai.app.invocations.flux.flux_denoise.FluxVaeEncodeInvocation.vae_encode",
        side_effect=lambda **kwargs: _encode().to(vae_dtype),
    ):
        img_cond = node._prep_flux_fill_img_cond(_context(), device=torch.device("cpu"), dtype=torch.bfloat16)

    assert img_cond.dtype == torch.bfloat16
    # The encoded image (16 channels packed 2x2: 64) and the packed mask (64 channels packed 2x2: 256) side by side.
    assert img_cond.shape == (1, (SIDE // 16) ** 2, 64 + 256)


@pytest.mark.parametrize("vae_dtype", [torch.float32, torch.bfloat16])
def test_the_control_lora_conditioning_is_in_the_transformers_dtype_whatever_the_vae_ran_in(
    vae_dtype: torch.dtype,
) -> None:
    node = FluxDenoiseInvocation.model_construct(
        width=SIDE,
        height=SIDE,
        control_lora=SimpleNamespace(img=SimpleNamespace(image_name="image")),
        controlnet_vae=SimpleNamespace(vae="vae"),
    )

    with patch(
        "invokeai.app.invocations.flux.flux_denoise.FluxVaeEncodeInvocation.vae_encode",
        side_effect=lambda **kwargs: _encode().to(vae_dtype),
    ):
        img_cond = node._prep_structural_control_img_cond(_context(), dtype=torch.bfloat16)

    assert img_cond is not None and img_cond.dtype == torch.bfloat16
