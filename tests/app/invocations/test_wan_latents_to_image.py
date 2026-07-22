"""Tests for ``WanLatentsToImageInvocation`` input validation (JPPhoto review 2026-07-21).

The bug: the node accepted any 5D latent tensor. Multi-frame video latents ran the full
multi-frame VAE decode (under a working-memory estimate that assumed one frame) and then
died in an opaque einops rank error at the final rearrange. The node must reject T>1
latents early — before the VAE is even loaded — and point users at the video decode node.
"""

from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.fields import LatentsField
from invokeai.app.invocations.model import ModelIdentifierField, VAEField
from invokeai.app.invocations.wan_latents_to_image import WanLatentsToImageInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def _make_invocation() -> WanLatentsToImageInvocation:
    vae_id = ModelIdentifierField(key="wan-vae", hash="h", name="wan-vae", base=BaseModelType.Wan, type=ModelType.VAE)
    return WanLatentsToImageInvocation(
        id="inv-1",
        latents=LatentsField(latents_name="latents-1"),
        vae=VAEField(vae=vae_id),
    )


def _make_context(latents: torch.Tensor) -> MagicMock:
    ctx = MagicMock()
    ctx.tensors.load.return_value = latents
    return ctx


@pytest.mark.parametrize("num_frames", [2, 5, 21])
def test_multi_frame_latents_rejected_before_vae_load(num_frames: int) -> None:
    ctx = _make_context(torch.zeros(1, 16, num_frames, 8, 8))

    with pytest.raises(ValueError, match="frames of video"):
        _make_invocation().invoke(ctx)

    # The rejection must be cheap: no model load, no decode.
    ctx.models.load.assert_not_called()


def test_error_points_at_the_video_decode_node() -> None:
    ctx = _make_context(torch.zeros(1, 16, 5, 8, 8))
    with pytest.raises(ValueError, match="wan_l2v"):
        _make_invocation().invoke(ctx)


@pytest.mark.parametrize("shape", [(16, 8, 8), (1, 1, 16, 1, 8, 8)], ids=["3d", "6d"])
def test_invalid_rank_rejected_before_vae_load(shape: tuple[int, ...]) -> None:
    ctx = _make_context(torch.zeros(*shape))

    with pytest.raises(ValueError, match="expects a 4D or 5D latent tensor"):
        _make_invocation().invoke(ctx)

    ctx.models.load.assert_not_called()


@pytest.mark.parametrize(
    "shape",
    [(1, 16, 8, 8), (1, 16, 1, 8, 8)],
    ids=["4d", "5d-single-frame"],
)
def test_valid_shapes_pass_validation(shape: tuple[int, ...]) -> None:
    """4D and single-frame 5D latents proceed past the temporal check. The mocked model
    load then fails the AutoencoderKLWan type check, proving execution got there."""
    ctx = _make_context(torch.zeros(*shape))
    ctx.models.load.return_value = MagicMock(model=object())

    with pytest.raises(TypeError, match="Expected AutoencoderKLWan"):
        _make_invocation().invoke(ctx)
