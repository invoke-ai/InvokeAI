from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.blend_latents import BlendLatentsInvocation
from invokeai.app.invocations.fields import LatentsField


def _blend(latents_a: torch.Tensor, latents_b: torch.Tensor, alpha: float) -> torch.Tensor:
    context = MagicMock()
    context.tensors.load.side_effect = {"a": latents_a, "b": latents_b}.__getitem__
    context.tensors.save.return_value = "blended"

    invocation = BlendLatentsInvocation(
        latents_a=LatentsField(latents_name="a"), latents_b=LatentsField(latents_name="b"), alpha=alpha
    )
    invocation.invoke(context)

    return context.tensors.save.call_args.kwargs["tensor"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_blend_latents_halfway_between_orthogonal_latents(dtype: torch.dtype):
    # Two orthogonal latents of equal norm: halfway along the arc is (a + b) * sin(pi/4) / sin(pi/2).
    checkerboard = (torch.arange(4 * 8 * 8).reshape(1, 4, 8, 8) % 2).to(torch.float32)
    latents_a = checkerboard.to(dtype)
    latents_b = (1 - checkerboard).to(dtype)

    blended = _blend(latents_a, latents_b, alpha=0.5)

    assert blended.dtype == dtype
    torch.testing.assert_close(blended.float(), torch.full_like(checkerboard, 0.5**0.5), atol=1e-2, rtol=0)


def test_blend_latents_keeps_identical_large_half_precision_latents():
    # A 1024x1024 SD latent has 65,536 values, so its squared norm overflows float16 (max 65,504).
    latents = torch.full((1, 4, 128, 128), 1.5, dtype=torch.float16)

    blended = _blend(latents, latents.clone(), alpha=0.5)

    assert blended.dtype == torch.float16
    torch.testing.assert_close(blended, latents)
