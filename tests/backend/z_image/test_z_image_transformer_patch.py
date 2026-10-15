"""Regional prompting's patched Z-Image forward, run on a tiny real `ZImageTransformer2DModel`.

diffusers returns no padding mask from `_build_unified_sequence` when every batch item has the same length -- always
the case at batch size 1, which is how the denoise node calls the transformer -- and the patched forward crashed on
`None.bool()` for every regional generation.
"""

import torch

from invokeai.backend.z_image.z_image_transformer_patch import patch_transformer_for_regional_prompting

LATENT_SIDE = 16  # 8x8 = 64 image tokens after the 2x2 patch, already a multiple of 32
IMAGE_TOKENS = (LATENT_SIDE // 2) ** 2


def _inputs(model, caption_tokens: int) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    x = [torch.randn(16, 1, LATENT_SIDE, LATENT_SIDE)]
    t = torch.full((1,), 0.5)
    cap = torch.randn(caption_tokens, model.config.cap_feat_dim)
    return x, t, cap


def _forward(model, x, t, cap) -> torch.Tensor:
    return model(x=x, t=t, cap_feats=[cap])[0][0]


@torch.no_grad()
def test_an_unrestricted_regional_mask_reproduces_the_plain_forward(tiny_z_image_transformer) -> None:
    """With nothing to restrict and no padding, the regional bias is all zeros."""
    model = tiny_z_image_transformer
    x, t, cap = _inputs(model, caption_tokens=32)
    plain = _forward(model, x, t, cap)
    mask = torch.ones(IMAGE_TOKENS + 32, IMAGE_TOKENS + 32, dtype=torch.float16)

    with patch_transformer_for_regional_prompting(model, mask, IMAGE_TOKENS, positive_cap_feats=cap):
        regional = _forward(model, x, t, cap)

    torch.testing.assert_close(regional, plain, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_a_restricting_mask_applies_to_the_positive_pass_only(tiny_z_image_transformer) -> None:
    """A padded caption, image and caption kept from attending to each other: the positive pass changes, and the
    negative pass -- a different caption tensor -- runs exactly as it would unpatched."""
    model = tiny_z_image_transformer
    x, t, cap = _inputs(model, caption_tokens=20)
    negative_cap = torch.randn(20, model.config.cap_feat_dim)
    plain = _forward(model, x, t, cap)
    plain_negative = _forward(model, x, t, negative_cap)
    mask = torch.ones(IMAGE_TOKENS + 20, IMAGE_TOKENS + 20, dtype=torch.float16)
    mask[:IMAGE_TOKENS, IMAGE_TOKENS:] = 0
    mask[IMAGE_TOKENS:, :IMAGE_TOKENS] = 0

    with patch_transformer_for_regional_prompting(model, mask, IMAGE_TOKENS, positive_cap_feats=cap):
        regional = _forward(model, x, t, cap)
        negative = _forward(model, x, t, negative_cap)

    assert torch.isfinite(regional).all()
    assert not torch.allclose(regional, plain, atol=1e-3)
    torch.testing.assert_close(negative, plain_negative, rtol=1e-4, atol=1e-5)
