"""Regional prompting's patched Z-Image forward, run on a tiny real `ZImageTransformer2DModel`.

diffusers returns no padding mask from `_build_unified_sequence` when every batch item has the same length -- always
the case at batch size 1, which is how the denoise node calls the transformer -- and the patched forward crashed on
`None.bool()` for every regional generation.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.fields import TensorField, ZImageConditioningField
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, ZImageConditioningInfo
from invokeai.backend.z_image.extensions.regional_prompting_extension import ZImageRegionalPromptingExtension
from invokeai.backend.z_image.text_conditioning import ZImageTextConditioning
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


@torch.no_grad()
@pytest.mark.parametrize("negative_tokens", [20, 41])
def test_cfg_passes_use_independent_regional_masks(tiny_z_image_transformer, negative_tokens: int) -> None:
    model = tiny_z_image_transformer
    x, t, positive = _inputs(model, caption_tokens=20)
    negative = torch.randn(negative_tokens, model.config.cap_feat_dim)
    positive_mask = torch.ones(IMAGE_TOKENS + 20, IMAGE_TOKENS + 20, dtype=torch.bool)
    positive_mask[: IMAGE_TOKENS // 2, IMAGE_TOKENS:] = False
    negative_mask = torch.ones(IMAGE_TOKENS + negative_tokens, IMAGE_TOKENS + negative_tokens, dtype=torch.bool)
    negative_mask[IMAGE_TOKENS // 2 : IMAGE_TOKENS, IMAGE_TOKENS:] = False

    # Each prompt's isolated forward is the reference for the shared CFG patch.
    with patch_transformer_for_regional_prompting(model, positive_mask, IMAGE_TOKENS, positive):
        expected_positive = _forward(model, x, t, positive)
    with patch_transformer_for_regional_prompting(model, negative_mask, IMAGE_TOKENS, negative):
        expected_negative = _forward(model, x, t, negative)
    with patch_transformer_for_regional_prompting(
        model, positive_mask, IMAGE_TOKENS, positive, negative_mask, negative
    ):
        for _ in range(2):
            torch.testing.assert_close(_forward(model, x, t, positive), expected_positive)
            torch.testing.assert_close(_forward(model, x, t, negative), expected_negative)


@torch.no_grad()
def test_negative_only_regions_leave_positive_unmasked_and_restore_on_failure(tiny_z_image_transformer) -> None:
    model = tiny_z_image_transformer
    x, t, positive = _inputs(model, caption_tokens=20)
    negative = torch.randn_like(positive)
    plain_positive = _forward(model, x, t, positive)
    plain_negative = _forward(model, x, t, negative)
    negative_mask = torch.ones(IMAGE_TOKENS + 20, IMAGE_TOKENS + 20, dtype=torch.bool)
    negative_mask[:IMAGE_TOKENS, IMAGE_TOKENS:] = False
    original_forward = model.forward

    with pytest.raises(RuntimeError, match="cancelled"):
        with patch_transformer_for_regional_prompting(model, None, IMAGE_TOKENS, positive, negative_mask, negative):
            torch.testing.assert_close(_forward(model, x, t, positive), plain_positive)
            restricted_negative = _forward(model, x, t, negative)
            assert torch.isfinite(restricted_negative).all()
            assert not torch.allclose(restricted_negative, plain_negative, atol=1e-3)
            raise RuntimeError("cancelled")

    assert model.forward == original_forward
    torch.testing.assert_close(_forward(model, x, t, negative), plain_negative)


def test_unmasked_text_covers_the_whole_image_only_in_negative_mode(monkeypatch) -> None:
    monkeypatch.setattr("invokeai.backend.util.devices.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    # Sequence: [region image token, background image token, global text, regional text].
    conditionings = [
        ZImageTextConditioning(prompt_embeds=torch.zeros(1, 4), mask=None),
        ZImageTextConditioning(prompt_embeds=torch.zeros(1, 4), mask=torch.tensor([[[1.0, 0.0]]])),
    ]
    positive = ZImageRegionalPromptingExtension.from_text_conditionings(conditionings, img_seq_len=2)
    negative = ZImageRegionalPromptingExtension.from_text_conditionings(
        conditionings, img_seq_len=2, unmasked_covers_whole_image=True
    )
    global_text, regional_text = 2, 3
    assert positive.regional_attn_mask is not None and negative.regional_attn_mask is not None
    assert positive.regional_attn_mask[:2, global_text].tolist() == [False, True]
    assert positive.regional_attn_mask[global_text, :2].tolist() == [False, True]
    assert negative.regional_attn_mask[:2, global_text].tolist() == [True, True]
    assert negative.regional_attn_mask[global_text, :2].tolist() == [True, True]
    # Regional text keeps its region in both modes.
    for extension in (positive, negative):
        assert extension.regional_attn_mask[:2, regional_text].tolist() == [True, False]
        assert extension.regional_attn_mask[regional_text, :2].tolist() == [True, False]


def _regional_denoise(model, monkeypatch, *, guidance_scale: float, scheduler: str = "euler"):
    """A denoise node on the tiny transformer, whose "region" tensor masks the left half of the image."""
    transformer_info = MagicMock(model=model)
    transformer_info.model_on_device.side_effect = lambda **kwargs: nullcontext((None, model))
    context = MagicMock()
    context.models.load.return_value = transformer_info
    context.models.get_config.return_value = SimpleNamespace(format=ModelFormat.Diffusers)
    positive = torch.randn(20, model.config.cap_feat_dim)
    negative = torch.randn_like(positive)

    def load_conditioning(name):
        if name == "negative":
            assert guidance_scale > 1.0, "CFG-off must not load or allocate negative conditioning"
        return ConditioningFieldData(
            conditionings=[ZImageConditioningInfo(prompt_embeds=positive if name == "positive" else negative)]
        )

    context.conditioning.load.side_effect = load_conditioning
    mask = torch.zeros(1, LATENT_SIDE, LATENT_SIDE)
    mask[:, :, : LATENT_SIDE // 2] = 1
    context.tensors.load.side_effect = lambda name: mask if name == "region" else None
    monkeypatch.setattr("invokeai.backend.util.devices.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        "invokeai.backend.util.devices.TorchDevice.choose_bfloat16_safe_dtype", lambda device: torch.float32
    )
    node = ZImageDenoiseInvocation.model_construct(
        id="regional",
        width=LATENT_SIDE * 8,
        height=LATENT_SIDE * 8,
        steps=2,
        guidance_scale=guidance_scale,
        positive_conditioning=ZImageConditioningField(conditioning_name="positive"),
        negative_conditioning=ZImageConditioningField(conditioning_name="negative"),
        transformer=MagicMock(transformer="transformer", loras=[]),
        seed=0,
        scheduler=scheduler,
    )
    return node, context


def _global_and_left_half(name: str) -> list[ZImageConditioningField]:
    return [
        ZImageConditioningField(conditioning_name=name),
        ZImageConditioningField(conditioning_name=name, mask=TensorField(tensor_name="region")),
    ]


def _record_patches(monkeypatch) -> list[dict]:
    patches: list[dict] = []

    def record_patch(**kwargs):
        patches.append(kwargs)
        return patch_transformer_for_regional_prompting(**kwargs)

    monkeypatch.setattr(
        "invokeai.app.invocations.z_image.z_image_denoise.patch_transformer_for_regional_prompting", record_patch
    )
    return patches


@torch.no_grad()
@pytest.mark.parametrize("scheduler", ["euler", "heun"])
@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_denoise_consumes_negative_masks_only_with_cfg(
    tiny_z_image_transformer, monkeypatch, scheduler: str, guidance_scale: float
) -> None:
    """Run both solvers through loading, regional attention and CFG with a real tiny transformer."""
    model = tiny_z_image_transformer
    original_forward = model.forward
    node, context = _regional_denoise(model, monkeypatch, guidance_scale=guidance_scale, scheduler=scheduler)
    plain = node._run_diffusion(context)
    node.negative_conditioning = _global_and_left_half("negative")
    patches = _record_patches(monkeypatch)
    regional = node._run_diffusion(context)

    assert model.forward == original_forward
    assert torch.isfinite(regional).all()
    if guidance_scale == 1.0:
        torch.testing.assert_close(regional, plain, rtol=0, atol=0)
        context.tensors.load.assert_not_called()
    else:
        assert not torch.allclose(regional, plain, atol=1e-3)
        # The global negative stays image-wide while the regional negative keeps to its region.
        negative_mask = patches[0]["negative_regional_attn_mask"]
        global_text = slice(IMAGE_TOKENS, IMAGE_TOKENS + 20)
        assert negative_mask[:IMAGE_TOKENS, global_text].all() and negative_mask[global_text, :IMAGE_TOKENS].all()
        assert not negative_mask[:IMAGE_TOKENS, IMAGE_TOKENS + 20 :].all()


@torch.no_grad()
def test_denoise_keeps_the_global_positive_prompt_on_the_background(tiny_z_image_transformer, monkeypatch) -> None:
    node, context = _regional_denoise(tiny_z_image_transformer, monkeypatch, guidance_scale=1.0)
    node.positive_conditioning = _global_and_left_half("positive")
    patches = _record_patches(monkeypatch)
    node._run_diffusion(context)

    positive_mask = patches[0]["regional_attn_mask"]
    # 8x8 token grid, row-major: the region holds columns 0-3 of every row, the background columns 4-7.
    background = torch.tensor([False, False, False, False, True, True, True, True]).repeat(8)
    global_text = slice(IMAGE_TOKENS, IMAGE_TOKENS + 20)
    assert torch.equal(positive_mask[:IMAGE_TOKENS, global_text], background[:, None].expand(IMAGE_TOKENS, 20))
    assert torch.equal(positive_mask[global_text, :IMAGE_TOKENS], background[None, :].expand(20, IMAGE_TOKENS))
