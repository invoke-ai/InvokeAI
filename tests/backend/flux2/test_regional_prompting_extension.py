from unittest.mock import patch

import torch

from invokeai.backend.flux2.extensions.regional_prompting_extension import Flux2RegionalPromptingExtension
from invokeai.backend.flux2.text_conditioning import Flux2TextConditioning
from invokeai.backend.util.devices import TorchDevice


def _cpu_device():
    return patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu"))


def test_preprocess_mask_none_returns_ones():
    with _cpu_device():
        mask = Flux2RegionalPromptingExtension.preprocess_regional_prompt_mask(
            mask=None, packed_height=2, packed_width=2, dtype=torch.float32, device=torch.device("cpu")
        )
    assert mask.shape == (1, 1, 4)
    assert torch.equal(mask, torch.ones(1, 1, 4))


def test_preprocess_mask_resizes_and_flattens():
    # A (h, w) mask covering the left half, at 2x the packed resolution.
    raw_mask = torch.zeros(4, 4, dtype=torch.bool)
    raw_mask[:, :2] = True
    with _cpu_device():
        mask = Flux2RegionalPromptingExtension.preprocess_regional_prompt_mask(
            mask=raw_mask, packed_height=2, packed_width=2, dtype=torch.float32, device=torch.device("cpu")
        )
    assert mask.shape == (1, 1, 4)
    # Packed grid rows are [left, right]: left half is masked.
    assert torch.equal(mask.view(2, 2), torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_no_masks_short_circuits_to_unmasked_attention():
    conditionings = [
        Flux2TextConditioning(txt_embeddings=torch.zeros(1, 2, 8), mask=None),
        Flux2TextConditioning(txt_embeddings=torch.zeros(1, 3, 8), mask=None),
    ]
    with _cpu_device():
        extension = Flux2RegionalPromptingExtension.from_text_conditionings(conditionings, img_seq_len=4)

    assert extension.restricted_attn_mask is None
    assert extension.get_joint_attention_kwargs(dtype=torch.float32) is None

    # Embeddings are concatenated and position IDs vary only in the L coordinate.
    tc = extension.regional_text_conditioning
    assert tc.txt_embeddings.shape == (1, 5, 8)
    assert tc.txt_ids.shape == (1, 5, 4)
    assert torch.equal(tc.txt_ids[..., 3], torch.arange(5).unsqueeze(0))
    assert torch.equal(tc.txt_ids[..., :3], torch.zeros(1, 5, 3, dtype=torch.long))
    assert [(r.start, r.end) for r in tc.embedding_ranges] == [(0, 2), (2, 5)]


def _build_two_region_extension() -> Flux2RegionalPromptingExtension:
    """One global prompt (len 2) and two single-token regional prompts on a 4-pixel image.

    Region A covers pixel 0, region B covers pixel 1, pixels 2 and 3 are background.
    Attention sequence layout: txt tokens 0-3 ([0:2] global, [2] region A, [3] region B),
    img tokens 4-7 (pixels 0-3).
    """
    mask_a = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    mask_b = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    conditionings = [
        Flux2TextConditioning(txt_embeddings=torch.zeros(1, 2, 8), mask=None),
        Flux2TextConditioning(txt_embeddings=torch.zeros(1, 1, 8), mask=mask_a),
        Flux2TextConditioning(txt_embeddings=torch.zeros(1, 1, 8), mask=mask_b),
    ]
    return Flux2RegionalPromptingExtension.from_text_conditionings(conditionings, img_seq_len=4)


def test_restricted_mask_semantics():
    with _cpu_device():
        extension = _build_two_region_extension()

    attn_mask = extension.restricted_attn_mask
    assert attn_mask is not None
    assert attn_mask.shape == (8, 8)
    assert attn_mask.dtype == torch.bool

    # txt self-attention: within a prompt yes, across prompts no.
    assert attn_mask[0, 1] and attn_mask[2, 2] and attn_mask[3, 3]
    assert not attn_mask[0, 2] and not attn_mask[2, 3]

    # Regional txt attends only to its own region's pixels.
    assert attn_mask[2, 4] and not attn_mask[2, 5] and not attn_mask[2, 6]
    assert attn_mask[3, 5] and not attn_mask[3, 4]
    assert attn_mask[4, 2] and not attn_mask[5, 2]

    # Global txt attends to background pixels only, not to regional pixels.
    assert not attn_mask[0, 4] and not attn_mask[0, 5]
    assert attn_mask[0, 6] and attn_mask[0, 7]
    assert attn_mask[6, 0] and not attn_mask[4, 0]

    # img self-attention: region pixels attend to themselves but not to other regions'
    # pixels; background rows/columns are open.
    assert attn_mask[4, 4] and attn_mask[5, 5]
    assert not attn_mask[4, 5] and not attn_mask[5, 4]
    assert attn_mask[4, 6] and attn_mask[6, 4] and attn_mask[6, 7]

    # No fully-blocked rows (an all-False row would produce NaNs in SDPA).
    assert bool(attn_mask.any(dim=1).all())


def test_get_joint_attention_kwargs_builds_additive_mask():
    with _cpu_device():
        extension = _build_two_region_extension()
        kwargs = extension.get_joint_attention_kwargs(dtype=torch.float32)

    assert kwargs is not None
    additive = kwargs["attention_mask"]
    assert additive.shape == (1, 1, 8, 8)
    assert additive.dtype == torch.float32
    # 0 where attention is allowed, -inf where blocked.
    assert additive[0, 0, 2, 4] == 0.0
    assert additive[0, 0, 2, 5] == float("-inf")
    assert additive[0, 0, 4, 5] == float("-inf")
    assert additive[0, 0, 0, 6] == 0.0
