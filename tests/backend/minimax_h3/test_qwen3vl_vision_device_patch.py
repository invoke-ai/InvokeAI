"""Tests for the Qwen3VLVisionModel pos-embed device patch.

The bug: ``fast_pos_embed_interpolate`` builds all of its tensors on
``self.pos_embed.weight.device``. Under partial loading that weight may sit on the CPU while
the forward computes on CUDA (nn.Embedding is autocast-wrapped), so ``hidden_states +
pos_embeds`` fails with a cross-device RuntimeError. The patch records the true compute device
from forward's ``hidden_states`` input and moves only the interpolation's result there.

CI has no GPU, so the cross-device move is exercised against the ``meta`` device; the
end-to-end CPU forward covers the no-op path (result already on the right device).
"""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_vl")

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel  # noqa: E402

from invokeai.backend.minimax_h3.qwen3vl_vision_device_patch import (  # noqa: E402
    _DEVICE_ATTR,
    _SENTINEL,
    apply_qwen3vl_vision_pos_embed_device_patch,
)


def _tiny_vision_model() -> Qwen3VLVisionModel:
    config = Qwen3VLVisionConfig(
        hidden_size=32,
        num_heads=2,
        depth=2,
        intermediate_size=64,
        num_position_embeddings=16,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        out_hidden_size=32,
        spatial_merge_size=2,
        deepstack_visual_indexes=[0],
    )
    torch.manual_seed(0)
    return Qwen3VLVisionModel(config)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    t, h, w = 1, 4, 4
    torch.manual_seed(1)
    hidden_states = torch.randn(t * h * w, 3 * 1 * 2 * 2)
    grid_thw = torch.tensor([[t, h, w]])
    return hidden_states, grid_thw


def test_patched_forward_matches_baseline_on_cpu():
    model = _tiny_vision_model()
    hidden_states, grid_thw = _inputs()
    baseline = model(hidden_states, grid_thw).last_hidden_state

    apply_qwen3vl_vision_pos_embed_device_patch()
    patched = model(hidden_states, grid_thw).last_hidden_state

    assert torch.equal(baseline, patched)


def test_forward_records_input_device():
    apply_qwen3vl_vision_pos_embed_device_patch()
    model = _tiny_vision_model()
    hidden_states, grid_thw = _inputs()
    model(hidden_states, grid_thw)
    assert getattr(model, _DEVICE_ATTR) == hidden_states.device


def test_interpolation_result_moved_to_recorded_device():
    """The core fix: with the compute device differing from pos_embed.weight's device, the
    interpolation result must land on the compute device (validated via `meta`, the only
    second device available on CI)."""
    apply_qwen3vl_vision_pos_embed_device_patch()
    model = _tiny_vision_model()
    _, grid_thw = _inputs()

    setattr(model, _DEVICE_ATTR, torch.device("meta"))
    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
    assert pos_embeds.device.type == "meta"


def test_interpolation_without_recorded_device_stays_on_weight_device():
    """Before any forward has run (attr unset), behavior is upstream's: the result stays on
    the weight's device rather than being moved to a stale or absent target."""
    apply_qwen3vl_vision_pos_embed_device_patch()
    model = _tiny_vision_model()
    _, grid_thw = _inputs()

    assert not hasattr(model, _DEVICE_ATTR)
    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
    assert pos_embeds.device == model.pos_embed.weight.device


def test_apply_is_idempotent():
    apply_qwen3vl_vision_pos_embed_device_patch()
    forward_after_first = Qwen3VLVisionModel.forward
    interpolate_after_first = Qwen3VLVisionModel.fast_pos_embed_interpolate
    assert getattr(Qwen3VLVisionModel, _SENTINEL) is True

    apply_qwen3vl_vision_pos_embed_device_patch()
    assert Qwen3VLVisionModel.forward is forward_after_first
    assert Qwen3VLVisionModel.fast_pos_embed_interpolate is interpolate_after_first
