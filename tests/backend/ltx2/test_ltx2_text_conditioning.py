"""LTX-2 text connectors under the model cache's partial loading."""

import pytest
import torch

from invokeai.backend.ltx2.text_conditioning import apply_connectors
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)


def _tiny_connectors() -> torch.nn.Module:
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    torch.manual_seed(0)
    return LTX2TextConnectors(
        caption_channels=16,
        text_proj_in_factor=3,
        video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=8,
        video_connector_num_layers=1,
        video_connector_num_learnable_registers=4,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=8,
        audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=4,
    ).eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_connectors_run_with_only_required_weights_on_the_device():
    """With no VRAM to spare, only the registers are resident and the projection weights stream from RAM."""
    connectors = _tiny_connectors()
    # Left-padded, as the Gemma tokenizer emits it, so the registers fill the tail.
    hidden_states = torch.randn(1, 8, 48)
    attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    expected = apply_connectors(connectors, hidden_states, attention_mask, device=torch.device("cpu"))

    apply_custom_layers_to_model(connectors)
    cached = CachedModelWithPartialLoad(connectors, torch.device("cuda"))
    cached.partial_load_to_vram(0)
    assert next(connectors.parameters()).device.type == "cpu"

    actual = apply_connectors(connectors, hidden_states, attention_mask, device=torch.device("cuda"))

    torch.testing.assert_close(actual.video_embeds, expected.video_embeds, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual.audio_embeds, expected.audio_embeds, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual.attention_mask, expected.attention_mask)
