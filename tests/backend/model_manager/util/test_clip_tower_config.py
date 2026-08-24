"""Tests for sub-tower config resolution on full-CLIP checkpoints."""

import pytest
import torch
from transformers import (
    CLIPConfig,
    CLIPModel,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from invokeai.backend.model_manager.util.clip_tower_config import clip_tower_config_override

# A full CLIPModel whose top-level projection_dim (8) differs from the nested
# configs' untouched default (512) — the exact shape of published repos like
# apple/DFN2B-CLIP-ViT-L-14-39B, where CLIPModel loads but the WithProjection
# tower classes build a wrong-sized head.
_TINY_FULL_CLIP = CLIPConfig(
    projection_dim=8,
    vision_config={
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 8,
        "patch_size": 4,
    },
    text_config={
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 12,
        "vocab_size": 99,
    },
)


@pytest.fixture
def full_clip_dir(tmp_path):
    torch.manual_seed(0)
    CLIPModel(_TINY_FULL_CLIP).save_pretrained(tmp_path)
    return tmp_path


def test_override_copies_top_level_projection_dim(full_clip_dir) -> None:
    for tower in ("vision", "text"):
        override = clip_tower_config_override(full_clip_dir, tower)
        assert override is not None
        assert override.projection_dim == 8


def test_towers_load_and_match_full_model(full_clip_dir) -> None:
    full = CLIPModel.from_pretrained(full_clip_dir).eval()
    vision = CLIPVisionModelWithProjection.from_pretrained(
        full_clip_dir, config=clip_tower_config_override(full_clip_dir, "vision")
    ).eval()
    text = CLIPTextModelWithProjection.from_pretrained(
        full_clip_dir, config=clip_tower_config_override(full_clip_dir, "text")
    ).eval()

    assert vision.visual_projection.out_features == 8
    assert text.text_projection.out_features == 8

    def as_tensor(features) -> torch.Tensor:
        # transformers >= 5 returns BaseModelOutputWithPooling from
        # CLIPModel.get_*_features; older versions returned a bare tensor.
        if isinstance(features, torch.Tensor):
            return features
        return features.pooler_output

    torch.manual_seed(1)
    pixel_values = torch.randn(1, 3, 8, 8)
    input_ids = torch.tensor([[1, 5, 9, 2]])
    with torch.no_grad():
        assert torch.allclose(
            vision(pixel_values=pixel_values).image_embeds, as_tensor(full.get_image_features(pixel_values))
        )
        assert torch.allclose(text(input_ids=input_ids).text_embeds, as_tensor(full.get_text_features(input_ids)))


def test_override_is_none_when_config_has_no_model_type(tmp_path) -> None:
    # The install probe accepts a config.json with architectures but no
    # model_type, and the tower classes load such a dir; AutoConfig refuses
    # it. The helper must return None (pre-override behavior), not raise.
    import json

    config = json.loads((_write_vision_only(tmp_path) / "config.json").read_text())
    del config["model_type"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert clip_tower_config_override(tmp_path, "vision") is None


def _write_vision_only(tmp_path):
    config = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
        projection_dim=8,
    )
    CLIPVisionModelWithProjection(config).save_pretrained(tmp_path)
    return tmp_path


def test_override_is_none_for_vision_only_checkpoint(tmp_path) -> None:
    # A vision-only dir (e.g. an IP-Adapter image encoder) has a correct config
    # of its own; the override must not interfere.
    _write_vision_only(tmp_path)
    assert clip_tower_config_override(tmp_path, "vision") is None
