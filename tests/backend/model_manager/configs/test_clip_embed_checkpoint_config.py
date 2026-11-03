"""Identifying ComfyUI's single-file CLIP-L and CLIP-G text encoders, through the whole config factory.

Each case is a captured header written back without its data (`write_header_only_safetensors`): identification
reads nothing else, so every config class sees the real key names, shapes and dtypes, and the test proves that no
other config claims the file and that the CLIP configs claim nothing they must not.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from invokeai.backend.model_manager.configs.clip_embed import (
    CLIPEmbed_Checkpoint_G_Config,
    CLIPEmbed_Checkpoint_L_Config,
)
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import ClipVariantType, ModelFormat
from tests.backend.model_manager.load.state_dicts import clip_g_comfy_keys, clip_l_comfy_keys
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/clip",
    "file_size": 1000,
    "name": "clip",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

Keys = dict[str, tuple[list[int], str]]


def _identify(tmp_path: Path, keys: Keys) -> Any:
    path = write_header_only_safetensors(tmp_path / "clip.safetensors", keys)
    return ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=False)


def _clip_l() -> Keys:
    return dict(clip_l_comfy_keys.state_dict_keys)


def _clip_g() -> Keys:
    return dict(clip_g_comfy_keys.state_dict_keys)


@pytest.mark.parametrize(
    ("keys", "config_class"),
    [(_clip_l(), CLIPEmbed_Checkpoint_L_Config), (_clip_g(), CLIPEmbed_Checkpoint_G_Config)],
    ids=["clip_l", "clip_g"],
)
def test_comfys_clip_files_are_single_file_clip_encoders_of_their_variant(
    tmp_path: Path, keys: Keys, config_class: type
) -> None:
    result = _identify(tmp_path, keys)

    assert type(result.config) is config_class
    assert result.config.format is ModelFormat.Checkpoint
    assert result.match_count == 1


@pytest.mark.parametrize("variant", [ClipVariantType.L, ClipVariantType.G])
def test_a_stored_record_reads_back_as_the_variant_it_was_saved_as(tmp_path: Path, variant: ClipVariantType) -> None:
    # The record goes through the database as a plain dict; its discriminator must include the variant,
    # or L and G records cannot be told apart and fail to load.
    keys = _clip_l() if variant is ClipVariantType.L else _clip_g()
    stored = _identify(tmp_path, keys).config.model_dump(mode="json")

    restored = TypeAdapter(AnyModelConfig).validate_python(stored)

    assert restored.variant is variant
    assert restored.format is ModelFormat.Checkpoint


def test_a_clip_l_with_the_full_models_projection_and_logit_scale_is_still_clip_l(tmp_path: Path) -> None:
    # Text-only exports of fine-tuned CLIP-L (e.g. zer0int's) keep these two from the full CLIP model.
    keys = {**_clip_l(), "logit_scale": ([], "F32"), "text_projection": ([768, 768], "F16")}

    assert isinstance(_identify(tmp_path, keys).config, CLIPEmbed_Checkpoint_L_Config)


def test_a_full_clip_model_with_its_vision_tower_is_not_a_text_encoder(tmp_path: Path) -> None:
    keys = {**_clip_l(), "vision_model.embeddings.patch_embedding.weight": ([1024, 3, 14, 14], "F16")}

    assert _identify(tmp_path, keys).config is None


def test_a_clip_text_tower_of_another_size_is_not_claimed(tmp_path: Path) -> None:
    # SD 2's OpenCLIP-H text tower in transformers names: width 1024, 24 layers.
    keys = {
        key.replace("768", "1024"): ([1024 if d == 768 else d for d in shape], dtype)
        for key, (shape, dtype) in _clip_l().items()
    }
    keys.update({f"text_model.encoder.layers.{i}.self_attn.q_proj.weight": ([1024, 1024], "F16") for i in range(24)})

    result = _identify(tmp_path, keys)

    assert result.config is None
    assert not result.invalid_matches


def test_a_long_clip_l_is_refused_with_the_reason(tmp_path: Path) -> None:
    # Long-CLIP-L (e.g. zer0int's text-only exports, a common FLUX CLIP swap) stretches the position table to 248.
    keys = {**_clip_l(), "text_model.embeddings.position_embedding.weight": ([248, 768], "F16")}

    result = _identify(tmp_path, keys)

    assert result.config is None
    assert len(result.invalid_matches) == 1
    assert "248 positions" in str(result.invalid_matches[0])


def test_a_clip_g_without_its_projection_is_refused_with_the_reason(tmp_path: Path) -> None:
    keys = _clip_g()
    del keys["text_projection.weight"]

    result = _identify(tmp_path, keys)

    assert result.config is None
    assert len(result.invalid_matches) == 1
    assert "no text_projection, which SD 3 needs" in str(result.invalid_matches[0])


@pytest.mark.parametrize("dtype", ["F8_E4M3", "I8"])
def test_a_quantized_clip_is_refused_at_install(tmp_path: Path, dtype: str) -> None:
    keys = {
        key: (shape, dtype if key.endswith("self_attn.q_proj.weight") else stored)
        for key, (shape, stored) in _clip_l().items()
    }

    result = _identify(tmp_path, keys)

    assert result.config is None
    assert len(result.invalid_matches) == 1
    assert "quantized weight(s)" in str(result.invalid_matches[0])
