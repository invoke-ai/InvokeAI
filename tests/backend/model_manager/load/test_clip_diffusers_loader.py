"""Which model a Diffusers CLIP Embed folder loads as, for each variant.

FLUX.1 encodes with a plain CLIPTextModel; SD 3 conditions on CLIP-G's projected pooled output, which only
CLIPTextModelWithProjection carries. A tiny model saved with `save_pretrained` stands in for the real folders.
"""

from pathlib import Path

import pytest
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from invokeai.backend.model_manager.configs.clip_embed import CLIPEmbed_Diffusers_G_Config, CLIPEmbed_Diffusers_L_Config
from invokeai.backend.model_manager.load.model_loaders.flux import CLIPDiffusersLoader
from invokeai.backend.model_manager.taxonomy import SubModelType

TINY = CLIPTextConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=8,
    projection_dim=16,
)


def _folder(tmp_path: Path, model_class: type) -> Path:
    model_class(TINY).save_pretrained(tmp_path / "text_encoder")
    return tmp_path


def _load(config) -> object:
    return object.__new__(CLIPDiffusersLoader)._load_model(config, SubModelType.TextEncoder)


def test_a_clip_l_folder_loads_as_the_plain_text_model_flux_needs(tmp_path: Path) -> None:
    path = _folder(tmp_path, CLIPTextModel)
    config = CLIPEmbed_Diffusers_L_Config.model_construct(path=str(path), name="clip-l")

    assert type(_load(config)) is CLIPTextModel


def test_a_clip_g_folder_loads_with_its_projection(tmp_path: Path) -> None:
    path = _folder(tmp_path, CLIPTextModelWithProjection)
    config = CLIPEmbed_Diffusers_G_Config.model_construct(path=str(path), name="clip-g")

    model = _load(config)

    assert type(model) is CLIPTextModelWithProjection
    saved = CLIPTextModelWithProjection.from_pretrained(path / "text_encoder")
    assert model.text_projection.weight.equal(saved.text_projection.weight)


def test_a_clip_g_folder_without_a_projection_is_refused(tmp_path: Path) -> None:
    path = _folder(tmp_path, CLIPTextModel)
    config = CLIPEmbed_Diffusers_G_Config.model_construct(path=str(path), name="clip-g")

    with pytest.raises(ValueError, match="has no text_projection, which SD 3 needs"):
        _load(config)
