"""Which model a Diffusers CLIP Embed folder loads as, for each variant.

FLUX.1 encodes with a plain CLIPTextModel; SD 3 conditions on CLIP-G's projected pooled output, which only
CLIPTextModelWithProjection carries. A tiny model saved with `save_pretrained` stands in for the real folders.
"""

from pathlib import Path

import pytest
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from invokeai.backend.clip.clip_text_encoder import load_bundled_clip_tokenizer
from invokeai.backend.model_manager.configs.clip_embed import CLIPEmbed_Diffusers_G_Config, CLIPEmbed_Diffusers_L_Config
from invokeai.backend.model_manager.load.model_loaders.flux import CLIPDiffusersLoader
from invokeai.backend.model_manager.taxonomy import ClipVariantType, SubModelType

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


def test_a_bare_encoder_folder_loads_from_its_root_with_the_bundled_tokenizer(tmp_path: Path) -> None:
    """An SD3.5 pipeline's `text_encoder_2` installed on its own: `config.json` at the root, no tokenizer. The
    expected ids are what that pipeline's own `tokenizer_2` gives: CLIP-G pads with "!" (id 0)."""
    CLIPTextModelWithProjection(TINY).save_pretrained(tmp_path)
    config = CLIPEmbed_Diffusers_G_Config.model_construct(path=str(tmp_path), name="text_encoder_2")
    loader = object.__new__(CLIPDiffusersLoader)

    model = loader._load_model(config, SubModelType.TextEncoder)
    tokenizer = loader._load_model(config, SubModelType.Tokenizer)

    saved = CLIPTextModelWithProjection.from_pretrained(tmp_path)
    assert type(model) is CLIPTextModelWithProjection
    assert model.text_projection.weight.equal(saved.text_projection.weight)
    assert (
        tokenizer("a photo!!", padding="max_length", max_length=77).input_ids
        == [49406, 320, 1125, 0, 0, 49407] + [0] * 71
    )
    assert loader.get_size_fs(config, tmp_path, SubModelType.Tokenizer) == 0


def test_a_bundled_folder_keeps_its_own_tokenizer(tmp_path: Path) -> None:
    """The folder's tokenizer, not the bundled CLIP-G one: this one pads with the end token."""
    _folder(tmp_path, CLIPTextModelWithProjection)
    load_bundled_clip_tokenizer(ClipVariantType.L).save_pretrained(tmp_path / "tokenizer")
    config = CLIPEmbed_Diffusers_G_Config.model_construct(path=str(tmp_path), name="clip-g")

    tokenizer = object.__new__(CLIPDiffusersLoader)._load_model(config, SubModelType.Tokenizer)

    assert tokenizer("a photo", padding="max_length", max_length=8).input_ids == [49406, 320, 1125] + [49407] * 5
