"""Identification of Ministral 3B, the text encoder ERNIE-Image encodes prompts with.

It reaches the same `MistralEncoder` type as FLUX.2's encoders but is a different architecture:
26 layers at hidden_size 3072 with YaRN RoPE, loaded as `Ministral3Model`. Geometry is the only
discriminator, and the variant recorded here is what the loader keys its model class on -- so a
misidentification does not fail loudly, it produces an encoder that loads and emits garbage.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs import mistral_encoder as configs_mistral_encoder
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.mistral_encoder import (
    MistralEncoder_Checkpoint_Config,
    MistralEncoder_Diffusers_Config,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import MistralVariantType, ModelFormat, ModelType

# The released geometries. Widths are what separate the families; the tensors below are toy-sized
# in every other dimension so the fixtures stay small.
_TEKKEN_VOCAB = 131072
_MINISTRAL_3B = (3072, 26)
_COW = (5120, 30)
_MISTRAL_24B = (5120, 40)


def _encoder_tensors(
    hidden_size: int,
    num_layers: int,
    vocab_size: int = _TEKKEN_VOCAB,
    *,
    with_vision_stack: bool = False,
) -> dict[str, torch.Tensor]:
    """A Mistral-family state dict at the given geometry, with one column per row kept tiny.

    Only the embedding carries the real width -- that and the layer count are what the probe
    reads. `torch.zeros` keeps the fixture a few MB rather than the size of a real encoder.

    `with_vision_stack` reproduces the Mistral3 multimodal layout the released Ministral 3B
    encoder ships. The FLUX.2 encoders are text-only, so it stays off for them.
    """
    tensors = {"model.embed_tokens.weight": torch.zeros(vocab_size, hidden_size, dtype=torch.bfloat16)}
    for layer in range(num_layers):
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    if with_vision_stack:
        tensors["vision_tower.transformer.layers.0.attention.q_proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
        tensors["multi_modal_projector.linear_1.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    return tensors


def _identify(path: Path, tensors: dict[str, torch.Tensor]):
    save_file(tensors, str(path))
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=True).config


def test_ministral_3b_is_identified_as_its_own_variant(tmp_path: Path) -> None:
    config = _identify(
        tmp_path / "ministral-3-3b.safetensors", _encoder_tensors(*_MINISTRAL_3B, with_vision_stack=True)
    )

    assert isinstance(config, MistralEncoder_Checkpoint_Config)
    assert config.type is ModelType.MistralEncoder
    assert config.format is ModelFormat.Checkpoint
    assert config.variant is MistralVariantType.Ministral3B


@pytest.mark.parametrize(
    "geometry, expected",
    [(_COW, MistralVariantType.Cow), (_MISTRAL_24B, MistralVariantType.Mistral24B)],
)
def test_the_flux2_encoders_keep_their_variants(
    tmp_path: Path, geometry: tuple[int, int], expected: MistralVariantType
) -> None:
    """Widening the probe must not reclassify the encoders that already installed."""
    config = _identify(tmp_path / "mistral.safetensors", _encoder_tensors(*geometry))

    assert isinstance(config, MistralEncoder_Checkpoint_Config)
    assert config.variant is expected


@pytest.mark.parametrize(
    "hidden_size, num_layers",
    [
        (3072, 30),  # Ministral's width at the cow depth
        (5120, 26),  # Mistral Small 3's width at Ministral's depth
        (2048, 26),  # a narrower model entirely
    ],
)
def test_a_geometry_between_the_families_is_refused(tmp_path: Path, hidden_size: int, num_layers: int) -> None:
    """The pairs are matched as pairs. Accepting a loose combination would hand the loader a
    variant whose model class does not fit the weights."""
    config = _identify(tmp_path / "mixed.safetensors", _encoder_tensors(hidden_size, num_layers))

    assert not isinstance(config, MistralEncoder_Checkpoint_Config)


def test_a_small_vocabulary_is_still_refused_at_ministral_geometry(tmp_path: Path) -> None:
    """The floor that keeps Llama-2 out has to cover the new geometry too -- it is the only thing
    separating a Mistral encoder from an unrelated causal LM with the same key names.

    The vision stack is present so the vocabulary is what decides; without it this would pass for
    the reason the test below covers.
    """
    config = _identify(
        tmp_path / "other-lm.safetensors",
        _encoder_tensors(*_MINISTRAL_3B, vocab_size=32000, with_vision_stack=True),
    )

    assert not isinstance(config, MistralEncoder_Checkpoint_Config)


def test_the_released_encoder_folder_is_identified_too(tmp_path: Path) -> None:
    """`baidu/ERNIE-Image`'s `text_encoder/` folder is both directly installable and where the
    tokenizer fallback points, so the folder probe has to recognize the same family the state-dict
    probe does. Its `config.json` names the multimodal wrapper and nests the geometry under
    `text_config`, which is the shape that decides here."""
    folder = tmp_path / "text_encoder"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Mistral3Model"],
                "model_type": "mistral3",
                "text_config": {"hidden_size": 3072, "model_type": "ministral3", "num_hidden_layers": 26},
            }
        ),
        encoding="utf-8",
    )

    config = ModelConfigFactory.from_model_on_disk(folder, allow_unknown=True).config

    assert isinstance(config, MistralEncoder_Diffusers_Config)
    assert config.variant is MistralVariantType.Ministral3B


def test_a_ministral_gguf_is_refused_rather_than_loaded_as_a_mistral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GGUF loader builds a `MistralModel`, which has neither Ministral's YaRN scaling nor its
    attention scale -- it would load cleanly and encode garbage. The refusal has to sit ahead of
    the variant lookup, so this pins the ordering rather than the message.

    `_has_ggml_tensors` stands in for a real GGUF file: building one here would cost far more than
    the ordering it protects.
    """
    monkeypatch.setattr(configs_mistral_encoder, "_has_ggml_tensors", lambda _state_dict: True)

    config = _identify(
        tmp_path / "ministral-3-3b-q8.gguf.safetensors",
        _encoder_tensors(*_MINISTRAL_3B, with_vision_stack=True),
    )

    assert not isinstance(config, MistralEncoder_Checkpoint_Config)
    assert getattr(config, "type", None) is not ModelType.MistralEncoder


def test_the_prompt_enhancer_is_not_claimed_as_an_encoder(tmp_path: Path) -> None:
    """ERNIE-Image's prompt enhancer sits in the same released folder as the encoder and carries
    the identical geometry, vocabulary and embedded Tekken vocab -- it is a bare `Ministral3-
    ForCausalLM` that rewrites prompts. Installing it as the encoder would load cleanly and
    condition every prompt with the wrong weights, so the missing vision tower has to refuse it."""
    config = _identify(
        tmp_path / "ernie-image-prompt-enhancer.safetensors",
        _encoder_tensors(*_MINISTRAL_3B, with_vision_stack=False),
    )

    assert not isinstance(config, MistralEncoder_Checkpoint_Config)


def test_the_prompt_enhancer_refusal_says_which_file_to_install(tmp_path: Path) -> None:
    """The generic geometry message is worse than unhelpful here -- it lists Ministral's 3072/26
    among the *expected* geometries, which is precisely what this file has. Whoever grabbed the
    wrong one of two files from the same folder has to be told so."""
    path = tmp_path / "ernie-image-prompt-enhancer.safetensors"
    save_file(_encoder_tensors(*_MINISTRAL_3B, with_vision_stack=False), str(path))

    with pytest.raises(NotAMatchError, match="prompt enhancer"):
        MistralEncoder_Checkpoint_Config.from_model_on_disk(ModelOnDisk(path), {})
