"""Identification of FLUX.2 [dev]'s Mistral Small 3 encoder against the models that share its shape.

Width, depth and the Tekken vocabulary do not settle it alone. Mistral NeMo 12B has all three and a
narrower MLP; Qwen3-14B has the width and depth under another llama.cpp architecture; Ministral 3B is
a Mistral the GGUF loader cannot build. Each would install, load cleanly and condition FLUX.2 with
the wrong model.

GGUF fixtures are headers only -- shapes and string metadata, in the identification suite's stripped
form -- because the real tensors run to gigabytes and identification never reads them. Real stripped
headers of unsloth's llama.cpp build (``llama``) and gguf-org's cow build live in
``tests/model_identification/stripped_models``.
"""

import json
from pathlib import Path

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.mistral_encoder import (
    MistralEncoder_Diffusers_Config,
    MistralEncoder_GGUF_Config,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import MistralVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.model_identification.stripped_model_on_disk import StrippedModelOnDisk

_TEKKEN_VOCAB = 131072
_MISTRAL_SMALL_3 = (5120, 40, 32768)
_MISTRAL_NEMO = (5120, 40, 14336)
_MINISTRAL_3B = (3072, 26, 9216)


def _ggml(rows: int, cols: int) -> GGMLTensor:
    return GGMLTensor(
        torch.empty((rows, 1), dtype=torch.uint8, device="meta"),
        gguf.GGMLQuantizationType.Q4_K,
        torch.Size((rows, cols)),
        torch.float32,
    )


def _llamacpp_tensors(hidden_size: int, num_layers: int, mlp_width: int) -> dict[str, GGMLTensor]:
    tensors = {
        "token_embd.weight": _ggml(_TEKKEN_VOCAB, hidden_size),
        "blk.0.ffn_gate.weight": _ggml(mlp_width, hidden_size),
    }
    tensors.update({f"blk.{layer}.attn_q.weight": _ggml(4096, hidden_size) for layer in range(num_layers)})
    return tensors


def _hf_named_tensors(hidden_size: int, num_layers: int, mlp_width: int) -> dict[str, GGMLTensor]:
    tensors = {
        "model.embed_tokens.weight": _ggml(_TEKKEN_VOCAB, hidden_size),
        "model.layers.0.mlp.gate_proj.weight": _ggml(mlp_width, hidden_size),
    }
    tensors.update(
        {f"model.layers.{layer}.self_attn.q_proj.weight": _ggml(4096, hidden_size) for layer in range(num_layers)}
    )
    return tensors


def _gguf_header(
    path: Path, tensors: dict[str, GGMLTensor], architecture: str | None, **strings: str
) -> StrippedModelOnDisk:
    """``strings`` are further string metadata, keyed under the architecture's prefix."""
    metadata = {} if architecture is None else {"general.architecture": architecture}
    metadata |= {f"{architecture}.{key}": value for key, value in strings.items()}
    path.write_text(json.dumps({**StrippedModelOnDisk.strip(tensors), StrippedModelOnDisk.METADATA_KEY: metadata}))
    return StrippedModelOnDisk(path)


def test_a_llamacpp_mistral_small_3_is_accepted_under_its_newer_architecture_name(tmp_path: Path) -> None:
    """Current llama.cpp writes Mistral Small 3 as ``mistral3``; older builds such as unsloth's say
    ``llama`` (a real stripped header covers that one)."""
    mod = _gguf_header(tmp_path / "mistral-small-3.gguf", _llamacpp_tensors(*_MISTRAL_SMALL_3), "mistral3")

    config = ModelConfigFactory.from_model_on_disk(mod, allow_unknown=False).config

    assert isinstance(config, MistralEncoder_GGUF_Config)
    assert config.variant is MistralVariantType.Mistral24B


@pytest.mark.parametrize("architecture", ["qwen3", None])
def test_a_llamacpp_gguf_of_another_architecture_is_not_a_mistral_encoder(
    tmp_path: Path, architecture: str | None
) -> None:
    """Mistral Small 3's exact geometry, MLP width included, so only the architecture can refuse it.
    The loader would also un-permute its attention rows, which a Qwen3 converter never permuted."""
    mod = _gguf_header(tmp_path / "other.gguf", _llamacpp_tensors(*_MISTRAL_SMALL_3), architecture)

    with pytest.raises(NotAMatchError, match="architecture"):
        MistralEncoder_GGUF_Config.from_model_on_disk(mod, {})


def test_a_yarn_scaled_model_of_mistral_small_3s_shape_is_not_a_mistral_encoder(tmp_path: Path) -> None:
    """Devstral Small 2: a ``mistral3`` llama.cpp build at Mistral Small 3's exact geometry, MLP
    included, but with YaRN RoPE. The loader builds plain RoPE, so it would encode at the wrong
    frequencies; it is a different model, so it installs as unknown rather than being refused."""
    mod = _gguf_header(
        tmp_path / "devstral-small-2.gguf",
        _llamacpp_tensors(*_MISTRAL_SMALL_3),
        "mistral3",
        **{"rope.scaling.type": "yarn"},
    )

    with pytest.raises(NotAMatchError, match="'yarn' RoPE scaling"):
        MistralEncoder_GGUF_Config.from_model_on_disk(mod, {})


def test_mistral_nemo_gguf_is_not_a_flux2_encoder(tmp_path: Path) -> None:
    """Mistral NeMo 12B GGUFs -- and their many finetunes -- are llama.cpp ``llama`` builds of Mistral
    Small 3's width, depth and vocabulary. The MLP is what differs."""
    mod = _gguf_header(tmp_path / "mistral-nemo-12b.gguf", _llamacpp_tensors(*_MISTRAL_NEMO), "llama")

    with pytest.raises(NotAMatchError, match="intermediate_size=14336"):
        MistralEncoder_GGUF_Config.from_model_on_disk(mod, {})


def _write_config(folder: Path, config: dict) -> Path:
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def test_a_mistral_nemo_folder_is_not_a_flux2_encoder(tmp_path: Path) -> None:
    """The folder probe recognizes the same geometries as the state-dict probe, MLP width included."""
    folder = _write_config(
        tmp_path / "Mistral-Nemo-Instruct-2407",
        {
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 5120,
            "intermediate_size": 14336,
            "num_hidden_layers": 40,
        },
    )

    with pytest.raises(NotAMatchError, match="recognized Mistral variant"):
        MistralEncoder_Diffusers_Config.from_model_on_disk(ModelOnDisk(folder), {})


def test_the_bfl_text_encoder_folder_keeps_its_variant(tmp_path: Path) -> None:
    """BFL's ``text_encoder/config.json`` nests the language model's geometry under ``text_config``
    and carries a vision tower whose own ``intermediate_size`` must not be the one read."""
    folder = _write_config(
        tmp_path / "text_encoder",
        {
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": {"hidden_size": 5120, "intermediate_size": 32768, "num_hidden_layers": 40},
            "vision_config": {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24},
        },
    )

    config = ModelConfigFactory.from_model_on_disk(folder, allow_unknown=False).config

    assert isinstance(config, MistralEncoder_Diffusers_Config)
    assert config.variant is MistralVariantType.Mistral24B


@pytest.mark.parametrize("layout", ["llama.cpp", "hf_named_with_vision_tower"])
def test_a_ministral_gguf_is_refused_with_what_to_install_instead(tmp_path: Path, layout: str) -> None:
    """Refused rather than left to fall back to an unknown model: the user tried to install an
    encoder, and the message leads with which file works. Either layout, so a repackaging that keeps
    the vision tower cannot slip past the llama.cpp case. Both declare YaRN, as mistralai's own
    Ministral 3B GGUF does, so this refusal has to win over the plain-RoPE one."""
    if layout == "llama.cpp":
        tensors, architecture = _llamacpp_tensors(*_MINISTRAL_3B), "mistral3"
    else:
        tensors = {
            **_hf_named_tensors(*_MINISTRAL_3B),
            "vision_tower.transformer.layers.0.attention.q_proj.weight": _ggml(4, 4),
        }
        architecture = "ministral3"
    mod = _gguf_header(tmp_path / "ministral-3-3b.gguf", tensors, architecture, **{"rope.scaling.type": "yarn"})

    with pytest.raises(InvalidMatchError, match="^Install the 'ERNIE-Image Ministral Encoder' starter model"):
        MistralEncoder_GGUF_Config.from_model_on_disk(mod, {})

    assert ModelConfigFactory.from_model_on_disk(mod, allow_unknown=True).config is None
