"""Identifying MOSS-Image's transformer, VAE and pipeline folder, through the whole config factory.

The single files are captured headers written back without their data (`write_header_only_safetensors`):
identification reads nothing else, so every config class sees the real key names, shapes and dtypes, and a second
config claiming the file shows up as a second match.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_MossImage_Config,
    Main_Diffusers_MossImage_Config,
)
from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.state_dicts import moss_image_dit_keys, moss_image_vae_keys
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

Keys = dict[str, tuple[list[int], str]]


def _identify(path: Path) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=False)


def _dit() -> Keys:
    return dict(moss_image_dit_keys.state_dict_keys)


def test_the_transformer_is_a_moss_image_checkpoint_and_nothing_else(tmp_path: Path) -> None:
    result = _identify(write_header_only_safetensors(tmp_path / "moss-image-dit.safetensors", _dit()))

    assert type(result.config) is Main_Checkpoint_MossImage_Config
    assert result.config.base is BaseModelType.MossImage
    assert result.config.format is ModelFormat.Checkpoint
    assert result.match_count == 1


def test_a_comfyui_prefixed_transformer_is_still_identified(tmp_path: Path) -> None:
    keys = {f"model.diffusion_model.{key}": value for key, value in _dit().items()}

    result = _identify(write_header_only_safetensors(tmp_path / "moss.safetensors", keys))

    assert type(result.config) is Main_Checkpoint_MossImage_Config
    assert result.match_count == 1


def test_the_vae_is_one_flux2_vae(tmp_path: Path) -> None:
    keys = dict(moss_image_vae_keys.state_dict_keys)

    result = _identify(write_header_only_safetensors(tmp_path / "moss-image-vae.safetensors", keys))

    assert type(result.config) is VAE_Checkpoint_Flux2_Config
    assert result.match_count == 1


def test_a_lora_for_the_transformer_is_not_a_main_model(tmp_path: Path) -> None:
    # A LoRA addresses the same modules, so the fingerprint's module names are all there.
    keys: Keys = {}
    for module in ("shared_block_modulation", "final_linear_video", "text_embedder.blocks.0.qkv", "layers.0.proj.q"):
        keys[f"diffusion_model.{module}.lora_A.weight"] = ([16, 3072], "BF16")
        keys[f"diffusion_model.{module}.lora_B.weight"] = ([3072, 16], "BF16")
        keys[f"diffusion_model.{module}.weight"] = ([3072, 3072], "BF16")

    result = ModelConfigFactory.from_model_on_disk(
        write_header_only_safetensors(tmp_path / "moss-lora.safetensors", keys), allow_unknown=True
    )

    assert isinstance(result.details[Main_Checkpoint_MossImage_Config.__name__], NotAMatchError)


@pytest.mark.parametrize(
    "quantize",
    [
        pytest.param(lambda keys: keys.update({"layers.0.proj.q.weight_scale": ([], "F32")}), id="scale-side-channel"),
        pytest.param(lambda keys: keys.update({"layers.0.proj.q.weight": ([3072, 3072], "F8_E4M3")}), id="raw-fp8"),
        pytest.param(lambda keys: keys.update({"layers.0.proj.q.comfy_quant": ([32], "U8")}), id="comfy-quant-marker"),
    ],
)
def test_a_quantized_transformer_is_refused_not_registered(tmp_path: Path, quantize: Any) -> None:
    keys = _dit()
    quantize(keys)

    result = ModelConfigFactory.from_model_on_disk(
        write_header_only_safetensors(tmp_path / "moss-fp8.safetensors", keys), allow_unknown=True
    )

    # Final, not "unknown": nothing is written to the database.
    assert result.config is None
    refusal = result.details[Main_Checkpoint_MossImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert "not supported yet" in str(refusal)


def test_a_gguf_transformer_is_refused_with_what_to_install(tmp_path: Path) -> None:
    path = tmp_path / "moss-image-Q4_0.gguf"
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = {
        key: GGMLTensor(
            data=torch.zeros((1,), dtype=torch.uint8),
            ggml_quantization_type=gguf.GGMLQuantizationType.Q4_0,
            tensor_shape=torch.Size(shape),
            compute_dtype=torch.float32,
        )
        for key, (shape, _) in _dit().items()
    }

    with pytest.raises(InvalidMatchError, match="GGUF-quantized .* Install the bf16 transformer instead"):
        Main_Checkpoint_MossImage_Config.from_model_on_disk(mod, {})


def _pipeline_folder(root: Path, *, transformer_class: str = "MOSSImageTransformer", skip: str | None = None) -> Path:
    """A MOSS-Image pipeline folder with the configs identification reads and an empty shard per component."""
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline"}))
    files = {
        "transformer/config.json": json.dumps({"_class_name": transformer_class, "num_layers": 42}),
        "transformer/diffusion_pytorch_model-00001-of-00001.safetensors": None,
        "text_encoder/config.json": json.dumps({"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5"}),
        "text_encoder/model-00001-of-00001.safetensors": None,
        "tokenizer/tokenizer_config.json": json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        "tokenizer/tokenizer.json": json.dumps({"version": "1.0"}),
        "tokenizer/chat_template.jinja": "{{ messages }}",
        "vae/config.json": json.dumps({"_class_name": "AutoencoderKLFlux2"}),
        "vae/diffusion_pytorch_model.safetensors": None,
    }
    for relative, content in files.items():
        if skip is not None and (relative == skip or relative.startswith(f"{skip}/")):
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            write_header_only_safetensors(path, {"weight": ([1], "BF16")})
        else:
            path.write_text(content)
    return root


def test_the_pipeline_folder_is_a_moss_image_diffusers_model(tmp_path: Path) -> None:
    result = _identify(_pipeline_folder(tmp_path / "MOSS-Image-Base"))

    assert type(result.config) is Main_Diffusers_MossImage_Config
    assert result.config.format is ModelFormat.Diffusers
    assert result.match_count == 1


@pytest.mark.parametrize(
    "missing",
    [
        "transformer",
        "text_encoder",
        "vae",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        # MOSS-Image ships its chat template as a file of its own, not as a `tokenizer_config.json` key, and the
        # prompt encoding cannot run without it.
        "tokenizer/chat_template.jinja",
    ],
)
def test_an_incomplete_pipeline_folder_names_what_is_missing(tmp_path: Path, missing: str) -> None:
    result = ModelConfigFactory.from_model_on_disk(
        _pipeline_folder(tmp_path / "MOSS-Image-Base", skip=missing), allow_unknown=True
    )

    assert result.config is None
    refusal = result.details[Main_Diffusers_MossImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert missing in str(refusal)


def test_a_pipeline_folder_with_another_transformer_is_refused(tmp_path: Path) -> None:
    result = ModelConfigFactory.from_model_on_disk(
        _pipeline_folder(tmp_path / "MOSS-Image-Base", transformer_class="SomethingElse"), allow_unknown=True
    )

    refusal = result.details[Main_Diffusers_MossImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert "SomethingElse" in str(refusal)
