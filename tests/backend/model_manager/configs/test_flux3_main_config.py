"""Identification of FLUX.3 Image models, end to end through `ModelConfigFactory`.

Run over the whole candidate set, because the question is that exactly one config claims each release:
FLUX.3 GGUFs call themselves `general.architecture = "flux"`, and the FLUX.1 and FLUX.2 detectors read some
of the same key names (`txt_in`, `time_in`, `single_blocks`).
"""

import json
from pathlib import Path

import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Flux3_Config,
    Main_Diffusers_Flux3_Config,
    Main_GGUF_Flux3_Config,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import Flux3VariantType, ModelFormat
from tests.backend.flux3.flux3_fixture import tiny_flux3_model, to_native, write_flux3_gguf
from tests.backend.model_manager.load.state_dicts import flux3_klein_base_keys, flux3_klein_keys
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

# The released folder's JSON files (black-forest-labs/flux-3-image-klein, 2026-10-15), shortened where nothing
# reads the rest. The root `config.json` is BFL's own package config, which sits beside the pipeline.
_MODULAR_INDEX = {
    "_blocks_class_name": "Flux3DistilledAutoBlocks",
    "_class_name": "Flux3DistilledModularPipeline",
    "_diffusers_version": "0.41.0.dev0",
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", {"subfolder": "text_encoder"}],
    "tokenizer": ["transformers", "Qwen3VLProcessor", {"subfolder": "text_encoder"}],
    "transformer": ["diffusers", "Flux3Transformer2DModel", {"subfolder": "transformer"}],
    "vae": ["diffusers", "AutoencoderKLFlux2", {"subfolder": "vae"}],
}
_BFL_PACKAGE_CONFIG = {
    "format_version": 1,
    "model": "flux-3-image-klein",
    "quantization": None,
    "dit": {"flow_map": True},
}
_TRANSFORMER_CONFIG = {
    "_class_name": "Flux3Transformer2DModel",
    "attention_head_dim": 128,
    "joint_attention_dim": 20480,
    "num_layers": 5,
    "num_single_layers": 28,
}


def _identify(path: Path):
    return ModelConfigFactory.from_model_on_disk(ModelOnDisk(path, "random"), {}, allow_unknown=True)


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def _write_folder(root: Path, *, flow_map: bool, base: bool = False) -> Path:
    index = dict(_MODULAR_INDEX)
    if base:
        index["_class_name"] = "Flux3ModularPipeline"
    _write_json(root / "modular_model_index.json", index)
    _write_json(root / "config.json", _BFL_PACKAGE_CONFIG)
    _write_json(root / "transformer" / "config.json", {**_TRANSFORMER_CONFIG, "flow_map": flow_map})
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    _write_json(root / "vae" / "config.json", {"_class_name": "AutoencoderKLFlux2", "latent_channels": 32})
    _write_json(root / "text_encoder" / "config.json", {"architectures": ["Qwen3VLForConditionalGeneration"]})
    (root / "text_encoder" / "tokenizer.json").write_text("{}")
    return root


@pytest.mark.parametrize(
    ("flow_map", "base", "variant"),
    [(True, False, Flux3VariantType.Klein), (False, True, Flux3VariantType.KleinBase)],
    ids=["klein", "base"],
)
def test_a_diffusers_folder_installs_with_the_variant_its_transformer_declares(
    tmp_path: Path, flow_map: bool, base: bool, variant: Flux3VariantType
) -> None:
    result = _identify(_write_folder(tmp_path / "flux-3-image-klein", flow_map=flow_map, base=base))

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Diffusers_Flux3_Config)
    assert result.config.variant is variant
    assert result.config.default_settings is not None
    assert result.config.default_settings.steps == (16 if flow_map else 50)


@pytest.mark.parametrize("missing", ["vae/config.json", "text_encoder/config.json", "text_encoder/tokenizer.json"])
def test_a_partial_diffusers_folder_is_refused_by_name(tmp_path: Path, missing: str) -> None:
    root = _write_folder(tmp_path / "flux3", flow_map=True)
    (root / missing).unlink()

    with pytest.raises(InvalidMatchError):
        Main_Diffusers_Flux3_Config.from_model_on_disk(ModelOnDisk(root, "random"), {})


@pytest.mark.parametrize(
    ("fixture", "variant"),
    [(flux3_klein_keys, Flux3VariantType.Klein), (flux3_klein_base_keys, Flux3VariantType.KleinBase)],
    ids=["klein", "base"],
)
def test_a_released_single_file_installs_as_flux3(tmp_path: Path, fixture, variant: Flux3VariantType) -> None:
    path = write_header_only_safetensors(tmp_path / "flux3_image_klein.safetensors", fixture.state_dict_keys)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Checkpoint_Flux3_Config)
    assert result.config.variant is variant


# What each quantized release changes on one linear, reduced to the header entries the refusal reads.
_QUANTIZED_LAYERS = {
    # BFL `variants/fp8r`: per-row E4M3 with a scale vector.
    "fp8r": {"weight": ([3072, 3072], "F8_E4M3"), "weight_scale": ([3072], "F32")},
    # A plain fp8 cast with no side channel.
    "fp8-raw": {"weight": ([3072, 3072], "F8_E4M3")},
    # ComfyUI `int8_convrot`: int8 weights tagged by a `.comfy_quant` marker.
    "int8-convrot": {"weight": ([3072, 3072], "I8"), "comfy_quant": ([16], "U8")},
    # nvfp4 (ComfyUI and ModelOpt): two values per byte, block and tensor scales.
    "nvfp4": {"weight": ([3072, 1536], "U8"), "weight_scale": ([3072, 192], "F8_E4M3"), "weight_scale_2": ([], "F32")},
}


@pytest.mark.parametrize("layout", sorted(_QUANTIZED_LAYERS))
def test_a_quantized_single_file_is_refused_with_a_reason(tmp_path: Path, layout: str) -> None:
    keys = dict(flux3_klein_keys.state_dict_keys)
    for suffix, entry in _QUANTIZED_LAYERS[layout].items():
        keys[f"single_blocks.0.q_proj.{suffix}"] = entry
    path = write_header_only_safetensors(tmp_path / f"flux3_{layout}.safetensors", keys)

    result = _identify(path)

    # Refused outright: neither installed as FLUX.3 nor taken by another config, nor registered as unknown.
    assert result.config is None
    assert any("quantized" in str(error) for error in result.invalid_matches)


def test_a_lora_with_flux3_names_is_not_a_main_model(tmp_path: Path) -> None:
    keys = {
        f"{name[: -len('.weight')]}.lora_A.weight": ([4, 3072], "BF16")
        for name in flux3_klein_keys.state_dict_keys
        if name.endswith(".weight")
    }
    path = write_header_only_safetensors(tmp_path / "flux3_lora.safetensors", keys)

    with pytest.raises(NotAMatchError):
        Main_Checkpoint_Flux3_Config.from_model_on_disk(ModelOnDisk(path, "random"), {})


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."], ids=["bare", "comfy-prefixed"])
def test_a_gguf_installs_as_flux3_and_not_as_flux1(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "flux3_image_klein_sd_Q8_0.gguf"
    write_flux3_gguf(path, to_native(tiny_flux3_model(flow_map=True)), prefix=prefix)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_GGUF_Flux3_Config)
    assert result.config.format is ModelFormat.GGUFQuantized
    assert result.config.variant is Flux3VariantType.Klein


def test_a_full_repo_download_sizes_the_tokenizer_without_the_root_package(tmp_path: Path) -> None:
    """The tokenizer has no folder of its own; sizing it by the folder root counted BFL's 11 GB single file there.

    The model cache reserves that size before a load, so a few-MB tokenizer evicted the transformer.
    """
    from unittest.mock import MagicMock

    from invokeai.backend.model_manager.load.model_loaders.flux3 import Flux3DiffusersModel
    from invokeai.backend.model_manager.taxonomy import SubModelType

    root = _write_folder(tmp_path / "flux-3-image-klein", flow_map=True)
    (root / "model.safetensors").write_bytes(b"\0" * 2_000_000)
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 1_000)
    config = _identify(root).config
    assert isinstance(config, Main_Diffusers_Flux3_Config)
    loader = Flux3DiffusersModel(app_config=MagicMock(), logger=MagicMock(), ram_cache=MagicMock())

    assert loader.get_size_fs(config, root, SubModelType.Tokenizer) == 0
    assert 1_000 <= loader.get_size_fs(config, root, SubModelType.Transformer) < 2_000_000
