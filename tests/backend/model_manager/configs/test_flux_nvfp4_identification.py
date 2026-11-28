"""Probe FLUX checkpoints through the real safetensors and config factory paths."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Flux2_Config, Main_Checkpoint_FLUX_Config
from invokeai.backend.model_manager.taxonomy import Flux2VariantType, FluxVariantType

_FAMILIES = [
    pytest.param(Main_Checkpoint_FLUX_Config, FluxVariantType.Dev, "FLUX.1", id="flux1-dev"),
    pytest.param(Main_Checkpoint_Flux2_Config, Flux2VariantType.Klein4B, "FLUX.2", id="flux2-klein-4b"),
]
_BLOCK = "double_blocks.0.img_attn.qkv"
_PREFIX = "model.diffusion_model."
_REFUSAL = (
    "FLUX transformer checkpoints with NVFP4 are not supported. "
    "Choose a supported FLUX.1 or FLUX.2 checkpoint from the Model Manager."
)


def _layer(name: str, quantization: str, *, out_features: int = 768, in_features: int = 256) -> dict[str, torch.Tensor]:
    match quantization:
        case "nvfp4":
            tensors = {
                "weight": torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
                "weight_scale": torch.ones(out_features, in_features // 16).to(torch.float8_e4m3fn),
                "weight_scale_2": torch.ones(()),
            }
        case "int8_tensorwise":
            tensors = {
                "weight": torch.zeros(out_features, in_features, dtype=torch.int8),
                "weight_scale": torch.ones(out_features, 1),
            }
        case "float8_e4m3fn":
            tensors = {
                "weight": torch.zeros(out_features, in_features).to(torch.float8_e4m3fn),
                "weight_scale": torch.ones(()),
            }
        case _:
            return {f"{name}.weight": torch.zeros(out_features, in_features, dtype=torch.bfloat16)}

    marker = {"format": quantization}
    if quantization == "int8_tensorwise":
        marker.update(convrot=True, convrot_groupsize=256)
    tensors["comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8)
    return {f"{name}.{suffix}": tensor for suffix, tensor in tensors.items()}


def _write_checkpoint(
    path: Path,
    family: str,
    quantization: str,
    *,
    prefix: str = "",
    bundle: bool = False,
    depth: bool = False,
    packed_context: bool = False,
) -> None:
    # Keep the BFL input dimensions used for identification and shrink the hidden width.
    in_channels = 64 if family == "FLUX.1" and not depth else 128
    context_dim = 4096 if family == "FLUX.1" else 7680
    tensors = {
        "img_in.weight": torch.zeros(256, in_channels, dtype=torch.bfloat16),
        **_layer("txt_in", "nvfp4" if packed_context else "bf16", out_features=256, in_features=context_dim),
        "double_blocks.0.img_attn.norm.key_norm.scale": torch.ones(128, dtype=torch.bfloat16),
        **_layer(_BLOCK, quantization),
    }
    if family == "FLUX.1":
        tensors["guidance_in.out_layer.weight"] = torch.zeros(256, 256, dtype=torch.bfloat16)
    tensors = {f"{prefix}{key}": value for key, value in tensors.items()}
    if bundle:
        encoder = (
            "t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q"
            if family == "FLUX.1"
            else "qwen3_4b.transformer.model.layers.0.self_attn.q_proj"
        )
        tensors.update(_layer(f"text_encoders.{encoder}", "nvfp4"))
    save_file(tensors, path)


@pytest.mark.parametrize(("config_type", "_variant", "family"), _FAMILIES)
@pytest.mark.parametrize("prefix", ["", _PREFIX], ids=["bare", "wrapped"])
def test_nvfp4_transformer_is_refused_by_name(tmp_path: Path, config_type, _variant, family: str, prefix: str) -> None:
    path = tmp_path / "transformer.safetensors"
    _write_checkpoint(path, family, "nvfp4", prefix=prefix)

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

    assert result.config is None, type(result.config).__name__
    refusal = result.details[config_type.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert result.invalid_matches == [refusal]
    assert str(refusal) == _REFUSAL


def test_nvfp4_depth_checkpoint_refusal_does_not_guess_a_flux_family(tmp_path: Path) -> None:
    path = tmp_path / "transformer.safetensors"
    # Comfy-Org's FLUX.1 Depth header has 128 image channels and a 4096-wide text input.
    _write_checkpoint(path, "FLUX.1", "nvfp4", depth=True)

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

    assert result.config is None
    assert [str(refusal) for refusal in result.invalid_matches] == [_REFUSAL]


def test_flux2_refuses_nvfp4_in_the_packed_text_projection(tmp_path: Path) -> None:
    path = tmp_path / "transformer.safetensors"
    # Packing halves the physical text width to 3840. The dense image projection still identifies FLUX.2.
    _write_checkpoint(path, "FLUX.2", "bf16", packed_context=True)

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

    assert result.config is None
    assert [str(refusal) for refusal in result.invalid_matches] == [_REFUSAL]


@pytest.mark.parametrize(("config_type", "variant", "family"), _FAMILIES)
@pytest.mark.parametrize(
    ("quantization", "prefix", "bundle"),
    [
        pytest.param("bf16", "", False, id="bf16"),
        pytest.param("int8_tensorwise", "", False, id="int8-convrot"),
        pytest.param("float8_e4m3fn", "", False, id="scaled-fp8"),
        pytest.param("bf16", _PREFIX, True, id="wrapped-nvfp4-encoder-sidecar"),
        pytest.param("bf16", "", True, id="bare-nvfp4-encoder-sidecar"),
    ],
)
def test_supported_transformer_installs(
    tmp_path: Path, config_type, variant, family: str, quantization: str, prefix: str, bundle: bool
) -> None:
    path = tmp_path / "transformer.safetensors"
    _write_checkpoint(path, family, quantization, prefix=prefix, bundle=bundle)

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

    assert isinstance(result.config, config_type)
    assert result.config.variant is variant
    assert not result.invalid_matches
