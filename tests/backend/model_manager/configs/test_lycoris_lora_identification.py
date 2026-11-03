from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

import invokeai.backend.model_manager.configs.lora as lora_config
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import (
    LoRA_LyCORIS_Flux2_Config,
    LoRA_LyCORIS_FLUX_Config,
    LoRA_LyCORIS_SDXL_Config,
    _lokr_in_dim,
    _lokr_out_dim,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux2VariantType
from invokeai.backend.model_manager.util.model_util import lora_token_vector_length

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/lycoris.safetensors",
    "file_size": 1000,
    "name": "lycoris",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _tensor(*shape: int) -> torch.Tensor:
    return torch.empty(shape, device="meta")


def _add_lokr_operand(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    operand: str,
    shape: tuple[int, int],
    factorized: bool,
) -> None:
    out_dim, in_dim = shape
    if factorized:
        rank = 2
        state_dict[f"{prefix}.lokr_{operand}_a"] = _tensor(out_dim, rank)
        state_dict[f"{prefix}.lokr_{operand}_b"] = _tensor(rank, in_dim)
    else:
        state_dict[f"{prefix}.lokr_{operand}"] = _tensor(out_dim, in_dim)


def _lokr_layer(
    prefix: str,
    out_dim: int,
    in_dim: int,
    *,
    w1_factorized: bool,
    w2_factorized: bool,
) -> dict[str, torch.Tensor]:
    assert out_dim % 4 == 0
    assert in_dim % 8 == 0
    state_dict: dict[str, torch.Tensor] = {}
    _add_lokr_operand(state_dict, prefix, "w1", (4, 8), w1_factorized)
    _add_lokr_operand(state_dict, prefix, "w2", (out_dim // 4, in_dim // 8), w2_factorized)
    return state_dict


def _loha_layer(prefix: str, out_dim: int, in_dim: int) -> dict[str, torch.Tensor]:
    rank = 2
    return {
        f"{prefix}.hada_w1_a": _tensor(out_dim, rank),
        f"{prefix}.hada_w1_b": _tensor(rank, in_dim),
        f"{prefix}.hada_w2_a": _tensor(out_dim, rank),
        f"{prefix}.hada_w2_b": _tensor(rank, in_dim),
    }


_LYCORIS_BUILDERS: dict[str, Callable[[str, int, int], dict[str, torch.Tensor]]] = {
    "lokr": lambda prefix, out_dim, in_dim: _lokr_layer(
        prefix,
        out_dim,
        in_dim,
        w1_factorized=False,
        w2_factorized=True,
    ),
    "loha": _loha_layer,
}


def _flux_state_dict(layout: str, algorithm: str, hidden_size: int, mlp_size: int) -> dict[str, torch.Tensor]:
    builder = _LYCORIS_BUILDERS[algorithm]
    prefixes = {
        "bfl": (
            "diffusion_model.double_blocks.0.img_attn.proj",
            "diffusion_model.double_blocks.0.img_mlp.0",
        ),
        "kohya": (
            "lora_unet_double_blocks_0_img_attn_proj",
            "lora_unet_double_blocks_0_img_mlp_0",
        ),
    }
    attn_prefix, mlp_prefix = prefixes[layout]
    state_dict = builder(attn_prefix, hidden_size, hidden_size)
    state_dict.update(builder(mlp_prefix, mlp_size, hidden_size))
    return state_dict


def _model_on_disk(state_dict: dict[str, torch.Tensor]) -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = state_dict
    mod.metadata.return_value = {}
    return mod


@pytest.mark.parametrize(
    ("w1_factorized", "w2_factorized"),
    [
        pytest.param(False, False, id="both-full"),
        pytest.param(False, True, id="w2-factorized"),
        pytest.param(True, False, id="w1-factorized"),
        pytest.param(True, True, id="both-factorized"),
    ],
)
def test_lokr_dimensions_allow_each_operand_to_be_factorized_independently(
    w1_factorized: bool, w2_factorized: bool
) -> None:
    prefix = "layer"
    state_dict = _lokr_layer(
        prefix,
        out_dim=320,
        in_dim=768,
        w1_factorized=w1_factorized,
        w2_factorized=w2_factorized,
    )

    assert _lokr_in_dim(state_dict, prefix) == 768
    assert _lokr_out_dim(state_dict, prefix) == 320


@pytest.mark.parametrize(
    ("w1_factorized", "w2_factorized"),
    [
        pytest.param(False, False, id="both-full"),
        pytest.param(False, True, id="w2-factorized"),
        pytest.param(True, False, id="w1-factorized"),
        pytest.param(True, True, id="both-factorized"),
    ],
)
@pytest.mark.parametrize("token_vector_length", [768, 1024], ids=["sd1", "sd2"])
def test_sd_lokr_token_vector_length_supports_full_and_factorized_operands(
    w1_factorized: bool, w2_factorized: bool, token_vector_length: int
) -> None:
    prefix = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k"
    state_dict = _lokr_layer(
        prefix,
        out_dim=320,
        in_dim=token_vector_length,
        w1_factorized=w1_factorized,
        w2_factorized=w2_factorized,
    )

    assert lora_token_vector_length(state_dict) == token_vector_length


@pytest.mark.parametrize("algorithm", ["lokr", "loha"])
@pytest.mark.parametrize("layout", ["bfl", "kohya"])
@pytest.mark.parametrize(
    ("hidden_size", "mlp_size", "variant"),
    [
        pytest.param(3072, 18432, Flux2VariantType.Klein4B, id="klein-4b"),
        pytest.param(4096, 16384, Flux2VariantType.Klein9B, id="klein-9b"),
    ],
)
def test_flux2_config_identifies_lycoris_layouts(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    algorithm: str,
    hidden_size: int,
    mlp_size: int,
    variant: Flux2VariantType,
) -> None:
    monkeypatch.setattr(lora_config, "raise_if_not_file", lambda _mod: None)
    mod = _model_on_disk(_flux_state_dict(layout, algorithm, hidden_size, mlp_size))

    config = LoRA_LyCORIS_Flux2_Config.from_model_on_disk(mod, dict(_REQUIRED_FIELDS))

    assert config.base is BaseModelType.Flux2
    assert config.variant is variant


@pytest.mark.parametrize("algorithm", ["lokr", "loha"])
@pytest.mark.parametrize("layout", ["bfl", "kohya"])
def test_flux1_lycoris_layout_is_not_misidentified_as_flux2(
    monkeypatch: pytest.MonkeyPatch, layout: str, algorithm: str
) -> None:
    monkeypatch.setattr(lora_config, "raise_if_not_file", lambda _mod: None)
    mod = _model_on_disk(_flux_state_dict(layout, algorithm, hidden_size=3072, mlp_size=12288))

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Flux2_Config.from_model_on_disk(mod, dict(_REQUIRED_FIELDS))

    config = LoRA_LyCORIS_FLUX_Config.from_model_on_disk(mod, dict(_REQUIRED_FIELDS))
    assert config.base is BaseModelType.Flux


@pytest.mark.parametrize("algorithm", ["lokr", "loha"])
def test_sdxl_lycoris_layout_is_not_misidentified_as_flux2(monkeypatch: pytest.MonkeyPatch, algorithm: str) -> None:
    monkeypatch.setattr(lora_config, "raise_if_not_file", lambda _mod: None)
    prefix = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k"
    mod = _model_on_disk(_LYCORIS_BUILDERS[algorithm](prefix, 320, 2048))

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Flux2_Config.from_model_on_disk(mod, dict(_REQUIRED_FIELDS))

    config = LoRA_LyCORIS_SDXL_Config.from_model_on_disk(mod, dict(_REQUIRED_FIELDS))
    assert config.base is BaseModelType.StableDiffusionXL
