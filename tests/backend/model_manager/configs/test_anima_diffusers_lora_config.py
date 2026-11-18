import pytest
import torch
from safetensors.torch import save_file

import invokeai.backend.model_manager.load  # noqa: F401
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Anima_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType

PREFIXES = ["", "transformer.", "base_model.model.", "base_model.model.transformer."]
PREFIX_IDS = ["bare", "diffusers", "peft", "peft-transformer"]


def _lora_layers(prefix: str, layers: dict[str, tuple[int, int]], block: int = 0) -> dict[str, torch.Tensor]:
    """Rank-4 low-rank layers of `transformer_blocks.<block>`, each given as (input width, output width)."""
    state_dict: dict[str, torch.Tensor] = {}
    for name, (in_features, out_features) in layers.items():
        layer = f"{prefix}transformer_blocks.{block}.{name}"
        state_dict[f"{layer}.lora_A.weight"] = torch.zeros(4, in_features)
        state_dict[f"{layer}.lora_B.weight"] = torch.zeros(out_features, 4)
    return state_dict


def _identify(tmp_path, state_dict: dict[str, torch.Tensor]):
    path = tmp_path / "lora.safetensors"
    save_file(state_dict, path)
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=True).config


# Anima: 2048 wide, text context 1024.
ANIMA_LAYERS = {
    "attn1.to_q": (2048, 2048),
    "attn1.to_out.0": (2048, 2048),
    "attn2.to_q": (2048, 2048),
    "attn2.to_k": (1024, 2048),
    "attn2.to_v": (1024, 2048),
    "attn2.to_out.0": (2048, 2048),
    "ff.net.0.proj": (2048, 8192),
}


@pytest.mark.parametrize("model_prefix", PREFIXES, ids=PREFIX_IDS)
def test_diffusers_cosmos_lora_identifies_as_anima(tmp_path, model_prefix: str) -> None:
    config = _identify(tmp_path, _lora_layers(model_prefix, ANIMA_LAYERS))

    assert isinstance(config, LoRA_LyCORIS_Anima_Config)
    assert config.base == BaseModelType.Anima


@pytest.mark.parametrize("model_prefix", PREFIXES, ids=PREFIX_IDS)
def test_video_only_ltx2_lora_is_not_identified_as_anima(tmp_path, model_prefix: str) -> None:
    """LTX-2's video stream (4096 wide) names its layers as Anima's diffusers layout does."""
    layers = {
        "attn1.to_q": (4096, 4096),
        "attn1.to_out.0": (4096, 4096),
        "attn2.to_q": (4096, 4096),
        "attn2.to_k": (4096, 4096),
        "ff.net.0.proj": (4096, 16384),
    }

    config = _identify(tmp_path, _lora_layers(model_prefix, layers))

    assert config.base == BaseModelType.Unknown


# Widths read from the released models' configs: (attn1 width, attn2 width or None, attn2 text context).
FOREIGN_TRANSFORMERS = {
    "cogview4": (4096, None, None),
    "mochi": (3072, None, None),
    "cogvideox-2b": (1920, None, None),
    "cogvideox-5b": (3072, None, None),
    "pixart-sigma": (1152, 1152, 1152),
    "sana-600m": (1152, 1152, 1152),
    "sana-1600m": (2240, 2240, 2240),
    # Self-attention is as wide as Anima's; only the text context tells them apart.
    "ltx-video": (2048, 2048, 2048),
}


def _foreign_layers(hidden: int, attn2_hidden: int | None, context: int | None) -> dict[str, tuple[int, int]]:
    layers = {"attn1.to_q": (hidden, hidden), "attn1.to_out.0": (hidden, hidden)}
    if attn2_hidden is not None and context is not None:
        layers |= {
            "attn2.to_q": (attn2_hidden, attn2_hidden),
            "attn2.to_k": (context, attn2_hidden),
            "attn2.to_out.0": (attn2_hidden, attn2_hidden),
        }
    return layers


@pytest.mark.parametrize("model_prefix", PREFIXES, ids=PREFIX_IDS)
@pytest.mark.parametrize("family", FOREIGN_TRANSFORMERS)
def test_diffusers_lora_of_another_transformer_is_not_identified_as_anima(
    tmp_path, family: str, model_prefix: str
) -> None:
    layers = _foreign_layers(*FOREIGN_TRANSFORMERS[family])

    config = _identify(tmp_path, _lora_layers(model_prefix, layers))

    assert config.base == BaseModelType.Unknown


def test_attention_that_reads_the_wrong_width_names_the_layer(tmp_path) -> None:
    """LTX-Video's `attn1` is Anima's width; its `attn2.to_k` reads a 2048 context instead of 1024."""
    path = tmp_path / "ltx_video.safetensors"
    save_file(_lora_layers("transformer.", _foreign_layers(*FOREIGN_TRANSFORMERS["ltx-video"])), path)

    with pytest.raises(NotAMatchError, match=r"attn2\.to_k reads 2048 features; Anima's reads 1024"):
        LoRA_LyCORIS_Anima_Config._validate_looks_like_lora(ModelOnDisk(path))
    with pytest.raises(NotAMatchError, match=r"attn2\.to_k"):
        LoRA_LyCORIS_Anima_Config._get_base_or_raise(ModelOnDisk(path))


def test_sd35_medium_lora_is_identified_as_sd3_not_anima(tmp_path) -> None:
    layers = {"attn.to_q": (1536, 1536), "attn2.to_q": (1536, 1536), "attn2.to_k": (1536, 1536)}

    config = _identify(tmp_path, _lora_layers("transformer.", layers))

    assert config.base == BaseModelType.StableDiffusion3


def _lokr_layers(prefix: str, layers: dict[str, int]) -> dict[str, torch.Tensor]:
    """LoKR layers whose input width is the product of two factors (2 * width // 2)."""
    state_dict: dict[str, torch.Tensor] = {}
    for name, width in layers.items():
        layer = f"{prefix}transformer_blocks.0.{name}"
        state_dict[f"{layer}.lokr_w1"] = torch.zeros(8, 2)
        state_dict[f"{layer}.lokr_w2"] = torch.zeros(8, width // 2)
    return state_dict


def test_lokr_whose_factors_multiply_to_anima_widths_identifies_as_anima(tmp_path) -> None:
    config = _identify(tmp_path, _lokr_layers("transformer.", {"attn1.to_q": 2048, "attn2.to_k": 1024}))

    assert isinstance(config, LoRA_LyCORIS_Anima_Config)


def test_lokr_whose_factors_multiply_to_another_width_is_not_identified_as_anima(tmp_path) -> None:
    config = _identify(tmp_path, _lokr_layers("transformer.", {"attn1.to_q": 4096, "attn2.to_k": 4096}))

    assert config.base == BaseModelType.Unknown


def test_attention_layers_without_a_width_do_not_reject_an_anima_lora(tmp_path) -> None:
    """A DoRA magnitude carries no input width, so it neither confirms nor refutes one."""
    state_dict = _lora_layers("transformer.", {"attn2.to_k": (1024, 2048), "ff.net.0.proj": (2048, 8192)})
    state_dict["transformer.transformer_blocks.0.attn1.to_q.dora_scale"] = torch.zeros(1, 2048)

    config = _identify(tmp_path, state_dict)

    assert isinstance(config, LoRA_LyCORIS_Anima_Config)


def test_sd_unet_transformer_block_is_not_identified_as_anima(tmp_path) -> None:
    prefix = "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
    state_dict = {
        f"{prefix}.lora_A.weight": torch.randn(4, 320),
        f"{prefix}.lora_B.weight": torch.randn(320, 4),
    }

    path = tmp_path / "sd_unet_lora.safetensors"
    save_file(state_dict, path)

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Anima_Config._validate_looks_like_lora(ModelOnDisk(path))


def test_diffusers_flux_ff_net_is_not_identified_as_anima(tmp_path) -> None:
    prefix = "transformer.transformer_blocks.0.ff.net.0.proj"
    state_dict = {
        f"{prefix}.lora_A.weight": torch.randn(4, 3072),
        f"{prefix}.lora_B.weight": torch.randn(12288, 4),
    }

    path = tmp_path / "flux_diffusers_lora.safetensors"
    save_file(state_dict, path)

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Anima_Config._validate_looks_like_lora(ModelOnDisk(path))


def test_transformer_prefixed_ltx2_lora_is_not_identified_as_anima(tmp_path) -> None:
    state_dict = {
        "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(4, 2048),
        "transformer.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(2048, 4),
        "transformer.transformer_blocks.0.audio_attn1.to_q.lora_A.weight": torch.randn(4, 2048),
        "transformer.transformer_blocks.0.audio_attn1.to_q.lora_B.weight": torch.randn(2048, 4),
    }
    path = tmp_path / "ltx2_transformer_prefix.safetensors"
    save_file(state_dict, path)

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Anima_Config._validate_looks_like_lora(ModelOnDisk(path))


def test_kohya_anima_lora_still_identifies_as_anima(tmp_path) -> None:
    """Kohya keys name `blocks_N`, not the diffusers layout the width check applies to."""
    state_dict = {
        "lora_unet_blocks_0_cross_attn_q_proj.lora_down.weight": torch.zeros(4, 2048),
        "lora_unet_blocks_0_cross_attn_q_proj.lora_up.weight": torch.zeros(2048, 4),
        "lora_unet_blocks_0_mlp_layer1.lora_down.weight": torch.zeros(4, 2048),
        "lora_unet_blocks_0_mlp_layer1.lora_up.weight": torch.zeros(8192, 4),
    }

    assert isinstance(_identify(tmp_path, state_dict), LoRA_LyCORIS_Anima_Config)
