import pytest
import torch
from safetensors.torch import save_file

import invokeai.backend.model_manager.load  # noqa: F401
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Anima_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType


@pytest.mark.parametrize(
    "model_prefix",
    ["", "transformer.", "base_model.model.", "base_model.model.transformer."],
    ids=["bare", "diffusers", "peft", "peft-transformer"],
)
def test_diffusers_cosmos_lora_identifies_as_anima(tmp_path, model_prefix: str) -> None:
    state_dict: dict[str, torch.Tensor] = {}
    for name in ("attn1.to_q", "attn2.to_k", "ff.net.0.proj"):
        prefix = f"{model_prefix}transformer_blocks.0.{name}"
        state_dict[f"{prefix}.lora_A.weight"] = torch.randn(4, 2048)
        state_dict[f"{prefix}.lora_B.weight"] = torch.randn(2048, 4)

    path = tmp_path / "anima_diffusers_lora.safetensors"
    save_file(state_dict, path)

    config = ModelConfigFactory.from_model_on_disk(path, allow_unknown=False).config

    assert isinstance(config, LoRA_LyCORIS_Anima_Config)
    assert config.base == BaseModelType.Anima


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
