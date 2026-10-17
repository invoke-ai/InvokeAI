"""Identification of Qwen-Image-2.1 single-file transformers, and of LoRAs against Qwen-Image's.

Qwen-Image-2.1 reuses Qwen-Image's module names almost everywhere, so both directions matter: a 2.1 file must
not be claimed as Qwen-Image, and the quantizations the loader cannot build must be refused at install
(`InvalidMatchError`), not registered as a model that fails at the first render.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_QwenImage_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage21_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, QwenImage21VariantType

_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/model.safetensors",
    "file_size": 1000,
    "name": "model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _transformer(**extra: Any) -> dict[str, Any]:
    """The keys the probe reads, in ComfyUI's layout (prefixed, fused gate_up)."""
    sd: dict[str, Any] = {
        "model.diffusion_model.img_in.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        "model.diffusion_model.txt_in.text_norm.weight": torch.zeros(8, dtype=torch.bfloat16),
        "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
    }
    sd.update(extra)
    return sd


def _identify(sd: dict[str, Any], path: Path) -> Main_Checkpoint_QwenImage21_Config:
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    with (
        patch("invokeai.backend.model_manager.configs.main.raise_if_not_file"),
        patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields"),
    ):
        return Main_Checkpoint_QwenImage21_Config.from_model_on_disk(mod, dict(_FIELDS))


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("qwen_image_2.1_bf16.safetensors", QwenImage21VariantType.Base),
        ("Qwen-Image-2.1-Turbo_fp8_scaled.safetensors", QwenImage21VariantType.Turbo),
    ],
)
def test_a_single_file_transformer_takes_its_variant_from_the_name(name: str, variant: QwenImage21VariantType) -> None:
    config = _identify(_transformer(), Path(name))
    assert config.base is BaseModelType.QwenImage21
    assert config.variant is variant


def test_qwen_image_is_not_claimed() -> None:
    # Qwen-Image's txt_in is a plain Linear beside a top-level txt_norm, and its MLP is `img_mlp.net`.
    v1 = {
        "img_in.weight": torch.zeros(8, 4),
        "txt_norm.weight": torch.zeros(8),
        "txt_in.weight": torch.zeros(8, 8),
        "transformer_blocks.0.img_mlp.net.0.proj.weight": torch.zeros(16, 8),
    }
    with pytest.raises(NotAMatchError):
        _identify(v1, Path("qwen_image_2512.safetensors"))


def test_an_nvfp4_file_is_refused_at_install() -> None:
    sd = _transformer(**{"model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight_scale_2": torch.ones(())})
    with pytest.raises(InvalidMatchError, match="nvfp4"):
        _identify(sd, Path("qwen_image_2.1_nvfp4.safetensors"))


def test_a_torchao_file_is_refused_at_install() -> None:
    # unsloth's FP8 build: torchao Float8Tensors, saved as the payload and a per-row scale, no `.weight`.
    layer = "transformer_blocks.0.attn.to_q"
    sd = _transformer(
        **{
            f"{layer}._weight_qdata": torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
            f"{layer}._weight_scale": torch.ones(8, 1),
        }
    )
    with pytest.raises(InvalidMatchError, match="torchao"):
        _identify(sd, Path("Qwen-Image-2.1-FP8.safetensors"))


def _int8_gate_up(marker: dict | None) -> dict[str, Any]:
    layer = "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up"
    sd: dict[str, Any] = {
        f"{layer}.weight": torch.zeros(16, 8, dtype=torch.int8),
        f"{layer}.weight_scale": torch.ones(16, 1),
    }
    if marker is not None:
        sd[f"{layer}.comfy_quant"] = torch.frombuffer(bytearray(json.dumps(marker).encode()), dtype=torch.uint8).clone()
    return sd


def test_int8_weights_without_a_convrot_marker_are_refused_at_install(tmp_path: Path) -> None:
    # The markers are read from the file's header, so the file has to exist.
    sd = _transformer(**_int8_gate_up(marker=None))
    path = tmp_path / "qwen_image_2.1_int8_dynamic.safetensors"
    save_file(sd, path)
    with pytest.raises(InvalidMatchError, match="int8"):
        _identify(sd, path)


def test_comfy_int8_convrot_installs(tmp_path: Path) -> None:
    sd = _transformer(**_int8_gate_up(marker={"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}))
    path = tmp_path / "qwen_image_2.1_int8_convrot.safetensors"
    save_file(sd, path)
    assert _identify(sd, path).base is BaseModelType.QwenImage21


def _attention_lora(width: int) -> dict[str, torch.Tensor]:
    """PEFT's common `to_q/to_k/to_v` targets: module names Qwen-Image and Qwen-Image-2.1 share."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for projection in ("to_q", "to_k", "to_v"):
            prefix = f"transformer.transformer_blocks.{block}.attn.{projection}"
            sd[f"{prefix}.lora_A.weight"] = torch.zeros(4, width)
            sd[f"{prefix}.lora_B.weight"] = torch.zeros(width, 4)
    return sd


def _probe_qwen_image_lora(sd: dict[str, torch.Tensor], tmp_path: Path) -> bool:
    path = tmp_path / "lora.safetensors"
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    fields = {**_FIELDS, "path": str(path), "source": str(path)}
    try:
        LoRA_LyCORIS_QwenImage_Config.from_model_on_disk(mod, fields)
    except NotAMatchError:
        return False
    return True


def test_an_attention_only_lora_is_told_apart_by_its_width(tmp_path: Path) -> None:
    assert _probe_qwen_image_lora(_attention_lora(3072), tmp_path)
    assert not _probe_qwen_image_lora(_attention_lora(4096), tmp_path)


def test_a_lora_on_the_gated_mlp_is_not_qwen_image(tmp_path: Path) -> None:
    sd = {
        "lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_down.weight": torch.zeros(4, 3072),
        "lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_up.weight": torch.zeros(3072, 4),
    }
    assert not _probe_qwen_image_lora(sd, tmp_path)
