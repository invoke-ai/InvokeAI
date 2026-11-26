"""The Diffusers-layout Qwen2.5-VL encoder loaders leave the LM head in the file.

Conditioning reads the inner model's hidden states, so the head's weights would only take RAM and VRAM. A toy encoder
is saved with `save_pretrained`, which writes the legacy `model.layers.*` key layout the released encoders ship in, and
read back through both loaders. The single-file and GGUF loaders are covered beside their own formats.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors import safe_open
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from invokeai.backend.model_manager.configs.main import Main_Diffusers_QwenImage_Config
from invokeai.backend.model_manager.configs.qwen_vl_encoder import QwenVLEncoder_Diffusers_Config
from invokeai.backend.model_manager.load.model_loaders import qwen_image
from invokeai.backend.model_manager.load.model_loaders.qwen_image import QwenImageDiffusersModel, QwenVLEncoderLoader
from invokeai.backend.model_manager.taxonomy import SubModelType


def _tiny_config() -> Qwen2_5_VLConfig:
    return Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 64,
            "fullatt_block_indexes": [1],
        },
    )


@pytest.fixture
def reference(tmp_path: Path) -> Qwen2_5_VLForConditionalGeneration:
    """The full encoder, saved as `tmp_path / "text_encoder"`."""
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(_tiny_config()).eval()
    model.save_pretrained(tmp_path / "text_encoder")
    return model


def _load_standalone(root: Path) -> torch.nn.Module:
    loader = object.__new__(QwenVLEncoderLoader)
    config = QwenVLEncoder_Diffusers_Config.model_construct(path=str(root))
    return loader._load_model(config, SubModelType.TextEncoder)


def _load_from_main_model(root: Path) -> torch.nn.Module:
    index = {"text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"]}
    (root / "model_index.json").write_text(json.dumps(index))
    loader = object.__new__(QwenImageDiffusersModel)
    loader._logger = MagicMock()
    config = Main_Diffusers_QwenImage_Config.model_construct(path=str(root), repo_variant=None)
    return loader._load_model(config, SubModelType.TextEncoder)


@pytest.mark.parametrize("load", [_load_standalone, _load_from_main_model], ids=["standalone", "main_model"])
def test_the_encoder_loads_every_weight_but_the_lm_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reference: Qwen2_5_VLForConditionalGeneration, load
) -> None:
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    with safe_open(tmp_path / "text_encoder" / "model.safetensors", framework="pt") as saved:
        assert {"lm_head.weight", "model.layers.0.self_attn.q_proj.weight"} <= set(saved.keys())

    encoder = load(tmp_path)

    # Prompt nodes and the LoRA patcher expect the causal-LM class.
    assert isinstance(encoder, Qwen2_5_VLForConditionalGeneration)
    assert list(encoder.lm_head.parameters()) == []
    expected = reference.state_dict()
    del expected["lm_head.weight"]
    loaded = encoder.state_dict()
    assert loaded.keys() == expected.keys()
    for key, value in loaded.items():
        assert torch.equal(value, expected[key].to(value.dtype)), key
    assert not any(tensor.is_meta for tensor in (*encoder.parameters(), *encoder.buffers()))


def test_a_caller_of_the_full_forward_is_told_the_lm_head_was_not_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reference: Qwen2_5_VLForConditionalGeneration
) -> None:
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)

    encoder = _load_standalone(tmp_path)

    with pytest.raises(RuntimeError, match="without its LM head"):
        encoder(input_ids=torch.tensor([[1, 2]]))
