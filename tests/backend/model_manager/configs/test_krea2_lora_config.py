from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_Krea2_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/krea2-lora.safetensors",
    "file_size": 1000,
    "name": "krea2-lora",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _ambiguous_transformer_only_lora() -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": object(),
    }
    return mod


def _ambiguous_text_encoder_only_lora() -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "text_encoder.language_model.layers.0.self_attn.q_proj.lora_A.weight": object(),
        "text_encoder.language_model.layers.0.self_attn.q_proj.lora_B.weight": object(),
    }
    return mod


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_accepts_ambiguous_transformer_only_lora(_raise_if_not_file) -> None:
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(
        _ambiguous_transformer_only_lora(), {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2}
    )

    assert config.base is BaseModelType.Krea2


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_automatic_probe_rejects_ambiguous_transformer_only_lora(_raise_if_not_file) -> None:
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(_ambiguous_transformer_only_lora(), {**_REQUIRED_FIELDS})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_rejects_incomplete_lora_pair(_raise_if_not_file) -> None:
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": object(),
    }

    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2})


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_explicit_krea2_override_accepts_text_encoder_only_lora(_raise_if_not_file) -> None:
    config = LoRA_LyCORIS_Krea2_Config.from_model_on_disk(
        _ambiguous_text_encoder_only_lora(), {**_REQUIRED_FIELDS, "base": BaseModelType.Krea2}
    )

    assert config.base is BaseModelType.Krea2


@patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
def test_automatic_probe_rejects_ambiguous_text_encoder_only_lora(_raise_if_not_file) -> None:
    with pytest.raises(NotAMatchError):
        LoRA_LyCORIS_Krea2_Config.from_model_on_disk(_ambiguous_text_encoder_only_lora(), {**_REQUIRED_FIELDS})
