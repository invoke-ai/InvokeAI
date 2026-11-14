"""A complete Qwen3 causal-LM folder is a Text LLM that also serves as a Qwen3 encoder (`Qwen3EncoderLoader`'s layout)."""

import json
from pathlib import Path
from typing import Any

import pytest

from invokeai.backend.model_manager.configs.base import ModelRole
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import ModelFormat, ModelType, Qwen3VariantType

_FIELDS: dict[str, Any] = {
    "hash": "blake3:x",
    "path": "x",
    "file_size": 1,
    "name": "x",
    "source": "x",
    "source_type": "path",
    "key": "k",
}


def _folder(
    root: Path, *, architecture: str = "Qwen3ForCausalLM", hidden_size: int | None = 2560, **extra: Any
) -> Path:
    root.mkdir()
    config: dict[str, Any] = {"architectures": [architecture], **extra}
    if hidden_size is not None:
        config["hidden_size"] = hidden_size
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    return root


def _identify(root: Path) -> TextLLM_Diffusers_Config:
    return TextLLM_Diffusers_Config.from_model_on_disk(ModelOnDisk(root), dict(_FIELDS))


@pytest.mark.parametrize(
    ("hidden_size", "variant"),
    [(1024, Qwen3VariantType.Qwen3_06B), (2560, Qwen3VariantType.Qwen3_4B), (4096, Qwen3VariantType.Qwen3_8B)],
)
def test_a_qwen3_causal_lm_serves_as_the_qwen3_encoder_of_its_size(
    tmp_path: Path, hidden_size: int, variant: Qwen3VariantType
) -> None:
    config = _identify(_folder(tmp_path / "m", hidden_size=hidden_size))

    assert config.roles == [ModelRole(type=ModelType.Qwen3Encoder, format=ModelFormat.Qwen3Encoder, variant=variant)]


@pytest.mark.parametrize(
    "folder_args",
    [
        {"architecture": "Qwen2ForCausalLM"},
        {"quantization_config": {"quant_method": "bitsandbytes"}},
        {"hidden_size": 3072},
        {"hidden_size": None},
    ],
    ids=["qwen2", "quantization_config", "unknown size", "no size"],
)
def test_other_folders_stay_plain_text_llms(tmp_path: Path, folder_args: dict[str, Any]) -> None:
    assert _identify(_folder(tmp_path / "m", **folder_args)).roles == []


def test_an_sdnq_folder_stays_a_plain_text_llm(tmp_path: Path) -> None:
    root = _folder(tmp_path / "m")
    (root / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}))

    assert _identify(root).roles == []
