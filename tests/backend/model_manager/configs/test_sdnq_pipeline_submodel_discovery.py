"""Tests that SDNQ pipeline submodel discovery recognizes all compatible Qwen encoder/tokenizer
classes.

The Qwen3 encoder config accepts Qwen2VLForConditionalGeneration / Qwen2ForCausalLM /
Qwen3ForCausalLM and the slow/fast Qwen2 tokenizer classes. _get_submodels() must record the
TextEncoder / Tokenizer submodels for those same classes; otherwise a valid SDNQ pipeline whose
model_index.json advertises e.g. Qwen2ForCausalLM or Qwen2TokenizerFast is mis-recorded as partial
and is_self_contained_sdnq_pipeline() wrongly returns False, forcing separate VAE/Qwen3 sources.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import is_self_contained_sdnq_pipeline
from invokeai.backend.model_manager.configs.main import (
    Main_SDNQ_Diffusers_Flux2_Config,
    Main_SDNQ_Diffusers_ZImage_Config,
)
from invokeai.backend.model_manager.taxonomy import SubModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "file_size": 1000,
    "name": "sdnq-pipeline",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

_ENCODER_CLASSES = ["Qwen2ForCausalLM", "Qwen2VLForConditionalGeneration", "Qwen3ForCausalLM"]


def _write_sdnq_transformer(root: Path, transformer_config: dict) -> None:
    transformer_dir = root / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config), encoding="utf-8")
    (transformer_dir / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")


def _make_flux2_pipeline(root: Path, encoder_class: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2KleinPipeline",
                "transformer": ["diffusers", "Flux2Transformer2DModel"],
                "text_encoder": ["transformers", encoder_class],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            }
        ),
        encoding="utf-8",
    )
    _write_sdnq_transformer(root, {"attention_head_dim": 128, "num_attention_heads": 24, "joint_attention_dim": 7680})
    return root


def _make_zimage_pipeline(root: Path, encoder_class: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "text_encoder": ["transformers", encoder_class],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    _write_sdnq_transformer(root, {"_class_name": "ZImageTransformer2DModel"})
    scheduler_dir = root / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps({"shift": 3.0}), encoding="utf-8")
    return root


def _mod(root: Path) -> MagicMock:
    mod = MagicMock()
    mod.path = root
    mod.name = "sdnq-klein-4b"  # no "base" substring -> distilled variant
    return mod


def _assert_complete_pipeline(config) -> None:
    assert config.submodels is not None
    assert SubModelType.TextEncoder in config.submodels
    assert SubModelType.Tokenizer in config.submodels
    assert SubModelType.VAE in config.submodels
    assert is_self_contained_sdnq_pipeline(config)


@pytest.mark.parametrize("encoder_class", _ENCODER_CLASSES)
def test_flux2_sdnq_pipeline_records_compatible_encoder_and_fast_tokenizer(tmp_path: Path, encoder_class: str):
    root = _make_flux2_pipeline(tmp_path / "flux2", encoder_class)
    config = Main_SDNQ_Diffusers_Flux2_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_complete_pipeline(config)


@pytest.mark.parametrize("encoder_class", _ENCODER_CLASSES)
def test_zimage_sdnq_pipeline_records_compatible_encoder_and_fast_tokenizer(tmp_path: Path, encoder_class: str):
    root = _make_zimage_pipeline(tmp_path / "zimage", encoder_class)
    config = Main_SDNQ_Diffusers_ZImage_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_complete_pipeline(config)
