"""Tests that SDNQ pipeline submodel discovery recognizes the compatible Qwen encoder/tokenizer
classes the loader can actually load.

_get_submodels() must record the TextEncoder / Tokenizer submodels for the text-only Qwen causal-LM
classes and the slow/fast Qwen2 tokenizer classes; otherwise a valid SDNQ pipeline whose
model_index.json advertises e.g. Qwen2ForCausalLM or Qwen2TokenizerFast is mis-recorded as partial
and is_self_contained_sdnq_pipeline() wrongly returns False, forcing separate VAE/Qwen3 sources.

It must NOT record a TextEncoder for Qwen2VLForConditionalGeneration: that is a multimodal Qwen-VL
model (with a visual tower), but the SDNQ pipeline loaders instantiate a text-only Qwen3ForCausalLM,
so treating it as self-contained would mark the pipeline complete even though the loader would fail
on the visual-tower weights.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

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

# Text-only Qwen causal-LM classes the pipeline loader can instantiate. Qwen2VLForConditionalGeneration
# is intentionally excluded (multimodal, not loadable as text-only Qwen3ForCausalLM).
_ENCODER_CLASSES = ["Qwen2ForCausalLM", "Qwen3ForCausalLM"]


def _write_sdnq_transformer(root: Path, transformer_config: dict) -> None:
    transformer_dir = root / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(json.dumps(transformer_config), encoding="utf-8")
    (transformer_dir / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")


def _write_qwen_vl_text_encoder(root: Path) -> None:
    """Write a text_encoder/ folder that actually contains a Qwen-VL model: a config declaring the
    Qwen-VL architecture and SDNQ weights that include both language (model.*) and visual-tower
    (visual.*) keys. The pipeline loader's text-only Qwen3ForCausalLM cannot consume these."""
    te_dir = root / "text_encoder"
    te_dir.mkdir()
    (te_dir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2VLForConditionalGeneration"], "hidden_size": 2560}), encoding="utf-8"
    )
    (te_dir / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(1000, 2560, dtype=torch.uint8),
            "model.embed_tokens.scale": torch.zeros(1000, 1, dtype=torch.float32),
            "visual.patch_embed.proj.weight": torch.zeros(64, 32, dtype=torch.uint8),
            "visual.patch_embed.proj.scale": torch.zeros(64, 1, dtype=torch.float32),
        },
        str(te_dir / "model.safetensors"),
    )


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


def _assert_qwen_vl_pipeline_not_self_contained(config) -> None:
    # The VAE and Tokenizer are still recorded, but the multimodal Qwen-VL text encoder is NOT — so
    # the pipeline is not self-contained and readiness/invocation must require an explicit text-only
    # Qwen source instead of selecting the main model (which the loader could not load).
    assert config.submodels is not None
    assert SubModelType.TextEncoder not in config.submodels
    assert SubModelType.VAE in config.submodels
    assert not is_self_contained_sdnq_pipeline(config)


def test_flux2_sdnq_pipeline_with_qwen_vl_encoder_is_not_self_contained(tmp_path: Path):
    root = _make_flux2_pipeline(tmp_path / "flux2-vl", "Qwen2VLForConditionalGeneration")
    _write_qwen_vl_text_encoder(root)
    config = Main_SDNQ_Diffusers_Flux2_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_qwen_vl_pipeline_not_self_contained(config)


def test_zimage_sdnq_pipeline_with_qwen_vl_encoder_is_not_self_contained(tmp_path: Path):
    root = _make_zimage_pipeline(tmp_path / "zimage-vl", "Qwen2VLForConditionalGeneration")
    _write_qwen_vl_text_encoder(root)
    config = Main_SDNQ_Diffusers_ZImage_Config.from_model_on_disk(
        _mod(root), {**_REQUIRED_FIELDS, "path": root.as_posix()}
    )
    _assert_qwen_vl_pipeline_not_self_contained(config)
