"""Tests that Qwen3Encoder_SDNQ_Folder_Config only accepts real Qwen3 encoder folders.

An SDNQ transformer / VAE / other component folder also has a quantization_config.json with
quant_method="sdnq". Without verifying the folder is actually a Qwen3 encoder, such a folder would
be classified as type=qwen3_encoder and only fail later in the loader when Qwen-specific weights
are missing. These tests point the factory at non-Qwen3 SDNQ folders and assert they do NOT resolve
to Qwen3Encoder_SDNQ_Folder_Config, while a real SDNQ Qwen3 encoder folder still does.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Qwen3Encoder_Config,
    Qwen3Encoder_SDNQ_Folder_Config,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import ModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "file_size": 1000,
    "name": "sdnq-qwen3",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _write_sdnq_marker(folder: Path) -> None:
    (folder / "quantization_config.json").write_text(
        json.dumps({"quant_method": "sdnq", "weights_dtype": "uint4", "group_size": 128}), encoding="utf-8"
    )


def _make_sdnq_transformer_folder(root: Path) -> Path:
    """A standalone SDNQ transformer folder (e.g. a Z-Image / FLUX.2 transformer)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"_class_name": "ZImageTransformer2DModel"}), encoding="utf-8")
    _write_sdnq_marker(root)
    save_file(
        {
            "transformer_blocks.0.attn.to_q.weight": torch.zeros(64, 32, dtype=torch.uint8),
            "transformer_blocks.0.attn.to_q.scale": torch.zeros(64, 1, dtype=torch.float32),
        },
        str(root / "diffusion_pytorch_model.safetensors"),
    )
    return root


def _make_sdnq_vae_folder(root: Path) -> Path:
    """A standalone SDNQ VAE folder."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKL"}), encoding="utf-8")
    _write_sdnq_marker(root)
    save_file(
        {
            "decoder.conv_in.weight": torch.zeros(64, 32, dtype=torch.uint8),
            "decoder.conv_in.scale": torch.zeros(64, 1, dtype=torch.float32),
        },
        str(root / "diffusion_pytorch_model.safetensors"),
    )
    return root


def _make_sdnq_qwen3_encoder_folder(root: Path) -> Path:
    """A real SDNQ Qwen3 encoder folder."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2560}), encoding="utf-8"
    )
    _write_sdnq_marker(root)
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(1000, 2560, dtype=torch.uint8),
            "model.embed_tokens.scale": torch.zeros(1000, 1, dtype=torch.float32),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(64, 32, dtype=torch.uint8),
            "model.layers.0.self_attn.q_proj.scale": torch.zeros(64, 1, dtype=torch.float32),
        },
        str(root / "model.safetensors"),
    )
    return root


class TestQwen3EncoderSDNQFolderIdentification:
    def test_sdnq_transformer_folder_not_classified_as_qwen3(self, tmp_path: Path):
        root = _make_sdnq_transformer_folder(tmp_path / "sdnq-transformer")
        result = ModelConfigFactory.from_model_on_disk(root, allow_unknown=True)
        assert not isinstance(result.config, Qwen3Encoder_SDNQ_Folder_Config)
        assert not any(isinstance(m, Qwen3Encoder_SDNQ_Folder_Config) for m in result.all_matches)
        assert result.config is None or result.config.type is not ModelType.Qwen3Encoder

    def test_sdnq_vae_folder_not_classified_as_qwen3(self, tmp_path: Path):
        root = _make_sdnq_vae_folder(tmp_path / "sdnq-vae")
        result = ModelConfigFactory.from_model_on_disk(root, allow_unknown=True)
        assert not isinstance(result.config, Qwen3Encoder_SDNQ_Folder_Config)
        assert not any(isinstance(m, Qwen3Encoder_SDNQ_Folder_Config) for m in result.all_matches)
        assert result.config is None or result.config.type is not ModelType.Qwen3Encoder

    def test_real_sdnq_qwen3_encoder_folder_still_accepted(self, tmp_path: Path):
        """The SDNQ folder config must still accept a genuine SDNQ Qwen3 encoder folder."""
        root = _make_sdnq_qwen3_encoder_folder(tmp_path / "sdnq-qwen3")
        mod = ModelOnDisk(root)

        config = Qwen3Encoder_SDNQ_Folder_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": root.as_posix()})
        assert config.type is ModelType.Qwen3Encoder
        assert config.format.value == "sdnq_quantized"

    def test_plain_qwen3_config_rejects_sdnq_folder(self, tmp_path: Path):
        """The unquantized Qwen3 encoder config must reject an SDNQ Qwen3 folder (same Qwen3 class
        name), so it does not compete with Qwen3Encoder_SDNQ_Folder_Config."""
        root = _make_sdnq_qwen3_encoder_folder(tmp_path / "sdnq-qwen3")
        # Use a real ModelOnDisk so the guard's marker + state-dict fallback checks run for real.
        mod = ModelOnDisk(root)
        with pytest.raises(NotAMatchError):
            Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": root.as_posix()})

    def test_factory_resolves_sdnq_qwen3_folder_deterministically(self, tmp_path: Path):
        """With the plain config rejecting SDNQ, the factory resolves the folder unambiguously to
        the SDNQ config (previously a non-deterministic tie between the two Qwen3 configs)."""
        root = _make_sdnq_qwen3_encoder_folder(tmp_path / "sdnq-qwen3")
        result = ModelConfigFactory.from_model_on_disk(root, allow_unknown=True)
        assert isinstance(result.config, Qwen3Encoder_SDNQ_Folder_Config)
        assert not any(isinstance(m, Qwen3Encoder_Qwen3Encoder_Config) for m in result.all_matches)
