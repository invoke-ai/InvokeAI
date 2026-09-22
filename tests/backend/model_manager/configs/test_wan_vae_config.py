"""Tests for Wan 2.2 VAE config probes (checkpoint + diffusers)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_QwenImage_Config,
    VAE_Checkpoint_Wan_Config,
    VAE_Diffusers_Wan_Config,
    _wan_vae_z_dim,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat


def _build_overrides(model_path: Path, name: str) -> dict:
    return {
        "hash": "test-hash",
        "path": str(model_path),
        "file_size": 0,
        "name": name,
        "source": str(model_path),
        "source_type": "path",
    }


def _make_mod(model_path: Path, state_dict: dict | None = None) -> MagicMock:
    mod = MagicMock()
    mod.path = model_path
    if state_dict is not None:
        mod.load_state_dict.return_value = state_dict
    return mod


def _wan_vae_state_dict(z_dim: int) -> dict:
    """Synthetic 5D Wan-style VAE state dict."""
    return {
        "decoder.conv_in.weight": torch.zeros(96, z_dim, 1, 3, 3),
        "encoder.conv_in.weight": torch.zeros(z_dim, 3, 1, 3, 3),
    }


class TestZDimDetection:
    def test_detects_16_channel(self):
        assert _wan_vae_z_dim(_wan_vae_state_dict(16)) == 16

    def test_detects_48_channel(self):
        assert _wan_vae_z_dim(_wan_vae_state_dict(48)) == 48

    def test_rejects_unknown_z_dim(self):
        # Some other 5D conv weight (not Wan).
        sd = {"decoder.conv_in.weight": torch.zeros(96, 32, 1, 3, 3)}
        assert _wan_vae_z_dim(sd) is None

    def test_rejects_4d_conv(self):
        # Standard SD/SDXL 4D conv — not Wan.
        sd = {"decoder.conv_in.weight": torch.zeros(96, 16, 3, 3)}
        assert _wan_vae_z_dim(sd) is None


class TestVAECheckpointWanConfig:
    """Probe + filename-heuristic disambiguation from Qwen Image VAE."""

    def test_48_channel_unambiguous_wan(self):
        with TemporaryDirectory() as tmp:
            vae_path = Path(tmp) / "wan2.2-vae.safetensors"
            vae_path.touch()

            cfg = VAE_Checkpoint_Wan_Config.from_model_on_disk(
                _make_mod(vae_path, state_dict=_wan_vae_state_dict(48)),
                _build_overrides(vae_path, "Wan2.2-VAE"),
            )

            assert cfg.base == BaseModelType.Wan
            assert cfg.format == ModelFormat.Checkpoint
            assert cfg.latent_channels == 48

    def test_16_channel_with_wan_in_filename(self):
        with TemporaryDirectory() as tmp:
            vae_path = Path(tmp) / "wan-vae.safetensors"
            vae_path.touch()

            cfg = VAE_Checkpoint_Wan_Config.from_model_on_disk(
                _make_mod(vae_path, state_dict=_wan_vae_state_dict(16)),
                _build_overrides(vae_path, "Wan VAE"),
            )

            assert cfg.latent_channels == 16

    def test_16_channel_without_wan_in_filename_defers(self):
        """Filename without 'wan' should let Qwen Image VAE win."""
        with TemporaryDirectory() as tmp:
            vae_path = Path(tmp) / "qwen_vae.safetensors"
            vae_path.touch()

            with pytest.raises(NotAMatchError, match="deferring to Qwen Image"):
                VAE_Checkpoint_Wan_Config.from_model_on_disk(
                    _make_mod(vae_path, state_dict=_wan_vae_state_dict(16)),
                    _build_overrides(vae_path, "QwenImage VAE"),
                )

    def test_qwen_image_defers_when_filename_says_wan(self):
        """The mirror case — QwenImage config refuses files whose filenames suggest Wan."""
        with TemporaryDirectory() as tmp:
            vae_path = Path(tmp) / "wan-vae.safetensors"
            vae_path.touch()

            with pytest.raises(NotAMatchError, match="filename suggests a Wan"):
                VAE_Checkpoint_QwenImage_Config.from_model_on_disk(
                    _make_mod(vae_path, state_dict=_wan_vae_state_dict(16)),
                    _build_overrides(vae_path, "Wan VAE"),
                )

    def test_rejects_non_wan_state_dict(self):
        with TemporaryDirectory() as tmp:
            vae_path = Path(tmp) / "wan-junk.safetensors"
            vae_path.touch()
            sd = {"foo.bar": torch.zeros(1)}

            with pytest.raises(NotAMatchError):
                VAE_Checkpoint_Wan_Config.from_model_on_disk(
                    _make_mod(vae_path, state_dict=sd),
                    _build_overrides(vae_path, "junk"),
                )


class TestVAEDiffusersWanConfig:
    """Diffusers-folder probe; latent_channels read from vae/config.json."""

    def test_z_dim_from_config_json(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Wan2.2-VAE"
            root.mkdir()
            with (root / "config.json").open("w") as f:
                json.dump({"_class_name": "AutoencoderKLWan", "z_dim": 48}, f)

            cfg = VAE_Diffusers_Wan_Config.from_model_on_disk(
                _make_mod(root),
                _build_overrides(root, "Wan2.2-VAE"),
            )
            assert cfg.latent_channels == 48
            assert cfg.format == ModelFormat.Diffusers

    def test_default_to_16_when_z_dim_missing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Wan-VAE"
            root.mkdir()
            with (root / "config.json").open("w") as f:
                json.dump({"_class_name": "AutoencoderKLWan"}, f)  # No z_dim.

            cfg = VAE_Diffusers_Wan_Config.from_model_on_disk(
                _make_mod(root),
                _build_overrides(root, "Wan-VAE"),
            )
            assert cfg.latent_channels == 16

    def test_rejects_non_wan_class(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "FluxVAE"
            root.mkdir()
            with (root / "config.json").open("w") as f:
                json.dump({"_class_name": "AutoencoderKL"}, f)

            with pytest.raises(NotAMatchError):
                VAE_Diffusers_Wan_Config.from_model_on_disk(
                    _make_mod(root),
                    _build_overrides(root, "FluxVAE"),
                )
