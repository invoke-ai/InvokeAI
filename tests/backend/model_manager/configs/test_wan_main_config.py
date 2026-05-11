"""Tests for Wan 2.2 model identification (Main_Diffusers_Wan_Config)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.main import Main_Diffusers_Wan_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, WanVariantType


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def _build_a14b_layout(root: Path) -> None:
    """Synthetic on-disk layout for Wan-AI/Wan2.2-T2V-A14B: dual transformers, z_dim=16."""
    _write_json(root / "model_index.json", {"_class_name": "WanPipeline"})
    _write_json(root / "transformer" / "config.json", {"_class_name": "WanTransformer3DModel", "in_channels": 16})
    _write_json(root / "transformer_2" / "config.json", {"_class_name": "WanTransformer3DModel", "in_channels": 16})
    _write_json(root / "vae" / "config.json", {"_class_name": "AutoencoderKLWan", "z_dim": 16})


def _build_ti2v_5b_layout(root: Path) -> None:
    """Synthetic on-disk layout for Wan-AI/Wan2.2-TI2V-5B: single transformer, z_dim=48."""
    _write_json(root / "model_index.json", {"_class_name": "WanImageToVideoPipeline"})
    _write_json(root / "transformer" / "config.json", {"_class_name": "WanTransformer3DModel", "in_channels": 48})
    _write_json(root / "vae" / "config.json", {"_class_name": "AutoencoderKLWan", "z_dim": 48})


def _build_i2v_a14b_layout(root: Path) -> None:
    """Wan-AI/Wan2.2-I2V-A14B: dual transformers, z_dim=16, transformer in_channels=36.

    The Wan 2.2 I2V transformer concatenates noise latents (16) + ref-image
    latents (16) + first-frame mask (4) along the channel dim, so its
    ``in_channels`` is 36 vs 16 for T2V.
    """
    _write_json(root / "model_index.json", {"_class_name": "WanImageToVideoPipeline"})
    _write_json(
        root / "transformer" / "config.json",
        {"_class_name": "WanTransformer3DModel", "in_channels": 36, "image_dim": None},
    )
    _write_json(
        root / "transformer_2" / "config.json",
        {"_class_name": "WanTransformer3DModel", "in_channels": 36, "image_dim": None},
    )
    _write_json(root / "vae" / "config.json", {"_class_name": "AutoencoderKLWan", "z_dim": 16})


def _build_overrides(model_path: Path, name: str) -> dict:
    return {
        "hash": "test-hash",
        "path": str(model_path),
        "file_size": 0,
        "name": name,
        "source": str(model_path),
        "source_type": "path",
    }


def _make_mod(model_path: Path) -> MagicMock:
    mod = MagicMock()
    mod.path = model_path
    return mod


class TestWanDiffusersIdentification:
    """Wan diffusers probe: variant detection from transformer / VAE / dir layout."""

    def test_a14b_detected_from_dual_transformer(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Wan2.2-T2V-A14B"
            _build_a14b_layout(root)

            cfg = Main_Diffusers_Wan_Config.from_model_on_disk(_make_mod(root), _build_overrides(root, "A14B"))

            assert cfg.base == BaseModelType.Wan
            assert cfg.format == ModelFormat.Diffusers
            assert cfg.variant == WanVariantType.T2V_A14B
            assert cfg.has_dual_expert is True

    def test_i2v_a14b_detected_from_in_channels_36(self) -> None:
        """I2V-A14B has the same dual-expert + z_dim=16 layout as T2V, but its
        transformer's ``in_channels`` is 36 (16 noise + 16 ref-image latents +
        4 first-frame mask). That's the canonical Wan 2.2 differentiator."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Wan2.2-I2V-A14B"
            _build_i2v_a14b_layout(root)

            cfg = Main_Diffusers_Wan_Config.from_model_on_disk(_make_mod(root), _build_overrides(root, "I2V"))

            assert cfg.variant == WanVariantType.I2V_A14B
            assert cfg.has_dual_expert is True

    def test_t2v_a14b_kept_when_in_channels_is_16(self) -> None:
        """A14B layout with ``in_channels=16`` resolves to T2V (not I2V)."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Wan2.2-T2V-A14B"
            _build_a14b_layout(root)

            cfg = Main_Diffusers_Wan_Config.from_model_on_disk(_make_mod(root), _build_overrides(root, "T2V"))

            assert cfg.variant == WanVariantType.T2V_A14B

    def test_ti2v_5b_detected_from_z_dim(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Wan2.2-TI2V-5B"
            _build_ti2v_5b_layout(root)

            cfg = Main_Diffusers_Wan_Config.from_model_on_disk(_make_mod(root), _build_overrides(root, "TI2V-5B"))

            assert cfg.variant == WanVariantType.TI2V_5B
            assert cfg.has_dual_expert is False

    def test_filename_heuristic_when_vae_config_missing(self) -> None:
        """When ``vae/config.json`` is missing, fall back to the directory name."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Wan2.2-TI2V-5B"
            _write_json(root / "model_index.json", {"_class_name": "WanPipeline"})
            _write_json(root / "transformer" / "config.json", {"_class_name": "WanTransformer3DModel"})
            # No vae/config.json — single-transformer + dirname containing "5b" → TI2V-5B.

            cfg = Main_Diffusers_Wan_Config.from_model_on_disk(_make_mod(root), _build_overrides(root, "TI2V-5B"))

            assert cfg.variant == WanVariantType.TI2V_5B

    def test_explicit_variant_override_takes_precedence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "wan-something"
            _build_a14b_layout(root)
            overrides = _build_overrides(root, "Custom A14B")
            overrides["variant"] = WanVariantType.TI2V_5B  # Explicit override.

            cfg = Main_Diffusers_Wan_Config.from_model_on_disk(_make_mod(root), overrides)
            assert cfg.variant == WanVariantType.TI2V_5B
            # has_dual_expert is still detected from disk; the override only forces variant.
            assert cfg.has_dual_expert is True

    def test_rejects_non_wan_pipeline(self) -> None:
        """A model_index.json that isn't a Wan class name must not match."""
        from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "not-wan"
            _write_json(root / "model_index.json", {"_class_name": "FluxPipeline"})

            with pytest.raises(NotAMatchError):
                Main_Diffusers_Wan_Config.from_model_on_disk(_make_mod(root), _build_overrides(root, "fake"))
