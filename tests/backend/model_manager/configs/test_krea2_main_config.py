"""Tests for Krea-2 main-model identification and variant detection.

Krea-2-Turbo (distilled) and Krea-2-Raw (Base, undistilled) share the IDENTICAL transformer
architecture, so a single-file/GGUF checkpoint cannot be told apart from its weights. Detection:

1. Explicit ``variant`` in override_fields always wins.
2. Diffusers pipelines read the pipeline-level ``is_distilled`` flag from model_index.json
   (``false`` → Base, ``true``/absent → Turbo).
3. Single-file / GGUF fall back to a filename heuristic ("raw"/"base" → Base, else Turbo).
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.main import (
    MainModelDefaultSettings,
    _get_krea2_variant_from_name,
    _has_krea2_keys,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, Krea2VariantType

# Required fields for the Pydantic config models (mirrors the other config-probe tests).
_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


class TestKrea2VariantFromName:
    """The filename heuristic used for single-file / GGUF Krea-2 checkpoints."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Krea-2-Raw", Krea2VariantType.Base),
            ("krea2_raw_q4.gguf", Krea2VariantType.Base),
            ("Krea-2-RAW-Q8_0.gguf", Krea2VariantType.Base),  # case-insensitive
            ("krea2_base_fp8_scaled.safetensors", Krea2VariantType.Base),
            ("Krea-2-Turbo", Krea2VariantType.Turbo),
            ("krea2_turbo-Q3_K_M.gguf", Krea2VariantType.Turbo),
            ("Krea-2-Turbo-Q4_K_M.gguf", Krea2VariantType.Turbo),
            ("some-random-name.safetensors", Krea2VariantType.Turbo),  # default
        ],
    )
    def test_variant_from_name(self, name: str, expected: Krea2VariantType) -> None:
        assert _get_krea2_variant_from_name(name) == expected


class TestHasKrea2Keys:
    """The Krea-2 transformer state-dict signature (diffusers + native/GGUF naming)."""

    def test_diffusers_naming_matches(self) -> None:
        sd = {
            "text_fusion.layerwise_blocks.0.attn.to_q.weight": object(),
            "img_in.weight": object(),
            "transformer_blocks.0.attn.to_q.weight": object(),
        }
        assert _has_krea2_keys(sd) is True

    def test_native_gguf_naming_matches(self) -> None:
        # Compact ComfyUI/GGUF naming: txtfusion + first / tproj.
        sd = {
            "txtfusion.0.attn.wq.weight": object(),
            "first.weight": object(),
            "tproj.1.weight": object(),
        }
        assert _has_krea2_keys(sd) is True

    def test_comfyui_prefixed_keys_match(self) -> None:
        sd = {
            "model.diffusion_model.text_fusion.projector.weight": object(),
            "model.diffusion_model.img_in.weight": object(),
        }
        assert _has_krea2_keys(sd) is True

    def test_text_fusion_without_corroborator_does_not_match(self) -> None:
        # text-fusion alone is not enough — an image-input corroborator is required.
        sd = {"text_fusion.projector.weight": object()}
        assert _has_krea2_keys(sd) is False

    def test_non_krea2_state_dict_does_not_match(self) -> None:
        sd = {"double_blocks.0.img_attn.qkv.weight": object(), "img_in.weight": object()}
        assert _has_krea2_keys(sd) is False

    def test_lora_is_rejected(self) -> None:
        # LoRA suffixes must never be classified as a Krea-2 main model.
        sd = {
            "text_fusion.layerwise_blocks.0.attn.to_q.lora_down.weight": object(),
            "img_in.lora_up.weight": object(),
        }
        assert _has_krea2_keys(sd) is False


class TestKrea2GGUFVariantDetection:
    """Main_GGUF_Krea2_Config: variant from filename, explicit override wins."""

    def _make_mock_mod(self, filename: str) -> MagicMock:
        mod = MagicMock()
        mod.path = Path(f"/fake/models/{filename}")
        mod.load_state_dict.return_value = {}
        return mod

    @patch("invokeai.backend.model_manager.configs.main._has_krea2_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_raw_filename_sets_base_variant(self, _rfo, _rif, _hgt, _hkk) -> None:
        from invokeai.backend.model_manager.configs.main import Main_GGUF_Krea2_Config

        mod = self._make_mock_mod("Krea-2-Raw-Q4_K_M.gguf")
        config = Main_GGUF_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == Krea2VariantType.Base

    @patch("invokeai.backend.model_manager.configs.main._has_krea2_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_turbo_filename_defaults_to_turbo(self, _rfo, _rif, _hgt, _hkk) -> None:
        from invokeai.backend.model_manager.configs.main import Main_GGUF_Krea2_Config

        mod = self._make_mock_mod("krea2_turbo-Q3_K_M.gguf")
        config = Main_GGUF_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == Krea2VariantType.Turbo

    @patch("invokeai.backend.model_manager.configs.main._has_krea2_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_explicit_variant_override_wins(self, _rfo, _rif, _hgt, _hkk) -> None:
        from invokeai.backend.model_manager.configs.main import Main_GGUF_Krea2_Config

        # Filename says Raw, but an explicit Turbo override must not be overwritten.
        mod = self._make_mock_mod("Krea-2-Raw-Q4_K_M.gguf")
        config = Main_GGUF_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "variant": Krea2VariantType.Turbo})
        assert config.variant == Krea2VariantType.Turbo


class TestKrea2CheckpointVariantDetection:
    """Main_Checkpoint_Krea2_Config: variant from filename (non-GGUF single file)."""

    def _make_mock_mod(self, filename: str) -> MagicMock:
        mod = MagicMock()
        mod.path = Path(f"/fake/models/{filename}")
        mod.load_state_dict.return_value = {}
        return mod

    @patch("invokeai.backend.model_manager.configs.main._has_krea2_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_raw_filename_sets_base_variant(self, _rfo, _rif, _hgt, _hkk) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config

        mod = self._make_mock_mod("krea2_raw_fp8_scaled.safetensors")
        config = Main_Checkpoint_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == Krea2VariantType.Base

    @patch("invokeai.backend.model_manager.configs.main._has_krea2_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_turbo_filename_defaults_to_turbo(self, _rfo, _rif, _hgt, _hkk) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config

        mod = self._make_mock_mod("krea2_turbo_fp8_scaled.safetensors")
        config = Main_Checkpoint_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == Krea2VariantType.Turbo


class TestKrea2DiffusersVariantDetection:
    """Main_Diffusers_Krea2_Config._get_variant reads is_distilled from model_index.json."""

    def _make_mock_mod_with_model_index(self, tmpdir: str, model_index: dict) -> MagicMock:
        path = Path(tmpdir)
        (path / "model_index.json").write_text(json.dumps(model_index))
        mod = MagicMock()
        mod.path = path
        return mod

    def test_is_distilled_false_is_base(self) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Diffusers_Krea2_Config

        with TemporaryDirectory() as tmpdir:
            mod = self._make_mock_mod_with_model_index(tmpdir, {"_class_name": "Krea2Pipeline", "is_distilled": False})
            assert Main_Diffusers_Krea2_Config._get_variant(mod) == Krea2VariantType.Base

    def test_is_distilled_true_is_turbo(self) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Diffusers_Krea2_Config

        with TemporaryDirectory() as tmpdir:
            mod = self._make_mock_mod_with_model_index(tmpdir, {"_class_name": "Krea2Pipeline", "is_distilled": True})
            assert Main_Diffusers_Krea2_Config._get_variant(mod) == Krea2VariantType.Turbo

    def test_is_distilled_absent_defaults_to_turbo(self) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Diffusers_Krea2_Config

        with TemporaryDirectory() as tmpdir:
            mod = self._make_mock_mod_with_model_index(tmpdir, {"_class_name": "Krea2Pipeline"})
            assert Main_Diffusers_Krea2_Config._get_variant(mod) == Krea2VariantType.Turbo


class TestKrea2DefaultSettings:
    """Per-variant default generation settings."""

    def test_turbo_defaults(self) -> None:
        ds = MainModelDefaultSettings.from_base(BaseModelType.Krea2, Krea2VariantType.Turbo)
        assert ds is not None
        assert ds.steps == 8
        assert ds.cfg_scale == 1.0
        assert ds.width == 1024
        assert ds.height == 1024

    def test_base_defaults(self) -> None:
        ds = MainModelDefaultSettings.from_base(BaseModelType.Krea2, Krea2VariantType.Base)
        assert ds is not None
        assert ds.steps == 28
        assert ds.cfg_scale == 4.5
        assert ds.width == 1024
        assert ds.height == 1024
