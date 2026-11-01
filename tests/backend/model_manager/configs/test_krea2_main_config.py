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
import torch
from safetensors.torch import save_file

from invokeai.backend.architectures import resolve_default_settings
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    _get_krea2_variant_from_name,
    _has_krea2_keys,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
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
            # "turbo" wins outright, so a Turbo file whose name merely contains "base"/"raw" as a
            # substring is not misread as Base.
            ("krea2_turbo_baseline_q4.gguf", Krea2VariantType.Turbo),
            ("krea2-turbo-raw-export.safetensors", Krea2VariantType.Turbo),
            # "base"/"raw" only match as whole tokens, not arbitrary substrings.
            ("krea2_database_model.safetensors", Krea2VariantType.Turbo),
            ("krea2_baseline_q8.gguf", Krea2VariantType.Turbo),
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

    @patch("invokeai.backend.model_manager.configs.main._has_krea2_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_rejects_non_safetensors_checkpoint(self, _rfo, _rif, _hgt, _hkk) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config

        mod = self._make_mock_mod("krea2_turbo.bin")

        with pytest.raises(NotAMatchError, match="safetensors"):
            Main_Checkpoint_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})


class TestKrea2CheckpointQuantizationAtInstall:
    """What the loader would refuse is refused at install, with the format named: an int8 layer the file marks as
    something Krea-2 does not decode, or marks only ambiguously in its header. The int4 builds install. Driven
    through a real `ModelOnDisk` over a real file: identification reads the markers from the file, because its
    state dict is on the meta device."""

    @staticmethod
    def _mod(tmp_path: Path, layers: dict[str, dict | None], header: dict[str, dict] | None = None) -> ModelOnDisk:
        """One int8 layer per entry of `layers`, with that per-tensor marker (None: none), and `header` as the
        `_quantization_metadata` layers."""
        state_dict: dict[str, torch.Tensor] = {}
        for path, marker in layers.items():
            state_dict[f"{path}.weight"] = torch.zeros(4, 128, dtype=torch.int8)
            state_dict[f"{path}.weight_scale"] = torch.ones(4)
            if marker is not None:
                raw = json.dumps(marker).encode("utf-8")
                state_dict[f"{path}.comfy_quant"] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        metadata = {"_quantization_metadata": json.dumps({"layers": header})} if header else None
        path = tmp_path / "krea2_turbo_int4.safetensors"
        save_file(state_dict, path, metadata=metadata)
        return ModelOnDisk(path, "random")

    @staticmethod
    def _identify(mod: ModelOnDisk):
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config

        with (
            patch("invokeai.backend.model_manager.configs.main._has_krea2_keys", return_value=True),
            patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields"),
        ):
            return Main_Checkpoint_Krea2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

    @pytest.mark.parametrize(
        ("marker", "header"),
        [
            pytest.param(None, {"blocks.0.attn.wq": {"format": "convrot_w4a4"}}, id="header only, Starnodes"),
            pytest.param({"format": "int4_tensorwise", "convrot": True}, None, id="per-tensor, comfy-quants"),
        ],
    )
    def test_an_int4_build_installs(self, tmp_path: Path, marker: dict | None, header: dict | None) -> None:
        mod = self._mod(tmp_path, {"blocks.0.attn.wq": marker}, header)

        assert self._identify(mod).variant == Krea2VariantType.Turbo

    def test_a_header_naming_its_layers_without_the_checkpoint_prefix_still_names_them(self, tmp_path: Path) -> None:
        """The loader compares both after stripping the prefix, so identification has to as well."""
        mod = self._mod(
            tmp_path, {"model.diffusion_model.blocks.0.attn.wq": None}, {"blocks.0.attn.wq": {"format": "convrot_w4a4"}}
        )

        assert self._identify(mod).variant == Krea2VariantType.Turbo

    def test_a_format_krea2_does_not_decode_is_refused_by_name(self, tmp_path: Path) -> None:
        mod = self._mod(tmp_path, {"blocks.0.attn.wq": None}, {"blocks.0.attn.wq": {"format": "asym_w4a8_int8"}})

        with pytest.raises(InvalidMatchError, match="marked 'asym_w4a8_int8'"):
            self._identify(mod)

    def test_a_per_tensor_marker_wins_over_the_header_as_it_does_at_load(self, tmp_path: Path) -> None:
        mod = self._mod(
            tmp_path,
            {"blocks.0.attn.wq": {"format": "asym_w4a8_int8"}},
            {"blocks.0.attn.wq": {"format": "convrot_w4a4"}},
        )

        with pytest.raises(InvalidMatchError, match="marked 'asym_w4a8_int8'"):
            self._identify(mod)

    def test_a_header_entry_that_does_not_say_whether_the_layer_was_rotated_is_refused(self, tmp_path: Path) -> None:
        mod = self._mod(tmp_path, {"blocks.0.attn.wq": None}, {"blocks.0.attn.wq": {"format": "int8_tensorwise"}})

        with pytest.raises(InvalidMatchError, match="does not say whether the weight was rotated"):
            self._identify(mod)

    def test_the_final_block_projections_the_loader_drops_are_not_checked(self, tmp_path: Path) -> None:
        """One repack quantizes `last.up` in a scheme of its own; the loader discards that layer before it reads any
        marker, so it loads, and refusing it here would refuse a file that works."""
        mod = self._mod(tmp_path, {"blocks.0.attn.wq": {"format": "int4_tensorwise", "convrot": True}, "last.up": None})

        assert self._identify(mod).variant == Krea2VariantType.Turbo


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

    def test_is_distilled_absent_defaults_to_base(self) -> None:
        from invokeai.backend.model_manager.configs.main import Main_Diffusers_Krea2_Config

        with TemporaryDirectory() as tmpdir:
            mod = self._make_mock_mod_with_model_index(tmpdir, {"_class_name": "Krea2Pipeline"})
            assert Main_Diffusers_Krea2_Config._get_variant(mod) == Krea2VariantType.Base


class TestKrea2DefaultSettings:
    """Per-variant default generation settings."""

    def test_turbo_defaults(self) -> None:
        ds = resolve_default_settings(BaseModelType.Krea2, Krea2VariantType.Turbo)
        assert ds is not None
        assert ds.steps == 8
        assert ds.cfg_scale == 1.0
        assert ds.width == 1024
        assert ds.height == 1024

    def test_base_defaults(self) -> None:
        ds = resolve_default_settings(BaseModelType.Krea2, Krea2VariantType.Base)
        assert ds is not None
        assert ds.cfg_scale == 5.5
        assert ds.steps == 28
        assert ds.width == 1024
        assert ds.height == 1024
