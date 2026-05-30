"""Tests for Qwen Image single-file checkpoint variant detection.

Mirrors `test_qwen_image_gguf_variant_detection.py`. The Checkpoint and GGUF
configs share the same variant inference (`_infer_qwen_image_variant`):

1. Explicit `variant` in override_fields wins.
2. Presence of the `__index_timestep_zero__` tensor → Edit.
3. Filename heuristic: "edit" substring in the stem → Edit.
4. Otherwise default to Generate.

Also tests that the Checkpoint config rejects GGUF state dicts (and vice
versa), so the two configs don't both match the same file.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.taxonomy import QwenImageVariantType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test.safetensors",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


class TestCheckpointQwenImageVariantDetection:
    def _make_mock_mod(self, filename: str) -> MagicMock:
        mod = MagicMock()
        mod.path = Path(f"/fake/models/{filename}")
        return mod

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_edit_in_filename_sets_edit_variant(self, _rfo, _rif, _hgt, _hqk):
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config

        mod = self._make_mock_mod("qwen-image-edit-2511-fp8.safetensors")
        mod.load_state_dict.return_value = {}

        config = Main_Checkpoint_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == QwenImageVariantType.Edit

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_edit_case_insensitive(self, _rfo, _rif, _hgt, _hqk):
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config

        mod = self._make_mock_mod("Qwen-Image-EDIT-2511-fp8.safetensors")
        mod.load_state_dict.return_value = {}

        config = Main_Checkpoint_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == QwenImageVariantType.Edit

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_no_marker_no_edit_in_filename_defaults_to_generate(self, _rfo, _rif, _hgt, _hqk):
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config

        mod = self._make_mock_mod("qwen-image-2512-bf16.safetensors")
        mod.load_state_dict.return_value = {}

        config = Main_Checkpoint_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == QwenImageVariantType.Generate

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_marker_tensor_sets_edit_variant(self, _rfo, _rif, _hgt, _hqk):
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config

        mod = self._make_mock_mod("some-arbitrary-name.safetensors")
        mod.load_state_dict.return_value = {"__index_timestep_zero__": object()}

        config = Main_Checkpoint_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == QwenImageVariantType.Edit

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_explicit_variant_override_not_overwritten(self, _rfo, _rif, _hgt, _hqk):
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config

        mod = self._make_mock_mod("qwen-image-edit-2511-fp8.safetensors")
        mod.load_state_dict.return_value = {}

        config = Main_Checkpoint_QwenImage_Config.from_model_on_disk(
            mod, {**_REQUIRED_FIELDS, "variant": QwenImageVariantType.Generate}
        )
        assert config.variant == QwenImageVariantType.Generate

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_rejects_gguf_state_dict(self, _rfo, _rif, _hgt, _hqk):
        """Checkpoint config must NOT match files that look GGUF-quantized."""
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config

        mod = self._make_mock_mod("qwen-image-edit-2511-Q4_K_M.gguf")
        mod.load_state_dict.return_value = {}

        with pytest.raises(NotAMatchError):
            Main_Checkpoint_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=False)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_rejects_non_qwen_state_dict(self, _rfo, _rif, _hgt, _hqk):
        """Checkpoint config must NOT match files whose state dict isn't Qwen Image."""
        from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config

        mod = self._make_mock_mod("not-a-qwen-model.safetensors")
        mod.load_state_dict.return_value = {}

        with pytest.raises(NotAMatchError):
            Main_Checkpoint_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
