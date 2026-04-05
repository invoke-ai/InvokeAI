"""Tests for GGUF Qwen Image variant detection from filename."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from invokeai.backend.model_manager.taxonomy import QwenImageVariantType

# Required fields for the Pydantic config model
_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test.gguf",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


class TestGGUFQwenImageVariantDetection:
    """Test that GGUF Qwen Image models infer the edit variant from filename."""

    def _make_mock_mod(self, filename: str) -> MagicMock:
        """Create a mock ModelOnDisk with the given filename."""
        mod = MagicMock()
        mod.path = Path(f"/fake/models/{filename}")
        return mod

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_edit_in_filename_sets_edit_variant(self, _rfo, _rif, _hgt, _hqk):
        """A GGUF file with 'edit' in the name should be tagged as edit variant."""
        from invokeai.backend.model_manager.configs.main import Main_GGUF_QwenImage_Config

        mod = self._make_mock_mod("qwen-image-edit-2511-Q4_K_M.gguf")
        mod.load_state_dict.return_value = {}

        config = Main_GGUF_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == QwenImageVariantType.Edit

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_edit_case_insensitive(self, _rfo, _rif, _hgt, _hqk):
        """The 'edit' check should be case-insensitive."""
        from invokeai.backend.model_manager.configs.main import Main_GGUF_QwenImage_Config

        mod = self._make_mock_mod("Qwen-Image-EDIT-2511-Q8_0.gguf")
        mod.load_state_dict.return_value = {}

        config = Main_GGUF_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant == QwenImageVariantType.Edit

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_no_edit_in_filename_leaves_variant_none(self, _rfo, _rif, _hgt, _hqk):
        """A GGUF file without 'edit' should NOT be tagged as edit variant."""
        from invokeai.backend.model_manager.configs.main import Main_GGUF_QwenImage_Config

        mod = self._make_mock_mod("qwen-image-2512-Q4_K_M.gguf")
        mod.load_state_dict.return_value = {}

        config = Main_GGUF_QwenImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.variant is None

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_explicit_variant_override_not_overwritten(self, _rfo, _rif, _hgt, _hqk):
        """An explicit variant in override_fields should not be overwritten by filename heuristic."""
        from invokeai.backend.model_manager.configs.main import Main_GGUF_QwenImage_Config

        mod = self._make_mock_mod("qwen-image-edit-2511-Q4_K_M.gguf")
        mod.load_state_dict.return_value = {}

        config = Main_GGUF_QwenImage_Config.from_model_on_disk(
            mod, {**_REQUIRED_FIELDS, "variant": QwenImageVariantType.Generate}
        )
        assert config.variant == QwenImageVariantType.Generate
