"""Regression tests for the double-variant kwarg bug.

When override_fields contains a field (variant, repo_variant, prediction_type, etc.)
that is also computed and passed as an explicit kwarg to cls(), using .get() instead
of .pop() causes TypeError("got multiple values for keyword argument ...").

These tests verify that .pop() is used consistently, so override values don't conflict
with explicitly computed values.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from invokeai.backend.model_manager.taxonomy import QwenImageVariantType

# Required fields for the Pydantic config model
_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test-model",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _make_mock_dir(dirname: str = "test-model") -> MagicMock:
    """Create a mock ModelOnDisk for a Diffusers directory."""
    mod = MagicMock()
    mod.path = Path(f"/fake/models/{dirname}")
    return mod


class TestDoubleVariantRegression:
    """Verify that override_fields with variant/repo_variant don't cause double-kwarg errors."""

    @patch("invokeai.backend.model_manager.configs.main.raise_for_class_name")
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_dir")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_qwen_image_diffusers_with_variant_in_overrides(self, _rfo, _rid, _rfc):
        """Installing a Qwen Image Edit Diffusers model with variant in override_fields should not crash."""
        from invokeai.backend.model_manager.configs.main import Main_Diffusers_QwenImage_Config

        mod = _make_mock_dir("Qwen-Image-Edit-2511")

        # Simulate what happens when a starter model provides variant
        overrides = {
            **_REQUIRED_FIELDS,
            "variant": QwenImageVariantType.Edit,
        }

        from invokeai.backend.model_manager.configs.base import ModelRepoVariant

        with patch.object(
            Main_Diffusers_QwenImage_Config, "_get_repo_variant_or_raise", return_value=ModelRepoVariant("")
        ):
            with patch.object(
                Main_Diffusers_QwenImage_Config,
                "_get_qwen_image_variant",
                return_value=QwenImageVariantType.Edit,
            ):
                # This would previously raise: TypeError("got multiple values for keyword argument 'variant'")
                config = Main_Diffusers_QwenImage_Config.from_model_on_disk(mod, overrides)

        assert config.variant == QwenImageVariantType.Edit

    @patch("invokeai.backend.model_manager.configs.main.raise_for_class_name")
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_dir")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_qwen_image_diffusers_override_variant_takes_precedence(self, _rfo, _rid, _rfc):
        """An explicit variant override should take precedence over auto-detection."""
        from invokeai.backend.model_manager.configs.base import ModelRepoVariant
        from invokeai.backend.model_manager.configs.main import Main_Diffusers_QwenImage_Config

        mod = _make_mock_dir("Qwen-Image-2512")

        overrides = {
            **_REQUIRED_FIELDS,
            "variant": QwenImageVariantType.Edit,  # explicitly override to Edit
        }

        with patch.object(
            Main_Diffusers_QwenImage_Config, "_get_repo_variant_or_raise", return_value=ModelRepoVariant("")
        ):
            with patch.object(
                Main_Diffusers_QwenImage_Config,
                "_get_qwen_image_variant",
                return_value=QwenImageVariantType.Generate,  # auto-detect says Generate
            ):
                config = Main_Diffusers_QwenImage_Config.from_model_on_disk(mod, overrides)

        # Override should win over auto-detection
        assert config.variant == QwenImageVariantType.Edit

    @patch("invokeai.backend.model_manager.configs.main._has_qwen_image_keys", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main._has_ggml_tensors", return_value=True)
    @patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
    def test_qwen_image_gguf_with_variant_in_overrides(self, _rfo, _rif, _hgt, _hqk):
        """Installing a Qwen Image Edit GGUF with variant in override_fields should not crash."""
        from invokeai.backend.model_manager.configs.main import Main_GGUF_QwenImage_Config

        mod = MagicMock()
        mod.path = Path("/fake/models/qwen-image-edit-2511-Q4_K_M.gguf")
        mod.load_state_dict.return_value = {}

        overrides = {
            **_REQUIRED_FIELDS,
            "variant": QwenImageVariantType.Edit,
        }

        config = Main_GGUF_QwenImage_Config.from_model_on_disk(mod, overrides)
        assert config.variant == QwenImageVariantType.Edit
