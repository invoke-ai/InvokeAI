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


class TestHasQwenImageKeys:
    """Detection must agree with the loader, which strips ComfyUI prefixes before loading."""

    def test_bare_keys_detected(self):
        from invokeai.backend.model_manager.configs.main import _has_qwen_image_keys

        sd = {"txt_in.weight": 1, "txt_norm.weight": 1, "img_in.weight": 1}
        assert _has_qwen_image_keys(sd)

    @pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
    def test_comfyui_prefixed_keys_detected(self, prefix: str):
        """A ComfyUI checkpoint with prefixed keys must still be identified so it reaches the loader."""
        from invokeai.backend.model_manager.configs.main import _has_qwen_image_keys

        sd = {f"{prefix}txt_in.weight": 1, f"{prefix}txt_norm.weight": 1, f"{prefix}img_in.weight": 1}
        assert _has_qwen_image_keys(sd)

    def test_flux_rejected(self):
        from invokeai.backend.model_manager.configs.main import _has_qwen_image_keys

        sd = {"txt_in.weight": 1, "txt_norm.weight": 1, "img_in.weight": 1, "context_embedder.weight": 1}
        assert not _has_qwen_image_keys(sd)

    def test_prefixed_marker_sets_edit_variant(self):
        """The Edit marker tensor may also carry a ComfyUI prefix."""
        from invokeai.backend.model_manager.configs.main import _infer_qwen_image_variant
        from invokeai.backend.model_manager.taxonomy import QwenImageVariantType

        sd = {"model.diffusion_model.__index_timestep_zero__": object()}
        assert _infer_qwen_image_variant(sd, Path("/fake/plain-name.safetensors")) == QwenImageVariantType.Edit


class TestEditTokenHeuristic:
    """The filename "edit" heuristic must match the token, not any substring."""

    @pytest.mark.parametrize(
        "stem",
        ["qwen-image-edit-2511", "qwen_image_edit_2509", "Qwen-Image-EDIT", "model.edit"],
    )
    def test_edit_token_matches(self, stem: str):
        from invokeai.backend.model_manager.configs.main import _infer_qwen_image_variant
        from invokeai.backend.model_manager.taxonomy import QwenImageVariantType

        assert _infer_qwen_image_variant({}, Path(f"/fake/{stem}.safetensors")) == QwenImageVariantType.Edit

    @pytest.mark.parametrize("stem", ["credited-model", "edited-final", "unedited", "qwen-image"])
    def test_edit_substring_does_not_false_positive(self, stem: str):
        from invokeai.backend.model_manager.configs.main import _infer_qwen_image_variant
        from invokeai.backend.model_manager.taxonomy import QwenImageVariantType

        assert _infer_qwen_image_variant({}, Path(f"/fake/{stem}.safetensors")) == QwenImageVariantType.Generate
