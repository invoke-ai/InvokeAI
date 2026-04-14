from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_GGUF_QwenImage_Config
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def _build_ggml_tensor() -> GGMLTensor:
    return GGMLTensor(
        data=torch.zeros((1,), dtype=torch.uint8),
        ggml_quantization_type=gguf.GGMLQuantizationType.Q4_0,
        tensor_shape=torch.Size([1, 1]),
        compute_dtype=torch.float32,
    )


@pytest.mark.parametrize("is_edit_model", [True, False])
def test_qwen_gguf_config_sets_a_variant_for_imported_models(is_edit_model: bool) -> None:
    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / ("qwen-image-edit.gguf" if is_edit_model else "qwen-image.gguf")
        model_name = "Qwen Image Edit GGUF" if is_edit_model else "Qwen Image GGUF"
        model_path.touch()

        mod = MagicMock()
        mod.path = model_path
        mod.load_state_dict.return_value = {
            "txt_in.weight": _build_ggml_tensor(),
            "txt_norm.weight": _build_ggml_tensor(),
            "img_in.weight": _build_ggml_tensor(),
        }

        config = Main_GGUF_QwenImage_Config.from_model_on_disk(
            mod,
            {
                "hash": "test-hash",
                "path": str(model_path),
                "file_size": model_path.stat().st_size,
                "name": model_name,
                "source": str(model_path),
                "source_type": "path",
            },
        )

    if is_edit_model:
        assert config.variant == "edit"
    else:
        assert config.variant == "generate"
