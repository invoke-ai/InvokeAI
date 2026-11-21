import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import gguf
import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config, Main_GGUF_QwenImage_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import QwenImageVariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.state_dicts import qwen_image_int8_convrot_keys
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors


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


def _identify(path: Path):
    return ModelConfigFactory.from_model_on_disk(ModelOnDisk(path, "random"), {}, allow_unknown=True).config


def _tiny_int8_checkpoint(
    tmp_path: Path, marker: dict | None, header: dict[str, dict] | None = None, name: str = "qwen_image_int8"
) -> Path:
    """The three keys that identify a Qwen Image transformer, and one int8 layer with the given per-tensor marker."""
    state_dict = {
        "img_in.weight": torch.zeros(8, 4),
        "txt_in.weight": torch.zeros(8, 4),
        "txt_norm.weight": torch.zeros(4),
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(4, 64, dtype=torch.int8),
        "transformer_blocks.0.attn.to_q.weight_scale": torch.ones(4, 1),
    }
    if marker is not None:
        raw = json.dumps(marker).encode("utf-8")
        state_dict["transformer_blocks.0.attn.to_q.comfy_quant"] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    metadata = {"_quantization_metadata": json.dumps({"layers": header})} if header else None
    path = tmp_path / f"{name}.safetensors"
    save_file({k: v.clone() for k, v in state_dict.items()}, path, metadata=metadata)
    return path


def test_the_released_edit_2511_int8_convrot_layout_installs_as_the_edit_variant(tmp_path: Path) -> None:
    """Its markers are read from the file; the `__index_timestep_zero__` key makes it the Edit variant."""
    path = write_header_only_safetensors(
        tmp_path / "qwen_image_2511_int8_convrot.safetensors",
        qwen_image_int8_convrot_keys.state_dict_keys,
        qwen_image_int8_convrot_keys.markers,
    )

    config = _identify(path)

    assert isinstance(config, Main_Checkpoint_QwenImage_Config)
    assert config.variant is QwenImageVariantType.Edit


@pytest.mark.parametrize(
    ("marker", "header"),
    [
        pytest.param({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}, None, id="rotated"),
        pytest.param({"format": "int8_tensorwise"}, None, id="unrotated"),
        pytest.param(
            None,
            {"transformer_blocks.0.attn.to_q": {"format": "int8_tensorwise", "convrot": False}},
            id="header_only",
        ),
    ],
)
def test_the_int8_marker_shapes_the_loader_decodes_install(tmp_path: Path, marker, header) -> None:
    config = _identify(_tiny_int8_checkpoint(tmp_path, marker, header))

    assert isinstance(config, Main_Checkpoint_QwenImage_Config)
    assert config.variant is QwenImageVariantType.Generate


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        pytest.param(None, "carry no readable marker", id="unmarked"),
        pytest.param({"convrot": True}, "carry no readable marker", id="convrot_without_format"),
        pytest.param({"format": "int4_tensorwise", "convrot": True}, "marked 'int4_tensorwise'", id="int4"),
        pytest.param({"format": "asym_w4a8_int8"}, "marked 'asym_w4a8_int8'", id="other_scheme"),
    ],
)
def test_an_int8_weight_the_loader_cannot_decode_is_refused_at_install(tmp_path: Path, marker, message) -> None:
    path = _tiny_int8_checkpoint(tmp_path, marker)

    with pytest.raises(InvalidMatchError, match=message):
        Main_Checkpoint_QwenImage_Config.from_model_on_disk(ModelOnDisk(path, "random"), {})


def test_a_header_entry_that_does_not_say_whether_the_weight_was_rotated_is_refused_at_install(tmp_path: Path) -> None:
    path = _tiny_int8_checkpoint(tmp_path, None, {"transformer_blocks.0.attn.to_q": {"format": "int8_tensorwise"}})

    with pytest.raises(InvalidMatchError, match="does not say whether the weight was rotated"):
        Main_Checkpoint_QwenImage_Config.from_model_on_disk(ModelOnDisk(path, "random"), {})
