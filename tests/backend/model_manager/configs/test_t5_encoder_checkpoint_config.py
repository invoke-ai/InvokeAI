"""Identifying ComfyUI's single-file T5-XXL encoders, through the whole config factory.

Each case is a captured header written back without its data (`write_header_only_safetensors`):
identification reads nothing else, so every config class sees the real key names, shapes and dtypes and
the test proves no other config claims the file -- and that the T5 config claims nothing it must not.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import TypeAdapter
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    raise_if_quantized_beyond_fp8,
)
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.flux import T5EncoderSingleFileLoader
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType
from tests.backend.model_manager.load.state_dicts import (
    t5xxl_fp8_e4m3fn_comfy_keys,
    t5xxl_fp8_e4m3fn_scaled_comfy_keys,
    t5xxl_fp16_comfy_keys,
    umt5_xxl_fp8_scaled_comfy_keys,
)
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors
from tests.fixtures.quantized_payloads import comfy_quant_marker, mxfp8_marker

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/t5xxl",
    "file_size": 1000,
    "name": "t5xxl",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

Keys = dict[str, tuple[list[int], str]]


def _identify(tmp_path: Path, keys: Keys, name: str = "t5xxl.safetensors") -> Any:
    path = write_header_only_safetensors(tmp_path / name, keys)
    return ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=False)


@pytest.mark.parametrize(
    "keys",
    [
        t5xxl_fp16_comfy_keys.state_dict_keys,
        t5xxl_fp8_e4m3fn_comfy_keys.state_dict_keys,
        t5xxl_fp8_e4m3fn_scaled_comfy_keys.state_dict_keys,
    ],
    ids=["fp16", "fp8_e4m3fn", "fp8_e4m3fn_scaled"],
)
def test_each_comfy_t5xxl_build_is_a_single_file_t5_encoder(tmp_path: Path, keys: Keys) -> None:
    result = _identify(tmp_path, keys)

    assert isinstance(result.config, T5Encoder_Checkpoint_Config)
    assert (result.config.type, result.config.format, result.config.base) == (
        ModelType.T5Encoder,
        ModelFormat.Checkpoint,
        BaseModelType.Any,
    )
    assert result.match_count == 1


def test_a_stored_record_reads_back_and_resolves_to_the_single_file_loader(tmp_path: Path) -> None:
    stored = _identify(tmp_path, _fp16()).config.model_dump(mode="json")

    restored = TypeAdapter(AnyModelConfig).validate_python(stored)
    loader, _, _ = ModelLoaderRegistry.get_implementation(restored, SubModelType.TextEncoder2)

    assert isinstance(restored, T5Encoder_Checkpoint_Config)
    assert loader is T5EncoderSingleFileLoader


def test_the_first_shard_of_a_sharded_export_is_not_a_t5_encoder(tmp_path: Path) -> None:
    # It carries the embedding and block 0 at full width; only the last shard ends in the final norm.
    keys = _fp16()
    del keys["encoder.final_layer_norm.weight"]

    result = _identify(tmp_path, keys)

    assert result.config is None
    assert not result.invalid_matches


def test_wans_umt5_is_not_taken_for_t5(tmp_path: Path) -> None:
    result = _identify(tmp_path, umt5_xxl_fp8_scaled_comfy_keys.state_dict_keys, "umt5_xxl.safetensors")

    assert result.config is None
    assert not result.invalid_matches


def test_an_encoder_decoder_checkpoint_is_not_an_encoder(tmp_path: Path) -> None:
    keys = {**t5xxl_fp16_comfy_keys.state_dict_keys, "decoder.final_layer_norm.weight": ([4096], "F16")}

    assert _identify(tmp_path, keys).config is None


def test_a_t5_narrower_than_xxl_is_refused_with_the_width_flux_needs(tmp_path: Path) -> None:
    # T5 v1.1 XL: the same names at half the width. FLUX.1 and SD 3 cannot condition on it.
    keys = {key: ([2048 if d == 4096 else d for d in shape], dtype) for key, (shape, dtype) in _fp16().items()}

    result = _identify(tmp_path, keys)

    assert result.config is None
    assert [str(e) for e in result.invalid_matches] == [
        "this T5 encoder has width 2048 and vocabulary 32128, but FLUX.1 and SD 3 need T5 v1.1 XXL "
        "(width 4096, vocabulary 32128)."
    ]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(
            lambda keys: {
                k: (shape, "I8" if k.endswith("SelfAttention.q.weight") else dtype)
                for k, (shape, dtype) in keys.items()
            },
            "integer-quantized weight(s)",
            id="int8",
        ),
        pytest.param(
            lambda keys: {**keys, "encoder.block.0.layer.0.SelfAttention.q.weight_scale_2": ([], "F32")},
            "nvfp4-quantized",
            id="nvfp4",
        ),
    ],
)
def test_builds_quantized_past_fp8_are_refused_at_install(tmp_path: Path, change, reason: str) -> None:
    result = _identify(tmp_path, change(_fp16()))

    assert result.config is None
    assert len(result.invalid_matches) == 1
    assert reason in str(result.invalid_matches[0])


@pytest.mark.parametrize(
    ("marker", "refused"),
    [({"format": "float8_e4m3fn"}, False), (mxfp8_marker(), True)],
    ids=["fp8_marker", "mxfp8_marker"],
)
def test_a_format_declared_in_the_header_is_refused_unless_it_is_fp8(
    tmp_path: Path, marker: dict[str, Any], refused: bool
) -> None:
    # A real file: the per-layer `.comfy_quant` markers are read from tensor bytes, not from the header.
    path = tmp_path / "t5xxl.safetensors"
    save_file(
        {
            "encoder.block.0.layer.0.SelfAttention.q.weight": torch.zeros(2, 2),
            "encoder.block.0.layer.0.SelfAttention.q.comfy_quant": comfy_quant_marker(marker),
        },
        str(path),
    )
    mod = ModelOnDisk(path)

    def check() -> None:
        raise_if_quantized_beyond_fp8(mod, mod.load_state_dict(), "T5 encoder", "Use the fp16 build.")

    if refused:
        with pytest.raises(InvalidMatchError, match=r"Unsupported quantization format\(s\) \['mxfp8'\]"):
            check()
    else:
        check()


def _fp16() -> Keys:
    return dict(t5xxl_fp16_comfy_keys.state_dict_keys)
