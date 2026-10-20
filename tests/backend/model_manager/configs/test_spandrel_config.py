"""Spandrel upscaler configs record the model's native scale from the model the probe already loads.

The fixtures are tiny real Compact (Real-ESRGAN SRVGG) models, so identification runs through the whole config factory
and Spandrel's own architecture detection, as an install or a re-identify does.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import TypeAdapter
from safetensors.torch import save_file
from spandrel.architectures.Compact import Compact

from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.spandrel import Spandrel_Checkpoint_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/upscaler.safetensors",
    "file_size": 1000,
    "name": "upscaler",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def write_compact_upscaler(path: Path, scale: int) -> Path:
    model = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=scale)
    save_file(model.state_dict(), path)
    return path


def _identify(path: Path, overrides: dict[str, Any] | None = None) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, {**_OVERRIDE_FIELDS, **(overrides or {})}, allow_unknown=False)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_identification_records_the_upscalers_native_scale(tmp_path: Path, scale: int) -> None:
    path = write_compact_upscaler(tmp_path / f"{scale}x.safetensors", scale)

    result = _identify(path)

    assert type(result.config) is Spandrel_Checkpoint_Config
    assert result.config.scale == scale


def test_a_record_stored_before_scale_was_recorded_reads_back_with_no_scale(tmp_path: Path) -> None:
    stored = _identify(write_compact_upscaler(tmp_path / "4x.safetensors", 4)).config.model_dump(mode="json")
    del stored["scale"]

    restored = TypeAdapter(AnyModelConfig).validate_python(stored)

    assert type(restored) is Spandrel_Checkpoint_Config
    assert restored.scale is None


def test_a_file_spandrel_cannot_load_is_not_an_upscaler(tmp_path: Path) -> None:
    path = tmp_path / "not-an-upscaler.safetensors"
    save_file({"some.weight": torch.zeros(8, 3, 3, 3)}, path)

    with pytest.raises(NotAMatchError):
        Spandrel_Checkpoint_Config.from_model_on_disk(ModelOnDisk(path), dict(_OVERRIDE_FIELDS))
