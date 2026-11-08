"""Identification of alibaba-pai's FLUX.2 [dev] Fun-ControlNet-Union, through the config factory.

The adapter is a VACE-style control branch: `control_img_in` projects 260 packed channels into FLUX.2 [dev]'s
6144-wide stream, and four `control_transformer_blocks` copy the double-stream `Flux2TransformerBlock` with
`before_proj`/`after_proj` taps. Its Qwen-Image siblings use the same recipe under `control_blocks`. Every case is a
header written back without its data: identification reads nothing else, and each config sees the real key names,
shapes and dtypes.
"""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from tests.backend.model_manager.load.state_dicts import flux2_fun_controlnet_keys, qwen_image_fun_controlnet_keys
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

Keys = dict[str, tuple[list[int], str]]

FILE_NAME = "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/controlnet.safetensors",
    "file_size": 1000,
    "name": "controlnet",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _keys() -> Keys:
    return dict(flux2_fun_controlnet_keys.state_dict_keys)


def _write(tmp_path: Path, name: str, keys: Keys) -> Path:
    return write_header_only_safetensors(tmp_path / name, keys)


def test_the_released_file_is_a_flux2_controlnet_and_nothing_else(tmp_path: Path) -> None:
    result = ModelConfigFactory.from_model_on_disk(_write(tmp_path, FILE_NAME, _keys()), allow_unknown=False)

    assert type(result.config) is ControlNet_Checkpoint_Flux2_Config, result.details
    assert result.match_count == 1, result.details
    assert result.config.base is BaseModelType.Flux2
    assert result.config.type is ModelType.ControlNet
    assert result.config.format is ModelFormat.Checkpoint


def test_a_stored_record_reads_back_as_a_flux2_controlnet(tmp_path: Path) -> None:
    """The record crosses the database and the API as a plain dict, so the config has to be a member of
    `AnyModelConfig` under a discriminator of its own."""
    stored = ModelConfigFactory.from_model_on_disk(
        _write(tmp_path, FILE_NAME, _keys()), allow_unknown=False
    ).config.model_dump(mode="json")

    restored = TypeAdapter(AnyModelConfig).validate_python(stored)

    assert type(restored) is ControlNet_Checkpoint_Flux2_Config
    assert restored.base is BaseModelType.Flux2


def test_the_qwen_image_sibling_is_refused(tmp_path: Path) -> None:
    mod = ModelOnDisk(
        _write(
            tmp_path,
            "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors",
            qwen_image_fun_controlnet_keys.state_dict_keys,
        )
    )

    with pytest.raises(NotAMatchError):
        ControlNet_Checkpoint_Flux2_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def _under_qwen_names(keys: Keys) -> Keys:
    return {key.replace("control_transformer_blocks.", "control_blocks.", 1): value for key, value in keys.items()}


@pytest.mark.parametrize(
    "keys",
    [_under_qwen_names(_keys()), {**_keys(), "control_img_in.weight": ([4096, 260], "BF16")}],
    ids=["control_blocks_names", "klein_9b_width"],
)
def test_a_branch_with_one_half_of_the_fingerprint_is_refused(tmp_path: Path, keys: Keys) -> None:
    """Both halves carry weight: the projection shape alone does not name the recipe, which the FLUX.2 block naming
    does, and the FLUX.2 names behind a 4096-wide projection would be a branch for Klein 9B, which nobody publishes
    and the denoise could not run."""
    mod = ModelOnDisk(_write(tmp_path, FILE_NAME, keys))

    with pytest.raises(NotAMatchError):
        ControlNet_Checkpoint_Flux2_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
