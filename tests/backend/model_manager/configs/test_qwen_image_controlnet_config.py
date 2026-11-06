"""Identification of the two Qwen-Image ControlNet designs, through the config factory.

alibaba-pai's Fun-ControlNet-Union is a VACE-style control branch: `control_img_in` projects 132 packed channels into
Qwen-Image's 3072-wide stream, and five `control_blocks` copy the dual-stream `QwenImageTransformerBlock` with
`before_proj`/`after_proj` taps. Its siblings reuse the naming -- the Qwen-Image-2.1 branch is single-stream at 4096
wide, FLUX.2's keeps `control_transformer_blocks` at 6144.

InstantX's ControlNet-Union is a five-block copy of the Qwen-Image transformer with `controlnet_x_embedder` and
`controlnet_blocks`: the key family of InstantX's FLUX ControlNets, and with `img_in`/`txt_in`/`txt_norm` the keys the
Qwen-Image main model probe reads. Both used to claim it, and the factory ranks main models first.

Both are one config, the design recorded as `control_architecture`. Every case is a header written back without its
data: identification reads nothing else, and each config sees the real key names, shapes and dtypes.
"""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from invokeai.backend.model_manager.configs.controlnet import (
    ControlNet_Checkpoint_FLUX_Config,
    ControlNet_Checkpoint_QwenImage_Config,
)
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from tests.backend.flux.controlnet.instantx_flux_controlnet_state_dict import instantx_sd_shapes
from tests.backend.model_manager.load.state_dicts import (
    flux2_fun_controlnet_keys,
    qwen_image_fun_controlnet_keys,
    qwen_image_instantx_controlnet_keys,
)
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

Keys = dict[str, tuple[list[int], str]]

FUN_FILE_NAME = "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors"
INSTANTX_FILE_NAME = "Qwen-Image-InstantX-ControlNet-Union.safetensors"

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


def _fun_keys() -> Keys:
    return dict(qwen_image_fun_controlnet_keys.state_dict_keys)


def _instantx_keys() -> Keys:
    return dict(qwen_image_instantx_controlnet_keys.state_dict_keys)


def _bf16(shapes: dict[str, list[int]]) -> Keys:
    return {key: (shape, "BF16") for key, shape in shapes.items()}


# Sibling layouts with no captured header, reduced to the keys the probes read.
QWEN_IMAGE_2_1_FUN_KEYS = _bf16(
    {
        "control_img_in.weight": [4096, 129],
        "control_img_in.bias": [4096],
        "control_blocks.0.before_proj.weight": [4096, 4096],
        "control_blocks.0.after_proj.weight": [4096, 4096],
        "control_blocks.0.attn.to_q.weight": [4096, 4096],
        "control_blocks.0.img_mlp.gate_layer.weight": [12288, 4096],
        "control_blocks.0.img_mlp.proj.weight": [12288, 4096],
        "control_blocks.0.img_mlp.out.weight": [4096, 12288],
        "control_blocks.15.after_proj.weight": [4096, 4096],
    }
)
Z_IMAGE_FUN_KEYS = _bf16(
    {
        "control_layers.0.attention.qkv.weight": [11520, 3840],
        "control_layers.0.attention.out.weight": [3840, 3840],
        "control_all_x_embedder.2-1.weight": [3840, 64],
        "control_noise_refiner.0.attention.qkv.weight": [11520, 3840],
    }
)
DIFFSYNTH_KEYS = _bf16(
    {
        "img_in.weight": [3072, 64],
        "controlnet_blocks.0.input_proj.weight": [3072, 3072],
        "controlnet_blocks.0.output_proj.weight": [3072, 3072],
        "controlnet_blocks.0.x_rms.weight": [3072],
        "controlnet_blocks.0.y_rms.weight": [3072],
    }
)
# InstantX's FLUX.1 ControlNet-Union, diffusers layout: the same `controlnet_*` names over FLUX blocks.
FLUX_INSTANTX_KEYS = _bf16(instantx_sd_shapes)


def _write(tmp_path: Path, name: str, keys: Keys) -> Path:
    return write_header_only_safetensors(tmp_path / name, keys)


def _instantx_inpainting_keys() -> Keys:
    """InstantX's inpainting sibling: the control input also takes the 4 packed features of a mask."""
    return {**_instantx_keys(), "controlnet_x_embedder.weight": ([3072, 68], "BF16")}


@pytest.mark.parametrize(
    ("name", "keys", "architecture"),
    [(FUN_FILE_NAME, _fun_keys(), "fun_union"), (INSTANTX_FILE_NAME, _instantx_keys(), "instantx_union")],
    ids=["fun", "instantx"],
)
def test_each_released_file_is_a_qwen_image_controlnet_of_its_design_and_nothing_else(
    tmp_path: Path, name: str, keys: Keys, architecture: str
) -> None:
    result = ModelConfigFactory.from_model_on_disk(_write(tmp_path, name, keys), allow_unknown=False)

    assert type(result.config) is ControlNet_Checkpoint_QwenImage_Config, result.details
    assert result.match_count == 1, result.details
    assert result.config.control_architecture == architecture
    assert result.config.base is BaseModelType.QwenImage
    assert result.config.type is ModelType.ControlNet
    assert result.config.format is ModelFormat.Checkpoint


def test_the_flux_instantx_controlnet_is_still_a_flux_controlnet_and_not_a_qwen_image_one(tmp_path: Path) -> None:
    path = _write(tmp_path, "FLUX.1-dev-Controlnet-Union.safetensors", FLUX_INSTANTX_KEYS)

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=False)

    assert type(result.config) is ControlNet_Checkpoint_FLUX_Config, result.details
    assert result.match_count == 1, result.details
    with pytest.raises(NotAMatchError):
        ControlNet_Checkpoint_QwenImage_Config.from_model_on_disk(ModelOnDisk(path), dict(_OVERRIDE_FIELDS))


def test_the_instantx_inpainting_layout_is_refused_with_a_reason_rather_than_registered(tmp_path: Path) -> None:
    """Recognised and unusable, so nothing is registered -- not even as an unknown model -- and the reason is the
    installer's message."""
    result = ModelConfigFactory.from_model_on_disk(
        _write(tmp_path, "Qwen-Image-InstantX-ControlNet-Inpainting.safetensors", _instantx_inpainting_keys())
    )

    assert result.config is None
    refusal = result.details[ControlNet_Checkpoint_QwenImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert "does not support yet" in str(refusal) and "ControlNet Union instead" in str(refusal)


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."], ids=["bare", "comfyui-prefixed"])
def test_the_qwen_image_main_probe_refuses_the_instantx_union(tmp_path: Path, prefix: str) -> None:
    """The probe strips a ComfyUI prefix before it looks for the transformer's keys, so it has to before it looks
    for the ControlNet's too, as the main loader does: otherwise a prefixed copy would be identified as a main
    model, refused at load, and identified as one again by Re-identify."""
    keys = {f"{prefix}{name}": spec for name, spec in _instantx_keys().items()}
    mod = ModelOnDisk(_write(tmp_path, INSTANTX_FILE_NAME, keys))

    with pytest.raises(NotAMatchError):
        Main_Checkpoint_QwenImage_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


@pytest.mark.parametrize(
    ("name", "preprocessor"),
    [(FUN_FILE_NAME, None), ("Qwen-Image-Fun-Controlnet-Depth.safetensors", "depth_anything_image_processor")],
    ids=["union", "depth"],
)
def test_the_preprocessor_is_chosen_from_the_name_as_for_every_controlnet(
    tmp_path: Path, name: str, preprocessor: str | None
) -> None:
    """A union adapter takes any control image, so none is pre-selected; a single-condition build picks its own."""
    config = ModelConfigFactory.from_model_on_disk(_write(tmp_path, name, _fun_keys()), allow_unknown=False).config

    assert isinstance(config, ControlNet_Checkpoint_QwenImage_Config)
    assert config.default_settings is not None
    assert config.default_settings.preprocessor == preprocessor


@pytest.mark.parametrize(
    ("name", "keys"), [(FUN_FILE_NAME, _fun_keys()), (INSTANTX_FILE_NAME, _instantx_keys())], ids=["fun", "instantx"]
)
def test_a_stored_record_reads_back_as_the_same_qwen_image_controlnet(tmp_path: Path, name: str, keys: Keys) -> None:
    """The record crosses the database and the API as a plain dict, so the config has to be a member of
    `AnyModelConfig` under a discriminator of its own, and the design has to survive the round trip."""
    identified = ModelConfigFactory.from_model_on_disk(_write(tmp_path, name, keys), allow_unknown=False).config
    assert identified is not None

    restored = TypeAdapter(AnyModelConfig).validate_python(identified.model_dump(mode="json"))

    assert type(restored) is ControlNet_Checkpoint_QwenImage_Config
    assert restored.base is BaseModelType.QwenImage
    assert restored.control_architecture == identified.control_architecture


def test_a_record_stored_before_the_design_was_recorded_reads_back_as_the_fun_union(tmp_path: Path) -> None:
    """Every Qwen-Image ControlNet installed until now is a Fun-ControlNet: no other design was identified."""
    identified = ModelConfigFactory.from_model_on_disk(
        _write(tmp_path, FUN_FILE_NAME, _fun_keys()), allow_unknown=False
    ).config
    assert identified is not None
    stored = identified.model_dump(mode="json")
    del stored["control_architecture"]

    restored = TypeAdapter(AnyModelConfig).validate_python(stored)

    assert type(restored) is ControlNet_Checkpoint_QwenImage_Config
    assert restored.control_architecture == "fun_union"


@pytest.mark.parametrize(
    ("name", "keys"),
    [
        ("Qwen-Image-2.1-Fun-Controlnet-Union.safetensors", QWEN_IMAGE_2_1_FUN_KEYS),
        ("FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors", dict(flux2_fun_controlnet_keys.state_dict_keys)),
        ("Z-Image-Turbo-Fun-Controlnet-Union.safetensors", Z_IMAGE_FUN_KEYS),
        ("Qwen-Image-DiffSynth-Controlnet.safetensors", DIFFSYNTH_KEYS),
    ],
    ids=["qwen_image_2_1_fun", "flux2_fun", "z_image_fun", "diffsynth"],
)
def test_the_other_control_designs_are_refused(tmp_path: Path, name: str, keys: Keys) -> None:
    mod = ModelOnDisk(_write(tmp_path, name, keys))

    with pytest.raises(NotAMatchError):
        ControlNet_Checkpoint_QwenImage_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def _without_img_mod(keys: Keys) -> Keys:
    return {key: value for key, value in keys.items() if ".img_mod." not in key}


def _with_control_img_in(keys: Keys, shape: list[int]) -> Keys:
    return {**keys, "control_img_in.weight": (shape, "BF16")}


def _without(keys: Keys, key: str) -> Keys:
    return {name: value for name, value in keys.items() if name != key}


@pytest.mark.parametrize(
    "keys",
    [
        _without_img_mod(_fun_keys()),
        _with_control_img_in(_fun_keys(), [4096, 129]),
        _without_img_mod(_instantx_keys()),
        _without(_instantx_keys(), "txt_norm.weight"),
        {**_instantx_keys(), "controlnet_x_embedder.weight": ([4096, 64], "BF16")},
    ],
    ids=[
        "fun_single_stream_blocks",
        "fun_wider_projection",
        "instantx_single_stream_blocks",
        "instantx_without_text_norm",
        "instantx_wider_embedder",
    ],
)
def test_a_controlnet_with_the_names_but_not_the_fingerprint_is_refused(tmp_path: Path, keys: Keys) -> None:
    """Every part of a fingerprint carries weight: `control_blocks` at Qwen-Image's width without `img_mod` would be
    a single-stream branch, `img_mod` behind a wider projection a branch for a wider base, and `controlnet_blocks`
    without Qwen-Image's dual-stream block and text norm a FLUX ControlNet."""
    mod = ModelOnDisk(_write(tmp_path, FUN_FILE_NAME, keys))

    with pytest.raises(NotAMatchError):
        ControlNet_Checkpoint_QwenImage_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
