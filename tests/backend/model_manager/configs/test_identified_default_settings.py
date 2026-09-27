"""The default settings a model is identified with.

FP8 Storage is switched on for a checkpoint whose denoiser weights are stored in float8, read from the weights'
dtypes and never from the file name: a Comfy "fp8_scaled" file that nobody renamed and a full-precision file that
somebody did must both come out right. Settings passed along with an install land on top of what identification chose.

Single files are tiny Qwen-Image checkpoints, the smallest main model identification recognises from real keys. Folders
are the installer's SDXL diffusers fixture, whose placeholder weight files have no readable header at all.
"""

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config
from invokeai.backend.model_manager.taxonomy import ModelType

_DIFFUSERS_FIXTURE = Path(__file__).parents[1] / "data" / "test_files" / "test-diffusers-main"


def _qwen_image_checkpoint(path: Path, weight_dtype: torch.dtype, **extra: torch.Tensor) -> Path:
    tensors = {
        "img_in.weight": torch.zeros(8, 4).to(weight_dtype),
        "txt_in.weight": torch.zeros(8, 4).to(weight_dtype),
        "txt_norm.weight": torch.ones(4),
        **extra,
    }
    save_file(tensors, str(path))
    return path


def _identify(path: Path, override_fields: dict | None = None) -> Main_Checkpoint_QwenImage_Config:
    result = ModelConfigFactory.from_model_on_disk(path, override_fields, allow_unknown=False)
    assert isinstance(result.config, Main_Checkpoint_QwenImage_Config), result.details
    return result.config


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_float8_denoiser_weights_turn_fp8_storage_on_whatever_the_file_is_called(tmp_path: Path, fp8_dtype) -> None:
    config = _identify(_qwen_image_checkpoint(tmp_path / "qwen_image.safetensors", fp8_dtype))

    assert config.default_settings is not None
    assert config.default_settings.fp8_storage is True


@pytest.mark.parametrize("scale_key", ["img_in.weight_scale", "img_in.scale_weight"])
def test_a_scaled_fp8_checkpoint_gets_it_too(tmp_path: Path, scale_key: str) -> None:
    """A Comfy "scaled fp8" file is float8 on disk like any other, and the loaders keep it in exactly that form when
    the setting is on -- scale included -- so there is nothing to protect it from. It briefly was excluded, back when
    storage folded the scale away and re-encoded the result unscaled.

    Both spellings of the scale key are exercised because matching only `.weight_scale` is the miss that keeps
    recurring (see `iter_weight_scale_pairs`).
    """
    path = _qwen_image_checkpoint(
        tmp_path / "qwen_image.safetensors",
        torch.float8_e4m3fn,
        **{scale_key: torch.ones(1)},
    )

    config = _identify(path)

    assert config.default_settings is not None
    assert config.default_settings.fp8_storage is True


def test_a_full_precision_file_named_fp8_is_left_alone(tmp_path: Path) -> None:
    config = _identify(_qwen_image_checkpoint(tmp_path / "qwen_image_fp8_scaled.safetensors", torch.bfloat16))

    assert config.default_settings is None or config.default_settings.fp8_storage is None


def test_float8_weights_of_a_bundled_text_encoder_do_not_count(tmp_path: Path) -> None:
    """All-in-one checkpoints ship an fp8 text encoder beside a full-precision denoiser; FP8 Storage would re-cast
    that denoiser, which is not what the file is."""
    encoder_weight = torch.zeros(8, 4).to(torch.float8_e4m3fn)
    path = _qwen_image_checkpoint(
        tmp_path / "qwen_image_all_in_one.safetensors",
        torch.bfloat16,
        **{"text_encoders.qwen25_7b.transformer.model.layers.0.mlp.down_proj.weight": encoder_weight},
    )

    config = _identify(path)

    assert config.default_settings is None or config.default_settings.fp8_storage is None


def _diffusers_folder_with_float8(tmp_path: Path, weight_file: str | None) -> Path:
    folder = tmp_path / "sdxl-diffusers"
    shutil.copytree(_DIFFUSERS_FIXTURE, folder)
    if weight_file is not None:
        save_file({"layer.weight": torch.zeros(8, 4).to(torch.float8_e4m3fn)}, str(folder / weight_file))
    return folder


def test_a_denoiser_file_whose_header_is_not_a_json_object_does_not_fail_identification(tmp_path: Path) -> None:
    folder = _diffusers_folder_with_float8(tmp_path, None)
    header = b"[]"
    (folder / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)

    result = ModelConfigFactory.from_model_on_disk(folder, allow_unknown=False)

    assert result.config is not None and result.config.type is ModelType.Main, result.details


@pytest.mark.parametrize(
    "weight_file, expected",
    [
        ("unet/diffusion_pytorch_model.safetensors", True),
        # The denoiser is what FP8 Storage casts; an FP8 text encoder beside a full-precision UNet does not count.
        ("text_encoder/model.safetensors", None),
        # Unreadable placeholder headers everywhere: nothing to go on, and no reason to fail identification.
        (None, None),
    ],
)
def test_a_diffusers_folder_is_judged_by_its_denoiser_weights(
    tmp_path: Path, weight_file: str | None, expected: bool | None
) -> None:
    folder = _diffusers_folder_with_float8(tmp_path, weight_file)

    result = ModelConfigFactory.from_model_on_disk(folder, allow_unknown=False)

    assert result.config is not None and result.config.type is ModelType.Main, result.details
    settings = result.config.default_settings
    assert (settings.fp8_storage if settings is not None else None) is expected


def test_an_install_setting_wins_over_detection_in_both_directions(tmp_path: Path) -> None:
    fp8_file = _qwen_image_checkpoint(tmp_path / "fp8.safetensors", torch.float8_e4m3fn)
    full_precision_file = _qwen_image_checkpoint(tmp_path / "bf16.safetensors", torch.bfloat16)

    kept_off = _identify(fp8_file, {"default_settings": {"fp8_storage": False}})
    turned_on = _identify(full_precision_file, {"default_settings": {"fp8_storage": True}})

    assert kept_off.default_settings is not None and kept_off.default_settings.fp8_storage is False
    assert turned_on.default_settings is not None and turned_on.default_settings.fp8_storage is True


def test_an_install_setting_keeps_the_architectures_other_defaults(tmp_path: Path) -> None:
    path = _qwen_image_checkpoint(tmp_path / "bf16.safetensors", torch.bfloat16)
    probed = _identify(path).default_settings

    overridden = _identify(path, {"default_settings": {"fp8_storage": True}}).default_settings

    assert probed is not None and overridden is not None
    assert overridden.model_dump(exclude={"fp8_storage"}) == probed.model_dump(exclude={"fp8_storage"})


def _anima_lllite_adapter(path: Path, weight_dtype: torch.dtype) -> Path:
    """The smallest file identification reads as an Anima ControlNet-LLLite adapter.

    Chosen because its loader is one of the seven that declare they do not implement FP8 Storage, and
    it is the cheapest of those to identify. No published LLLite adapter is float8 -- they are 8-66 MB
    bf16 files -- so this one is synthetic on purpose: the subject is the wiring, not the file.
    """
    save_file(
        {
            "lllite_conditioning1.conv1.weight": torch.zeros(8, 3, 4, 4).to(weight_dtype),
            "lllite_dit_blocks_0.down.weight": torch.zeros(4, 8).to(weight_dtype),
        },
        str(path),
    )
    return path


def test_fp8_storage_is_not_enabled_for_a_model_whose_loader_ignores_it(tmp_path: Path) -> None:
    """The float8 probe is not the whole question, and used to be treated as though it were.

    A Wan fp8 checkpoint or an Anima LLLite adapter is float8 on disk, so identification enabled FP8
    Storage; neither loader touched the setting, so the record promised half the memory and the load
    delivered none of it. `AnimaControlNetLLLiteModel` now declares `NotApplicable`, and identification
    asks (`load/fp8_capability.py`).
    """
    result = ModelConfigFactory.from_model_on_disk(
        _anima_lllite_adapter(tmp_path / "lllite.safetensors", torch.float8_e4m3fn), allow_unknown=False
    )

    assert result.config is not None and result.config.type is ModelType.ControlNet, result.details
    settings = result.config.default_settings
    assert (settings.fp8_storage if settings is not None else None) is None


def test_an_explicit_install_request_is_dropped_for_such_a_model_too(tmp_path: Path) -> None:
    """Add Models has one FP8 Storage checkbox for whatever is being installed, so the request arrives
    for models that cannot use it. Storing it would leave a value the detail panel no longer shows and
    therefore nobody can clear; the install's other settings are untouched."""
    result = ModelConfigFactory.from_model_on_disk(
        _anima_lllite_adapter(tmp_path / "lllite.safetensors", torch.bfloat16),
        {"default_settings": {"fp8_storage": True, "preprocessor": "canny_edge_detection"}},
        allow_unknown=False,
    )

    assert result.config is not None, result.details
    assert result.config.default_settings is not None
    assert result.config.default_settings.fp8_storage is None
    assert result.config.default_settings.preprocessor == "canny_edge_detection"
