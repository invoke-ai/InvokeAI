"""MOSS-Image LoRA identification against every other LoRA config.

`Config_Base.CONFIG_CLASSES` is a set, so the order the probes run in changes between processes: each published file
must be claimed by exactly one config, or which one claims it is up to iteration order. MOSS-Image keeps its blocks
under `layers.N` behind `diffusion_model.` / `transformer.`, as Z-Image does, and its kohya names are `lora_unet_`
ones like the SD families', so Z-Image's probe used to claim its LoRAs, and a file no probe knew installed as an unknown
model. The headers are those of published files (`lora_state_dicts/moss_image_lora_*`), read as identification reads a
file: on the meta device, with PEFT's adapter names stripped.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_Diffusers_MossImage_Config, LoRA_LyCORIS_MossImage_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.lora_conversions import moss_image_lora_constants as constants
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    anima_lora_kohya_format,
    ernie_image_lora_aitoolkit_format,
    flux3_lora_aitoolkit_format,
    flux3_lora_comfy_alpha_format,
    flux3_lora_diffusers_format,
    flux_lora_aitoolkit_format,
    flux_lora_kohya_format,
    ideogram4_lora_aitoolkit_format,
    krea2_lora_kohya_format,
    moss_image_lora_aitoolkit_dora_format,
    moss_image_lora_aitoolkit_format,
    moss_image_lora_aitoolkit_fused_format,
    moss_image_lora_bare_format,
    moss_image_lora_comfy_format,
    moss_image_lora_comfy_fused_format,
    moss_image_lora_comfy_fused_partial_format,
    moss_image_lora_control_format,
    moss_image_lora_diffsynth_format,
    moss_image_lora_diffusers_format,
    moss_image_lora_edit_format,
    moss_image_lora_kohya_format,
    moss_image_lora_kohya_fused_format,
    moss_image_lora_loha_format,
    moss_image_lora_lokr_auto_format,
    moss_image_lora_lokr_factor4_format,
    moss_image_lora_lokr_factor16_format,
    moss_image_lora_lycoris_dora_format,
    moss_image_lora_norm_diff_format,
    moss_image_lora_prefixed_alpha_format,
    qwen_image_lora_diffusers_format,
    qwen_image_lora_kohya_format,
    sd3_lora_diffusers_format,
    z_image_lora_aitoolkit_format,
)

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_AITOOLKIT_KEYS = moss_image_lora_aitoolkit_format.state_dict_keys
_MOSS_CONFIGS = {LoRA_LyCORIS_MossImage_Config.__name__, LoRA_Diffusers_MossImage_Config.__name__}


def _state_dict(keys: dict[str, Any]) -> dict[str, torch.Tensor]:
    """A fixture's keys on the meta device, as identification reads a file, PEFT's adapter names stripped. Values are a
    shape, or a captured `(shape, dtype)`."""
    tensors = {}
    for key, value in keys.items():
        shape, dtype = value if isinstance(value, tuple) else (value, "F32")
        tensors[key] = torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
    return normalize_peft_adapter_names(tensors)


def _probe(
    cls: type,
    state_dict: dict[str, torch.Tensor],
    tmp_path: Path,
    metadata: dict[str, str] | None = None,
    folder: bool = False,
):
    """The config `cls` builds for the file, or for a diffusers folder holding it, or the error it refuses it with."""
    if folder:
        path = tmp_path / "lora"
        path.mkdir(exist_ok=True)
        (path / "pytorch_lora_weights.safetensors").touch()
    else:
        path = tmp_path / "lora.safetensors"
        path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = state_dict
    mod.metadata.return_value = metadata or {}
    fields = {
        "hash": "h",
        "path": str(path),
        "file_size": 0,
        "name": "lora",
        "source": str(path),
        "source_type": "path",
    }
    try:
        return cls.from_model_on_disk(mod, fields)
    except (NotAMatchError, InvalidMatchError) as error:
        return error


def _claimed_by(
    state_dict: dict[str, torch.Tensor],
    tmp_path: Path,
    metadata: dict[str, str] | None = None,
    folder: bool = False,
) -> list[str]:
    probes = [cls for cls in Config_Base.CONFIG_CLASSES if cls.__name__.startswith(("LoRA_", "ControlLoRA_"))]
    assert {LoRA_LyCORIS_MossImage_Config, LoRA_Diffusers_MossImage_Config} <= set(probes)
    return sorted(
        cls.__name__ for cls in probes if isinstance(_probe(cls, state_dict, tmp_path, metadata, folder), Config_Base)
    )


_MOSS_IMAGE_LORAS = {
    "ai-toolkit": moss_image_lora_aitoolkit_format,
    "ai-toolkit-fused": moss_image_lora_aitoolkit_fused_format,
    "ai-toolkit-dora": moss_image_lora_aitoolkit_dora_format,
    "kohya": moss_image_lora_kohya_format,
    "kohya-fused": moss_image_lora_kohya_fused_format,
    "lycoris-dora": moss_image_lora_lycoris_dora_format,
    "prefixed-alpha": moss_image_lora_prefixed_alpha_format,
    "diffsynth": moss_image_lora_diffsynth_format,
    "bare": moss_image_lora_bare_format,
    "comfy": moss_image_lora_comfy_format,
    "comfy-fused": moss_image_lora_comfy_fused_format,
    "comfy-fused-partial": moss_image_lora_comfy_fused_partial_format,
    "lokr-factor-16": moss_image_lora_lokr_factor16_format,
    "lokr-factor-4": moss_image_lora_lokr_factor4_format,
    "lokr-auto": moss_image_lora_lokr_auto_format,
    "loha": moss_image_lora_loha_format,
    "diffusers": moss_image_lora_diffusers_format,
}


@pytest.mark.parametrize("layout", list(_MOSS_IMAGE_LORAS))
def test_a_published_moss_image_lora_is_claimed_by_the_moss_image_config_alone(layout: str, tmp_path: Path) -> None:
    fixture = _MOSS_IMAGE_LORAS[layout]
    state_dict = _state_dict(fixture.state_dict_keys)
    metadata = getattr(fixture, "metadata", None)

    assert _claimed_by(state_dict, tmp_path, metadata) == [LoRA_LyCORIS_MossImage_Config.__name__]
    config = _probe(LoRA_LyCORIS_MossImage_Config, state_dict, tmp_path, metadata)
    assert (config.base, config.format) == (BaseModelType.MossImage, ModelFormat.LyCORIS)


@pytest.mark.parametrize("layout", ["diffusers", "ai-toolkit"])
def test_a_moss_image_lora_folder_is_claimed_by_the_diffusers_config_alone(layout: str, tmp_path: Path) -> None:
    """A repository-id install of a diffusers trainer's output leaves a folder holding `pytorch_lora_weights.safetensors`."""
    fixture = _MOSS_IMAGE_LORAS[layout]
    state_dict = _state_dict(fixture.state_dict_keys)
    metadata = getattr(fixture, "metadata", None)

    assert _claimed_by(state_dict, tmp_path, metadata, folder=True) == [LoRA_Diffusers_MossImage_Config.__name__]
    config = _probe(LoRA_Diffusers_MossImage_Config, state_dict, tmp_path, metadata, folder=True)
    assert (config.base, config.format) == (BaseModelType.MossImage, ModelFormat.Diffusers)


@pytest.mark.parametrize(
    ("keys", "claimed_by"),
    [
        (z_image_lora_aitoolkit_format.state_dict_keys, "LoRA_LyCORIS_ZImage_Config"),
        (ernie_image_lora_aitoolkit_format.state_dict_keys, "LoRA_LyCORIS_ErnieImage_Config"),
        (ideogram4_lora_aitoolkit_format.state_dict_keys, "LoRA_LyCORIS_Ideogram4_Config"),
        (flux3_lora_aitoolkit_format.state_dict_keys, "LoRA_LyCORIS_Flux3_Config"),
        (flux3_lora_comfy_alpha_format.state_dict_keys, "LoRA_LyCORIS_Flux3_Config"),
        (krea2_lora_kohya_format.state_dict_keys, "LoRA_LyCORIS_Krea2_Config"),
        (qwen_image_lora_diffusers_format.state_dict_keys, "LoRA_LyCORIS_QwenImage_Config"),
        (qwen_image_lora_kohya_format.state_dict_keys, "LoRA_LyCORIS_QwenImage_Config"),
        (anima_lora_kohya_format.state_dict_keys, "LoRA_LyCORIS_Anima_Config"),
        (sd3_lora_diffusers_format.state_dict_keys, "LoRA_LyCORIS_SD3_Config"),
        (flux_lora_aitoolkit_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
        (flux_lora_kohya_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
    ],
    ids=[
        "z-image",
        "ernie-image",
        "ideogram4",
        "flux3-ai-toolkit",
        "flux3-comfy",
        "krea2",
        "qwen-image-diffusers",
        "qwen-image-kohya",
        "anima",
        "sd3",
        "flux1-ai-toolkit",
        "flux1-kohya",
    ],
)
def test_another_familys_lora_keeps_its_config_and_is_not_claimed_as_moss_image(
    keys: dict[str, Any], claimed_by: str, tmp_path: Path
) -> None:
    state_dict = _state_dict(keys)

    assert _claimed_by(state_dict, tmp_path) == [claimed_by]
    assert isinstance(_probe(LoRA_LyCORIS_MossImage_Config, state_dict, tmp_path), NotAMatchError)


def test_a_flux3_diffusers_folder_is_not_claimed_as_moss_image(tmp_path: Path) -> None:
    state_dict = _state_dict(flux3_lora_diffusers_format.state_dict_keys)

    assert not _MOSS_CONFIGS & set(_claimed_by(state_dict, tmp_path, folder=True))


@pytest.mark.parametrize(
    ("fixture", "reasons"),
    [
        (
            moss_image_lora_norm_diff_format,
            {constants.NORM_WEIGHT, constants.NOT_A_LORA_PARAMETER},
        ),
        (moss_image_lora_control_format, {constants.NOT_A_LORA_PARAMETER}),
    ],
    ids=["norm-and-bias-diffs", "control-weights"],
)
def test_a_moss_image_lora_that_does_not_apply_in_full_is_refused_with_its_reason(
    fixture, reasons: set[str], tmp_path: Path
) -> None:
    """The patcher would skip what it cannot apply with a log line, so the file installs as nothing rather than as a
    LoRA that silently does less -- and not as an unknown model, which would hide why."""
    state_dict = _state_dict(fixture.state_dict_keys)

    refused = _probe(LoRA_LyCORIS_MossImage_Config, state_dict, tmp_path)

    assert isinstance(refused, InvalidMatchError)
    assert all(reason in str(refused) for reason in reasons)
    assert _claimed_by(state_dict, tmp_path) == []


def test_a_stray_text_encoder_layer_refuses_a_moss_image_lora(tmp_path: Path) -> None:
    keys = dict(moss_image_lora_aitoolkit_format.state_dict_keys)
    keys["lora_te_text_model_layers_0_mlp_fc1.lora_down.weight"] = ([16, 4096], "F16")
    keys["lora_te_text_model_layers_0_mlp_fc1.lora_up.weight"] = ([4096, 16], "F16")

    refused = _probe(LoRA_LyCORIS_MossImage_Config, _state_dict(keys), tmp_path)

    assert isinstance(refused, InvalidMatchError)
    assert "lora_te_text_model_layers_0_mlp_fc1" in str(refused)


def test_a_header_asking_for_a_scaling_this_cannot_apply_refuses_the_lora(tmp_path: Path) -> None:
    metadata = {"lora_adapter_metadata": '{"transformer.r": 4, "transformer.alpha_pattern": {"layers.0.proj.q": 2}}'}

    refused = _probe(
        LoRA_LyCORIS_MossImage_Config,
        _state_dict(moss_image_lora_diffusers_format.state_dict_keys),
        tmp_path,
        metadata,
    )

    assert isinstance(refused, InvalidMatchError)
    assert "This MOSS-Image LoRA sets its rank or alpha per module" in str(refused)


def test_an_edit_lora_stays_unknown(tmp_path: Path) -> None:
    """MOSS-Image's edit LoRAs name the same modules in bare `lora_A`/`lora_B`, without PEFT's `.weight`, and condition
    on reference images: no config claims them, so they install as unknown instead of as a LoRA that ignores that."""
    state_dict = _state_dict(moss_image_lora_edit_format.state_dict_keys)

    assert isinstance(_probe(LoRA_LyCORIS_MossImage_Config, state_dict, tmp_path), NotAMatchError)
    assert _claimed_by(state_dict, tmp_path) == []


def test_a_moss_image_checkpoint_is_no_lora(tmp_path: Path) -> None:
    """Its weights carry MOSS-Image's module names but no LoRA parameter: nothing to refuse either."""
    state_dict = _state_dict(
        {
            "layers.0.proj.q.weight": [3072, 3072],
            "layers.0.proj.ffn_down.weight": [3072, 8192],
            "video_embedder.bias": [3072],
        }
    )

    assert isinstance(_probe(LoRA_LyCORIS_MossImage_Config, state_dict, tmp_path), NotAMatchError)


@pytest.mark.parametrize("prefix", ["base_model.model.", "base_model.model.transformer."])
def test_a_raw_peft_moss_image_lora_is_refused_and_not_taken_for_z_image(prefix: str, tmp_path: Path) -> None:
    """Z-Image's probe admits `base_model.model.transformer.layers.`: the file must not install as a Z-Image LoRA,
    which would then do nothing, and says why it is not a MOSS-Image one."""
    keys = {f"{prefix}{key.removeprefix('diffusion_model.')}": value for key, value in _AITOOLKIT_KEYS.items()}
    state_dict = _state_dict(keys)

    refused = _probe(LoRA_LyCORIS_MossImage_Config, state_dict, tmp_path)

    assert isinstance(refused, InvalidMatchError)
    assert constants.RAW_PEFT in str(refused)
    assert _claimed_by(state_dict, tmp_path) == []
