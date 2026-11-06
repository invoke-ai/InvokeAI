"""FLUX.3 LoRA identification against every other LoRA config.

`Config_Base.CONFIG_CLASSES` is a set, so the order the probes run in changes between processes: each published file
must be claimed by exactly one config, or which one claims it is up to iteration order. FLUX.3's native single blocks
share `diffusion_model.single_blocks.N.` with FLUX.1's and FLUX.2's native layout, and FLUX.1's ai-toolkit and
BFL-PEFT probes accept whatever module follows, so until FLUX.3 had a config of its own they claimed its ai-toolkit
LoRAs as FLUX.1 ones. The FLUX.1 and FLUX.2 files must keep their configs in turn, files and diffusers folders alike,
and so must another family whose module names FLUX.3 shares. The headers are those of published files
(`lora_state_dicts/flux3_*`), read as identification reads a file: on the meta device, with PEFT's adapter names
stripped.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_Diffusers_Flux3_Config, LoRA_LyCORIS_Flux3_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.lora_conversions import flux3_lora_constants as constants
from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names
from tests.backend.model_manager.load.state_dicts import flux3_klein_keys
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    flux3_lora_aitoolkit_format,
    flux3_lora_comfy_alpha_format,
    flux3_lora_diffusers_format,
    flux3_lora_distilled_extract_format,
    flux3_lora_peft_adapter_format,
    flux_dora_onetrainer_format,
    flux_lokr_bfl_format,
    flux_lora_aitoolkit_format,
    flux_lora_diffusers_format,
    flux_lora_kohya_format,
    flux_lora_xlabs_format,
)

_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


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
):
    probes = [cls for cls in Config_Base.CONFIG_CLASSES if cls.__name__.startswith(("LoRA_", "ControlLoRA_"))]
    assert {LoRA_LyCORIS_Flux3_Config, LoRA_Diffusers_Flux3_Config} <= set(probes)
    return sorted(
        cls.__name__ for cls in probes if isinstance(_probe(cls, state_dict, tmp_path, metadata, folder), Config_Base)
    )


_FLUX3_LORAS = {
    "diffusers": (flux3_lora_diffusers_format.state_dict_keys, None),
    "ai-toolkit": (flux3_lora_aitoolkit_format.state_dict_keys, flux3_lora_aitoolkit_format.metadata),
    "comfy-alpha": (flux3_lora_comfy_alpha_format.state_dict_keys, None),
    "peft-adapter": (flux3_lora_peft_adapter_format.state_dict_keys, None),
}


@pytest.mark.parametrize("layout", list(_FLUX3_LORAS))
def test_a_published_flux3_lora_is_claimed_by_the_flux3_config_alone(layout: str, tmp_path: Path) -> None:
    keys, metadata = _FLUX3_LORAS[layout]
    state_dict = _state_dict(keys)

    assert _claimed_by(state_dict, tmp_path, metadata) == [LoRA_LyCORIS_Flux3_Config.__name__]
    assert _probe(LoRA_LyCORIS_Flux3_Config, state_dict, tmp_path, metadata).base is BaseModelType.Flux3
    # The gate that keeps FLUX.1's probes off it: no FLUX.1/FLUX.2 format, metadata and all.
    assert flux_format_from_state_dict(state_dict, metadata) is None


def _flux2_klein_diffusers() -> dict[str, list[int]]:
    """A FLUX.2 Klein 4B LoRA in diffusers' names: its double blocks' `ff.linear_in` and its single blocks' fused
    projection, beside FLUX.3's single-block `ff` and separate `to_out` in name, and a text embedder 7680 wide."""
    return {
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight": [16, 3072],
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight": [27648, 16],
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight": [16, 12288],
        "transformer.single_transformer_blocks.0.attn.to_out.lora_B.weight": [3072, 16],
        "transformer.transformer_blocks.0.ff.linear_in.lora_A.weight": [16, 3072],
        "transformer.transformer_blocks.0.ff.linear_in.lora_B.weight": [18432, 16],
        "transformer.context_embedder.lora_A.weight": [16, 7680],
        "transformer.context_embedder.lora_B.weight": [3072, 16],
    }


def _flux2_klein_native() -> dict[str, list[int]]:
    """A FLUX.2 Klein 4B LoRA in BFL's names under `diffusion_model.`, single blocks only, as FLUX.3's ai-toolkit
    LoRAs are, beside a text embedder 7680 wide: `linear1`/`linear2` where FLUX.3 has q/k/v and the MLP apart."""
    return {
        "diffusion_model.single_blocks.0.linear1.lora_A.weight": [16, 3072],
        "diffusion_model.single_blocks.0.linear1.lora_B.weight": [27648, 16],
        "diffusion_model.single_blocks.0.linear2.lora_A.weight": [16, 12288],
        "diffusion_model.single_blocks.0.linear2.lora_B.weight": [3072, 16],
        "diffusion_model.txt_in.lora_A.weight": [16, 7680],
        "diffusion_model.txt_in.lora_B.weight": [3072, 16],
    }


@pytest.mark.parametrize(
    ("keys", "claimed_by"),
    [
        (flux_lora_aitoolkit_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
        (flux_lora_diffusers_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
        (flux_lora_kohya_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
        (flux_lora_xlabs_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
        (flux_dora_onetrainer_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
        (flux_lokr_bfl_format.state_dict_keys, "LoRA_LyCORIS_Flux2_Config"),
        (_flux2_klein_diffusers(), "LoRA_LyCORIS_Flux2_Config"),
        (_flux2_klein_native(), "LoRA_LyCORIS_Flux2_Config"),
    ],
    ids=[
        "flux1-ai-toolkit",
        "flux1-diffusers",
        "flux1-kohya",
        "flux1-xlabs",
        "flux1-onetrainer-dora",
        "flux2-bfl-lokr",
        "flux2-klein-diffusers",
        "flux2-klein-native",
    ],
)
def test_a_flux1_or_flux2_lora_keeps_its_config(keys: dict[str, list[int]], claimed_by: str, tmp_path: Path) -> None:
    assert _claimed_by(_state_dict(keys), tmp_path) == [claimed_by]


def test_a_distillation_extract_is_refused_with_its_reason(tmp_path: Path) -> None:
    """Its norm-scale deltas and flow-map embedder weights are no LoRA layers: the installer names them, and what to
    install instead, rather than registering a file nothing applies in full as an unknown model."""
    state_dict = _state_dict(flux3_lora_distilled_extract_format.state_dict_keys)

    refused = _probe(LoRA_LyCORIS_Flux3_Config, state_dict, tmp_path)

    assert isinstance(refused, InvalidMatchError)
    message = str(refused)
    assert constants.DISTILLATION_EXTRACT in message and "'delta." in message
    # What to install instead comes after the reasons and their example keys.
    assert message.endswith("To sample with distilled weights, install FLUX.3 Klein itself.")
    # The install queue shows the message as plain text.
    assert "`" not in message
    assert _claimed_by(state_dict, tmp_path) == []


def _auraflow() -> dict[str, list[int]]:
    """An AuraFlow LoRA on its single blocks' attention: 3072 wide, named as FLUX.3's diffusers single blocks are."""
    keys: dict[str, list[int]] = {}
    for projection in ("to_q", "to_k", "to_v", "to_out.0"):
        keys[f"transformer.single_transformer_blocks.0.attn.{projection}.lora_A.weight"] = [16, 3072]
        keys[f"transformer.single_transformer_blocks.0.attn.{projection}.lora_B.weight"] = [3072, 16]
    return keys


def test_another_family_named_like_flux3_is_not_claimed_as_flux3(tmp_path: Path) -> None:
    """Only FLUX.3's single-block MLP is its own; claimed for the attention alone, AuraFlow's LoRA would install as a
    FLUX.3 one of the right shapes."""
    assert isinstance(_probe(LoRA_LyCORIS_Flux3_Config, _state_dict(_auraflow()), tmp_path), NotAMatchError)
    assert LoRA_LyCORIS_Flux3_Config.__name__ not in _claimed_by(_state_dict(_auraflow()), tmp_path)


@pytest.mark.parametrize(
    ("keys", "claimed_by"),
    [
        (flux3_lora_diffusers_format.state_dict_keys, "LoRA_Diffusers_Flux3_Config"),
        (flux_lora_diffusers_format.state_dict_keys, "LoRA_Diffusers_FLUX_Config"),
        (_flux2_klein_diffusers(), "LoRA_Diffusers_Flux2_Config"),
    ],
    ids=["flux3", "flux1", "flux2-klein"],
)
def test_a_diffusers_lora_folder_is_claimed_by_its_family_alone(
    keys: dict[str, Any], claimed_by: str, tmp_path: Path
) -> None:
    """A repository-id install of a diffusers trainer's output leaves a folder holding `pytorch_lora_weights.safetensors`."""
    assert _claimed_by(_state_dict(keys), tmp_path, folder=True) == [claimed_by]


def test_a_flux3_checkpoint_is_no_lora(tmp_path: Path) -> None:
    """Its weights carry FLUX.3's module names but no LoRA parameter: not a FLUX.3 LoRA, so nothing to refuse."""
    refused = _probe(LoRA_LyCORIS_Flux3_Config, _state_dict(flux3_klein_keys.state_dict_keys), tmp_path)

    assert isinstance(refused, NotAMatchError)
