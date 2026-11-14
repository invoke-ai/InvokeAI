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
from invokeai.backend.model_manager.configs.lora import (
    LoRA_Diffusers_Flux3_Config,
    LoRA_LyCORIS_Flux2_Config,
    LoRA_LyCORIS_Flux3_Config,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux2VariantType
from invokeai.backend.patches.lora_conversions import flux3_lora_constants as constants
from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict
from invokeai.backend.patches.lora_conversions.peft_adapter_utils import normalize_peft_adapter_names
from tests.backend.model_manager.load.state_dicts import flux3_klein_keys
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    flux2_lora_kohya_non_block_format,
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
    flux_lora_onetrainer_non_block_format,
    flux_lora_xlabs_format,
    qwen_image_lora_kohya_format,
)
from tests.backend.patches.lora_conversions.test_flux_onetrainer_lora_conversion_utils import (
    _meta_sd3,
    _onetrainer_legacy_state_dict,
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
        (flux2_lora_kohya_non_block_format.state_dict_keys, "LoRA_LyCORIS_Flux2_Config"),
        (flux_lora_onetrainer_non_block_format.state_dict_keys, "LoRA_LyCORIS_FLUX_Config"),
        (qwen_image_lora_kohya_format.state_dict_keys, "LoRA_LyCORIS_QwenImage_Config"),
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
        "flux2-klein-kohya-with-non-block-layers",
        "flux1-onetrainer-with-non-block-layers",
        "qwen-image-kohya",
    ],
)
def test_a_flux1_or_flux2_lora_keeps_its_config(keys: dict[str, list[int]], claimed_by: str, tmp_path: Path) -> None:
    assert _claimed_by(_state_dict(keys), tmp_path) == [claimed_by]


def test_a_kohya_lora_with_non_block_layers_is_a_klein_4b_lora(tmp_path: Path) -> None:
    """`txt_in` of the layers outside the blocks is what tells Klein 4B (7680 inputs) from FLUX.1 (#10030)."""
    config = _probe(LoRA_LyCORIS_Flux2_Config, _state_dict(flux2_lora_kohya_non_block_format.state_dict_keys), tmp_path)

    assert config.base is BaseModelType.Flux2
    assert config.variant is Flux2VariantType.Klein4B


def _layers(keys: dict[str, Any], *layer_names: str) -> dict[str, Any]:
    """The fixture's keys of the named kohya layers only (a LoRA that trains just those)."""
    return {k: v for k, v in keys.items() if k.split(".")[0] in {f"lora_unet_{name}" for name in layer_names}}


_KLEIN_4B_KOHYA = flux2_lora_kohya_non_block_format.state_dict_keys


@pytest.mark.parametrize(
    ("keys", "claimed_by", "variant"),
    [
        # The shared modulation exists in FLUX.2 only; the fused linear1 of a Klein 4B single block is 27648 wide where
        # FLUX.1's is 21504 (a mismatch at apply time if it were read as FLUX.1).
        (
            _layers(
                _KLEIN_4B_KOHYA, "single_blocks_0_linear1", "single_blocks_0_linear2", "single_stream_modulation_lin"
            ),
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        (
            _layers(_KLEIN_4B_KOHYA, "single_blocks_0_linear1", "single_blocks_0_linear2"),
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        (
            _layers(_KLEIN_4B_KOHYA, "double_stream_modulation_img_lin", "double_blocks_0_txt_attn_qkv"),
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        (
            _layers(flux_lora_kohya_format.state_dict_keys, "single_blocks_0_linear1", "single_blocks_0_linear2"),
            "LoRA_LyCORIS_FLUX_Config",
            None,
        ),
        # LoKr with a whole `lokr_w1`, whose single key carries both widths: (4 x 6912) by (4 x 768) is Klein 4B's
        # 27648 x 3072 linear1, (4 x 5376) by (4 x 768) FLUX.1's 21504 x 3072.
        (
            {
                "lora_unet_single_blocks_0_linear1.lokr_w1": [4, 4],
                "lora_unet_single_blocks_0_linear1.lokr_w2": [6912, 768],
                "lora_unet_single_blocks_0_linear1.alpha": [],
            },
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        (
            {
                "lora_unet_single_blocks_0_linear1.lokr_w1": [4, 4],
                "lora_unet_single_blocks_0_linear1.lokr_w2": [5376, 768],
                "lora_unet_single_blocks_0_linear1.alpha": [],
            },
            "LoRA_LyCORIS_FLUX_Config",
            None,
        ),
    ],
    ids=[
        "klein-single-block-and-modulation",
        "klein-single-block-only",
        "klein-modulation-and-txt-attn",
        "flux1-single-block-only",
        "klein-lokr-whole-w1",
        "flux1-lokr-whole-w1",
    ],
)
def test_a_partial_kohya_lora_is_identified_by_the_layers_it_has(
    keys: dict[str, Any], claimed_by: str, variant: Flux2VariantType | None, tmp_path: Path
) -> None:
    assert keys
    state_dict = _state_dict(keys)

    assert _claimed_by(state_dict, tmp_path) == [claimed_by]
    if variant is not None:
        assert _probe(LoRA_LyCORIS_Flux2_Config, state_dict, tmp_path).variant is variant


def _kohya(layer: str, in_dim: int, out_dim: int, rank: int = 16) -> dict[str, Any]:
    return {
        f"lora_unet_{layer}.alpha": [],
        f"lora_unet_{layer}.lora_down.weight": [rank, in_dim],
        f"lora_unet_{layer}.lora_up.weight": [out_dim, rank],
    }


_ATTN_PROJ = "double_blocks_0_img_attn_proj"  # 3072 x 3072 in FLUX.1 and Klein 4B: says nothing on its own
# A block layer that names no hidden size outright (a Klein 9B MLP output: 6 x 4096 in, 4096 out)
_MLP_OUT = "double_blocks_0_img_mlp_2"


@pytest.mark.parametrize(
    ("keys", "claimed_by", "variant"),
    [
        # `img_in` takes 128 latent channels in FLUX.2 and 64 in FLUX.1 (reviewer's reproducer, #10206).
        (
            _layers(_KLEIN_4B_KOHYA, "img_in", _ATTN_PROJ),
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        (
            _layers(_KLEIN_4B_KOHYA, "final_layer_linear", _ATTN_PROJ),
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        (
            {**_kohya("img_in", 64, 3072), **_kohya(_ATTN_PROJ, 3072, 3072)},
            "LoRA_LyCORIS_FLUX_Config",
            None,
        ),
        (
            {**_kohya("final_layer_linear", 3072, 64), **_kohya(_ATTN_PROJ, 3072, 3072)},
            "LoRA_LyCORIS_FLUX_Config",
            None,
        ),
        # FLUX.1 Fill's `img_in` takes 384 channels (latents, masked image and mask): not FLUX.2's 128.
        (
            {**_kohya("img_in", 384, 3072), **_kohya(_ATTN_PROJ, 3072, 3072)},
            "LoRA_LyCORIS_FLUX_Config",
            None,
        ),
        # Same shapes in FLUX.1 and Klein 4B: either model takes them, so FLUX.1 stays the answer.
        (
            _layers(
                _KLEIN_4B_KOHYA,
                "time_in_in_layer",
                "time_in_out_layer",
                "final_layer_adaLN_modulation_1",
                _ATTN_PROJ,
            ),
            "LoRA_LyCORIS_FLUX_Config",
            None,
        ),
        # Klein 9B (hidden size 4096) shows in the hidden-sized axis of any non-block layer, and `img_in`'s 128 channels.
        (
            {**_kohya("time_in_out_layer", 4096, 4096), **_kohya(_MLP_OUT, 24576, 4096)},
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein9B,
        ),
        (
            {**_kohya("img_in", 128, 4096), **_kohya(_MLP_OUT, 24576, 4096)},
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein9B,
        ),
        (
            {**_kohya("final_layer_adaLN_modulation_1", 4096, 8192), **_kohya(_MLP_OUT, 24576, 4096)},
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein9B,
        ),
        # The variant comes from the non-block layer alone (the block layer beside it has no hidden size to read).
        (
            {**_kohya("img_in", 128, 3072), **_kohya(_MLP_OUT, 9216, 3072)},
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        # LoHa: (3072 x r) by (r x 128), twice.
        (
            {
                "lora_unet_img_in.hada_w1_a": [3072, 4],
                "lora_unet_img_in.hada_w1_b": [4, 128],
                "lora_unet_img_in.hada_w2_a": [3072, 4],
                "lora_unet_img_in.hada_w2_b": [4, 128],
                **_kohya(_ATTN_PROJ, 3072, 3072),
            },
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        # LoKr: whole or factorized operands, the 128 inputs of `img_in` as 2 x 64 (FLUX.1's 64 as 2 x 32).
        (
            {
                "lora_unet_img_in.lokr_w1": [3, 2],
                "lora_unet_img_in.lokr_w2": [1024, 64],
                **_kohya(_ATTN_PROJ, 3072, 3072),
            },
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        (
            {
                "lora_unet_img_in.lokr_w1_a": [3, 4],
                "lora_unet_img_in.lokr_w1_b": [4, 2],
                "lora_unet_img_in.lokr_w2_a": [1024, 4],
                "lora_unet_img_in.lokr_w2_b": [4, 64],
                **_kohya(_ATTN_PROJ, 3072, 3072),
            },
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
        # Klein 9B: img_in is (4 x 1024) by (2 x 64) = 4096 x 128.
        (
            {
                "lora_unet_img_in.lokr_w1": [4, 2],
                "lora_unet_img_in.lokr_w2": [1024, 64],
                **_kohya(_MLP_OUT, 24576, 4096),
            },
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein9B,
        ),
        (
            {
                "lora_unet_img_in.lokr_w1": [3, 2],
                "lora_unet_img_in.lokr_w2": [1024, 32],
                **_kohya(_ATTN_PROJ, 3072, 3072),
            },
            "LoRA_LyCORIS_FLUX_Config",
            None,
        ),
        (
            {
                "lora_unet_final_layer_linear.lokr_w1": [2, 3],
                "lora_unet_final_layer_linear.lokr_w2": [64, 1024],
                **_kohya(_ATTN_PROJ, 3072, 3072),
            },
            "LoRA_LyCORIS_Flux2_Config",
            Flux2VariantType.Klein4B,
        ),
    ],
    ids=[
        "klein-img-in",
        "klein-final-layer-linear",
        "flux1-img-in",
        "flux1-final-layer-linear",
        "flux1-fill-img-in",
        "shared-shapes-stay-flux1",
        "klein9b-time-in",
        "klein9b-img-in",
        "klein9b-final-layer-adaln",
        "klein-variant-from-img-in",
        "klein-loha-img-in",
        "klein-lokr-img-in-whole",
        "klein-lokr-img-in-factorized",
        "klein9b-lokr-img-in",
        "flux1-lokr-img-in",
        "klein-lokr-final-layer-linear",
    ],
)
def test_a_kohya_lora_is_identified_by_the_shape_of_its_non_block_layers(
    keys: dict[str, Any], claimed_by: str, variant: Flux2VariantType | None, tmp_path: Path
) -> None:
    """A layer outside the blocks that takes other widths in FLUX.2 than in FLUX.1 identifies the family, though the
    block layers beside it fit both."""
    state_dict = _state_dict(keys)

    assert _claimed_by(state_dict, tmp_path) == [claimed_by]
    if variant is not None:
        assert _probe(LoRA_LyCORIS_Flux2_Config, state_dict, tmp_path).variant is variant


def test_an_sd3_kohya_folder_with_a_final_layer_is_not_claimed_as_flux2(tmp_path: Path) -> None:
    """SD3 names its last layer `final_layer_linear` too, 1536 wide: not a Klein hidden size. Only a FLUX kohya file is
    read for its non-block layers."""
    keys = {**_kohya("joint_blocks_0_x_block_attn_qkv", 1536, 4608), **_kohya("final_layer_linear", 1536, 64)}

    assert _claimed_by(_state_dict(keys), tmp_path, folder=True) == ["LoRA_Diffusers_SD3_Config"]


def test_an_sd3_onetrainer_lora_with_top_level_layers_is_not_claimed_as_flux(tmp_path: Path) -> None:
    """SD3's top-level layers and double blocks are named like FLUX's, which OneTrainer's LEGACY_LORA keys keep; only
    FLUX has single blocks."""
    state_dict = _onetrainer_legacy_state_dict(_meta_sd3(), lambda name: True)

    assert "LoRA_LyCORIS_FLUX_Config" not in _claimed_by(state_dict, tmp_path)


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
