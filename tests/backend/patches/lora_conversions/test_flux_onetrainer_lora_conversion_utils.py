from typing import Callable

import accelerate
import pytest
import torch
from diffusers import QwenImageTransformer2DModel, SD3Transformer2DModel

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.util import get_flux_transformers_params
from invokeai.backend.model_manager.taxonomy import FluxLoRAFormat, FluxVariantType
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.lora_conversions.flux_lora_constants import (
    FLUX_LORA_CLIP_PREFIX,
    FLUX_LORA_T5_PREFIX,
    FLUX_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.flux_onetrainer_lora_conversion_utils import (
    is_state_dict_likely_in_flux_onetrainer_format,
    lora_model_from_flux_onetrainer_state_dict,
)
from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_dora_onetrainer_format import (
    state_dict_keys as flux_onetrainer_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_format import (
    state_dict_keys as flux_diffusers_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_kohya_format import (
    state_dict_keys as flux_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_kohya_with_te1_format import (
    state_dict_keys as flux_kohya_te1_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_onetrainer_non_block_format import (
    state_dict_keys as flux_onetrainer_non_block_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


def test_is_state_dict_likely_in_flux_onetrainer_format_true():
    """Test that is_state_dict_likely_in_flux_onetrainer_format() can identify a state dict in the OneTrainer
    FLUX LoRA format.
    """
    # Construct a state dict that is in the OneTrainer FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(flux_onetrainer_state_dict_keys)

    assert is_state_dict_likely_in_flux_onetrainer_format(state_dict)


@pytest.mark.parametrize(
    "sd_keys",
    [
        flux_kohya_state_dict_keys,
        flux_kohya_te1_state_dict_keys,
        flux_diffusers_state_dict_keys,
    ],
)
def test_is_state_dict_likely_in_flux_onetrainer_format_false(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_in_flux_onetrainer_format() returns False for a state dict that is in the Diffusers
    FLUX LoRA format.
    """
    state_dict = keys_to_mock_state_dict(sd_keys)
    assert not is_state_dict_likely_in_flux_onetrainer_format(state_dict)


def test_lora_model_from_flux_onetrainer_state_dict():
    state_dict = keys_to_mock_state_dict(flux_onetrainer_state_dict_keys)

    lora_model = lora_model_from_flux_onetrainer_state_dict(state_dict)

    # Check that the model has the correct number of LoRA layers.
    expected_lora_layers: set[str] = set()
    for k in flux_onetrainer_state_dict_keys:
        k = k.replace(".lora_up.weight", "")
        k = k.replace(".lora_down.weight", "")
        k = k.replace(".alpha", "")
        k = k.replace(".dora_scale", "")
        expected_lora_layers.add(k)
    # Drop the K/V/proj_mlp weights because these are all concatenated into a single layer in the BFL format (we keep
    # the Q weights so that we count these layers once).
    concatenated_weights = ["to_k", "to_v", "proj_mlp", "add_k_proj", "add_v_proj"]
    expected_lora_layers = {k for k in expected_lora_layers if not any(w in k for w in concatenated_weights)}

    assert len(lora_model.layers) == len(expected_lora_layers)

    # Check that all of the layers have the expected prefix.
    assert all(
        k.startswith((FLUX_LORA_TRANSFORMER_PREFIX, FLUX_LORA_CLIP_PREFIX, FLUX_LORA_T5_PREFIX))
        for k in lora_model.layers.keys()
    )


def _non_block_fixture_state_dict() -> dict[str, torch.Tensor]:
    return keys_to_mock_state_dict(
        {k: shape for k, (shape, _dtype) in flux_onetrainer_non_block_state_dict_keys.items()}
    )


def test_flux_onetrainer_format_with_layers_outside_the_blocks_is_detected():
    """A full-layer-filter OneTrainer FLUX LoRA, with `context_embedder` and the other non-block layers, is OneTrainer
    FLUX and not rejected for them (#10030)."""
    assert is_state_dict_likely_in_flux_onetrainer_format(_non_block_fixture_state_dict())


def test_flux_onetrainer_format_needs_a_block_next_to_the_layers_outside_the_blocks():
    """`x_embedder`, `context_embedder`, `proj_out` and `norm_out` are Qwen-Image, SD3 and Z-Image module names too:
    alone they do not make a FLUX LoRA."""
    state_dict = _non_block_fixture_state_dict()
    only_non_block = {k: v for k, v in state_dict.items() if "_blocks_" not in k}
    assert len(only_non_block) == 10 * 3  # ten layers, alpha and both factors each

    assert not is_state_dict_likely_in_flux_onetrainer_format(only_non_block)
    state_dict["lora_transformer_unknown_layer.lora_down.weight"] = torch.empty(4, 4)
    assert not is_state_dict_likely_in_flux_onetrainer_format(state_dict)


def test_flux_onetrainer_layers_outside_the_blocks_land_on_their_bfl_modules():
    """Each layer of the real file reaches the BFL module of its role, the shapes fit the FLUX.1 model, and the last
    layer's modulation, which BFL orders (shift, scale) against diffusers' (scale, shift), is swapped."""
    lora_model = lora_model_from_flux_onetrainer_state_dict(_non_block_fixture_state_dict())

    non_block = {
        k.removeprefix(FLUX_LORA_TRANSFORMER_PREFIX): v for k, v in lora_model.layers.items() if "blocks" not in k
    }
    assert set(non_block) == {
        "img_in",
        "txt_in",
        "time_in.in_layer",
        "time_in.out_layer",
        "vector_in.in_layer",
        "vector_in.out_layer",
        "guidance_in.in_layer",
        "guidance_in.out_layer",
        "final_layer.linear",
        "final_layer.adaLN_modulation.1",
    }
    assert isinstance(non_block["final_layer.adaLN_modulation.1"], SwappedHalvesLayerPatch)
    assert not any(
        isinstance(v, SwappedHalvesLayerPatch) for k, v in non_block.items() if k != "final_layer.adaLN_modulation.1"
    )

    with accelerate.init_empty_weights():
        model = Flux(get_flux_transformers_params(FluxVariantType.Dev))
    for name, layer in non_block.items():
        weight = model.get_submodule(name).weight
        assert layer.get_parameters({"weight": weight}, weight=1.0)["weight"].shape == weight.shape, name


def _onetrainer_legacy_state_dict(model: torch.nn.Module, keep: Callable[[str], bool]) -> dict[str, torch.Tensor]:
    """The keys OneTrainer's LEGACY_LORA output writes for a diffusers transformer: each Linear's module path with the
    periods turned into underscores under `lora_transformer_`."""
    state_dict: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and keep(name):
            key = "lora_transformer_" + name.replace(".", "_")
            state_dict[f"{key}.lora_down.weight"] = torch.empty(4, module.in_features)
            state_dict[f"{key}.lora_up.weight"] = torch.empty(module.out_features, 4)
    return state_dict


def _meta_sd3() -> torch.nn.Module:
    with accelerate.init_empty_weights():
        return SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            caption_projection_dim=16,
            pooled_projection_dim=16,
            out_channels=4,
            pos_embed_max_size=8,
            dual_attention_layers=(0,),
        )


def _meta_qwen_image() -> torch.nn.Module:
    with accelerate.init_empty_weights():
        return QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            axes_dims_rope=(2, 2, 4),
        )


_TOP_LEVEL = tuple(f"lora_transformer_{name}" for name in ("norm_out", "proj_out", "time_text_embed"))


@pytest.mark.parametrize("make_model", [_meta_sd3, _meta_qwen_image], ids=["sd3", "qwen-image"])
@pytest.mark.parametrize("scope", ["full-filter", "blocks-and-top-level"])
def test_sd3_and_qwen_image_onetrainer_files_with_top_level_layers_are_not_flux(make_model, scope: str):
    """SD3 and Qwen-Image name their top-level layers and double blocks like FLUX; they have no single blocks, which
    is what tells their OneTrainer files with those layers from FLUX's."""
    model = make_model()
    state_dict = _onetrainer_legacy_state_dict(model, lambda name: True)
    if scope == "blocks-and-top-level":
        state_dict = {k: v for k, v in state_dict.items() if "_blocks_" in k or k.startswith(_TOP_LEVEL)}
    assert any(k.startswith("lora_transformer_transformer_blocks_") for k in state_dict)
    assert any(k.startswith("lora_transformer_norm_out") for k in state_dict)

    assert not is_state_dict_likely_in_flux_onetrainer_format(state_dict)
    assert flux_format_from_state_dict(state_dict) is not FluxLoRAFormat.OneTrainer


def test_flux_onetrainer_fixture_is_the_onetrainer_format():
    assert flux_format_from_state_dict(_non_block_fixture_state_dict()) is FluxLoRAFormat.OneTrainer
