"""Regression coverage for saved alpha on PEFT A/B weights (#10019)."""

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.anima_lora_constants import ANIMA_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.anima_lora_conversion_utils import lora_model_from_anima_state_dict
from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
    lora_model_from_qwen_image_state_dict,
)
from invokeai.backend.patches.lora_conversions.z_image_lora_conversion_utils import lora_model_from_z_image_state_dict
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear


@pytest.fixture(
    params=[
        pytest.param(
            (
                lora_model_from_z_image_state_dict,
                "diffusion_model.layers.0.attention.to_q",
                "lora_transformer-layers.0.attention.to_q",
            ),
            id="z-image",
        ),
        pytest.param(
            (
                lora_model_from_qwen_image_state_dict,
                "transformer_blocks.0.attn.to_q",
                "lora_transformer-transformer_blocks.0.attn.to_q",
            ),
            id="qwen-image",
        ),
        pytest.param(
            (
                lora_model_from_anima_state_dict,
                "diffusion_model.blocks.0.self_attn.q_proj",
                "lora_transformer-blocks.0.self_attn.q_proj",
            ),
            id="anima",
        ),
    ],
)
def converter_and_paths(request):
    return request.param


@pytest.mark.parametrize(
    ("weight_keys", "file_alpha", "fallback", "expected_alpha"),
    [
        pytest.param(("lora_A", "lora_B"), 1.0, None, 1.0, id="file-alpha-1-rank-16"),
        pytest.param(("lora_A", "lora_B"), 1.0 + 2**-30, 8.0, 1.0 + 2**-30, id="file-alpha-precedence-and-precision"),
        pytest.param(("lora_A", "lora_B"), 0.0, 8.0, 0.0, id="file-alpha-zero"),
        pytest.param(("lora_A", "lora_B"), None, 8.0, 8.0, id="caller-alpha-fallback"),
        pytest.param(("lora_A", "lora_B"), None, None, 16.0, id="rank-default"),
        pytest.param(("lora_down", "lora_up"), 2.0, 8.0, 2.0, id="internal-file-alpha"),
        pytest.param(("lora_down", "lora_up"), None, 8.0, 16.0, id="internal-rank-default"),
    ],
)
def test_lora_patch_applies_layer_alpha(converter_and_paths, weight_keys, file_alpha, fallback, expected_alpha):
    convert, source, target = converter_and_paths
    down_key, up_key = weight_keys
    state_dict = {
        f"{source}.{down_key}.weight": torch.ones(16, 20, dtype=torch.float64),
        f"{source}.{up_key}.weight": torch.ones(18, 16, dtype=torch.float64),
    }
    if file_alpha is not None:
        state_dict[f"{source}.alpha"] = torch.tensor(file_alpha, dtype=torch.float64)

    layer = convert(state_dict, alpha=fallback).layers[target]
    assert isinstance(layer, LoRALayer)
    original = torch.zeros(18, 20, dtype=torch.float64)
    actual = layer.get_parameters({"weight": original}, weight=0.5)["weight"]

    # Each entry of B @ A is the rank, so alpha/rank cancels the rank.
    torch.testing.assert_close(actual, torch.full_like(original, expected_alpha / 2), rtol=0, atol=0)


def test_incomplete_peft_pair_still_raises(converter_and_paths):
    convert, source, _ = converter_and_paths
    with pytest.raises(KeyError, match=r"lora_B\.weight"):
        convert({f"{source}.lora_A.weight": torch.ones(16, 20)})


def test_saved_alpha_scales_existing_lora_sidecar_on_packed_int8():
    in_features, out_features, rank = 16, 12, 16
    packed_weight = torch.full((out_features, in_features), 2, dtype=torch.int8)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].self_attn = torch.nn.Module()
    model.blocks[0].self_attn.q_proj = Int8ConvrotLinear(
        weight=packed_weight, weight_scale=torch.full((out_features, 1), 0.25), convrot=False
    )
    apply_custom_layers_to_model(model)
    projection = model.blocks[0].self_attn.q_proj

    source = "diffusion_model.blocks.0.self_attn.q_proj"
    patch = lora_model_from_anima_state_dict(
        {
            f"{source}.lora_A.weight": torch.ones(rank, in_features),
            f"{source}.lora_B.weight": torch.ones(out_features, rank),
            f"{source}.alpha": torch.tensor(1),
        }
    )
    x = torch.ones(1, in_features)
    original = projection(x)
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(patch, 0.5)],
        prefix=ANIMA_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_sidecar_patching=True,
    ):
        actual_delta = projection(x) - original
        assert projection.weight.dtype == torch.int8

    torch.testing.assert_close(projection(x), original)
    # All entries of B @ A equal rank. With alpha 1 and strength 0.5, each input contributes 0.5.
    torch.testing.assert_close(actual_delta, torch.full_like(original, in_features * 0.5))
