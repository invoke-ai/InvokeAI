import math

import accelerate
import pytest
import torch
from diffusers import Flux2Transformer2DModel

from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
    is_state_dict_likely_in_flux_bfl_peft_format,
    lora_model_from_flux2_bfl_peft_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_kohya_lora_conversion_utils import (
    lora_model_from_flux_kohya_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

_TINY_FLUX2_CONFIG = {
    "patch_size": 1,
    "in_channels": 128,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 8,
    "timestep_guidance_channels": 16,
    "axes_dims_rope": (2, 2, 2, 2),
}


@pytest.fixture(scope="module")
def flux2_transformer() -> Flux2Transformer2DModel:
    with accelerate.init_empty_weights():
        return Flux2Transformer2DModel(**_TINY_FLUX2_CONFIG)


def _tensor(shape: tuple[int, ...], start: float) -> torch.Tensor:
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    return values / max(values.numel(), 1) + start


def _lycoris_state_dict(
    variant: str, fused_out_features: int, in_features: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rank = 2
    alpha = torch.tensor(3.0)

    if variant == "lokr_w1_full":
        assert in_features % 4 == 0
        w1 = _tensor((3, 4), 0.25)
        w2_a = _tensor((fused_out_features // 3, rank), 0.5)
        w2_b = _tensor((rank, in_features // 4), 0.75)
        state_dict = {"lokr_w1": w1, "lokr_w2_a": w2_a, "lokr_w2_b": w2_b, "alpha": alpha}
        expected = torch.kron(w1, w2_a @ w2_b) * (alpha.item() / rank)
    elif variant == "lokr_w2_full":
        assert in_features % 4 == 0
        w1_a = _tensor((3, rank), 0.25)
        w1_b = _tensor((rank, 4), 0.5)
        w2 = _tensor((fused_out_features // 3, in_features // 4), 0.75)
        state_dict = {"lokr_w1_a": w1_a, "lokr_w1_b": w1_b, "lokr_w2": w2, "alpha": alpha}
        expected = torch.kron(w1_a @ w1_b, w2) * (alpha.item() / rank)
    elif variant == "loha":
        w1_a = _tensor((fused_out_features, rank), 0.25)
        w1_b = _tensor((rank, in_features), 0.5)
        w2_a = _tensor((fused_out_features, rank), 0.75)
        w2_b = _tensor((rank, in_features), 1.0)
        state_dict = {
            "hada_w1_a": w1_a,
            "hada_w1_b": w1_b,
            "hada_w2_a": w2_a,
            "hada_w2_b": w2_b,
            "alpha": alpha,
        }
        expected = ((w1_a @ w1_b) * (w2_a @ w2_b)) * (alpha.item() / rank)
    else:
        raise AssertionError(f"Unhandled test variant: {variant}")

    return state_dict, expected


def _convert_patch(source_format: str, bfl_layer: str, layer_state_dict: dict[str, torch.Tensor]) -> ModelPatchRaw:
    if source_format == "bfl":
        state_dict = {
            f"diffusion_model.double_blocks.0.{bfl_layer}.{suffix}": value for suffix, value in layer_state_dict.items()
        }
        assert is_state_dict_likely_in_flux_bfl_peft_format(state_dict)
        return lora_model_from_flux2_bfl_peft_state_dict(state_dict)

    if source_format == "kohya":
        kohya_layer = f"lora_unet_double_blocks_0_{bfl_layer.replace('.', '_')}"
        state_dict = {f"{kohya_layer}.{suffix}": value for suffix, value in layer_state_dict.items()}
        return convert_bfl_lora_patch_to_diffusers(lora_model_from_flux_kohya_state_dict(state_dict))

    raise AssertionError(f"Unhandled source format: {source_format}")


@pytest.mark.parametrize("source_format", ["bfl", "kohya"])
@pytest.mark.parametrize("variant", ["lokr_w1_full", "lokr_w2_full", "loha"])
@pytest.mark.parametrize(
    ("bfl_layer", "target_modules"),
    [
        (
            "img_attn.qkv",
            (
                "transformer_blocks.0.attn.to_q",
                "transformer_blocks.0.attn.to_k",
                "transformer_blocks.0.attn.to_v",
            ),
        ),
        (
            "txt_attn.qkv",
            (
                "transformer_blocks.0.attn.add_q_proj",
                "transformer_blocks.0.attn.add_k_proj",
                "transformer_blocks.0.attn.add_v_proj",
            ),
        ),
    ],
    ids=["image", "text"],
)
def test_flux2_lycoris_fused_qkv_resolves_and_preserves_update(
    source_format: str,
    variant: str,
    bfl_layer: str,
    target_modules: tuple[str, str, str],
    flux2_transformer: Flux2Transformer2DModel,
) -> None:
    target = flux2_transformer.get_submodule(target_modules[0])
    assert isinstance(target, torch.nn.Linear)
    fused_out_features = target.out_features * 3
    layer_state_dict, expected_weight = _lycoris_state_dict(variant, fused_out_features, target.in_features)

    patch = _convert_patch(source_format, bfl_layer, layer_state_dict)

    expected_keys = {f"{FLUX_LORA_TRANSFORMER_PREFIX}{module_name}" for module_name in target_modules}
    assert set(patch.layers) == expected_keys
    assert not any("qkv" in layer_key for layer_key in patch.layers)

    split_weights: list[torch.Tensor] = []
    input_vector = _tensor((target.in_features,), 0.125)
    split_outputs: list[torch.Tensor] = []
    for module_name in target_modules:
        module = flux2_transformer.get_submodule(module_name)
        layer = patch.layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{module_name}"]

        assert isinstance(module, torch.nn.Linear)
        expected_layer_type = LoHALayer if variant == "loha" else LoKRLayer
        assert isinstance(layer, expected_layer_type)
        assert layer.bias is None

        applied_parameters = layer.get_parameters({"weight": torch.empty(module.weight.shape)}, weight=1.0)
        assert set(applied_parameters) == {"weight"}
        split_weight = applied_parameters["weight"].detach()
        split_weights.append(split_weight)
        split_outputs.append(split_weight @ input_vector)

    actual_weight = torch.cat(split_weights, dim=0)
    assert torch.allclose(actual_weight, expected_weight)
    assert torch.allclose(torch.cat(split_outputs), expected_weight @ input_vector)
    assert (
        sum(layer.calc_size() for layer in patch.layers.values())
        < expected_weight.numel() * expected_weight.element_size()
    )


@pytest.mark.parametrize("source_format", ["bfl", "kohya"])
@pytest.mark.parametrize("variant", ["lokr", "loha"])
def test_klein_9b_qkv_split_keeps_factorized_storage(source_format: str, variant: str) -> None:
    hidden_size = 4096
    fused_out_features = hidden_size * 3
    rank = 4
    if variant == "lokr":
        # This is the mixed layout from the issue: w1 is full and w2 is factorized.
        layer_state_dict = {
            "lokr_w1": torch.empty((24, 8), device="meta"),
            "lokr_w2_a": torch.empty((512, rank), device="meta"),
            "lokr_w2_b": torch.empty((rank, 512), device="meta"),
            "alpha": torch.tensor(float(rank)),
        }
        expected_type = LoKRLayer
    else:
        layer_state_dict = {
            "hada_w1_a": torch.empty((fused_out_features, rank), device="meta"),
            "hada_w1_b": torch.empty((rank, hidden_size), device="meta"),
            "hada_w2_a": torch.empty((fused_out_features, rank), device="meta"),
            "hada_w2_b": torch.empty((rank, hidden_size), device="meta"),
            "alpha": torch.tensor(float(rank)),
        }
        expected_type = LoHALayer

    patch = _convert_patch(source_format, "img_attn.qkv", layer_state_dict)

    assert len(patch.layers) == 3
    assert all(isinstance(layer, expected_type) for layer in patch.layers.values())
    dense_update_bytes = fused_out_features * hidden_size * torch.empty((), dtype=torch.float32).element_size()
    assert sum(layer.calc_size() for layer in patch.layers.values()) < dense_update_bytes // 100


@pytest.mark.parametrize("source_format", ["bfl", "kohya"])
def test_non_aligned_lokr_uses_ranges_and_preserves_alpha_and_bias(source_format: str) -> None:
    w1 = _tensor((2, 2), 0.25)
    w2_a = _tensor((9, 2), 0.5)
    w2_b = _tensor((2, 3), 0.75)
    alpha = torch.tensor(3.0)
    fused_bias = _tensor((18,), 1.0)
    state_dict = {
        "lokr_w1": w1,
        "lokr_w2_a": w2_a,
        "lokr_w2_b": w2_b,
        "alpha": alpha,
        "bias_indices": torch.arange(18).unsqueeze(0),
        "bias_values": fused_bias,
        "bias_size": torch.tensor([18]),
    }
    expected_scale = alpha.item() / w2_b.shape[0]
    expected_weight = torch.kron(w1, w2_a @ w2_b) * expected_scale
    target_modules = (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
    )

    patch = _convert_patch(source_format, "img_attn.qkv", state_dict)

    split_layers = [patch.layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{key}"] for key in target_modules]
    assert isinstance(split_layers[0], LoKRLayer)
    assert isinstance(split_layers[1], MergedLayerPatch)
    assert isinstance(split_layers[2], LoKRLayer)

    actual_weights: list[torch.Tensor] = []
    actual_biases: list[torch.Tensor] = []
    for layer in split_layers:
        parameters = layer.get_parameters(
            {"weight": torch.empty((6, 6)), "bias": torch.empty(6)},
            weight=1.0,
        )
        actual_weights.append(parameters["weight"])
        actual_biases.append(parameters["bias"])

    assert torch.allclose(torch.cat(actual_weights), expected_weight)
    assert torch.allclose(torch.cat(actual_biases), fused_bias * expected_scale)
    assert sum(layer.calc_size() for layer in split_layers) < expected_weight.numel() * expected_weight.element_size()


@pytest.mark.parametrize("source_format", ["bfl", "kohya"])
def test_loha_split_preserves_alpha_and_bias(source_format: str) -> None:
    rank = 2
    alpha = torch.tensor(3.0)
    w1_a = _tensor((18, rank), 0.25)
    w1_b = _tensor((rank, 6), 0.5)
    w2_a = _tensor((18, rank), 0.75)
    w2_b = _tensor((rank, 6), 1.0)
    fused_bias = _tensor((18,), 1.25)
    state_dict = {
        "hada_w1_a": w1_a,
        "hada_w1_b": w1_b,
        "hada_w2_a": w2_a,
        "hada_w2_b": w2_b,
        "alpha": alpha,
        "bias_indices": torch.arange(18).unsqueeze(0),
        "bias_values": fused_bias,
        "bias_size": torch.tensor([18]),
    }
    expected_scale = alpha.item() / rank
    expected_weight = ((w1_a @ w1_b) * (w2_a @ w2_b)) * expected_scale

    patch = _convert_patch(source_format, "img_attn.qkv", state_dict)

    assert all(isinstance(layer, LoHALayer) for layer in patch.layers.values())
    actual_weights: list[torch.Tensor] = []
    actual_biases: list[torch.Tensor] = []
    for layer in patch.layers.values():
        parameters = layer.get_parameters(
            {"weight": torch.empty((6, 6)), "bias": torch.empty(6)},
            weight=1.0,
        )
        actual_weights.append(parameters["weight"])
        actual_biases.append(parameters["bias"])

    assert torch.allclose(torch.cat(actual_weights), expected_weight)
    assert torch.allclose(torch.cat(actual_biases), fused_bias * expected_scale)
