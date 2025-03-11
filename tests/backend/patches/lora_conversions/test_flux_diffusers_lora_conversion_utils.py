import pytest
import torch


from invokeai.backend.patches.layers.utils import swap_shift_scale_for_linear_weight
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import (
    is_state_dict_likely_in_flux_diffusers_format,
    lora_model_from_flux_diffusers_state_dict,
    approximate_flux_adaLN_lora_layer_from_diffusers_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_dora_onetrainer_format import (
    state_dict_keys as flux_onetrainer_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_base_model_format import (
    state_dict_keys as flux_diffusers_base_model_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_format import (
    state_dict_keys as flux_diffusers_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_no_proj_mlp_format import (
    state_dict_keys as flux_diffusers_no_proj_mlp_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_diffusers_with_norm_out_format import (
    state_dict_keys as flux_diffusers_with_norm_out_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_kohya_format import (
    state_dict_keys as flux_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


@pytest.mark.parametrize(
    "sd_keys",
    [
        flux_diffusers_state_dict_keys,
        flux_diffusers_no_proj_mlp_state_dict_keys,
        flux_diffusers_with_norm_out_state_dict_keys,
        flux_diffusers_base_model_state_dict_keys,
    ],
)
def test_is_state_dict_likely_in_flux_diffusers_format_true(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_in_flux_diffusers_format() can identify a state dict in the Diffusers FLUX LoRA format."""
    # Construct a state dict that is in the Diffusers FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(sd_keys)

    assert is_state_dict_likely_in_flux_diffusers_format(state_dict)


@pytest.mark.parametrize("sd_keys", [flux_kohya_state_dict_keys, flux_onetrainer_state_dict_keys])
def test_is_state_dict_likely_in_flux_diffusers_format_false(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_in_flux_diffusers_format() returns False for a state dict that is in the Kohya
    FLUX LoRA format.
    """
    # Construct a state dict that is not in the Kohya FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(sd_keys)

    assert not is_state_dict_likely_in_flux_diffusers_format(state_dict)


@pytest.mark.parametrize(
    "sd_keys",
    [
        flux_diffusers_state_dict_keys,
        flux_diffusers_no_proj_mlp_state_dict_keys,
        flux_diffusers_with_norm_out_state_dict_keys,
        flux_diffusers_base_model_state_dict_keys,
    ],
)
def test_lora_model_from_flux_diffusers_state_dict(sd_keys: dict[str, list[int]]):
    """Test that lora_model_from_flux_diffusers_state_dict() can load a state dict in the Diffusers FLUX LoRA format."""
    # Construct a state dict that is in the Diffusers FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(sd_keys)
    # Load the state dict into a LoRAModelRaw object.
    model = lora_model_from_flux_diffusers_state_dict(state_dict, alpha=8.0)

    # Check that the model has the correct number of LoRA layers.
    expected_lora_layers: set[str] = set()
    for k in sd_keys:
        k = k.replace("lora_A.weight", "")
        k = k.replace("lora_B.weight", "")
        expected_lora_layers.add(k)
    # Drop the K/V/proj_mlp weights because these are all concatenated into a single layer in the BFL format (we keep
    # the Q weights so that we count these layers once).
    concatenated_weights = ["to_k", "to_v", "proj_mlp", "add_k_proj", "add_v_proj"]
    expected_lora_layers = {k for k in expected_lora_layers if not any(w in k for w in concatenated_weights)}
    assert len(model.layers) == len(expected_lora_layers)
    assert all(k.startswith(FLUX_LORA_TRANSFORMER_PREFIX) for k in model.layers.keys())


def test_lora_model_from_flux_diffusers_state_dict_extra_keys_error():
    """Test that lora_model_from_flux_diffusers_state_dict() raises an error if the input state_dict contains unexpected
    keys that we don't handle.
    """
    # Construct a state dict that is in the Diffusers FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(flux_diffusers_state_dict_keys)
    # Add an unexpected key.
    state_dict["transformer.single_transformer_blocks.0.unexpected_key.lora_A.weight"] = torch.empty(1)

    # Check that an error is raised.
    with pytest.raises(AssertionError):
        lora_model_from_flux_diffusers_state_dict(state_dict, alpha=8.0)


@pytest.mark.parametrize("layer_sd_keys",[
    {}, # no keys
    {'lora_A.weight': [1024, 8], 'lora_B.weight': [8, 512]}, # wrong keys
    {'lora_up.weight': [1024, 8],}, # missing key
    {'lora_down.weight': [8, 512],}, # missing key
])
def test_approximate_adaLN_from_state_dict_should_only_accept_vanilla_LoRA_format(layer_sd_keys: dict[str, list[int]]):
    """Should only accept the valid state dict"""
    layer_state_dict = keys_to_mock_state_dict(layer_sd_keys)

    with pytest.raises(ValueError):
        approximate_flux_adaLN_lora_layer_from_diffusers_state_dict(layer_state_dict)


@pytest.mark.parametrize("dtype, rtol", [
   (torch.float32, 1e-4),
   (torch.half, 1e-3),
])
def test_approximate_adaLN_from_state_dict_should_work(dtype: torch.dtype, rtol: float, rate: float = 0.99):
    """Test that we should approximate good enough adaLN layer from diffusers state dict.
    This should tolorance some kind of errorness respect to input dtype"""
    input_dim = 1024
    output_dim = 512
    rank = 8  # Low rank
    total = input_dim * output_dim

    up = torch.randn(input_dim, rank, dtype=dtype)
    down = torch.randn(rank, output_dim, dtype=dtype)

    layer_state_dict = {
        'lora_up.weight': up,
        'lora_down.weight': down
    }

    # XXX Layer patcher cast things to f32
    original = up.float() @ down.float()
    swapped = swap_shift_scale_for_linear_weight(original)

    layer = approximate_flux_adaLN_lora_layer_from_diffusers_state_dict(layer_state_dict)
    weight = layer.get_weight(original).float()

    print(weight.dtype, swapped.dtype, layer.up.dtype)

    close_count = torch.isclose(weight, swapped, rtol=rtol).sum().item()
    close_rate = close_count / total

    assert close_rate > rate



