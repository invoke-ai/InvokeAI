import pytest
import torch
from diffusers.models.normalization import AdaLayerNormContinuous

from invokeai.backend.flux.modules.layers import LastLayer
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import (
    is_state_dict_likely_in_flux_diffusers_format,
    lora_model_from_flux_diffusers_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import (
    FLUX_LORA_CLIP_PREFIX,
    FLUX_LORA_T5_PREFIX,
    FLUX_LORA_TRANSFORMER_PREFIX,
)
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
from tests.backend.patches.lora_conversions.lora_state_dicts.flux_lora_kohya_format import (
    state_dict_keys as flux_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


@pytest.mark.parametrize(
    "sd_keys",
    [
        flux_diffusers_state_dict_keys,
        flux_diffusers_no_proj_mlp_state_dict_keys,
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


class _BflFinalLayer(torch.nn.Module):
    """The part of the BFL FLUX model that the final-layer LoRA keys resolve against."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.final_layer = LastLayer(hidden_size, patch_size=1, out_channels=out_channels)


@torch.no_grad()
def test_lora_model_from_flux_diffusers_state_dict_final_layer_modulation():
    """A LoRA on diffusers' norm_out.linear, which orders its output (scale, shift), must patch BFL's
    final_layer.adaLN_modulation.1, which orders it (shift, scale), so that the model computes what the LoRA does on
    the diffusers model (#10022).
    """
    torch.manual_seed(0)
    hidden_size, out_channels, rank = 8, 4, 2
    diffusers_norm_out = AdaLayerNormContinuous(hidden_size, hidden_size, elementwise_affine=False, eps=1e-6)
    bfl_model = _BflFinalLayer(hidden_size, out_channels)
    # The base weights as the FLUX loader converts them: diffusers' halves swapped.
    bfl_modulation = bfl_model.final_layer.adaLN_modulation[1]
    for name in ("weight", "bias"):
        diffusers_param = getattr(diffusers_norm_out.linear, name)
        getattr(bfl_modulation, name).copy_(torch.cat(diffusers_param.chunk(2, dim=0)[::-1], dim=0))

    def diffusers_forward(x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return bfl_model.final_layer.linear(diffusers_norm_out(x, vec))

    x = torch.randn(2, 3, hidden_size)
    vec = torch.randn(2, hidden_size)
    unpatched = bfl_model.final_layer(x, vec)
    assert torch.allclose(unpatched, diffusers_forward(x, vec), atol=1e-5)

    down = torch.randn(rank, hidden_size)
    up = torch.randn(2 * hidden_size, rank)
    state_dict = {"transformer.norm_out.linear.lora_A.weight": down, "transformer.norm_out.linear.lora_B.weight": up}
    patch = lora_model_from_flux_diffusers_state_dict(state_dict, alpha=None)
    with LayerPatcher.apply_smart_model_patches(
        model=bfl_model,
        patches=[(patch, 1.0)],
        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
        dtype=torch.float32,
        force_direct_patching=True,
    ):
        patched = bfl_model.final_layer(x, vec)

    diffusers_norm_out.linear.weight += up @ down
    expected = diffusers_forward(x, vec)
    assert not torch.allclose(expected, unpatched, atol=1e-2)
    assert torch.allclose(patched, expected, atol=1e-5)


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


def test_lora_model_from_flux_diffusers_state_dict_guidance_embedder():
    """Test that the two guidance-embedder layers map onto guidance_in.in_layer and guidance_in.out_layer, the same way
    the timestep and text embedders map onto time_in and vector_in, rather than both onto guidance_in.
    """
    rank = 4
    state_dict: dict[str, torch.Tensor] = {}
    for name, in_dim in (("linear_1", 256), ("linear_2", 3072)):
        prefix = f"transformer.time_text_embed.guidance_embedder.{name}"
        state_dict[f"{prefix}.lora_A.weight"] = torch.randn(rank, in_dim)
        state_dict[f"{prefix}.lora_B.weight"] = torch.randn(3072, rank)

    model = lora_model_from_flux_diffusers_state_dict(state_dict, alpha=None)

    assert set(model.layers.keys()) == {
        f"{FLUX_LORA_TRANSFORMER_PREFIX}guidance_in.in_layer",
        f"{FLUX_LORA_TRANSFORMER_PREFIX}guidance_in.out_layer",
    }

    # Each layer must produce a weight delta with the shape of the Linear it targets.
    in_layer = model.layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}guidance_in.in_layer"]
    out_layer = model.layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}guidance_in.out_layer"]
    assert in_layer.get_parameters({"weight": torch.empty(3072, 256)}, 1.0)["weight"].shape == (3072, 256)
    assert out_layer.get_parameters({"weight": torch.empty(3072, 3072)}, 1.0)["weight"].shape == (3072, 3072)


def test_lora_model_from_flux_diffusers_state_dict_with_text_encoder_layers():
    """Test that a diffusers-format FLUX LoRA that also trains the text encoders loads, routing the
    text_encoder.* keys to the CLIP prefix and text_encoder_2.* keys to the T5 prefix.
    """
    state_dict = keys_to_mock_state_dict(flux_diffusers_state_dict_keys)
    state_dict["text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight"] = torch.empty([4, 768])
    state_dict["text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight"] = torch.empty([768, 4])
    state_dict["text_encoder_2.encoder.block.0.layer.0.SelfAttention.q.lora_A.weight"] = torch.empty([4, 4096])
    state_dict["text_encoder_2.encoder.block.0.layer.0.SelfAttention.q.lora_B.weight"] = torch.empty([4096, 4])

    model = lora_model_from_flux_diffusers_state_dict(state_dict, alpha=8.0)

    assert any(k.startswith(FLUX_LORA_TRANSFORMER_PREFIX) for k in model.layers)
    assert f"{FLUX_LORA_CLIP_PREFIX}text_model.encoder.layers.0.self_attn.q_proj" in model.layers
    assert f"{FLUX_LORA_T5_PREFIX}encoder.block.0.layer.0.SelfAttention.q" in model.layers
