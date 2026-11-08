import accelerate
import pytest
import torch
from diffusers import Flux2Transformer2DModel
from diffusers.models.normalization import AdaLayerNormContinuous

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.modules.layers import LastLayer
from invokeai.backend.flux.util import get_flux_transformers_params
from invokeai.backend.model_manager.taxonomy import FluxVariantType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
)
from invokeai.backend.patches.lora_conversions.flux_kohya_lora_conversion_utils import (
    _convert_flux_transformer_kohya_state_dict_to_invoke_format,
    is_state_dict_likely_in_flux_kohya_format,
    lora_model_from_flux_kohya_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import (
    FLUX_LORA_CLIP_PREFIX,
    FLUX_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from tests.backend.patches.lora_conversions.lora_state_dicts.flux2_lora_kohya_non_block_format import (
    state_dict_keys as flux2_kohya_non_block_state_dict_keys,
)
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
from tests.backend.patches.lora_conversions.lora_state_dicts.qwen_image_lora_kohya_format import (
    state_dict_keys as qwen_image_kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict


@pytest.mark.parametrize("sd_keys", [flux_kohya_state_dict_keys, flux_kohya_te1_state_dict_keys])
def test_is_state_dict_likely_in_flux_kohya_format_true(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_in_flux_kohya_format() can identify a state dict in the Kohya FLUX LoRA format."""
    # Construct a state dict that is in the Kohya FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(sd_keys)

    assert is_state_dict_likely_in_flux_kohya_format(state_dict)


@pytest.mark.parametrize("sd_keys", [flux_diffusers_state_dict_keys, flux_onetrainer_state_dict_keys])
def test_is_state_dict_likely_in_flux_kohya_format_false(sd_keys: dict[str, list[int]]):
    """Test that is_state_dict_likely_in_flux_kohya_format() returns False for a state dict that is in the Diffusers
    FLUX LoRA format.
    """
    state_dict = keys_to_mock_state_dict(sd_keys)
    assert not is_state_dict_likely_in_flux_kohya_format(state_dict)


def test_convert_flux_transformer_kohya_state_dict_to_invoke_format():
    # Construct state_dict from state_dict_keys.
    state_dict = keys_to_mock_state_dict(flux_kohya_state_dict_keys)

    converted_state_dict = _convert_flux_transformer_kohya_state_dict_to_invoke_format(state_dict)

    # Extract the prefixes from the converted state dict (i.e. without the .lora_up.weight, .lora_down.weight, and
    # .alpha suffixes).
    converted_key_prefixes: list[str] = []
    for k in converted_state_dict.keys():
        k = k.replace(".lora_up.weight", "")
        k = k.replace(".lora_down.weight", "")
        k = k.replace(".alpha", "")
        converted_key_prefixes.append(k)

    # Initialize a FLUX model on the meta device.
    with accelerate.init_empty_weights():
        model = Flux(get_flux_transformers_params(FluxVariantType.Schnell))
    model_keys = set(model.state_dict().keys())

    # Assert that the converted state dict matches the keys in the actual model.
    for converted_key_prefix in converted_key_prefixes:
        found_match = False
        for model_key in model_keys:
            if model_key.startswith(converted_key_prefix):
                found_match = True
                break
        if not found_match:
            raise AssertionError(f"Could not find a match for the converted key prefix: {converted_key_prefix}")


def test_convert_flux_transformer_kohya_state_dict_to_invoke_format_error():
    """Test that an error is raised by _convert_flux_transformer_kohya_state_dict_to_invoke_format() if the input
    state_dict contains unexpected keys.
    """
    state_dict = {
        "unexpected_key.lora_up.weight": torch.empty(1),
    }

    with pytest.raises(ValueError):
        _convert_flux_transformer_kohya_state_dict_to_invoke_format(state_dict)


@pytest.mark.parametrize("sd_keys", [flux_kohya_state_dict_keys, flux_kohya_te1_state_dict_keys])
def test_lora_model_from_flux_kohya_state_dict(sd_keys: dict[str, list[int]]):
    """Test that a LoRAModelRaw can be created from a state dict in the Kohya FLUX LoRA format."""
    # Construct a state dict that is in the Kohya FLUX LoRA format.
    state_dict = keys_to_mock_state_dict(sd_keys)

    lora_model = lora_model_from_flux_kohya_state_dict(state_dict)

    # Prepare expected layer keys.
    expected_layer_keys: set[str] = set()
    for k in sd_keys:
        # Replace prefixes.
        k = k.replace("lora_unet_", FLUX_LORA_TRANSFORMER_PREFIX)
        k = k.replace("lora_te1_", FLUX_LORA_CLIP_PREFIX)
        # Remove suffixes.
        k = k.replace(".lora_up.weight", "")
        k = k.replace(".lora_down.weight", "")
        k = k.replace(".alpha", "")
        expected_layer_keys.add(k)

    # Assert that the lora_model has the expected layers.
    lora_model_keys = set(lora_model.layers.keys())
    lora_model_keys = {k.replace(".", "_") for k in lora_model_keys}
    assert lora_model_keys == expected_layer_keys


def _fixture_state_dict(keys: dict[str, tuple[list[int], str]]) -> dict[str, torch.Tensor]:
    # float32 whatever the file's dtype: CPU bf16 matmuls are not safe on every runner.
    return keys_to_mock_state_dict({k: shape for k, (shape, _dtype) in keys.items()})


def test_flux_kohya_format_with_layers_outside_the_blocks_is_detected():
    """A FLUX.2 kohya LoRA that trains `img_in`, `txt_in`, the time embedder, the last layer and the modulations is
    FLUX kohya, not unidentified (#10030)."""
    state_dict = _fixture_state_dict(flux2_kohya_non_block_state_dict_keys)

    assert is_state_dict_likely_in_flux_kohya_format(state_dict)


def test_flux_kohya_format_needs_a_block_next_to_the_layers_outside_the_blocks():
    """`img_in` and `txt_in` are also Qwen-Image and SD3 module names: alone they do not make a FLUX LoRA."""
    only_non_block = {
        k: v
        for k, v in _fixture_state_dict(flux2_kohya_non_block_state_dict_keys).items()
        if k.startswith(("lora_unet_img_in.", "lora_unet_txt_in."))
    }
    assert len(only_non_block) == 6

    assert not is_state_dict_likely_in_flux_kohya_format(only_non_block)
    assert not is_state_dict_likely_in_flux_kohya_format(keys_to_mock_state_dict(qwen_image_kohya_state_dict_keys))


def test_flux_kohya_format_rejects_an_unknown_layer_next_to_flux_blocks():
    state_dict = _fixture_state_dict(flux2_kohya_non_block_state_dict_keys)
    state_dict["lora_unet_final_layer_something_else.lora_down.weight"] = torch.empty(4, 4)

    assert not is_state_dict_likely_in_flux_kohya_format(state_dict)


def test_flux1_kohya_layers_outside_the_blocks_land_on_every_bfl_non_block_linear():
    """Every Linear of the FLUX.1 dev model outside its blocks, named as kohya names it (module path with the periods
    turned into underscores), converts to that module's path."""
    with accelerate.init_empty_weights():
        model = Flux(get_flux_transformers_params(FluxVariantType.Dev))
    non_block_linears = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and not name.startswith(("double_blocks", "single_blocks"))
    }
    assert {"img_in", "txt_in", "guidance_in.out_layer", "final_layer.adaLN_modulation.1"} <= set(non_block_linears)

    state_dict: dict[str, torch.Tensor] = {
        # One block layer, as the detector needs one.
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.empty(4, 3072),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.empty(3072, 4),
    }
    for name, module in non_block_linears.items():
        kohya_name = "lora_unet_" + name.replace(".", "_")
        state_dict[f"{kohya_name}.lora_down.weight"] = torch.empty(4, module.in_features)
        state_dict[f"{kohya_name}.lora_up.weight"] = torch.empty(module.out_features, 4)
    assert is_state_dict_likely_in_flux_kohya_format(state_dict)

    lora_model = lora_model_from_flux_kohya_state_dict(state_dict)

    assert set(lora_model.layers) == {
        FLUX_LORA_TRANSFORMER_PREFIX + name for name in [*non_block_linears, "double_blocks.0.img_attn.proj"]
    }


def test_flux2_kohya_lora_with_layers_outside_the_blocks_resolves_on_the_flux2_model():
    """The converted BFL names reach diffusers' Flux2 modules through the conversion the FLUX.2 denoiser applies, and
    every patch fits the module it lands on."""
    state_dict = _fixture_state_dict(flux2_kohya_non_block_state_dict_keys)
    with accelerate.init_empty_weights():
        # Klein 4B's widths, one block of each kind (the fixture holds block 0 of each).
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=128,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=7680,
        )

    patch = convert_bfl_lora_patch_to_diffusers(lora_model_from_flux_kohya_state_dict(state_dict))

    for key, layer in patch.layers.items():
        assert key.startswith(FLUX_LORA_TRANSFORMER_PREFIX)
        module = model.get_submodule(key.removeprefix(FLUX_LORA_TRANSFORMER_PREFIX))
        delta = layer.get_parameters({"weight": module.weight}, weight=1.0)["weight"]
        assert delta.shape == module.weight.shape, key
    non_block_targets = {k.removeprefix(FLUX_LORA_TRANSFORMER_PREFIX) for k in patch.layers if "blocks" not in k}
    assert non_block_targets == {
        "x_embedder",
        "context_embedder",
        "time_guidance_embed.timestep_embedder.linear_1",
        "time_guidance_embed.timestep_embedder.linear_2",
        "proj_out",
        "norm_out.linear",
        "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear",
    }


class _Flux2FinalLayer(torch.nn.Module):
    """The tail of Flux2Transformer2DModel: AdaLayerNormContinuous, then the output projection."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_out = AdaLayerNormContinuous(hidden_size, hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = torch.nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.norm_out(x, vec))


@torch.no_grad()
@pytest.mark.parametrize("flux_version", ["flux1", "flux2"])
def test_kohya_final_layer_lora_changes_the_output_as_it_does_on_the_bfl_layer(flux_version: str):
    """A kohya LoRA on the last layer applies with alpha / rank to BFL's (shift, scale) modulation halves. FLUX.2's
    `norm_out` orders the halves the other way, and must still give the BFL layer's output."""
    hidden, out_channels, rank, alpha = 8, 4, 2, 1.0
    torch.manual_seed(0)
    bfl_layer = LastLayer(hidden, patch_size=1, out_channels=out_channels)
    down = torch.randn(rank, hidden)
    up = torch.randn(2 * hidden, rank)
    linear_down = torch.randn(rank, hidden)
    linear_up = torch.randn(out_channels, rank)
    state_dict = {
        "lora_unet_final_layer_adaLN_modulation_1.lora_down.weight": down,
        "lora_unet_final_layer_adaLN_modulation_1.lora_up.weight": up,
        "lora_unet_final_layer_adaLN_modulation_1.alpha": torch.tensor(alpha),
        "lora_unet_final_layer_linear.lora_down.weight": linear_down,
        "lora_unet_final_layer_linear.lora_up.weight": linear_up,
        "lora_unet_final_layer_linear.alpha": torch.tensor(alpha),
        # A block layer, as the detector needs one.
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.randn(rank, hidden),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.randn(hidden, rank),
    }
    assert is_state_dict_likely_in_flux_kohya_format(state_dict)
    patch = lora_model_from_flux_kohya_state_dict(state_dict)
    prefix = FLUX_LORA_TRANSFORMER_PREFIX

    x = torch.randn(2, 3, hidden)
    vec = torch.randn(2, hidden)
    unpatched = bfl_layer(x, vec)
    if flux_version == "flux1":
        model: torch.nn.Module = bfl_layer
        layers = {
            f"{prefix}adaLN_modulation.1": patch.layers[f"{prefix}final_layer.adaLN_modulation.1"],
            f"{prefix}linear": patch.layers[f"{prefix}final_layer.linear"],
        }
    else:
        model = _Flux2FinalLayer(hidden, out_channels)
        # The base weights as the FLUX.2 loader converts them: BFL's modulation halves swapped.
        model.proj_out.load_state_dict(bfl_layer.linear.state_dict())
        for name in ("weight", "bias"):
            bfl_param = getattr(bfl_layer.adaLN_modulation[1], name)
            getattr(model.norm_out.linear, name).copy_(torch.cat(bfl_param.chunk(2, dim=0)[::-1], dim=0))
        assert torch.allclose(model(x, vec), unpatched, atol=1e-5)
        converted = convert_bfl_lora_patch_to_diffusers(patch)
        layers = {key: converted.layers[key] for key in (f"{prefix}norm_out.linear", f"{prefix}proj_out")}

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(ModelPatchRaw(layers=layers), 1.0)],
        prefix=prefix,
        dtype=torch.float32,
        force_direct_patching=True,
    ):
        patched = model(x, vec)

    # The LoRA by hand, on the BFL layer: W += up @ down * alpha / rank.
    bfl_layer.adaLN_modulation[1].weight += up @ down * (alpha / rank)
    bfl_layer.linear.weight += linear_up @ linear_down * (alpha / rank)
    expected = bfl_layer(x, vec)
    assert not torch.allclose(expected, unpatched, atol=1e-2)
    assert torch.allclose(patched, expected, atol=1e-4)


def test_flux2_dev_kohya_guidance_embedder_resolves_on_the_flux2_model():
    """FLUX.2 [dev] has a guidance embedder, which kohya names `guidance_in_*` (BFL's `guidance_in`)."""
    with accelerate.init_empty_weights():
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=16,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=True,
        )
    embedder = model.time_guidance_embed.guidance_embedder
    state_dict = {
        # A block layer, as the detector needs one.
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.empty(4, 16),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.empty(16, 4),
        "lora_unet_guidance_in_in_layer.lora_down.weight": torch.empty(4, embedder.linear_1.in_features),
        "lora_unet_guidance_in_in_layer.lora_up.weight": torch.empty(embedder.linear_1.out_features, 4),
        "lora_unet_guidance_in_out_layer.lora_down.weight": torch.empty(4, embedder.linear_2.in_features),
        "lora_unet_guidance_in_out_layer.lora_up.weight": torch.empty(embedder.linear_2.out_features, 4),
    }
    assert is_state_dict_likely_in_flux_kohya_format(state_dict)

    patch = convert_bfl_lora_patch_to_diffusers(lora_model_from_flux_kohya_state_dict(state_dict))

    for name, module in [
        ("time_guidance_embed.guidance_embedder.linear_1", embedder.linear_1),
        ("time_guidance_embed.guidance_embedder.linear_2", embedder.linear_2),
    ]:
        layer = patch.layers[FLUX_LORA_TRANSFORMER_PREFIX + name]
        assert layer.get_parameters({"weight": module.weight}, weight=1.0)["weight"].shape == module.weight.shape
