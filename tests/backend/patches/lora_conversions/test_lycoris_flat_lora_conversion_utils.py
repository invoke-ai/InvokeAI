"""Conversion of SimpleTuner's LyCORIS flat-key LoRAs (issue #10174) onto each family's modules.

A layer key that resolves to nothing is not an error at runtime: the patcher logs it and skips it, and the LoRA
quietly does less than it should. So every converted layer is resolved here against the model it will patch —
Invoke's BFL FLUX class for FLUX.1, and diffusers' models built from tiny configs whose module tree matches the
released ones for FLUX.2 Klein, Z-Image, Wan and SDXL (the SDXL UNet below carries the real
`stabilityai/stable-diffusion-xl-base-1.0` block structure at test sizes). The fixtures are the real key layouts
of the issue #10174 files, so each layer's weight delta is also checked against the (out, in) shape of the module
the real file patches, computed from its own LoKR factors.
"""

import torch

from invokeai.backend.flux.model import Flux, FluxParams
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.lycoris_flat_lora_conversion_utils import (
    lora_model_from_lycoris_flat_state_dict,
)
from invokeai.backend.patches.lora_conversions.wan_lora_constants import WAN_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from tests.backend.patches.lora_conversions.lora_state_dicts import (
    flux1_lycoris_flat_lokr_double_underscore_format,
    flux1_lycoris_flat_lokr_format,
    flux2_lycoris_flat_lokr_format,
    sdxl_lycoris_flat_lokr_format,
    wan_lycoris_flat_lokr_format,
    z_image_lycoris_flat_lokr_format,
)


def _shape(value) -> list[int]:
    return list(value[0]) if isinstance(value, tuple) else list(value)


def _state_dict(keys: dict) -> dict[str, torch.Tensor]:
    """The fixture's layout at its real shapes. LoKR's math runs on these, so they carry real storage."""
    return {key: torch.zeros(_shape(value)) for key, value in keys.items()}


def _layer_out_in(keys: dict, layer: str) -> tuple[int, int]:
    """The (out, in) shape of the weight a fixture layer patches, from its own LoKR factors."""
    w1, w2 = _shape(keys[f"{layer}.lokr_w1"]), _shape(keys[f"{layer}.lokr_w2"])
    return w1[0] * w2[0], w1[1] * w2[1]


def test_the_flux1_flat_layout_lands_on_the_bfl_modules() -> None:
    """toxicwind's file: the flattened diffusers names resolve to the same BFL keys the diffusers converter maps
    them to, with the Q/K/V projections merged into BFL's fused `qkv`."""
    keys = flux1_lycoris_flat_lokr_format.state_dict_keys
    patch = lora_model_from_lycoris_flat_state_dict(_state_dict(keys))
    prefix = FLUX_LORA_TRANSFORMER_PREFIX

    assert set(patch.layers) == {
        f"{prefix}double_blocks.0.img_attn.qkv",
        f"{prefix}double_blocks.0.img_attn.proj",
        f"{prefix}double_blocks.0.img_mlp.0",
        f"{prefix}double_blocks.0.img_mlp.2",
        f"{prefix}double_blocks.0.txt_attn.qkv",
        f"{prefix}double_blocks.0.txt_attn.proj",
        f"{prefix}double_blocks.0.txt_mlp.0",
        f"{prefix}double_blocks.0.txt_mlp.2",
        f"{prefix}single_blocks.0.linear1",
    }

    with torch.device("meta"):
        model = Flux(
            FluxParams(
                in_channels=3072,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=64,
                mlp_ratio=4.0,
                num_heads=2,
                depth=1,
                depth_single_blocks=1,
                axes_dim=[8, 12, 12],
                theta=10000,
                qkv_bias=True,
                guidance_embed=True,
                out_channels=64,
            )
        )
    for key in patch.layers:
        assert model.get_submodule(key.removeprefix(prefix)) is not None, key

    # Every layer's delta is the shape of the module the real file patches (FLUX.1: hidden 3072, MLP 12288).
    expected = {
        "double_blocks.0.img_attn.proj": (3072, 3072),
        "double_blocks.0.img_mlp.0": (12288, 3072),
        "double_blocks.0.img_mlp.2": (3072, 12288),
        "double_blocks.0.txt_attn.proj": (3072, 3072),
        "double_blocks.0.txt_mlp.0": (12288, 3072),
        "double_blocks.0.txt_mlp.2": (3072, 12288),
    }
    for name, shape in expected.items():
        assert patch.layers[f"{prefix}{name}"].get_weight(torch.empty(0)).shape == shape, name

    # The merged img qkv holds to_q, to_k and to_v, each 3072 wide; single_blocks.linear1 the same three.
    for name in ("double_blocks.0.img_attn.qkv", "double_blocks.0.txt_attn.qkv", "single_blocks.0.linear1"):
        merged = patch.layers[f"{prefix}{name}"]
        assert isinstance(merged, MergedLayerPatch) and len(merged.lora_layers) == 3, name
        for _sub, sub_range in zip(merged.lora_layers, merged.ranges, strict=True):
            assert sub_range.end - sub_range.start == 3072, name


def test_the_flux1_double_underscore_layout_also_lands_on_the_bfl_modules() -> None:
    """jimmycarter's `lycoris__` file: same mapping, plus the top-level modules and the swapped final modulation."""
    keys = flux1_lycoris_flat_lokr_double_underscore_format.state_dict_keys
    patch = lora_model_from_lycoris_flat_state_dict(_state_dict(keys))
    prefix = FLUX_LORA_TRANSFORMER_PREFIX

    assert set(patch.layers) == {
        f"{prefix}txt_in",
        f"{prefix}img_in",
        f"{prefix}final_layer.linear",
        f"{prefix}final_layer.adaLN_modulation.1",
        f"{prefix}time_in.in_layer",
        f"{prefix}time_in.out_layer",
        f"{prefix}vector_in.in_layer",
        f"{prefix}vector_in.out_layer",
        f"{prefix}guidance_in.in_layer",
        f"{prefix}guidance_in.out_layer",
        f"{prefix}single_blocks.0.linear1",
        f"{prefix}single_blocks.0.linear2",
        f"{prefix}single_blocks.0.modulation.lin",
        f"{prefix}double_blocks.0.img_attn.qkv",
        f"{prefix}double_blocks.0.img_attn.proj",
        f"{prefix}double_blocks.0.img_mlp.0",
        f"{prefix}double_blocks.0.img_mlp.2",
        f"{prefix}double_blocks.0.txt_attn.qkv",
        f"{prefix}double_blocks.0.txt_attn.proj",
        f"{prefix}double_blocks.0.txt_mlp.0",
        f"{prefix}double_blocks.0.txt_mlp.2",
        f"{prefix}double_blocks.0.img_mod.lin",
        f"{prefix}double_blocks.0.txt_mod.lin",
    }
    assert isinstance(patch.layers[f"{prefix}final_layer.adaLN_modulation.1"], SwappedHalvesLayerPatch)

    # Every layer's delta is the shape of the module the real file patches.
    expected = {
        "txt_in": (3072, 4096),
        "img_in": (3072, 64),
        "final_layer.linear": (64, 3072),
        "time_in.in_layer": (3072, 256),
        "time_in.out_layer": (3072, 3072),
        "vector_in.in_layer": (3072, 768),
        "vector_in.out_layer": (3072, 3072),
        "guidance_in.in_layer": (3072, 256),
        "guidance_in.out_layer": (3072, 3072),
        # BFL's linear2 consumes the single block's concatenated attention+MLP hidden (h + 4h).
        "single_blocks.0.linear2": (3072, 15360),
        "single_blocks.0.modulation.lin": (9216, 3072),
        "double_blocks.0.img_attn.proj": (3072, 3072),
        "double_blocks.0.img_mlp.0": (12288, 3072),
        "double_blocks.0.img_mlp.2": (3072, 12288),
        "double_blocks.0.txt_attn.proj": (3072, 3072),
        "double_blocks.0.txt_mlp.0": (12288, 3072),
        "double_blocks.0.txt_mlp.2": (3072, 12288),
        "double_blocks.0.img_mod.lin": (18432, 3072),
        "double_blocks.0.txt_mod.lin": (18432, 3072),
    }
    for name, shape in expected.items():
        assert patch.layers[f"{prefix}{name}"].get_weight(torch.empty(0)).shape == shape, name


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


def test_the_flux2_flat_layout_lands_on_the_flux2_transformer() -> None:
    from diffusers import Flux2Transformer2DModel

    keys = flux2_lycoris_flat_lokr_format.state_dict_keys
    patch = lora_model_from_lycoris_flat_state_dict(_state_dict(keys))
    prefix = FLUX_LORA_TRANSFORMER_PREFIX

    assert set(patch.layers) == {
        f"{prefix}single_transformer_blocks.0.attn.to_qkv_mlp_proj",
        f"{prefix}single_transformer_blocks.0.attn.to_out",
        f"{prefix}transformer_blocks.0.attn.to_q",
        f"{prefix}transformer_blocks.0.attn.to_k",
        f"{prefix}transformer_blocks.0.attn.to_v",
        f"{prefix}transformer_blocks.0.attn.to_out.0",
        f"{prefix}transformer_blocks.0.attn.add_q_proj",
        f"{prefix}transformer_blocks.0.attn.add_k_proj",
        f"{prefix}transformer_blocks.0.attn.add_v_proj",
        f"{prefix}transformer_blocks.0.attn.to_add_out",
        f"{prefix}transformer_blocks.0.ff.linear_in",
        f"{prefix}transformer_blocks.0.ff.linear_out",
        f"{prefix}transformer_blocks.0.ff_context.linear_in",
        f"{prefix}transformer_blocks.0.ff_context.linear_out",
    }

    with torch.device("meta"):
        model = Flux2Transformer2DModel(**_TINY_FLUX2_CONFIG)
    for key in patch.layers:
        assert isinstance(model.get_submodule(key.removeprefix(prefix)), torch.nn.Linear), key


def test_the_z_image_flat_layout_lands_on_the_z_image_transformer() -> None:
    from diffusers import ZImageTransformer2DModel

    keys = z_image_lycoris_flat_lokr_format.state_dict_keys
    patch = lora_model_from_lycoris_flat_state_dict(_state_dict(keys))
    prefix = Z_IMAGE_LORA_TRANSFORMER_PREFIX

    assert set(patch.layers) == {
        f"{prefix}layers.0.attention.to_q",
        f"{prefix}layers.0.attention.to_k",
        f"{prefix}layers.0.attention.to_v",
        f"{prefix}layers.0.attention.to_out.0",
        f"{prefix}layers.0.feed_forward.w1",
        f"{prefix}layers.0.feed_forward.w2",
        f"{prefix}layers.0.feed_forward.w3",
        f"{prefix}layers.0.adaLN_modulation.0",
    }

    with torch.device("meta"):
        model = ZImageTransformer2DModel(
            in_channels=4,
            dim=64,
            n_layers=1,
            n_refiner_layers=1,
            n_heads=2,
            n_kv_heads=2,
            cap_feat_dim=8,
            axes_dims=[8, 12, 12],
            axes_lens=[64, 32, 32],
        )
    for key in patch.layers:
        assert isinstance(model.get_submodule(key.removeprefix(prefix)), torch.nn.Linear), key


def test_the_wan_flat_layout_lands_on_the_wan_transformer() -> None:
    from diffusers import WanTransformer3DModel

    keys = wan_lycoris_flat_lokr_format.state_dict_keys
    patch = lora_model_from_lycoris_flat_state_dict(_state_dict(keys))
    prefix = WAN_LORA_TRANSFORMER_PREFIX

    assert set(patch.layers) == {
        f"{prefix}blocks.0.attn1.to_q",
        f"{prefix}blocks.0.attn1.to_k",
        f"{prefix}blocks.0.attn1.to_v",
        f"{prefix}blocks.0.attn1.to_out.0",
        f"{prefix}blocks.0.attn2.to_q",
        f"{prefix}blocks.0.attn2.to_k",
        f"{prefix}blocks.0.attn2.to_v",
        f"{prefix}blocks.0.attn2.to_out.0",
    }

    with torch.device("meta"):
        model = WanTransformer3DModel(
            num_layers=1, num_attention_heads=2, attention_head_dim=8, ffn_dim=16, text_dim=16, in_channels=4
        )
    for key in patch.layers:
        assert isinstance(model.get_submodule(key.removeprefix(prefix)), torch.nn.Linear), key


# The SDXL UNet's block structure, from `stabilityai/stable-diffusion-xl-base-1.0`'s unet config: the first down
# block and the last up block carry no attention, the deep attention blocks hold 10 transformer blocks. At test
# sizes, so only the names matter.
_SDXL_UNET_CONFIG = {
    "sample_size": 32,
    "in_channels": 4,
    "out_channels": 4,
    "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    "block_out_channels": (32, 64, 64),
    "layers_per_block": 2,
    "transformer_layers_per_block": (1, 2, 10),
    "attention_head_dim": (2, 2, 2),
    "cross_attention_dim": 16,
    "use_linear_projection": True,
    "norm_num_groups": 8,
}


def test_the_sdxl_flat_layout_lands_on_the_unet_by_flattened_name() -> None:
    """SD/SDXL patches keep kohya's flattened keys (`lora_unet_` + the name with `.` as `_`); the patcher resolves
    them against the UNet by the same flattening."""
    from diffusers import UNet2DConditionModel

    from invokeai.backend.patches.layer_patcher import LayerPatcher

    keys = sdxl_lycoris_flat_lokr_format.state_dict_keys
    patch = lora_model_from_lycoris_flat_state_dict(_state_dict(keys))

    expected = {
        "lora_unet_" + key.rsplit(".", 1)[0].removeprefix("lycoris_") for key in keys if key.endswith(".lokr_w1")
    }
    assert set(patch.layers) == expected

    with torch.device("meta"):
        unet = UNet2DConditionModel(**_SDXL_UNET_CONFIG)
    for key in patch.layers:
        _, module = LayerPatcher._get_submodule(unet, key.removeprefix("lora_unet_"), layer_key_is_flattened=True)
        assert isinstance(module, torch.nn.Linear), key


def test_a_flat_file_with_a_key_of_no_family_is_refused() -> None:
    keys = dict(flux1_lycoris_flat_lokr_format.state_dict_keys)
    keys["lycoris_some_module_nobody_has.lokr_w1"] = ([8, 8], "BF16")
    keys["lycoris_some_module_nobody_has.lokr_w2"] = ([384, 384], "BF16")

    try:
        lora_model_from_lycoris_flat_state_dict(_state_dict(keys))
    except ValueError as error:
        assert "some_module_nobody_has" in str(error)
    else:
        raise AssertionError("a layer that matches no module must be refused, not dropped")
