from typing import Any, Dict

import torch


def is_state_dict_xlabs_controlnet(sd: Dict[str, Any]) -> bool:
    """Is the state dict for an XLabs ControlNet model?

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision.
    """
    # If all of the expected keys are present, then this is very likely an XLabs ControlNet model.
    expected_keys = {
        "controlnet_blocks.0.bias",
        "controlnet_blocks.0.weight",
        "input_hint_block.0.bias",
        "input_hint_block.0.weight",
        "pos_embed_input.bias",
        "pos_embed_input.weight",
    }

    if expected_keys.issubset(sd.keys()):
        return True
    return False


def is_state_dict_instantx_controlnet(sd: Dict[str, Any]) -> bool:
    """Is the state dict for an InstantX ControlNet model?

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision.
    """
    # If all of the expected keys are present, then this is very likely an InstantX ControlNet model.
    expected_keys = {
        "controlnet_blocks.0.bias",
        "controlnet_blocks.0.weight",
        "controlnet_single_blocks.0.bias",
        "controlnet_single_blocks.0.weight",
        "controlnet_x_embedder.bias",
        "controlnet_x_embedder.weight",
    }

    if expected_keys.issubset(sd.keys()):
        return True
    return False


def _fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # TODO(ryand): Double check dim=0 is correct.
    return torch.cat((q, k, v), dim=0)


def _convert_flux_double_block_sd_from_diffusers_to_bfl_format(
    sd: Dict[str, torch.Tensor], double_block_index: int
) -> Dict[str, torch.Tensor]:
    """Convert the state dict for a double block from diffusers format to BFL format."""

    # double_blocks.0.img_attn.norm.key_norm.scale
    # double_blocks.0.img_attn.norm.query_norm.scale
    # double_blocks.0.img_attn.proj.bias
    # double_blocks.0.img_attn.proj.weight
    # double_blocks.0.img_attn.qkv.bias
    # double_blocks.0.img_attn.qkv.weight
    # double_blocks.0.img_mlp.0.bias
    # double_blocks.0.img_mlp.0.weight
    # double_blocks.0.img_mlp.2.bias
    # double_blocks.0.img_mlp.2.weight
    # double_blocks.0.img_mod.lin.bias
    # double_blocks.0.img_mod.lin.weight
    # double_blocks.0.txt_attn.norm.key_norm.scale
    # double_blocks.0.txt_attn.norm.query_norm.scale
    # double_blocks.0.txt_attn.proj.bias
    # double_blocks.0.txt_attn.proj.weight
    # double_blocks.0.txt_attn.qkv.bias
    # double_blocks.0.txt_attn.qkv.weight
    # double_blocks.0.txt_mlp.0.bias
    # double_blocks.0.txt_mlp.0.weight
    # double_blocks.0.txt_mlp.2.bias
    # double_blocks.0.txt_mlp.2.weight
    # double_blocks.0.txt_mod.lin.bias
    # double_blocks.0.txt_mod.lin.weight

    # "transformer_blocks.0.attn.add_k_proj.bias",
    # "transformer_blocks.0.attn.add_k_proj.weight",
    # "transformer_blocks.0.attn.add_q_proj.bias",
    # "transformer_blocks.0.attn.add_q_proj.weight",
    # "transformer_blocks.0.attn.add_v_proj.bias",
    # "transformer_blocks.0.attn.add_v_proj.weight",
    # "transformer_blocks.0.attn.norm_added_k.weight",
    # "transformer_blocks.0.attn.norm_added_q.weight",
    # "transformer_blocks.0.attn.norm_k.weight",
    # "transformer_blocks.0.attn.norm_q.weight",
    # "transformer_blocks.0.attn.to_add_out.bias",
    # "transformer_blocks.0.attn.to_add_out.weight",
    # "transformer_blocks.0.attn.to_k.bias",
    # "transformer_blocks.0.attn.to_k.weight",
    # "transformer_blocks.0.attn.to_out.0.bias",
    # "transformer_blocks.0.attn.to_out.0.weight",
    # "transformer_blocks.0.attn.to_q.bias",
    # "transformer_blocks.0.attn.to_q.weight",
    # "transformer_blocks.0.attn.to_v.bias",
    # "transformer_blocks.0.attn.to_v.weight",
    # "transformer_blocks.0.ff.net.0.proj.bias",
    # "transformer_blocks.0.ff.net.0.proj.weight",
    # "transformer_blocks.0.ff.net.2.bias",
    # "transformer_blocks.0.ff.net.2.weight",
    # "transformer_blocks.0.ff_context.net.0.proj.bias",
    # "transformer_blocks.0.ff_context.net.0.proj.weight",
    # "transformer_blocks.0.ff_context.net.2.bias",
    # "transformer_blocks.0.ff_context.net.2.weight",
    # "transformer_blocks.0.norm1.linear.bias",
    # "transformer_blocks.0.norm1.linear.weight",
    # "transformer_blocks.0.norm1_context.linear.bias",
    # "transformer_blocks.0.norm1_context.linear.weight",

    new_sd: dict[str, torch.Tensor] = {}

    new_sd[f"double_blocks.{double_block_index}.txt_attn.qkv.bias"] = _fuse_qkv(
        sd.pop(f"transformer_blocks.{double_block_index}.attn.add_q_proj.bias"),
        sd.pop(f"transformer_blocks.{double_block_index}.attn.add_k_proj.bias"),
        sd.pop(f"transformer_blocks.{double_block_index}.attn.add_v_proj.bias"),
    )
    new_sd[f"double_blocks.{double_block_index}.txt_attn.qkv.weight"] = _fuse_qkv(
        sd.pop(f"transformer_blocks.{double_block_index}.attn.add_q_proj.weight"),
        sd.pop(f"transformer_blocks.{double_block_index}.attn.add_k_proj.weight"),
        sd.pop(f"transformer_blocks.{double_block_index}.attn.add_v_proj.weight"),
    )

    return new_sd


def convert_diffusers_instantx_state_dict_to_bfl_format(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert an InstantX ControlNet state dict to the format that can be loaded by our internal
    DiffusersControlNetFlux.

    The original InstantX ControlNet model was developed to be used in diffusers. We have ported the original
    implementation to DiffusersControlNetFlux to make it compatible with BFL-style models. This function converts the
    original state dict to the format expected by DiffusersControlNetFlux.
    """
    # Shallow copy sd so that we can pop keys from it without modifying the original.
    sd = sd.copy()

    new_sd: dict[str, torch.Tensor] = {}

    # Handle basic 1-to-1 key conversions.
    basic_key_map = {
        # Base model keys.
        # ----------------
        # txt_in keys.
        "context_embedder.bias": "txt_in.bias",
        "context_embedder.weight": "txt_in.weight",
        # guidance_in MLPEmbedder keys.
        "time_text_embed.guidance_embedder.linear_1.bias": "guidance_in.in_layer.bias",
        "time_text_embed.guidance_embedder.linear_1.weight": "guidance_in.in_layer.weight",
        "time_text_embed.guidance_embedder.linear_2.bias": "guidance_in.out_layer.bias",
        "time_text_embed.guidance_embedder.linear_2.weight": "guidance_in.out_layer.weight",
        # vector_in MLPEmbedder keys.
        "time_text_embed.text_embedder.linear_1.bias": "vector_in.in_layer.bias",
        "time_text_embed.text_embedder.linear_1.weight": "vector_in.in_layer.weight",
        "time_text_embed.text_embedder.linear_2.bias": "vector_in.out_layer.bias",
        "time_text_embed.text_embedder.linear_2.weight": "vector_in.out_layer.weight",
        # time_in MLPEmbedder keys.
        "time_text_embed.timestep_embedder.linear_1.bias": "time_in.in_layer.bias",
        "time_text_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
        "time_text_embed.timestep_embedder.linear_2.bias": "time_in.out_layer.bias",
        "time_text_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        # img_in keys.
        "x_embedder.bias": "img_in.bias",
        "x_embedder.weight": "img_in.weight",
    }
    for old_key, new_key in basic_key_map.items():
        v = sd.pop(old_key, None)
        if v is not None:
            new_sd[new_key] = v

    # Handle the double_blocks.
    block_index = 0
    while True:
        converted_double_block_sd = _convert_flux_double_block_sd_from_diffusers_to_bfl_format(sd, block_index)
        if len(converted_double_block_sd) == 0:
            break
        new_sd.update(converted_double_block_sd)
        block_index += 1

    # Handle the single_blocks.
    ...

    # Transfer controlnet keys as-is.
    for k in sd:
        if k.startswith("controlnet_"):
            new_sd[k] = sd[k]

    # Assert that all keys have been handled.
    assert len(sd) == 0
    return new_sd
