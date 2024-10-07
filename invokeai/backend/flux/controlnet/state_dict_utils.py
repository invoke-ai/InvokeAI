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


def _fuse_weights(*t: torch.Tensor) -> torch.Tensor:
    """Fuse weights along dimension 0.

    Used to fuse q, k, v attention weights into a single qkv tensor when converting from diffusers to BFL format.
    """
    # TODO(ryand): Double check dim=0 is correct.
    return torch.cat(t, dim=0)


def _convert_flux_double_block_sd_from_diffusers_to_bfl_format(
    sd: Dict[str, torch.Tensor], double_block_index: int
) -> Dict[str, torch.Tensor]:
    """Convert the state dict for a double block from diffusers format to BFL format."""
    to_prefix = f"double_blocks.{double_block_index}"
    from_prefix = f"transformer_blocks.{double_block_index}"

    new_sd: dict[str, torch.Tensor] = {}

    # Check one key to determine if this block exists.
    if f"{from_prefix}.attn.add_q_proj.bias" not in sd:
        return new_sd

    # txt_attn.qkv
    new_sd[f"{to_prefix}.txt_attn.qkv.bias"] = _fuse_weights(
        sd.pop(f"{from_prefix}.attn.add_q_proj.bias"),
        sd.pop(f"{from_prefix}.attn.add_k_proj.bias"),
        sd.pop(f"{from_prefix}.attn.add_v_proj.bias"),
    )
    new_sd[f"{to_prefix}.txt_attn.qkv.weight"] = _fuse_weights(
        sd.pop(f"{from_prefix}.attn.add_q_proj.weight"),
        sd.pop(f"{from_prefix}.attn.add_k_proj.weight"),
        sd.pop(f"{from_prefix}.attn.add_v_proj.weight"),
    )

    # img_attn.qkv
    new_sd[f"{to_prefix}.img_attn.qkv.bias"] = _fuse_weights(
        sd.pop(f"{from_prefix}.attn.to_q.bias"),
        sd.pop(f"{from_prefix}.attn.to_k.bias"),
        sd.pop(f"{from_prefix}.attn.to_v.bias"),
    )
    new_sd[f"{to_prefix}.img_attn.qkv.weight"] = _fuse_weights(
        sd.pop(f"{from_prefix}.attn.to_q.weight"),
        sd.pop(f"{from_prefix}.attn.to_k.weight"),
        sd.pop(f"{from_prefix}.attn.to_v.weight"),
    )

    # Handle basic 1-to-1 key conversions.
    key_map = {
        # img_attn
        "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
        "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
        "attn.to_out.0.weight": "img_attn.proj.weight",
        "attn.to_out.0.bias": "img_attn.proj.bias",
        # img_mlp
        "ff.net.0.proj.weight": "img_mlp.0.weight",
        "ff.net.0.proj.bias": "img_mlp.0.bias",
        "ff.net.2.weight": "img_mlp.2.weight",
        "ff.net.2.bias": "img_mlp.2.bias",
        # img_mod
        "norm1.linear.weight": "img_mod.lin.weight",
        "norm1.linear.bias": "img_mod.lin.bias",
        # txt_attn
        "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
        "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
        "attn.to_add_out.weight": "txt_attn.proj.weight",
        "attn.to_add_out.bias": "txt_attn.proj.bias",
        # txt_mlp
        "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
        "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
        "ff_context.net.2.weight": "txt_mlp.2.weight",
        "ff_context.net.2.bias": "txt_mlp.2.bias",
        # txt_mod
        "norm1_context.linear.weight": "txt_mod.lin.weight",
        "norm1_context.linear.bias": "txt_mod.lin.bias",
    }
    for from_key, to_key in key_map.items():
        new_sd[f"{to_prefix}.{to_key}"] = sd.pop(f"{from_prefix}.{from_key}")

    return new_sd


def _convert_flux_single_block_sd_from_diffusers_to_bfl_format(
    sd: Dict[str, torch.Tensor], single_block_index: int
) -> Dict[str, torch.Tensor]:
    """Convert the state dict for a single block from diffusers format to BFL format."""
    to_prefix = f"single_blocks.{single_block_index}"
    from_prefix = f"single_transformer_blocks.{single_block_index}"

    new_sd: dict[str, torch.Tensor] = {}

    # Check one key to determine if this block exists.
    if f"{from_prefix}.attn.to_q.bias" not in sd:
        return new_sd

    # linear1 (qkv)
    new_sd[f"{to_prefix}.linear1.bias"] = _fuse_weights(
        sd.pop(f"{from_prefix}.attn.to_q.bias"),
        sd.pop(f"{from_prefix}.attn.to_k.bias"),
        sd.pop(f"{from_prefix}.attn.to_v.bias"),
        sd.pop(f"{from_prefix}.proj_mlp.bias"),
    )
    new_sd[f"{to_prefix}.linear1.weight"] = _fuse_weights(
        sd.pop(f"{from_prefix}.attn.to_q.weight"),
        sd.pop(f"{from_prefix}.attn.to_k.weight"),
        sd.pop(f"{from_prefix}.attn.to_v.weight"),
        sd.pop(f"{from_prefix}.proj_mlp.weight"),
    )

    # Handle basic 1-to-1 key conversions.
    key_map = {
        # linear2
        "proj_out.weight": "linear2.weight",
        "proj_out.bias": "linear2.bias",
        # modulation
        "norm.linear.weight": "modulation.lin.weight",
        "norm.linear.bias": "modulation.lin.bias",
        # norm
        "attn.norm_k.weight": "norm.key_norm.scale",
        "attn.norm_q.weight": "norm.query_norm.scale",
    }
    for from_key, to_key in key_map.items():
        new_sd[f"{to_prefix}.{to_key}"] = sd.pop(f"{from_prefix}.{from_key}")

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
    block_index = 0
    while True:
        converted_singe_block_sd = _convert_flux_single_block_sd_from_diffusers_to_bfl_format(sd, block_index)
        if len(converted_singe_block_sd) == 0:
            break
        new_sd.update(converted_singe_block_sd)
        block_index += 1

    # Transfer controlnet keys as-is.
    for k in list(sd.keys()):
        if k.startswith("controlnet_"):
            new_sd[k] = sd.pop(k)

    # Assert that all keys have been handled.
    assert len(sd) == 0
    return new_sd
