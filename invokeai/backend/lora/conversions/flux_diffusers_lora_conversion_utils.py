from typing import Dict

import torch

from invokeai.backend.lora.lora_model_raw import LoRAModelRaw


def convert_flux_transformer_checkpoint_to_diffusers(
    original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio=4.0
):
    converted_state_dict = {}

    ## time_text_embed.timestep_embedder <-  time_in
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
        "time_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
        "time_in.in_layer.bias"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
        "time_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
        "time_in.out_layer.bias"
    )

    ## time_text_embed.text_embedder <- vector_in
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop(
        "vector_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop(
        "vector_in.in_layer.bias"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop(
        "vector_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop(
        "vector_in.out_layer.bias"
    )

    # guidance
    has_guidance = any("guidance" in k for k in original_state_dict)
    if has_guidance:
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop(
            "guidance_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop(
            "guidance_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop(
            "guidance_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop(
            "guidance_in.out_layer.bias"
        )

    # context_embedder
    converted_state_dict["context_embedder.weight"] = original_state_dict.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = original_state_dict.pop("txt_in.bias")

    # x_embedder
    converted_state_dict["x_embedder.weight"] = original_state_dict.pop("img_in.weight")
    converted_state_dict["x_embedder.bias"] = original_state_dict.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms.
        ## norm1
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mod.lin.bias"
        )
        ## norm1_context
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mod.lin.bias"
        )
        # Q, K, V
        sample_q, sample_k, sample_v = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0
        )
        context_q, context_k, context_v = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
        )
        context_q_bias, context_k_bias, context_v_bias = torch.chunk(
            original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )
        # ff img_mlp
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_mlp.2.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear  <- single_blocks.0.modulation.lin
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        q, k, v, mlp = torch.split(original_state_dict.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        q_bias, k_bias, v_bias, mlp_bias = torch.split(
            original_state_dict.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.weight"
        )
        converted_state_dict[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(
            f"single_blocks.{i}.linear2.bias"
        )

    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        original_state_dict.pop("final_layer.adaLN_modulation.1.bias")
    )

    return converted_state_dict


def lora_model_from_flux_diffusers_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoRAModelRaw:
    # Group keys by layer.
    ...


# TODO(ryand): What alpha should we use? 1.0? Rank of the matrix?
def _convert_flux_diffusers_lora_state_dict_to_invoke_format(
    state_dict: Dict[str, torch.Tensor], alpha: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Converts a state dict from the Diffusers FLUX LoRA format to the format used internally by InvokeAI.

    This function is based on:
    https://github.com/huggingface/diffusers/blob/55ac421f7bb12fd00ccbef727be4dc2f3f920abb/scripts/convert_flux_to_diffusers.py
    """

    grouped_state_dict = _group_by_layer(state_dict)
    del state_dict

    num_double_layers = 19
    num_single_layers = 38

    original_state_dict = {}

    def convert_if_present(src_key: str, dst_key: str) -> None:
        if src_key in grouped_state_dict:
            src_layer_dict = grouped_state_dict.pop(src_key)
            original_state_dict[dst_key]["lora_down.weight"] = src_layer_dict["lora_A.weight"]
            original_state_dict[dst_key]["lora_up.weight"] = src_layer_dict["lora_B.weight"]
            original_state_dict[dst_key]["alpha"] = alpha

    def convert_qkv_if_present(src_q_key: str, src_k_key: str, src_v_key: str, dst_qkv_key: str) -> None:
        """Convert the Q, K, V matrices for a transformer block. We need special handling because the diffusers format
        stores them in separate matrices, whereas the BFL format used internally by InvokeAI concatenates them.
        """
        # Check for the presence of the q key to decide whether to convert this layer.
        # We assume that either all three keys are present or none of them are. If this assumption turns out to be
        # wrong, we will catch it at the end of the conversion process when we verify that all the keys have been
        # processed.
        if src_q_key not in grouped_state_dict:
            return

        src_q_layer_dict = grouped_state_dict.pop(src_q_key)
        src_k_layer_dict = grouped_state_dict.pop(src_k_key)
        src_v_layer_dict = grouped_state_dict.pop(src_v_key)

        original_state_dict[dst_qkv_key]["lora_down.weight"] = src_q_layer_dict["lora_A.weight"]
        original_state_dict[dst_key]["lora_up.weight"] = src_layer_dict["lora_B.weight"]
        original_state_dict[dst_key]["alpha"] = alpha

    # time_text_embed.timestep_embedder -> time_in.
    convert_if_present("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer")
    convert_if_present("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer")

    # time_text_embed.text_embedder -> vector_in.
    convert_if_present("time_text_embed.text_embedder.linear_1", "vector_in.in_layer")
    convert_if_present("time_text_embed.text_embedder.linear_2", "vector_in.out_layer")

    # time_text_embed.guidance_embedder -> guidance_in.
    convert_if_present("time_text_embed.guidance_embedder.linear_1", "guidance_in")
    convert_if_present("time_text_embed.guidance_embedder.linear_2", "guidance_in")

    # context_embedder -> txt_in.
    convert_if_present("context_embedder", "txt_in")

    # x_embedder -> img_in.
    convert_if_present("x_embedder", "img_in")

    # Double transformer blocks.
    for i in range(num_double_layers):
        # norms.
        convert_if_present(f"transformer_blocks.{i}.norm1.linear", f"double_blocks.{i}.img_mod.lin")
        convert_if_present(f"transformer_blocks.{i}.norm1_context.linear", f"double_blocks.{i}.txt_mod.lin")

        # Q, K, V
        # TODO(ryand): Implement the chunking/merging logic for the Q, K, V matrices.

        # >>> sd_k["lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight"].shape
        # torch.Size([9216, 16])
        # >>> sd_k["lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"].shape
        # torch.Size([16, 3072])
        # >>> 9216/3
        # 3072.0
        # >>> sd_d["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"].shape
        # torch.Size([32, 3072])
        # >>> sd_d["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"].shape
        # torch.Size([3072, 32])

    # Convert transformer blocks
    for key in list(state_dict.keys()):
        if key.startswith("transformer_blocks."):
            parts = key.split(".")
            block_num = int(parts[1])

            if "attn.to_q" in key:
                if "single_blocks.{}.linear1.weight".format(block_num) not in original_state_dict:
                    original_state_dict["single_blocks.{}.linear1.weight".format(block_num)] = torch.cat(
                        [
                            state_dict.pop("transformer_blocks.{}.attn.to_q.weight".format(block_num)),
                            state_dict.pop("transformer_blocks.{}.attn.to_k.weight".format(block_num)),
                            state_dict.pop("transformer_blocks.{}.attn.to_v.weight".format(block_num)),
                            state_dict.pop("transformer_blocks.{}.proj_mlp.weight".format(block_num)),
                        ]
                    )
                    original_state_dict["single_blocks.{}.linear1.bias".format(block_num)] = torch.cat(
                        [
                            state_dict.pop("transformer_blocks.{}.attn.to_q.bias".format(block_num)),
                            state_dict.pop("transformer_blocks.{}.attn.to_k.bias".format(block_num)),
                            state_dict.pop("transformer_blocks.{}.attn.to_v.bias".format(block_num)),
                            state_dict.pop("transformer_blocks.{}.proj_mlp.bias".format(block_num)),
                        ]
                    )
            elif "attn.norm_q.weight" in key:
                convert_if_present(key, "single_blocks.{}.norm.query_norm.scale".format(block_num))
            elif "attn.norm_k.weight" in key:
                convert_if_present(key, "single_blocks.{}.norm.key_norm.scale".format(block_num))
            elif "proj_out.weight" in key:
                convert_if_present(key, "single_blocks.{}.linear2.weight".format(block_num))
            elif "proj_out.bias" in key:
                convert_if_present(key, "single_blocks.{}.linear2.bias".format(block_num))

    # Convert final layer
    convert_if_present("proj_out.weight", "final_layer.linear.weight")
    convert_if_present("proj_out.bias", "final_layer.linear.bias")
    convert_if_present("norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight")
    convert_if_present("norm_out.linear.bias", "final_layer.adaLN_modulation.1.bias")

    assert len(grouped_state_dict) == 0
    return original_state_dict


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Groups the keys in the state dict by layer."""
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key in state_dict:
        parts = key.rsplit(".", maxsplit=2)
        layer_name = parts[0]
        key_name = ".".join(parts[1:])
        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][key_name] = state_dict[key]
    return layer_dict
