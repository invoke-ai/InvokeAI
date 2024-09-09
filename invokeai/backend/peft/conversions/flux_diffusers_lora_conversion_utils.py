from typing import Dict

import torch

from invokeai.backend.peft.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.peft.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.peft.layers.lora_layer import LoRALayer
from invokeai.backend.peft.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.peft.lora import LoRAModelRaw

# def convert_flux_transformer_checkpoint_to_diffusers(
#     original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio=4.0
# ):
#     converted_state_dict = {}

#     ## time_text_embed.timestep_embedder <-  time_in
#     converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
#         "time_in.in_layer.weight"
#     )
#     converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
#         "time_in.in_layer.bias"
#     )
#     converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
#         "time_in.out_layer.weight"
#     )
#     converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
#         "time_in.out_layer.bias"
#     )

#     ## time_text_embed.text_embedder <- vector_in
#     converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop(
#         "vector_in.in_layer.weight"
#     )
#     converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop(
#         "vector_in.in_layer.bias"
#     )
#     converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop(
#         "vector_in.out_layer.weight"
#     )
#     converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop(
#         "vector_in.out_layer.bias"
#     )

#     # guidance
#     has_guidance = any("guidance" in k for k in original_state_dict)
#     if has_guidance:
#         converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop(
#             "guidance_in.in_layer.weight"
#         )
#         converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop(
#             "guidance_in.in_layer.bias"
#         )
#         converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop(
#             "guidance_in.out_layer.weight"
#         )
#         converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop(
#             "guidance_in.out_layer.bias"
#         )

#     # context_embedder
#     converted_state_dict["context_embedder.weight"] = original_state_dict.pop("txt_in.weight")
#     converted_state_dict["context_embedder.bias"] = original_state_dict.pop("txt_in.bias")

#     # x_embedder
#     converted_state_dict["x_embedder.weight"] = original_state_dict.pop("img_in.weight")
#     converted_state_dict["x_embedder.bias"] = original_state_dict.pop("img_in.bias")

#     # double transformer blocks
#     for i in range(num_layers):
#         block_prefix = f"transformer_blocks.{i}."
#         # norms.
#         ## norm1
#         converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_mod.lin.weight"
#         )
#         converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_mod.lin.bias"
#         )
#         ## norm1_context
#         converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_mod.lin.weight"
#         )
#         converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_mod.lin.bias"
#         )
#         # Q, K, V
#         sample_q, sample_k, sample_v = torch.chunk(
#             original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0
#         )
#         context_q, context_k, context_v = torch.chunk(
#             original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
#         )
#         sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
#             original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
#         )
#         context_q_bias, context_k_bias, context_v_bias = torch.chunk(
#             original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
#         )
#         converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
#         converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
#         converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
#         converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
#         converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
#         converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
#         converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
#         converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
#         converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
#         converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
#         converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
#         converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
#         # qk_norm
#         converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_attn.norm.query_norm.scale"
#         )
#         converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_attn.norm.key_norm.scale"
#         )
#         converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
#         )
#         converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
#         )
#         # ff img_mlp
#         converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_mlp.0.weight"
#         )
#         converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_mlp.0.bias"
#         )
#         converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_mlp.2.weight"
#         )
#         converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_mlp.2.bias"
#         )
#         converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_mlp.0.weight"
#         )
#         converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_mlp.0.bias"
#         )
#         converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_mlp.2.weight"
#         )
#         converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_mlp.2.bias"
#         )
#         # output projections.
#         converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_attn.proj.weight"
#         )
#         converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.img_attn.proj.bias"
#         )
#         converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_attn.proj.weight"
#         )
#         converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
#             f"double_blocks.{i}.txt_attn.proj.bias"
#         )

#     # single transfomer blocks
#     for i in range(num_single_layers):
#         block_prefix = f"single_transformer_blocks.{i}."
#         # norm.linear  <- single_blocks.0.modulation.lin
#         converted_state_dict[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(
#             f"single_blocks.{i}.modulation.lin.weight"
#         )
#         converted_state_dict[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(
#             f"single_blocks.{i}.modulation.lin.bias"
#         )
#         # Q, K, V, mlp
#         mlp_hidden_dim = int(inner_dim * mlp_ratio)
#         split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
#         q, k, v, mlp = torch.split(original_state_dict.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
#         q_bias, k_bias, v_bias, mlp_bias = torch.split(
#             original_state_dict.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
#         )
#         converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
#         converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
#         converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
#         converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
#         converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
#         converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
#         converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
#         converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
#         # qk norm
#         converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
#             f"single_blocks.{i}.norm.query_norm.scale"
#         )
#         converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
#             f"single_blocks.{i}.norm.key_norm.scale"
#         )
#         # output projections.
#         converted_state_dict[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(
#             f"single_blocks.{i}.linear2.weight"
#         )
#         converted_state_dict[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(
#             f"single_blocks.{i}.linear2.bias"
#         )

#     converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
#     converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
#     converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
#         original_state_dict.pop("final_layer.adaLN_modulation.1.weight")
#     )
#     converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
#         original_state_dict.pop("final_layer.adaLN_modulation.1.bias")
#     )

#     return converted_state_dict


# TODO(ryand): What alpha should we use? 1.0? Rank of the matrix?
def lora_model_from_flux_diffusers_state_dict(state_dict: Dict[str, torch.Tensor], alpha: float = 1.0) -> LoRAModelRaw:  # pyright: ignore[reportRedeclaration] (state_dict is intentionally re-declared)
    """Loads a state dict in the Diffusers FLUX LoRA format into a LoRAModelRaw object.

    This function is based on:
    https://github.com/huggingface/diffusers/blob/55ac421f7bb12fd00ccbef727be4dc2f3f920abb/scripts/convert_flux_to_diffusers.py
    """
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = _group_by_layer(state_dict)

    # Constants for FLUX.1
    num_double_layers = 19
    num_single_layers = 38
    # inner_dim = 3072
    # mlp_ratio = 4.0

    layers: dict[str, AnyLoRALayer] = {}

    def add_lora_layer_if_present(src_key: str, dst_key: str) -> None:
        if src_key in grouped_state_dict:
            src_layer_dict = grouped_state_dict.pop(src_key)
            layers[dst_key] = LoRALayer(
                dst_key,
                {
                    "lora_down.weight": src_layer_dict.pop("lora_A.weight"),
                    "lora_up.weight": src_layer_dict.pop("lora_B.weight"),
                    "alpha": torch.tensor(alpha),
                },
            )
            assert len(src_layer_dict) == 0

    def add_qkv_lora_layer_if_present(src_keys: list[str], dst_qkv_key: str) -> None:
        """Handle the Q, K, V matrices for a transformer block. We need special handling because the diffusers format
        stores them in separate matrices, whereas the BFL format used internally by InvokeAI concatenates them.
        """
        # We expect that either all src keys are present or none of them are. Verify this.
        keys_present = [key in grouped_state_dict for key in src_keys]
        assert all(keys_present) or not any(keys_present)

        # If none of the keys are present, return early.
        if not any(keys_present):
            return

        src_layer_dicts = [grouped_state_dict.pop(key) for key in src_keys]
        sub_layers: list[LoRALayerBase] = []
        for src_layer_dict in src_layer_dicts:
            sub_layers.append(
                LoRALayer(
                    layer_key="",
                    values={
                        "lora_down.weight": src_layer_dict.pop("lora_A.weight"),
                        "lora_up.weight": src_layer_dict.pop("lora_B.weight"),
                        "alpha": torch.tensor(alpha),
                    },
                )
            )
            assert len(src_layer_dict) == 0
        layers[dst_qkv_key] = ConcatenatedLoRALayer(layer_key=dst_qkv_key, lora_layers=sub_layers, concat_axis=0)

    # time_text_embed.timestep_embedder -> time_in.
    add_lora_layer_if_present("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer")
    add_lora_layer_if_present("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer")

    # time_text_embed.text_embedder -> vector_in.
    add_lora_layer_if_present("time_text_embed.text_embedder.linear_1", "vector_in.in_layer")
    add_lora_layer_if_present("time_text_embed.text_embedder.linear_2", "vector_in.out_layer")

    # time_text_embed.guidance_embedder -> guidance_in.
    add_lora_layer_if_present("time_text_embed.guidance_embedder.linear_1", "guidance_in")
    add_lora_layer_if_present("time_text_embed.guidance_embedder.linear_2", "guidance_in")

    # context_embedder -> txt_in.
    add_lora_layer_if_present("context_embedder", "txt_in")

    # x_embedder -> img_in.
    add_lora_layer_if_present("x_embedder", "img_in")

    # Double transformer blocks.
    for i in range(num_double_layers):
        # norms.
        add_lora_layer_if_present(f"transformer_blocks.{i}.norm1.linear", f"double_blocks.{i}.img_mod.lin")
        add_lora_layer_if_present(f"transformer_blocks.{i}.norm1_context.linear", f"double_blocks.{i}.txt_mod.lin")

        # Q, K, V
        add_qkv_lora_layer_if_present(
            [
                f"transformer_blocks.{i}.attn.to_q",
                f"transformer_blocks.{i}.attn.to_k",
                f"transformer_blocks.{i}.attn.to_v",
            ],
            f"double_blocks.{i}.img_attn.qkv",
        )
        add_qkv_lora_layer_if_present(
            [
                f"transformer_blocks.{i}.attn.add_q_proj",
                f"transformer_blocks.{i}.attn.add_k_proj",
                f"transformer_blocks.{i}.attn.add_v_proj",
            ],
            f"double_blocks.{i}.txt_attn.qkv",
        )

        # ff img_mlp
        add_lora_layer_if_present(
            f"transformer_blocks.{i}.ff.net.0.proj",
            f"double_blocks.{i}.img_mlp.0",
        )
        add_lora_layer_if_present(
            f"transformer_blocks.{i}.ff.net.2",
            f"double_blocks.{i}.img_mlp.2",
        )

        # ff txt_mlp
        add_lora_layer_if_present(
            f"transformer_blocks.{i}.ff_context.net.0.proj",
            f"double_blocks.{i}.txt_mlp.0",
        )
        add_lora_layer_if_present(
            f"transformer_blocks.{i}.ff_context.net.2",
            f"double_blocks.{i}.txt_mlp.2",
        )

        # output projections.
        add_lora_layer_if_present(
            f"transformer_blocks.{i}.attn.to_out.0",
            f"double_blocks.{i}.img_attn.proj",
        )
        add_lora_layer_if_present(
            f"transformer_blocks.{i}.attn.to_add_out",
            f"double_blocks.{i}.txt_attn.proj",
        )

    # Single transformer blocks.
    for i in range(num_single_layers):
        # norms
        add_lora_layer_if_present(
            f"single_transformer_blocks.{i}.norm.linear",
            f"single_blocks.{i}.modulation.lin",
        )

        # Q, K, V, mlp
        add_qkv_lora_layer_if_present(
            [
                f"single_transformer_blocks.{i}.attn.to_q",
                f"single_transformer_blocks.{i}.attn.to_k",
                f"single_transformer_blocks.{i}.attn.to_v",
                f"single_transformer_blocks.{i}.proj_mlp",
            ],
            f"single_blocks.{i}.linear1",
        )

        # Output projections.
        add_lora_layer_if_present(
            f"single_transformer_blocks.{i}.proj_out",
            f"single_blocks.{i}.linear2",
        )

    # Final layer.
    add_lora_layer_if_present("proj_out", "final_layer.linear")

    return LoRAModelRaw(layers=layers)


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
