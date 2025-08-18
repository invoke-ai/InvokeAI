from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def is_state_dict_likely_in_flux_diffusers_format(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely in the Diffusers FLUX LoRA format.

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)
    """
    # First, check that all keys end in "lora_A.weight" or "lora_B.weight" (i.e. are in PEFT format).
    all_keys_in_peft_format = all(k.endswith(("lora_A.weight", "lora_B.weight")) for k in state_dict.keys())

    # Check if keys use transformer prefix
    transformer_prefix_keys = [
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight",
    ]
    transformer_keys_present = all(k in state_dict for k in transformer_prefix_keys)

    # Check if keys use base_model.model prefix
    base_model_prefix_keys = [
        "base_model.model.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "base_model.model.single_transformer_blocks.0.attn.to_q.lora_B.weight",
        "base_model.model.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "base_model.model.transformer_blocks.0.attn.add_q_proj.lora_B.weight",
    ]
    base_model_keys_present = all(k in state_dict for k in base_model_prefix_keys)

    return all_keys_in_peft_format and (transformer_keys_present or base_model_keys_present)


def lora_model_from_flux_diffusers_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None
) -> ModelPatchRaw:
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = _group_by_layer(state_dict)
    layers = lora_layers_from_flux_diffusers_grouped_state_dict(grouped_state_dict, alpha)
    return ModelPatchRaw(layers=layers)


def lora_layers_from_flux_diffusers_grouped_state_dict(
    grouped_state_dict: Dict[str, Dict[str, torch.Tensor]], alpha: float | None
) -> dict[str, BaseLayerPatch]:
    """Converts a grouped state dict with Diffusers FLUX LoRA keys to LoRA layers with BFL keys (i.e. the module key
    format used by Invoke).

    This function is based on:
    https://github.com/huggingface/diffusers/blob/55ac421f7bb12fd00ccbef727be4dc2f3f920abb/scripts/convert_flux_to_diffusers.py
    """

    # Determine which prefix is used and remove it from all keys.
    # Check if any key starts with "base_model.model." prefix
    has_base_model_prefix = any(k.startswith("base_model.model.") for k in grouped_state_dict.keys())

    if has_base_model_prefix:
        # Remove the "base_model.model." prefix from all keys.
        grouped_state_dict = {k.replace("base_model.model.", ""): v for k, v in grouped_state_dict.items()}
    else:
        # Remove the "transformer." prefix from all keys.
        grouped_state_dict = {k.replace("transformer.", ""): v for k, v in grouped_state_dict.items()}

    # Constants for FLUX.1
    num_double_layers = 19
    num_single_layers = 38
    hidden_size = 3072
    mlp_ratio = 4.0
    mlp_hidden_dim = int(hidden_size * mlp_ratio)

    layers: dict[str, BaseLayerPatch] = {}

    def get_lora_layer_values(src_layer_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "lora_A.weight" in src_layer_dict:
            # The LoRA keys are in PEFT format.
            values = {
                "lora_down.weight": src_layer_dict.pop("lora_A.weight"),
                "lora_up.weight": src_layer_dict.pop("lora_B.weight"),
            }
            if alpha is not None:
                values["alpha"] = torch.tensor(alpha)
            assert len(src_layer_dict) == 0
            return values
        else:
            # Assume that the LoRA keys are in Kohya format.
            return src_layer_dict

    def add_lora_layer_if_present(src_key: str, dst_key: str) -> None:
        if src_key in grouped_state_dict:
            src_layer_dict = grouped_state_dict.pop(src_key)
            values = get_lora_layer_values(src_layer_dict)
            layers[dst_key] = any_lora_layer_from_state_dict(values)

    def add_qkv_lora_layer_if_present(
        src_keys: list[str],
        src_weight_shapes: list[tuple[int, int]],
        dst_qkv_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        """Handle the Q, K, V matrices for a transformer block. We need special handling because the diffusers format
        stores them in separate matrices, whereas the BFL format used internally by InvokeAI concatenates them.
        """
        # If none of the keys are present, return early.
        keys_present = [key in grouped_state_dict for key in src_keys]
        if not any(keys_present):
            return

        dim_0_offset = 0
        sub_layers: list[BaseLayerPatch] = []
        sub_layer_ranges: list[Range] = []
        for src_key, src_weight_shape in zip(src_keys, src_weight_shapes, strict=True):
            src_layer_dict = grouped_state_dict.pop(src_key, None)
            if src_layer_dict is not None:
                values = get_lora_layer_values(src_layer_dict)
                # assert values["lora_down.weight"].shape[1] == src_weight_shape[1]
                # assert values["lora_up.weight"].shape[0] == src_weight_shape[0]
                sub_layers.append(any_lora_layer_from_state_dict(values))
                sub_layer_ranges.append(Range(dim_0_offset, dim_0_offset + src_weight_shape[0]))
            else:
                if not allow_missing_keys:
                    raise ValueError(f"Missing LoRA layer: '{src_key}'.")

            dim_0_offset += src_weight_shape[0]

        layers[dst_qkv_key] = MergedLayerPatch(sub_layers, sub_layer_ranges)

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
            [(hidden_size, hidden_size), (hidden_size, hidden_size), (hidden_size, hidden_size)],
            f"double_blocks.{i}.img_attn.qkv",
        )
        add_qkv_lora_layer_if_present(
            [
                f"transformer_blocks.{i}.attn.add_q_proj",
                f"transformer_blocks.{i}.attn.add_k_proj",
                f"transformer_blocks.{i}.attn.add_v_proj",
            ],
            [(hidden_size, hidden_size), (hidden_size, hidden_size), (hidden_size, hidden_size)],
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
            [
                (hidden_size, hidden_size),
                (hidden_size, hidden_size),
                (hidden_size, hidden_size),
                (mlp_hidden_dim, hidden_size),
            ],
            f"single_blocks.{i}.linear1",
            allow_missing_keys=True,
        )

        # Output projections.
        add_lora_layer_if_present(
            f"single_transformer_blocks.{i}.proj_out",
            f"single_blocks.{i}.linear2",
        )

    # Final layer.
    add_lora_layer_if_present("proj_out", "final_layer.linear")

    # Assert that all keys were processed.
    assert len(grouped_state_dict) == 0

    layers_with_prefix = {f"{FLUX_LORA_TRANSFORMER_PREFIX}{k}": v for k, v in layers.items()}

    return layers_with_prefix


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Groups the keys in the state dict by layer."""
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key in state_dict:
        # Split the 'lora_A.weight' or 'lora_B.weight' suffix from the layer name.
        parts = key.rsplit(".", maxsplit=2)
        layer_name = parts[0]
        key_name = ".".join(parts[1:])
        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][key_name] = state_dict[key]
    return layer_dict
