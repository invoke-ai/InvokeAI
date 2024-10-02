from typing import Dict

import torch

from invokeai.backend.lora.conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw


def is_state_dict_likely_in_flux_diffusers_format(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely in the Diffusers FLUX LoRA format.

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)
    """
    # First, check that all keys end in "lora_A.weight" or "lora_B.weight" (i.e. are in PEFT format).
    all_keys_in_peft_format = all(k.endswith(("lora_A.weight", "lora_B.weight")) for k in state_dict.keys())

    # Next, check that this is likely a FLUX model by spot-checking a few keys.
    expected_keys = [
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight",
    ]
    all_expected_keys_present = all(k in state_dict for k in expected_keys)

    return all_keys_in_peft_format and all_expected_keys_present


def lora_model_from_flux_diffusers_state_dict(state_dict: Dict[str, torch.Tensor], alpha: float | None) -> LoRAModelRaw:
    """Loads a state dict in the Diffusers FLUX LoRA format into a LoRAModelRaw object.

    This function is based on:
    https://github.com/huggingface/diffusers/blob/55ac421f7bb12fd00ccbef727be4dc2f3f920abb/scripts/convert_flux_to_diffusers.py
    """
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = _group_by_layer(state_dict)

    # Remove the "transformer." prefix from all keys.
    grouped_state_dict = {k.replace("transformer.", ""): v for k, v in grouped_state_dict.items()}

    # Constants for FLUX.1
    num_double_layers = 19
    num_single_layers = 38
    # inner_dim = 3072
    # mlp_ratio = 4.0

    layers: dict[str, AnyLoRALayer] = {}

    def add_lora_layer_if_present(src_key: str, dst_key: str) -> None:
        if src_key in grouped_state_dict:
            src_layer_dict = grouped_state_dict.pop(src_key)
            value = {
                "lora_down.weight": src_layer_dict.pop("lora_A.weight"),
                "lora_up.weight": src_layer_dict.pop("lora_B.weight"),
            }
            if alpha is not None:
                value["alpha"] = torch.tensor(alpha)
            layers[dst_key] = LoRALayer.from_state_dict_values(values=value)
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
        sub_layers: list[LoRALayer] = []
        for src_layer_dict in src_layer_dicts:
            values = {
                "lora_down.weight": src_layer_dict.pop("lora_A.weight"),
                "lora_up.weight": src_layer_dict.pop("lora_B.weight"),
            }
            if alpha is not None:
                values["alpha"] = torch.tensor(alpha)
            sub_layers.append(LoRALayer.from_state_dict_values(values=values))
            assert len(src_layer_dict) == 0
        layers[dst_qkv_key] = ConcatenatedLoRALayer(lora_layers=sub_layers, concat_axis=0)

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

    # Assert that all keys were processed.
    assert len(grouped_state_dict) == 0

    layers_with_prefix = {f"{FLUX_LORA_TRANSFORMER_PREFIX}{k}": v for k, v in layers.items()}

    return LoRAModelRaw(layers=layers_with_prefix)


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
