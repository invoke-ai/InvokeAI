from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def is_state_dict_likely_in_flux_diffusers_format(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely in the Diffusers FLUX LoRA format.

    This detects both Flux.1 diffusers format (separate to_q/to_k/to_v, ff.net.0.proj) and
    Flux2 Klein diffusers format (fused to_qkv_mlp_proj, ff.linear_in).

    This is intended to be a reasonably high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)
    """
    # Check that all keys are LoRA weight keys (either PEFT or standard format).
    # Some LoRAs use a mix of formats (PEFT for some layers, standard for others).
    _LORA_SUFFIXES = ("lora_A.weight", "lora_B.weight", "lora.down.weight", "lora.up.weight")
    all_keys_are_lora = all(k.endswith(_LORA_SUFFIXES) for k in state_dict.keys() if isinstance(k, str))
    if not all_keys_are_lora:
        return False

    # --- Flux.1 diffusers key patterns (separate Q/K/V, ff.net.0.proj) ---
    # Check if keys use transformer prefix
    flux1_transformer_keys = [
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight",
    ]
    flux1_transformer_present = all(k in state_dict for k in flux1_transformer_keys)

    # Check if keys use base_model.model prefix
    flux1_base_model_keys = [
        "base_model.model.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "base_model.model.single_transformer_blocks.0.attn.to_q.lora_B.weight",
        "base_model.model.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "base_model.model.transformer_blocks.0.attn.add_q_proj.lora_B.weight",
    ]
    flux1_base_model_present = all(k in state_dict for k in flux1_base_model_keys)

    if flux1_transformer_present or flux1_base_model_present:
        return True

    # --- Flux2 Klein diffusers key patterns (fused QKV+MLP, ff.linear_in) ---
    # These use Flux2Transformer2DModel naming which differs from Flux.1.
    for prefix in ["transformer.", "base_model.model."]:
        has_single = any(
            k.startswith(f"{prefix}single_transformer_blocks.") and "to_qkv_mlp_proj" in k for k in state_dict
        )
        has_double = any(k.startswith(f"{prefix}transformer_blocks.") for k in state_dict if isinstance(k, str))
        if has_single or has_double:
            # Verify it's actually Flux2 naming by checking for a Flux2-specific key pattern.
            # Flux2 uses ff.linear_in (not ff.net.0.proj) and attn.to_add_out (not attn.to_add_out in Flux.1 too,
            # but fused to_qkv_mlp_proj is unique to Flux2).
            has_flux2_keys = any(
                ("to_qkv_mlp_proj" in k or "ff.linear_in" in k or "ff_context.linear_in" in k)
                for k in state_dict
                if isinstance(k, str)
            )
            if has_flux2_keys:
                return True

    return False


def is_state_dict_flux2_diffusers_format(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the state dict uses Flux2 Klein native diffusers naming (not Flux.1 diffusers naming).

    Returns True only for Flux2 Klein diffusers format (to_qkv_mlp_proj, ff.linear_in, etc.),
    NOT for Flux.1 diffusers format (to_q/to_k/to_v, ff.net.0.proj).
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    return any("to_qkv_mlp_proj" in k or "ff.linear_in" in k or "ff_context.linear_in" in k for k in str_keys)


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


def lora_model_from_flux2_diffusers_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None
) -> ModelPatchRaw:
    """Convert a Flux2 Klein native diffusers format LoRA state dict to a ModelPatchRaw.

    Flux2 Klein diffusers LoRAs use key names that match Flux2Transformer2DModel directly
    (e.g. transformer_blocks.0.attn.to_add_out, single_transformer_blocks.0.attn.to_qkv_mlp_proj).
    The conversion strips the model prefix (transformer. or base_model.model.) and adds
    the InvokeAI prefix.

    Some LoRAs use a mix of PEFT format (lora_A.weight/lora_B.weight) and standard format
    (lora.down.weight/lora.up.weight) for different layers. Both are handled here.
    """
    grouped_state_dict = _group_by_layer_mixed_format(state_dict)

    # Determine and strip prefix
    has_base_model_prefix = any(k.startswith("base_model.model.") for k in grouped_state_dict.keys())
    if has_base_model_prefix:
        grouped_state_dict = {k.replace("base_model.model.", "", 1): v for k, v in grouped_state_dict.items()}
    else:
        grouped_state_dict = {k.replace("transformer.", "", 1): v for k, v in grouped_state_dict.items()}

    layers: dict[str, BaseLayerPatch] = {}
    for layer_key, src_layer_dict in grouped_state_dict.items():
        # Normalize to InvokeAI naming (lora_down.weight / lora_up.weight)
        values: dict[str, torch.Tensor] = {}
        if "lora_A.weight" in src_layer_dict:
            values["lora_down.weight"] = src_layer_dict["lora_A.weight"]
            values["lora_up.weight"] = src_layer_dict["lora_B.weight"]
        elif "lora.down.weight" in src_layer_dict:
            values["lora_down.weight"] = src_layer_dict["lora.down.weight"]
            values["lora_up.weight"] = src_layer_dict["lora.up.weight"]
        else:
            values = src_layer_dict

        if alpha is not None and "alpha" not in values:
            values["alpha"] = torch.tensor(alpha)

        layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{layer_key}"] = any_lora_layer_from_state_dict(values)

    return ModelPatchRaw(layers=layers)


def _group_by_layer_mixed_format(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Groups keys by layer, handling both PEFT and standard LoRA suffixes.

    PEFT format:    layer_name.lora_A.weight → layer=layer_name, suffix=lora_A.weight
    Standard format: layer_name.lora.down.weight → layer=layer_name, suffix=lora.down.weight
    """
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key in state_dict:
        if not isinstance(key, str):
            continue

        # Determine suffix length based on the key ending
        if key.endswith((".lora_A.weight", ".lora_B.weight")):
            # PEFT format: split off 2 parts (lora_A + weight)
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            suffix = ".".join(parts[1:])
        elif key.endswith((".lora.down.weight", ".lora.up.weight")):
            # Standard format: split off 3 parts (lora + down/up + weight)
            parts = key.rsplit(".", maxsplit=3)
            layer_name = parts[0]
            suffix = ".".join(parts[1:])
        else:
            # Unknown format, use 2-part split as fallback
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            suffix = ".".join(parts[1:])

        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][suffix] = state_dict[key]

    return layer_dict


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
