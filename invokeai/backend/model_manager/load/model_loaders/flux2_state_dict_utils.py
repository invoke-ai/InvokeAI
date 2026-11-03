"""Pure state-dict conversion helpers for FLUX.2 single-file checkpoints.

These functions translate the BFL (black-forest-labs) key layout used by FLUX.2
single-file transformer and VAE checkpoints into the diffusers layout that the
`Flux2Transformer2DModel` / `AutoencoderKLFlux2` architectures expect.

They are intentionally free of any file/model-loading side effects so the key
remapping can be unit-tested against a synthetic state dict (see
`tests/backend/model_manager/load/`). Both the checkpoint and GGUF FLUX.2 loaders
delegate here; GGUF quantized tensors are dequantized only where a fused weight
must be split (diffusers uses separate Q/K/V projections).

Based on the diffusers `convert_flux2_to_diffusers.py` key mappings.
"""

import re
from typing import Any

import torch

from invokeai.backend.quantization.fp8_scaled import (
    QKV_SPLIT_SIDECHANNEL_SUFFIXES,
    split_qkv_sidechannel,
)


def _flux2_chunk_tensor(tensor, chunks: int):
    """Chunk a tensor along dim 0, dequantizing GGUF tensors first.

    diffusers uses separate Q/K/V projections, so a fused GGUF weight cannot stay
    quantized through the split.
    """
    if hasattr(tensor, "get_dequantized_tensor"):
        tensor = tensor.get_dequantized_tensor()
    return tensor.chunk(chunks, dim=0)


def _flux2_malformed_for_chunk(tensor, chunks: int) -> bool:
    """Return True if a plain tensor cannot be evenly split into `chunks` along dim 0.

    GGUF quantized tensors are always considered well-formed here (their logical
    shape is only known after dequantization, matching the original GGUF loader,
    which split them unconditionally).
    """
    if hasattr(tensor, "get_dequantized_tensor"):
        return False
    return tensor.dim() < 1 or tensor.shape[0] % chunks != 0


def _flux2_swap_scale_shift(weight):
    """Swap the scale/shift halves of an AdaLayerNorm weight (BFL vs diffusers order)."""
    if hasattr(weight, "get_dequantized_tensor"):
        weight = weight.get_dequantized_tensor()
    elif weight.dim() < 1 or weight.shape[0] % 2 != 0:
        # Defensive: leave malformed plain tensors untouched.
        return weight
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _convert_flux2_double_block_key(
    key: str, tensor, converted: dict, *, destinations: list[str] | None = None
) -> str | None:
    """Convert a `double_blocks.X.*` key to `transformer_blocks.X.*` format.

    Returns the new key, or None if the key was consumed by writing directly into
    `converted` (fused QKV split into separate projections). ``destinations``, when given, collects
    the keys written that way, and only those: a split this function *declines* to perform -- a
    fused qkv whose rows are not divisible by three -- leaves it empty and reports the unchanged key
    through the return value instead, which is what the caller records.
    """
    parts = key.split(".")
    block_idx = parts[1]
    rest = ".".join(parts[2:])

    prefix = f"transformer_blocks.{block_idx}"

    # Attention QKV: BFL uses fused qkv, diffusers uses separate Q/K/V.
    if "img_attn.qkv.weight" in rest:
        if _flux2_malformed_for_chunk(tensor, 3):
            return key
        targets = [
            f"{prefix}.attn.to_q.weight",
            f"{prefix}.attn.to_k.weight",
            f"{prefix}.attn.to_v.weight",
        ]
        for target, part in zip(targets, _flux2_chunk_tensor(tensor, 3), strict=True):
            converted[target] = part
        if destinations is not None:
            destinations.extend(targets)
        return None
    elif "txt_attn.qkv.weight" in rest:
        if _flux2_malformed_for_chunk(tensor, 3):
            return key
        targets = [
            f"{prefix}.attn.add_q_proj.weight",
            f"{prefix}.attn.add_k_proj.weight",
            f"{prefix}.attn.add_v_proj.weight",
        ]
        for target, part in zip(targets, _flux2_chunk_tensor(tensor, 3), strict=True):
            converted[target] = part
        if destinations is not None:
            destinations.extend(targets)
        return None

    # Attention output projection
    if "img_attn.proj.weight" in rest:
        return f"{prefix}.attn.to_out.0.weight"
    elif "txt_attn.proj.weight" in rest:
        return f"{prefix}.attn.to_add_out.weight"

    # Attention norms
    if "img_attn.norm.query_norm.scale" in rest or "img_attn.norm.query_norm.weight" in rest:
        return f"{prefix}.attn.norm_q.weight"
    elif "img_attn.norm.key_norm.scale" in rest or "img_attn.norm.key_norm.weight" in rest:
        return f"{prefix}.attn.norm_k.weight"
    elif "txt_attn.norm.query_norm.scale" in rest or "txt_attn.norm.query_norm.weight" in rest:
        return f"{prefix}.attn.norm_added_q.weight"
    elif "txt_attn.norm.key_norm.scale" in rest or "txt_attn.norm.key_norm.weight" in rest:
        return f"{prefix}.attn.norm_added_k.weight"

    # MLP layers
    if "img_mlp.0.weight" in rest:
        return f"{prefix}.ff.linear_in.weight"
    elif "img_mlp.2.weight" in rest:
        return f"{prefix}.ff.linear_out.weight"
    elif "txt_mlp.0.weight" in rest:
        return f"{prefix}.ff_context.linear_in.weight"
    elif "txt_mlp.2.weight" in rest:
        return f"{prefix}.ff_context.linear_out.weight"

    return key


def _convert_flux2_single_block_key(key: str, tensor, converted: dict) -> str | None:
    """Convert a `single_blocks.X.*` key to `single_transformer_blocks.X.*` format."""
    parts = key.split(".")
    block_idx = parts[1]
    rest = ".".join(parts[2:])

    prefix = f"single_transformer_blocks.{block_idx}"

    # linear1 is the fused QKV+MLP projection
    if "linear1.weight" in rest:
        return f"{prefix}.attn.to_qkv_mlp_proj.weight"
    elif "linear2.weight" in rest:
        return f"{prefix}.attn.to_out.weight"

    # Norms
    if "norm.query_norm.scale" in rest or "norm.query_norm.weight" in rest:
        return f"{prefix}.attn.norm_q.weight"
    elif "norm.key_norm.scale" in rest or "norm.key_norm.weight" in rest:
        return f"{prefix}.attn.norm_k.weight"

    return key


def _convert_flux2_weight_keys(sd: dict, *, key_map: dict[str, list[str]] | None = None) -> dict:
    """Convert the *weight* keys of a FLUX.2 BFL state dict to diffusers format.

    Quantization side-channel keys must not be routed through here. The block renames below are
    substring tests, so `img_attn.proj.weight_scale` satisfies `"img_attn.proj.weight" in rest`
    and would be written to the weight's destination key -- overwriting the weight with its own
    scale. `convert_flux2_bfl_to_diffusers` keeps them out and places them separately.

    ``key_map``, when given, records where each source key went: one destination for a rename, three
    for a fused ``qkv`` that was split, and the source key itself for one that was *not*. That record
    is what places the side channel. Asking a probe where a layer would go cannot see the
    shape-dependent branches: it answered "three destinations" for a fused weight the converter had
    left alone, so the scale was split three ways onto modules that did not exist while the weight
    kept none.
    """
    converted: dict = {}

    # Basic key renames
    key_renames = {
        "img_in.weight": "x_embedder.weight",
        "txt_in.weight": "context_embedder.weight",
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "guidance_in.in_layer.weight": "time_guidance_embed.guidance_embedder.linear_1.weight",
        "guidance_in.out_layer.weight": "time_guidance_embed.guidance_embedder.linear_2.weight",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
    }

    for old_key, tensor in sd.items():
        new_key = old_key
        split_destinations: list[str] = []

        # Apply basic renames
        if old_key in key_renames:
            new_key = key_renames[old_key]
            # Apply scale-shift swap for adaLN modulation weights (BFL and diffusers use
            # different parameter ordering for AdaLayerNorm).
            if old_key == "final_layer.adaLN_modulation.1.weight":
                tensor = _flux2_swap_scale_shift(tensor)
            converted[new_key] = tensor
            if key_map is not None:
                key_map[old_key] = [new_key]
            continue

        # Convert double_blocks.X.* to transformer_blocks.X.*
        if old_key.startswith("double_blocks."):
            new_key = _convert_flux2_double_block_key(old_key, tensor, converted, destinations=split_destinations)
        # Convert single_blocks.X.* to single_transformer_blocks.X.*
        elif old_key.startswith("single_blocks."):
            new_key = _convert_flux2_single_block_key(old_key, tensor, converted)

        if new_key is None:
            # The helper wrote into `converted` itself, and recorded where.
            if key_map is not None:
                key_map[old_key] = split_destinations
            continue

        if new_key != old_key or new_key not in converted:
            converted[new_key] = tensor
        if key_map is not None:
            key_map[old_key] = [new_key]

    return converted


def _convert_flux2_vae_mid_attention_key(rest: str, tensor, block: str):
    """Map a `{enc,dec}.mid.attn_1.*` key to the diffusers mid_block attention layout.

    BFL uses Conv2d (shape `[out, in, 1, 1]`), diffusers uses Linear (`[out, in]`), so
    weight tensors are squeezed. Returns `(new_key, tensor)`.
    """
    attn_prefix = f"{block}.mid_block.attentions.0"
    if rest.startswith("q."):
        new_key = f"{attn_prefix}.to_q.{rest[2:]}"
    elif rest.startswith("k."):
        new_key = f"{attn_prefix}.to_k.{rest[2:]}"
    elif rest.startswith("v."):
        new_key = f"{attn_prefix}.to_v.{rest[2:]}"
    elif rest.startswith("proj_out."):
        new_key = f"{attn_prefix}.to_out.0.{rest[9:]}"
    elif rest.startswith("norm."):
        return f"{attn_prefix}.group_norm.{rest[5:]}", tensor
    else:
        return f"{attn_prefix}.{rest}", tensor

    if rest.endswith(".weight") and hasattr(tensor, "dim") and tensor.dim() == 4:
        tensor = tensor.squeeze(-1).squeeze(-1)
    return new_key, tensor


def convert_flux2_vae_bfl_to_diffusers(sd: dict) -> dict:
    """Convert a FLUX.2 VAE BFL-format state dict to diffusers format.

    Key differences:
    - encoder.down.X.block.Y -> encoder.down_blocks.X.resnets.Y
    - encoder.down.X.downsample.conv -> encoder.down_blocks.X.downsamplers.0.conv
    - encoder.mid.block_1/2 -> encoder.mid_block.resnets.0/1
    - encoder.mid.attn_1.q/k/v -> encoder.mid_block.attentions.0.to_q/k/v
    - encoder.norm_out -> encoder.conv_norm_out
    - encoder.quant_conv -> quant_conv (top-level)
    - decoder.up.X -> decoder.up_blocks.(num_blocks-1-X) (reversed order!)
    - decoder.post_quant_conv -> post_quant_conv (top-level)
    - *.nin_shortcut -> *.conv_shortcut
    """
    converted: dict = {}
    num_up_blocks = 4  # Standard VAE has 4 up blocks

    for old_key, tensor in sd.items():
        new_key = old_key

        # Encoder down blocks: encoder.down.X.block.Y -> encoder.down_blocks.X.resnets.Y
        match = re.match(r"encoder\.down\.(\d+)\.block\.(\d+)\.(.*)", old_key)
        if match:
            block_idx, resnet_idx, rest = match.groups()
            rest = rest.replace("nin_shortcut", "conv_shortcut")
            new_key = f"encoder.down_blocks.{block_idx}.resnets.{resnet_idx}.{rest}"
            converted[new_key] = tensor
            continue

        # Encoder downsamplers: encoder.down.X.downsample.conv -> encoder.down_blocks.X.downsamplers.0.conv
        match = re.match(r"encoder\.down\.(\d+)\.downsample\.conv\.(.*)", old_key)
        if match:
            block_idx, rest = match.groups()
            new_key = f"encoder.down_blocks.{block_idx}.downsamplers.0.conv.{rest}"
            converted[new_key] = tensor
            continue

        # Encoder mid block resnets: encoder.mid.block_1/2 -> encoder.mid_block.resnets.0/1
        match = re.match(r"encoder\.mid\.block_(\d+)\.(.*)", old_key)
        if match:
            block_num, rest = match.groups()
            resnet_idx = int(block_num) - 1  # block_1 -> resnets.0, block_2 -> resnets.1
            new_key = f"encoder.mid_block.resnets.{resnet_idx}.{rest}"
            converted[new_key] = tensor
            continue

        # Encoder mid block attention: encoder.mid.attn_1.* -> encoder.mid_block.attentions.0.*
        match = re.match(r"encoder\.mid\.attn_1\.(.*)", old_key)
        if match:
            new_key, tensor = _convert_flux2_vae_mid_attention_key(match.group(1), tensor, "encoder")
            converted[new_key] = tensor
            continue

        # Encoder norm_out -> conv_norm_out
        if old_key.startswith("encoder.norm_out."):
            new_key = old_key.replace("encoder.norm_out.", "encoder.conv_norm_out.")
            converted[new_key] = tensor
            continue

        # Encoder quant_conv -> quant_conv (move to top level)
        if old_key.startswith("encoder.quant_conv."):
            new_key = old_key.replace("encoder.quant_conv.", "quant_conv.")
            converted[new_key] = tensor
            continue

        # Decoder up blocks (reversed order!): decoder.up.X -> decoder.up_blocks.(num_blocks-1-X)
        match = re.match(r"decoder\.up\.(\d+)\.block\.(\d+)\.(.*)", old_key)
        if match:
            block_idx, resnet_idx, rest = match.groups()
            new_block_idx = num_up_blocks - 1 - int(block_idx)
            rest = rest.replace("nin_shortcut", "conv_shortcut")
            new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.{rest}"
            converted[new_key] = tensor
            continue

        # Decoder upsamplers (reversed order!)
        match = re.match(r"decoder\.up\.(\d+)\.upsample\.conv\.(.*)", old_key)
        if match:
            block_idx, rest = match.groups()
            new_block_idx = num_up_blocks - 1 - int(block_idx)
            new_key = f"decoder.up_blocks.{new_block_idx}.upsamplers.0.conv.{rest}"
            converted[new_key] = tensor
            continue

        # Decoder mid block resnets: decoder.mid.block_1/2 -> decoder.mid_block.resnets.0/1
        match = re.match(r"decoder\.mid\.block_(\d+)\.(.*)", old_key)
        if match:
            block_num, rest = match.groups()
            resnet_idx = int(block_num) - 1
            new_key = f"decoder.mid_block.resnets.{resnet_idx}.{rest}"
            converted[new_key] = tensor
            continue

        # Decoder mid block attention: decoder.mid.attn_1.* -> decoder.mid_block.attentions.0.*
        match = re.match(r"decoder\.mid\.attn_1\.(.*)", old_key)
        if match:
            new_key, tensor = _convert_flux2_vae_mid_attention_key(match.group(1), tensor, "decoder")
            converted[new_key] = tensor
            continue

        # Decoder norm_out -> conv_norm_out
        if old_key.startswith("decoder.norm_out."):
            new_key = old_key.replace("decoder.norm_out.", "decoder.conv_norm_out.")
            converted[new_key] = tensor
            continue

        # Decoder post_quant_conv -> post_quant_conv (move to top level)
        if old_key.startswith("decoder.post_quant_conv."):
            new_key = old_key.replace("decoder.post_quant_conv.", "post_quant_conv.")
            converted[new_key] = tensor
            continue

        # Keep other keys as-is (like encoder.conv_in, decoder.conv_in, decoder.conv_out, bn.*)
        converted[new_key] = tensor

    return converted


def _flux2_sidechannel_parts(key: Any) -> tuple[str, str] | None:
    """Split ``<module path>.<suffix>`` for a quantization side-channel key, else None."""
    if not isinstance(key, str):
        return None
    for suffix in QKV_SPLIT_SIDECHANNEL_SUFFIXES:
        if key.endswith(f".{suffix}"):
            return key[: -len(suffix) - 1], suffix
    return None


# Converter transforms that reorder weight *rows*. A per-output-channel weight scale has one entry
# per row, so it has to be reordered identically or every row ends up scaled by another row's
# factor. The fused-QKV split is handled separately (`split_qkv_sidechannel`); this is the only
# other row-reordering transform the converter applies.
_ROW_PERMUTED_BY_CONVERSION = {"final_layer.adaLN_modulation.1"}


def _mirror_row_permutation(base: str, suffix: str, value: Any) -> Any:
    """Apply the converter's row reordering to a per-output-channel weight scale.

    Only weight scales carry per-row structure: the activation scale is per-tensor and the
    `comfy_quant` marker is a byte blob, both of which describe the whole layer and must be copied
    verbatim. A per-tensor weight scale is likewise unaffected.
    """
    if base not in _ROW_PERMUTED_BY_CONVERSION or suffix not in ("weight_scale", "scale_weight"):
        return value
    tensor = torch.as_tensor(value) if hasattr(value, "shape") else value
    if not hasattr(tensor, "shape") or tensor.dim() < 1 or tensor.numel() <= 1 or tensor.shape[0] % 2 != 0:
        return value
    return _flux2_swap_scale_shift(tensor)


def convert_flux2_bfl_to_diffusers(sd: dict, *, module_map: dict[str, list[str]] | None = None) -> dict:
    """Convert a FLUX.2 transformer BFL-format state dict to diffusers format.

    Quantization scales and markers are carried to wherever their weight landed. Doing that is not
    optional for a scaled-fp8 checkpoint: a scale left on the fused ``qkv`` path is keyed on a
    module the diffusers model does not have, so `attach_fp8_scales` finds nothing and the three
    split weights stay quantized but *unscaled* -- off by 1/weight_scale, with nothing logged.

    ``module_map``, when given, receives the record the placement uses: each source module to the
    module(s) its weight became. Callers with something else to re-key in the checkpoint's own scheme
    -- the header's per-layer hints and its int8 markers -- read it, rather than asking a probe what
    the rename *would* do, which cannot see the branches that depend on the tensor. It is updated,
    not adopted, so a caller that reuses a dict cannot end up with two records interleaved.

    A module named only in the header and absent from the file gets no entry, and the caller keeps
    the name as it stands. What happens then is the caller's business and differs: the fp8
    extraction ignores a hint that names no layer, while the int8 swap refuses the checkpoint by name
    ("is marked int8_tensorwise but is missing its weight"). Both predate this record.
    """
    weights = {k: v for k, v in sd.items() if _flux2_sidechannel_parts(k) is None}
    key_map: dict[str, list[str]] = {}
    converted = _convert_flux2_weight_keys(weights, key_map=key_map)

    # The conversion records where each *key* went; a side channel is keyed on the *module*. Read the
    # module off whatever the source key was rather than assuming it was `<module>.weight`: BFL
    # stores a norm's parameter as `.scale`, and this converter accepts both spellings. (The probe
    # this replaces got that right; a first draft of this map looked the destination up under
    # `<base>.weight` and would have left every such scale behind. Caught in review, not shipped.)
    #
    # Only a destination ending `.weight` names a module a scale can sit beside. A `.bias` is not
    # renamed by this converter at all, so without the filter it contributes the module a second
    # time and the side channel is taken for a fused-qkv split.
    modules: dict[str, list[str]] = {}
    for source, moved_to in key_map.items():
        stem = source.rsplit(".", 1)[0] if "." in source else source
        landed = [target[: -len(".weight")] for target in moved_to if target.endswith(".weight")]
        if landed:
            modules.setdefault(stem, []).extend(landed)
    if module_map is not None:
        module_map.update(modules)

    for key, value in sd.items():
        parts = _flux2_sidechannel_parts(key)
        if parts is None:
            continue
        base, suffix = parts
        destinations = modules.get(base, [])
        if not destinations:
            # Unknown layer: keep the key as it is. `extract_fp8_scaled_layers` drops a scale with
            # no matching fp8 weight, which is the safe outcome -- better than guessing a target.
            converted[key] = value
        elif len(destinations) == 1:
            converted[f"{destinations[0]}.{suffix}"] = _mirror_row_permutation(base, suffix, value)
        else:
            for destination, part in zip(destinations, split_qkv_sidechannel(key, value), strict=True):
                converted[f"{destination}.{suffix}"] = part

    return converted
