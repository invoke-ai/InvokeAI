"""Native FLUX.3 state dicts, converted to diffusers' ``Flux3Transformer2DModel``.

Every single-file FLUX.3 build InvokeAI loads uses Black Forest Labs' own key names: BFL's
``model.safetensors``, the ComfyUI repackage (byte-identical to it), and the GGUF conversions. The renames
are a port of diffusers' ``convert_flux3_transformer_checkpoint_to_diffusers`` (``single_file_utils.py`` in
diffusers 0.42.1). Reading the structure from the weights, rather than assuming the released one, is InvokeAI's:
a finetune that changes the depth must not load as the first N blocks of the released model.
"""

import re
from collections.abc import Mapping
from typing import Any

import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

FLUX3_NATIVE_FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
"""Only the step-distilled checkpoint has it: the embedder of the timestep each step jumps to."""

# The rotary axes are the one structural parameter the weights do not record. Every released FLUX.3
# checkpoint uses these, and they must add up to the attention head width the weights do record.
FLUX3_AXES_DIMS_ROPE = (32, 32, 32, 32)

# Inner-module renames shared by the per-mode blocks and the joint single blocks.
_BLOCK_KEY_MAP = {
    "q_proj.weight": "attn.to_q.weight",
    "k_proj.weight": "attn.to_k.weight",
    "v_proj.weight": "attn.to_v.weight",
    "attn_out.weight": "attn.to_out.0.weight",
    "norm.query_norm.scale": "attn.norm_q.weight",
    "norm.key_norm.scale": "attn.norm_k.weight",
    "mlp_in.weight": "ff.linear_in.weight",
    "mlp_out.weight": "ff.linear_out.weight",
}

_PREFIX_MAP = {
    "emb_in.": "x_embedders.",
    "txt_in.": "context_embedder.",
    "content_mode_blocks.": "mode_blocks.",
    "single_blocks.": "single_transformer_blocks.",
}

_ADALN_KEY = "final_layer.image.adaLN_modulation.1.weight"

_KEY_RENAMES = {
    "time_in.in_layer.weight": "timestep_embedder.linear_1.weight",
    "time_in.out_layer.weight": "timestep_embedder.linear_2.weight",
    FLUX3_NATIVE_FLOW_MAP_KEY: "target_timestep_embedder.linear_1.weight",
    "ref_time_in.out_layer.weight": "target_timestep_embedder.linear_2.weight",
    "final_layer.image.linear.weight": "proj_out.weight",
    _ADALN_KEY: "norm_out.linear.weight",
}

# Training-only REPA projection heads; not part of the inference model.
_TRAINING_ONLY_PREFIX = "repa_projectors."


def _swap_shift_scale(weight: torch.Tensor) -> torch.Tensor:
    """BFL's output adaLN produces [shift, scale]; ``Flux3AdaLayerNorm`` expects [scale, shift].

    The halves are output rows, so a packed GGML tensor is unpacked first: a row swap inside quantized
    blocks is not something a byte shuffle can express for every quantization type.
    """
    if isinstance(weight, GGMLTensor):
        weight = weight.get_dequantized_tensor()
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def flux3_native_key_to_diffusers(key: str) -> str:
    """The diffusers name of one native FLUX.3 parameter (identity for names it does not know)."""
    if key in _KEY_RENAMES:
        return _KEY_RENAMES[key]
    new_key = key
    for old_prefix, new_prefix in _PREFIX_MAP.items():
        if new_key.startswith(old_prefix):
            new_key = new_prefix + new_key[len(old_prefix) :]
            break
    new_key = new_key.replace(".lin.weight", ".linear.weight")
    for old_suffix, new_suffix in _BLOCK_KEY_MAP.items():
        if new_key.endswith(old_suffix):
            return new_key[: -len(old_suffix)] + new_suffix
    return new_key


def convert_flux3_native_to_diffusers(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Rename a native FLUX.3 transformer state dict to the diffusers module, in place of the input.

    The input dict is emptied as it is converted, so the caller's reference does not keep a second copy of
    every tensor alive.
    """
    converted: dict[str, Any] = {}
    for key in list(state_dict.keys()):
        value = state_dict.pop(key)
        if key.startswith(_TRAINING_ONLY_PREFIX):
            continue
        if key == _ADALN_KEY:
            value = _swap_shift_scale(value)
        converted[flux3_native_key_to_diffusers(key)] = value
    return converted


def _shape(state_dict: Mapping[str, Any], key: str) -> tuple[int, ...]:
    if key not in state_dict:
        raise ValueError(f"The FLUX.3 checkpoint has no '{key}'; it is truncated or not a FLUX.3 transformer.")
    # GGMLTensor.shape is the logical (unpacked) shape, which is the one needed here.
    return tuple(state_dict[key].shape)


def _count_blocks(state_dict: Mapping[str, Any], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    indices = {int(m.group(1)) for key in state_dict if (m := pattern.match(key))}
    if not indices:
        raise ValueError(f"The FLUX.3 checkpoint has no '{prefix}N' blocks.")
    if indices != set(range(len(indices))):
        missing = sorted(set(range(max(indices) + 1)) - indices)
        raise ValueError(f"The FLUX.3 checkpoint is missing '{prefix}' blocks {missing}; it is truncated.")
    return len(indices)


def flux3_transformer_config(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """The ``Flux3Transformer2DModel`` config of a native FLUX.3 state dict, read from its tensor shapes.

    Only shapes are read, so this works on meta tensors and on packed GGUF tensors alike. The feedforward
    width is taken from ``mlp_in``'s output rows, which no supported quantization packs.
    """
    hidden_size, in_channels = _shape(state_dict, "emb_in.image.weight")
    cond_in_channels = _shape(state_dict, "emb_in.image_cond.weight")[1]
    joint_attention_dim = _shape(state_dict, "txt_in.weight")[1]
    timestep_channels = _shape(state_dict, "time_in.in_layer.weight")[1]
    head_dim = _shape(state_dict, "single_blocks.0.norm.query_norm.scale")[0]
    mlp_hidden = _shape(state_dict, "single_blocks.0.mlp_in.weight")[0] // 2

    if hidden_size % head_dim != 0:
        raise ValueError(f"FLUX.3 hidden size {hidden_size} is not a multiple of its head width {head_dim}.")
    if sum(FLUX3_AXES_DIMS_ROPE) != head_dim:
        raise ValueError(
            f"FLUX.3 attention head width {head_dim} does not match the rotary axes {FLUX3_AXES_DIMS_ROPE} of every "
            "released checkpoint; this checkpoint is not supported."
        )

    num_layers = _count_blocks(state_dict, "txt_mode_blocks.")
    for mode in ("image", "image_cond"):
        mode_layers = _count_blocks(state_dict, f"content_mode_blocks.{mode}.")
        if mode_layers != num_layers:
            raise ValueError(
                f"The FLUX.3 checkpoint has {mode_layers} '{mode}' blocks but {num_layers} text blocks; the per-mode "
                "stacks must have the same depth."
            )

    return {
        "in_channels": in_channels,
        "cond_in_channels": cond_in_channels,
        "num_layers": num_layers,
        "num_single_layers": _count_blocks(state_dict, "single_blocks."),
        "attention_head_dim": head_dim,
        "num_attention_heads": hidden_size // head_dim,
        "joint_attention_dim": joint_attention_dim,
        "timestep_channels": timestep_channels,
        "mlp_ratio": mlp_hidden / hidden_size,
        "axes_dims_rope": FLUX3_AXES_DIMS_ROPE,
        "flow_map": FLUX3_NATIVE_FLOW_MAP_KEY in state_dict,
    }
