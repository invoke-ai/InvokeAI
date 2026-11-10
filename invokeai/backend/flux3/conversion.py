"""Native FLUX.3 state dicts, converted to diffusers' ``Flux3Transformer2DModel``.

Every single-file FLUX.3 build InvokeAI loads uses Black Forest Labs' own key names: BFL's
``model.safetensors`` and its quantized ``variants/``, the ComfyUI repackages, and the GGUF conversions. The
renames are a port of diffusers' ``convert_flux3_transformer_checkpoint_to_diffusers`` (``single_file_utils.py``
in diffusers 0.42.1). Reading the structure from the weights, rather than assuming the released one, is
InvokeAI's: a finetune that changes the depth must not load as the first N blocks of the released model.

Comfy-Org's ``int8_convrot`` build fuses each block's query, key and value into one ``qkv`` linear, which
diffusers keeps apart: it is split here, its quantization side channel with it. Where a native module lands is a
function of its name alone (:func:`flux3_native_module_to_diffusers`), because a fused ``qkv`` is split or
refused, never left fused. A loader therefore re-keys what it read under the native names -- int8 markers, header
entries, popped nvfp4 layers -- through that same function, and cannot drift from the conversion.
"""

import re
from collections.abc import Mapping
from typing import Any, TypeVar

import torch

from invokeai.backend.quantization.fp8_scaled import (
    FP8_WEIGHT_DTYPES,
    LAYER_SIDECHANNEL_SUFFIXES,
    WEIGHT_SCALE_SUFFIXES,
    split_qkv_sidechannel,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.nvfp4 import WEIGHT_SCALE_2_SUFFIX, NVFP4Payload, split_nvfp4_rows
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

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

_ADALN_MODULE = "final_layer.image.adaLN_modulation.1"
_ADALN_KEY = f"{_ADALN_MODULE}.weight"

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

# ComfyUI's per-block attention-kernel hint, `{"attention": "comfy_kitchen_int8"}` in the int8_convrot build. It
# selects one of ComfyUI's own kernels and says nothing about the weights.
_COMFY_ATTENTION_CONFIG_SUFFIX = ".comfy_attention.config"

# Comfy-Org's int8 build stores a block's query, key and value as one linear, rows in that order.
_FUSED_QKV_SUFFIX = ".qkv"
_QKV_PARTS = ("q_proj", "k_proj", "v_proj")

# A layer's quantization side channel, carried to wherever its weight lands.
_SIDE_CHANNEL_SUFFIXES = (WEIGHT_SCALE_2_SUFFIX, *LAYER_SIDECHANNEL_SUFFIXES)

_Entry = TypeVar("_Entry")


def _swap_shift_scale(weight: torch.Tensor) -> torch.Tensor:
    """BFL's output adaLN produces [shift, scale]; ``Flux3AdaLayerNorm`` expects [scale, shift].

    The halves are output rows. A packed GGML tensor is unpacked first: a row swap inside quantized blocks is not
    something a byte shuffle can express for every quantization type. int8 and float8 codes are swapped as stored,
    float8 as its bytes, which every supported torch build can concatenate; their per-row scale moves with them
    (:func:`_swap_scale_rows`).
    """
    if isinstance(weight, GGMLTensor):
        weight = weight.get_dequantized_tensor()
    if weight.dtype in FP8_WEIGHT_DTYPES:
        return _swap_shift_scale(weight.view(torch.uint8)).view(weight.dtype)
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _swap_scale_rows(suffix: str, value: Any) -> Any:
    """The output adaLN's weight scale, swapped with the rows it scales.

    A per-row scale (``[N]`` or ``[N, 1]``) has one entry per output row. A per-tensor scale, an activation scale
    and a marker describe the whole layer and stay as they are.
    """
    if suffix not in WEIGHT_SCALE_SUFFIXES or not isinstance(value, torch.Tensor) or value.numel() <= 1:
        return value
    return _swap_shift_scale(value)


def _is_discarded(path: str) -> bool:
    """Whether the conversion drops a native module, and with it its side channel and any packed form."""
    return path.startswith(_TRAINING_ONLY_PREFIX)


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


def _native_module_to_diffusers(path: str) -> str:
    """One native module's diffusers path. A norm keeps its parameter as ``scale`` and every other module as
    ``weight``; diffusers calls both ``weight``."""
    renamed = flux3_native_key_to_diffusers(f"{path}.scale")
    if not renamed.endswith(".weight"):
        renamed = flux3_native_key_to_diffusers(f"{path}.weight")
    return renamed[: -len(".weight")]


def flux3_native_module_to_diffusers(path: str) -> list[str]:
    """The diffusers module(s) a native FLUX.3 module becomes: a block's fused ``qkv`` three, any other one."""
    if path.endswith(_FUSED_QKV_SUFFIX):
        block = path[: -len(_FUSED_QKV_SUFFIX)]
        return [_native_module_to_diffusers(f"{block}.{part}") for part in _QKV_PARTS]
    return [_native_module_to_diffusers(path)]


def convert_flux3_layer_entries(entries: Mapping[str, _Entry]) -> dict[str, _Entry]:
    """Re-key per-layer entries read under native module paths -- int8 markers, ``_quantization_metadata`` hints --
    to the diffusers modules their weights become. A fused ``qkv``'s entry describes each of its three parts."""
    return {
        target: entry
        for path, entry in entries.items()
        if not _is_discarded(path)
        for target in flux3_native_module_to_diffusers(path)
    }


def _side_channel_parts(key: str) -> tuple[str, str] | None:
    """``(module path, suffix)`` for a quantization side-channel key, else None."""
    for suffix in _SIDE_CHANNEL_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    return None


def _split_fused_qkv(key: str, weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The query, key and value rows of a fused ``qkv`` weight.

    Rows split cleanly for int8 and float8 codes: their scales are per row or per tensor, and ``convrot`` rotates
    along the input dimension. A packed weight is refused: GGML blocks and nvfp4's two codes per byte are not split
    here. (A header-named nvfp4 ``qkv`` never gets here; :func:`convert_flux3_nvfp4_payloads` splits its packed form
    on whole scale tiles.)
    """
    if isinstance(weight, GGMLTensor) or weight.dtype is torch.uint8:
        raise ValueError(
            f"The FLUX.3 checkpoint stores the fused '{key}' packed, and a packed fused projection cannot be split "
            "into the separate query, key and value the model has. Use a build with q_proj, k_proj and v_proj, or the "
            "bf16 file."
        )
    if weight.dim() != 2 or weight.shape[0] % 3 != 0:
        raise ValueError(
            f"The FLUX.3 checkpoint's fused '{key}' has shape {tuple(weight.shape)}, which does not split into equal "
            "query, key and value rows."
        )
    return weight.chunk(3, dim=0)


def convert_flux3_native_to_diffusers(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Rename a native FLUX.3 transformer state dict to the diffusers module, in place of the input.

    A quantization side channel (weight and input scales, ``.comfy_quant`` markers) follows its weight: a fused
    ``qkv``'s is split with it (``split_qkv_sidechannel``), and the output adaLN's per-row weight scale is swapped
    with its rows. Left behind under a name the model does not have, a scale would be dropped and its weight loaded
    off by ``1/weight_scale``, with nothing in the log.

    The input dict is emptied as it is converted, so the caller's reference does not keep a second copy of every
    tensor alive.
    """
    converted: dict[str, Any] = {}
    attention_hints = 0
    for key in list(state_dict.keys()):
        value = state_dict.pop(key)
        if key.endswith(_COMFY_ATTENTION_CONFIG_SUFFIX):
            attention_hints += 1
            continue
        side_channel = _side_channel_parts(key)
        module = side_channel[0] if side_channel is not None else key.rpartition(".")[0]
        if _is_discarded(module):
            continue
        targets = flux3_native_module_to_diffusers(module)
        if side_channel is not None:
            suffix = side_channel[1]
            if module == _ADALN_MODULE:
                value = _swap_scale_rows(suffix, value)
            parts = split_qkv_sidechannel(key, value) if len(targets) > 1 else (value,)
            converted.update({f"{target}{suffix}": part for target, part in zip(targets, parts, strict=True)})
        elif len(targets) > 1 and key == f"{module}.weight":
            projections = _split_fused_qkv(key, value)
            converted.update({f"{target}.weight": part for target, part in zip(targets, projections, strict=True)})
        else:
            converted[flux3_native_key_to_diffusers(key)] = _swap_shift_scale(value) if key == _ADALN_KEY else value
    if attention_hints:
        logger.debug(f"FLUX.3: dropped {attention_hints} ComfyUI attention-kernel hint(s) (comfy_attention.config).")
    return converted


def convert_flux3_nvfp4_payloads(payloads: Mapping[str, NVFP4Payload]) -> dict[str, NVFP4Payload]:
    """Re-key packed nvfp4 layers, taken out of a native state dict, to the diffusers modules they become.

    The packed counterpart of :func:`convert_flux3_native_to_diffusers`, for the layers a loader takes out before
    converting: a fused ``qkv`` splits into thirds and the output adaLN swaps its halves, both on whole 128-row bands
    (``split_nvfp4_rows``). The codes stay as stored, and so does the grid, in either of its layouts: a band of
    ComfyUI's cuBLAS tiles is a band of rows, as it is of ModelOpt's row-major grid, which is therefore re-laid out
    after this as well as before.
    """
    converted: dict[str, NVFP4Payload] = {}
    for path, payload in payloads.items():
        if _is_discarded(path):
            continue
        if path == _ADALN_MODULE:
            shift, scale = split_nvfp4_rows(path, payload, 2)
            # The block scales as bytes, which every supported torch build can concatenate.
            grid = torch.cat([scale.weight_scale.view(torch.uint8), shift.weight_scale.view(torch.uint8)])
            payload = NVFP4Payload(
                weight=torch.cat([scale.weight, shift.weight]),
                weight_scale=grid.view(payload.weight_scale.dtype),
                weight_scale_2=payload.weight_scale_2,
            )
        targets = flux3_native_module_to_diffusers(path)
        parts = split_nvfp4_rows(path, payload, len(targets)) if len(targets) > 1 else [payload]
        converted.update(zip(targets, parts, strict=True))
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

    Only shapes are read, so this works on meta tensors and on packed GGUF tensors alike. The feedforward width is
    taken from ``mlp_in``'s output rows, which no supported quantization packs. A loader that takes packed nvfp4
    layers out of the state dict reads this first: ``mlp_in`` is one of them. The input widths are read off the
    embedders, which no supported build packs (``flux3/checkpoint_quantization.py`` refuses one that does).
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
