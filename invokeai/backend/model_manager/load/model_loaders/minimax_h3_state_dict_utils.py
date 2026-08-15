"""State-dict conversion for MiniMax H3 single-file transformer checkpoints.

The MiniMax/Comfy-Org single-file H3 transformers (both bf16 and int8-convrot
quantized, full and AdaLN-pruned) use the original remote-code key layout. This
module renames it to the vendored diffusers-style
``MiniMaxH3Transformer3DModel`` / ``MiniMaxH3PrunedTransformer3DModel`` layout:

- ``blocks.N.*``                     -> ``transformer_blocks.N.*``
- ``token_refiner.blocks.N.*``       -> ``token_refiner.refiner_blocks.N.*``
- ``attn.qkv_proj`` (fused)          -> row-split into ``attn.to_q/to_k/to_v``
  (``weight_scale`` rows split identically — scales are per output channel)
- ``attn.out_proj``                  -> ``attn.to_out.0``
- ``attn.q_norm`` / ``attn.k_norm``  -> ``attn.norm_q`` / ``attn.norm_k``
- ``mlp.fc1`` (fused SwiGLU)         -> ``ff.net.0.proj`` (halves SWAPPED: [gate; value] -> [value; gate])
- ``mlp.fc2``                        -> ``ff.net.2``
- ``final_layer.norm``               -> ``norm_out.norm``
- ``final_layer.adaln_proj.linear``  -> ``norm_out.linear``
- ``final_layer.video_out``          -> ``proj_out``
- ``final_layer.audio_out``          -> ``audio_proj_out``
- ``video_patch_proj``               -> ``proj_in``
- ``audio_patch_proj``               -> ``audio_proj_in``
- ``condition_proj``                 -> ``context_embedder``
- ``rope.inv_freq``                  -> dropped (computed, non-persistent buffer)
- ``adaln_t_table``                  -> unchanged (pruned checkpoints only)
- ``<layer>.comfy_quant``            -> removed from the state dict; returned as a
  parsed marker dict keyed by the CONVERTED module name (a fused qkv marker fans
  out to all three of ``to_q``/``to_k``/``to_v``).
"""

import json
import struct
from pathlib import Path
from typing import Any

import torch

from invokeai.backend.minimax_h3.int8_convrot import parse_comfy_quant_marker


def read_comfy_quant_markers(path: Path) -> dict[str, dict[str, Any]]:
    """Read every ``<layer>.comfy_quant`` marker from a safetensors file WITHOUT loading tensor
    data - header parse plus a seek per marker blob. Keys are the raw (un-renamed) layer names.

    Lets the loader reject unsupported quantization formats (e.g. the fp8_scaled repacks, which
    share this key layout) before committing to a ~20 GiB read.
    """
    markers: dict[str, dict[str, Any]] = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        for key, entry in header.items():
            if not key.endswith(".comfy_quant"):
                continue
            start, end = entry["data_offsets"]
            f.seek(8 + header_len + start)
            markers[key[: -len(".comfy_quant")]] = json.loads(f.read(end - start).decode("utf-8"))
    return markers


_QKV_SUFFIXES = (".weight", ".weight_scale")

# Applied after the block-prefix renames; needles include the surrounding dots so they can
# only match whole path segments.
_SUBKEY_RENAMES = (
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
    (".attn.out_proj.", ".attn.to_out.0."),
    (".mlp.fc1.", ".ff.net.0.proj."),
    (".mlp.fc2.", ".ff.net.2."),
)

_PREFIX_RENAMES = (
    # Order matters: token_refiner.blocks. must be rewritten before the bare blocks. prefix.
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("blocks.", "transformer_blocks."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.video_out.", "proj_out."),
    ("final_layer.audio_out.", "audio_proj_out."),
    ("video_patch_proj.", "proj_in."),
    ("audio_patch_proj.", "audio_proj_in."),
    ("condition_proj.", "context_embedder."),
    # Full (non-pruned) checkpoints only: the timestep MLP maps onto diffusers'
    # TimestepEmbedding attribute names (linear_1: freq_dim -> hidden, linear_2: hidden -> out).
    ("time_embedder.proj_in.", "time_embedder.linear_1."),
    ("time_embedder.proj_out.", "time_embedder.linear_2."),
)


def _rename_key(key: str) -> str:
    for old, new in _PREFIX_RENAMES:
        if key.startswith(old):
            key = new + key[len(old) :]
            break
    for old, new in _SUBKEY_RENAMES:
        key = key.replace(old, new)
    return key


def convert_minimax_h3_checkpoint_to_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Convert a remote-code-layout H3 state dict to the vendored diffusers layout.

    Returns ``(converted_state_dict, quant_markers)`` where ``quant_markers`` maps each
    converted quantized MODULE name (e.g. ``transformer_blocks.0.attn.to_q``) to its parsed
    ``comfy_quant`` JSON dict. Non-quantized checkpoints return an empty marker dict.
    """
    converted: dict[str, torch.Tensor] = {}
    markers: dict[str, dict[str, Any]] = {}

    for key, tensor in state_dict.items():
        if key == "rope.inv_freq":
            continue

        if key.endswith(".comfy_quant"):
            module_name = _rename_key(key[: -len(".comfy_quant")] + ".weight")[: -len(".weight")]
            marker = parse_comfy_quant_marker(tensor)
            if module_name.endswith(".attn.qkv_proj"):
                stem = module_name[: -len("qkv_proj")]
                for proj in ("to_q", "to_k", "to_v"):
                    markers[stem + proj] = marker
            else:
                markers[module_name] = marker
            continue

        new_key = _rename_key(key)

        qkv_at = new_key.find(".attn.qkv_proj")
        if qkv_at != -1:
            suffix = new_key[qkv_at + len(".attn.qkv_proj") :]
            if suffix not in _QKV_SUFFIXES:
                raise ValueError(f"Unexpected fused-qkv key {key!r} (suffix {suffix!r})")
            if tensor.shape[0] % 3 != 0:
                raise ValueError(f"Fused qkv tensor {key!r} has {tensor.shape[0]} rows, not divisible by 3")
            rows = tensor.shape[0] // 3
            stem = new_key[: qkv_at + len(".attn.")]
            for i, proj in enumerate(("to_q", "to_k", "to_v")):
                converted[stem + proj + suffix] = tensor[i * rows : (i + 1) * rows]
            continue

        # The fused SwiGLU input projection's halves are ordered [gate; value] in the
        # remote-code layout (silu on the FIRST half) but [value; gate] in diffusers'
        # SwiGLU (silu on the SECOND half) - verified bit-exactly against the diffusers
        # folder: file[:H] == folder[H:] and file[H:] == folder[:H]. Swap the halves,
        # including the per-output-row scales of quantized fc1 layers. Without this swap
        # every block's MLP computes silu(value)*gate and the model emits garbage.
        if ".ff.net.0.proj." in new_key:
            if tensor.shape[0] % 2 != 0:
                raise ValueError(f"Fused SwiGLU tensor {key!r} has odd row count {tensor.shape[0]}")
            half = tensor.shape[0] // 2
            converted[new_key] = torch.cat([tensor[half:], tensor[:half]], dim=0)
            continue

        converted[new_key] = tensor

    return converted, markers
