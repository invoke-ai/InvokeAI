"""The single-file layouts MOSS-Image ships in, and how they become the vendored transformer's modules.

Comfy-Org's export and leejet's GGUFs fuse each block's attention projections into one `layers.N.proj.qkv` tensor
(`q`, `k`, `v` stacked along the output dimension, in that order). The diffusers checkpoint and unsloth's GGUFs keep
`layers.N.proj.q`, `.k` and `.v`, which is what `MOSSImageTransformer` has. The text refiner's `text_embedder.blocks.N.qkv`
is fused in every layout and is a module of its own, so it is left alone.
"""

import re
from typing import Any

import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.row_ops import split_rows

_FUSED_QKV = re.compile(r"^(layers\.\d+\.proj)\.qkv\.(.+)$")

# Sidecars that describe the whole tensor, not its rows, and so belong to all three projections unchanged.
_SHARED_SIDECARS = ("input_scale", "scale_input", "comfy_quant")

_COMFY_ATTENTION_CONFIG = ".comfy_attention.config"
"""ComfyUI's per-layer record of the attention kernel an export was calibrated with. Nothing here reads it."""


def split_fused_qkv(sd: dict[str, Any]) -> dict[str, Any]:
    """Split every fused `layers.N.proj.qkv` into `proj.q`, `proj.k` and `proj.v`, in place; returns `sd`.

    Runs before geometry detection and before any quantization handling, so the loaders see one module per Linear:

    - a dense or int8 weight splits into per-part copies;
    - a per-output-channel sidecar (an int8 `weight_scale` of shape `[3 * dim, 1]`) splits with its rows, and a
      per-tensor one, like the `comfy_quant` marker, is shared by all three;
    - a GGML tensor is cut on its packed rows without being dequantized: GGML blocks run along a row, never across
      rows (see `row_ops.split_rows`).

    Every part owns its storage, whatever the dtype: a view would keep the whole fused tensor alive for as long as any
    one projection lives (the dense `q` pins the rows of `k` and `v` after they were cast or swapped), memory the model
    cache never sees. The fused tensor is released once its layer is done, so only one layer's copies exist at a time.

    A state dict without fused tensors passes through untouched.
    """
    fused = {key: match.groups() for key in list(sd) if (match := _FUSED_QKV.match(key))}
    if not fused:
        return sd
    rows_by_block: dict[str, int] = {}
    for block, _ in fused.values():
        if block in rows_by_block:
            continue
        if f"{block}.qkv.weight" not in sd:
            raise ValueError(f"{block}.qkv has sidecars but no fused weight to split with")
        rows = sd[f"{block}.qkv.weight"].shape[0]
        if rows % 3:
            raise ValueError(f"{block}.qkv.weight has {rows} rows, which do not split into q, k and v")
        rows_by_block[block] = rows
    for key, (block, suffix) in fused.items():
        rows = rows_by_block[block]
        dim = rows // 3
        value = sd.pop(key)
        per_row = (
            suffix not in _SHARED_SIDECARS
            and isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == rows
        )
        if per_row:
            q, rest = split_rows(value, dim)
            k, v = split_rows(rest, dim)
            if not isinstance(value, GGMLTensor):
                # `split_rows` hands plain tensors back as views; GGML parts are already clones.
                q, k, v = q.clone(), k.clone(), v.clone()
        else:
            q = k = v = value
        for name, part in zip("qkv", (q, k, v), strict=True):
            part_key = f"{block}.{name}.{suffix}"
            if part_key in sd:
                raise ValueError(f"{key} is fused but the checkpoint also holds {part_key}")
            sd[part_key] = part
    return sd


def drop_comfy_attention_config(sd: dict[str, Any]) -> None:
    """Remove ComfyUI's per-layer attention-kernel records, which no loader here consumes."""
    for key in [key for key in sd if isinstance(key, str) and key.endswith(_COMFY_ATTENTION_CONFIG)]:
        del sd[key]
