"""Shared helpers for folding single-file checkpoint quantization scales into weights.

Both scaled formats we support - ComfyUI "scaled fp8" / MXFP8 (:mod:`.scaled_fp8`) and NVFP4
(:mod:`.nvfp4`) - dequantize by multiplying a low-precision weight by one or more scale tensors.
They share three concerns, collected here: reading tensors that may still be GGML-wrapped, choosing
the dtype the dequantized result is *stored* at, and reporting what a checkpoint says it contains.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

#: Metadata key ComfyUI/ModelOpt exports use to declare per-layer quantization.
QUANTIZATION_METADATA_KEY = "_quantization_metadata"

#: Module-path regexes for precision-sensitive layers that should never be stored as fp8. Mirrors
#: diffusers' ``DEFAULT_SKIP_MODULES_PATTERN`` - without these, FLUX RMSNorm.scale and similar tiny
#: learned scalars get crushed to fp8 and inference quality degrades. Matched both against
#: ``named_modules()`` dotted paths by the layerwise-casting pass and against state-dict key prefixes
#: by the dequantizers here, so that both routes to fp8 storage make the same exclusions.
FP8_STORAGE_SKIP_PATTERNS: tuple[str, ...] = (
    "pos_embed",
    "patch_embed",
    "norm",
    r"^proj_in$",
    r"^proj_out$",
)


def to_plain_tensor(value: Any) -> Any:
    """Dequantize a GGMLTensor to a plain tensor; pass anything else through."""
    if hasattr(value, "get_dequantized_tensor"):
        return value.get_dequantized_tensor()
    return value


def resolve_target_dtype(
    name: str,
    compute_dtype: torch.dtype,
    storage_dtype: torch.dtype | None,
    skip_patterns: Sequence[str],
) -> torch.dtype:
    """Pick the dtype a dequantized weight should be *stored* at.

    ``storage_dtype`` is how the caller intends to keep the weight resident - fp8, for the
    fp8-storage setting. Casting to it here, one tensor at a time as it is dequantized, is what
    keeps peak host RAM at the fp8 model size. Dequantizing the whole checkpoint to
    ``compute_dtype`` and re-quantizing afterwards transiently needs the full bf16 model, which for
    a 12.8 B-parameter Krea-2 transformer is 23.9 GiB - unloadable on a 32 GB host, which also has
    to hold a ~5 GiB text encoder. The two are numerically equivalent: both round the same
    dequantized value to fp8 once.

    ``skip_patterns`` mirrors the caller's layerwise-casting skip list, so a weight it would have
    deliberately preserved at full precision (norms and embeddings, whose tiny learned scalars are
    crushed by fp8) is not pre-emptively downcast here instead.
    """
    if storage_dtype is None:
        return compute_dtype
    if any(re.search(pattern, name) for pattern in skip_patterns):
        return compute_dtype
    return storage_dtype


def read_declared_quantization_formats(model_path: Path) -> set[str]:
    """Return the lower-cased quantization formats a safetensors file declares, if any.

    ComfyUI and NVIDIA TensorRT Model Optimizer both stamp a ``_quantization_metadata`` JSON blob
    into the safetensors header, mapping each quantized layer to a ``format`` (``"nvfp4"``,
    ``"mxfp8"``, ...). Reading it lets a loader reject a format it cannot handle with the format's
    name, instead of letting a wrongly-shaped state dict reach ``load_state_dict`` and emit one
    "size mismatch for ..." line per quantized layer - hundreds of lines that name every tensor and
    diagnose nothing.

    Returns an empty set for unquantized checkpoints, for quantized ones that ship no metadata, and
    for files that are not safetensors at all (an empty set means "nothing declared", never "nothing
    quantized" - detection still has to be driven by the actual keys, so this is only ever used to
    reject, never to accept).
    """
    from safetensors import safe_open

    if model_path.suffix != ".safetensors":
        return set()

    # Best-effort throughout: this only ever *rejects* a load, so an unreadable or malformed header
    # must fall through to the normal path rather than become a new failure mode of its own. Whatever
    # is actually wrong with the file will surface with a better message from the real load.
    try:
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata() or {}
        parsed = json.loads(metadata.get(QUANTIZATION_METADATA_KEY) or "")
    except Exception:
        return set()

    layers = parsed.get("layers") if isinstance(parsed, dict) else None
    if not isinstance(layers, dict):
        return set()

    return {
        str(info["format"]).lower()
        for info in layers.values()
        if isinstance(info, dict) and info.get("format") is not None
    }
