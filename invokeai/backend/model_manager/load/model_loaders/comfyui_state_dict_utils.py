"""Helpers for normalising ComfyUI-flavoured single-file checkpoints.

Community single-file releases (CivitAI, ComfyUI-oriented Hugging Face repos)
share a small set of conventions regardless of which architecture they wrap:
an optional ``model.diffusion_model.`` key prefix, and optional fp8 weights
paired with per-tensor scale factors. These helpers undo both so a state dict
can be handed to a plain diffusers module.

Originally written for the Qwen Image loader; shared so the Wan loader doesn't
need a second copy.
"""

import torch

from invokeai.backend.quantization.fp8_scaled import (
    expand_weight_scale,
    is_scale_metadata_key,
    reject_undecoded_mx_scale,
)
from invokeai.backend.quantization.int8_convrot import reject_int8_layers_a_plain_fold_cannot_decode
from invokeai.backend.quantization.nvfp4 import reject_nvfp4_layers_a_plain_fold_cannot_decode


def _dequantize_comfyui_fp8(sd: dict, compute_dtype: torch.dtype, what: str = "This checkpoint") -> int:
    """Dequantize ComfyUI-style fp8_scaled weights in-place. Returns count of dequantized tensors.

    Weights are dequantized directly to `compute_dtype` (typically bf16) instead of via a
    full-precision float32 intermediate. The previous float32 path materialised a complete
    4-byte/param copy of the model before a separate downcast pass, spiking peak RAM to ~2x the
    final bf16 size (~80GB for the 20B Qwen-Image transformer). Multiplying in the target dtype
    keeps the dict at the bf16 model size plus a single transient tensor. fp8 has only 3 mantissa
    bits and bf16 shares float32's exponent range, so the bf16 multiply loses no meaningful
    precision here.

    Two key naming schemes are in the wild:
      - `<path>.weight` + `<path>.weight_scale`  (FLUX, Z-Image style)
      - `<path>.weight` + `<path>.scale_weight`  (Qwen2.5-VL fp8_scaled style, also
        emits `<path>.scale_input` for activation scaling that we discard).

    The scale is applied whatever dtype the weight is stored in. There is deliberately no
    "only if the weight is fp8" gate: not every checkpoint using these keys stores fp8
    weights, and skipping the multiply for those would produce a silently wrong model in
    the same way applying a stale scale would. The assumption is that a scale key present
    in the file is a scale that still needs applying — i.e. a checkpoint must not ship
    already-dequantized weights alongside their scales.
    """
    reject_nvfp4_layers_a_plain_fold_cannot_decode(sd, what)
    reject_int8_layers_a_plain_fold_cannot_decode(sd, what)
    scale_suffixes = (".weight_scale", ".scale_weight")
    weight_scale_keys = [k for k in sd.keys() if isinstance(k, str) and k.endswith(scale_suffixes)]
    count = 0
    for scale_key in weight_scale_keys:
        for suffix in scale_suffixes:
            if scale_key.endswith(suffix):
                weight_key = scale_key[: -len(suffix)] + ".weight"
                break
        if weight_key not in sd:
            continue
        # Before the cast: `.to(compute_dtype)` on an E8M0 grid converts the exponent bytes to
        # floats and loses the only evidence of what they were.
        reject_undecoded_mx_scale(weight_key[: -len(".weight")], sd[scale_key])
        weight = sd[weight_key].to(compute_dtype)
        scale = sd[scale_key].to(compute_dtype)
        # Through the shared expansion rather than a local copy of it. The copy that used to live
        # here compared `scale.shape[dim] != weight.shape[dim]`, so a per-output-channel scale --
        # whose length already equals the number of rows -- was expanded by nothing and left to
        # broadcasting, which aligns on the *last* axis. That scales columns instead of rows: a
        # shape error on a non-square weight, and on a square one (every attention out-projection)
        # a model that loads and generates noise.
        sd[weight_key] = weight * expand_weight_scale(weight, scale, weight_key)
        count += 1
    return count


def _strip_quantization_metadata(sd: dict) -> None:
    """Strip ComfyUI fp8 quantization metadata keys in-place.

    Through the shared predicate rather than a second list of suffixes. The list that used to stand
    here was a clause-for-clause copy of it that had fallen one spelling behind: it dropped
    ``.scale_input`` but not ``.input_scale``, which is the spelling Comfy actually writes, so on
    the Wan path an ordinary scaled-fp8 checkpoint was rejected outright by
    `_raise_for_incompatible_keys` as an unsupported variant. A copy that has to be kept in step by
    hand will fall behind again the next time a spelling is added.
    """
    for key in [k for k in sd if is_scale_metadata_key(k)]:
        del sd[key]
