"""The working memory of one Z-Image transformer forward, as the denoise node reserves it.

The transformer attends over one unified sequence of image and caption tokens, each run padded to a multiple of
``SEQ_MULTI_OF``. On a fused SDPA kernel its activations scale linearly with that sequence: `k_denoise_<dtype>` bytes
per attended token plus a base of `b_denoise_<dtype>` bytes, shipped per platform in
`backend/working_memory/data/denoise.json`. Conditional and unconditional passes run one after the other, so only the
longer caption counts. The platform keys are those of the VAE estimators; for a transformer, `conv_backend` just names
the CUDA (`cudnn`) or ROCm (`miopen`) stack it runs on.

That line is `denoise_<dtype>` over the attended tokens (`_line`), and only the plain forward fits it: every analytic
term below comes to zero, and no attention mask is passed, which narrows the kernels SDPA can pick. `_line` reads that
off the terms themselves, so a term added here takes the forwards it prices out of the fit. The fit takes a point's
demand as measured, so a forward with an analytic term bounds its estimate but fits nothing. A materialized score
matrix is no exception, although its term could be subtracted from the demand instead: the term is an upper bound
(`SDPA_MATH_BYTES_PER_SCORE_ELEMENT` against the 9.6 to 16.4 bytes per score measured, and a whole chunk under the ROCm
guard), so the line would come out low by its slack, and as the platform keys do not tell a build that materializes
from one with a fused kernel, that low line would price the fused build short. The line is the shipped fit's alone:
the spec does not opt in to user calibration (`EstimatorSpec.user_calibrated`), so a calibration run records the
forward's windows but no user line moves the estimate.

The lines are fitted per platform and dtype on fresh-process points of `scripts/calibrate_z_image_working_memory.py`
(their provenance is in the data): an RTX 4090 with the native allocator and with `cudaMallocAsync`, and an RX 9060 XT
under ROCm 10 with AOTriton's fused kernels, with expandable segments and with the native allocator on Windows. bf16,
fp16 and fp32 run at 512 to 2048px with a 512-token caption, and at 1024 and 1440px with a 77-token one, on a
random-init transformer of the real geometry with 2 main blocks: a no-grad forward frees each block's intermediates, so
4 blocks measured no more. bf16 and fp16 take the same per token, fp32 1.7 to 1.9 times that; on ROCm a process's first
forward maps ~250 MiB more than its second, which the base carries. In a running server, the real 30-block Z-Image
Turbo measured below its estimate on two of these configurations: the RTX 4090 under `cudaMallocAsync` in bf16, and
the RX 9060 XT with expandable segments as an int8 build, whose dequantization transient sits on top. That in-server
check of the native allocators is pending.

ROCm with the native allocator on every OS but Windows is not measured, and keeps the `legacy` constants the node
reserved by hand: 0.25 MiB per attended token and a 1 GiB base, the same for every dtype, from peak reserved memory in
bf16 on the RTX 4090. They cover bf16 and fp16 everywhere measured since, but not fp32 from 1440px up, whose measured
lines are steeper. Their base also covers the fp8 weights cast per forward, which an fp8-stored model reserves as a
term of its own as well, so it pays that cast twice wherever they price.

Every other platform nothing measures is priced as a measured one (`priced_as`): XPU, MPS, the CPU and cuDNN with
expandable segments as cuDNN with the native allocator, MIOpen under `hipMallocAsync` (and with expandable segments off
Windows) as MIOpen with the native allocator on its OS. A fit describes the card it was measured on, so the two fitted
entries they borrow, the RTX 4090's and the RX 9060 XT's on Windows with the native allocator, floor every value at
the legacy constants there (`unmeasured_floor`): no unmeasured platform reserves less than the node did by hand.

The analytic terms:

- Where SDPA has no fused kernel for these shapes it materializes the score matrices, and those dominate: on ROCm
  under Windows, whose fused kernels are disabled, a 1024px generation spilled 7.7 GB into shared system memory while
  the node reserved only the default working memory. `sdpa_score_matrix_bytes` prices the term only where it is really
  built. An attention mask narrows which fused kernels a build can use, so the call is told whether one is passed:
  regional prompting's additive bias, or the padding mask the ControlNet forward always builds. The plain forward
  passes none at batch size 1.
- Regional prompting's attention bias (`_regional_attention_bias_bytes`).
- A ControlNet adapter first runs its own blocks over a sequence of the same length and keeps one hint per block,
  alive through the main layers. Measured over the plain forward at 1024px: 3.3, 5.9 and 18.9 hint-sized tensors
  (sequence x width x element size) for 1, 6 and 15 blocks in the script's first runs, and at 15 blocks 19.5 on the
  RTX 4090 and 18.4 on the RX 9060 XT since. n + 4 are charged; with the line's margin that covers each of them.
- Half a GiB per LoRA.
- The dequantization copy an int8_tensorwise, nvfp4 or GGUF build materializes per quantized linear and forward,
  alive alongside the activations and invisible to them (`peak_dequant_transient_bytes`); zero for every other build.
- The compute-dtype copies of a layer's float8-stored weights that FP8 Storage, or a kept fp8 checkpoint, widens per
  forward (`peak_fp8_cast_transient_bytes`); zero for every other build.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Optional, cast

import torch

from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.fp8 import peak_fp8_cast_transient_bytes
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import Line
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.spec import EstimatorSpec, register
from invokeai.backend.z_image.z_image_patchify_utils import SEQ_MULTI_OF

Z_IMAGE_HIDDEN_SIZE: Final = 3840
"""The transformer's width, which its ControlNet adapters share."""
Z_IMAGE_ATTENTION_HEADS: Final = 30
"""Heads of 128."""

_CONTROL_HINTS_BEYOND_BLOCKS: Final = 4
_LORA_PATCH_BYTES: Final = 2**29
_DTYPE_NAMES: Final[Mapping[torch.dtype, str]] = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
"""The dtypes `TorchDevice.choose_bfloat16_safe_dtype` runs the transformer in."""
_ELEMENT_SIZES: Final[Mapping[str, int]] = {"bf16": 2, "fp16": 2, "fp32": 4}
_COEFFICIENTS: Final = frozenset(f"{part}_denoise_{dtype}" for part in ("k", "b") for dtype in _ELEMENT_SIZES)
_UNPRICED: Final[Mapping[str, float]] = MappingProxyType(dict.fromkeys(_COEFFICIENTS, 0.0))
"""Coefficient values that price the fitted terms at nothing: what remains of a forward's terms is what it adds on top
of its line."""


def padded_tokens(tokens: int) -> int:
    """The transformer pads the image and the caption token runs to a multiple of `SEQ_MULTI_OF` each."""
    return tokens + (-tokens) % SEQ_MULTI_OF


def _regional_attention_bias_bytes(
    regional_attn_mask: Optional[torch.Tensor], image_seq_len: int, dtype: torch.dtype
) -> int:
    """What `regional_forward` allocates per positive pass for regional prompting: a boolean copy of the mask, the
    additive bias over the padded unified sequence, and the image-to-image block `torch.where` fills it from. The mask
    itself is already on the device when the estimate is taken. Measured at 1024px on CUDA: 68 MiB over the plain
    forward, against 93 MiB charged."""
    if regional_attn_mask is None:
        return 0
    mask_side = regional_attn_mask.shape[0]
    unified_seq_len = padded_tokens(image_seq_len) + padded_tokens(mask_side - image_seq_len)
    return mask_side * mask_side + (unified_seq_len * unified_seq_len + image_seq_len * image_seq_len) * dtype.itemsize


def _line(features: Mapping[str, FeatureValue]) -> Optional[Line]:
    """The forward's line, `denoise_<dtype>` over the attended tokens. Only the plain forward fits it: no attention
    mask, and every analytic term of `_terms` zero (see the module docstring). The label names what puts a forward off
    the line: the mask, and each analytic term that is not zero, in bytes. A dtype without a line names none."""
    dtype, tokens = features.get("dtype"), features.get("tokens")
    if not isinstance(dtype, str) or dtype not in _ELEMENT_SIZES or not isinstance(tokens, int):
        return None
    on_top = [term for term in _terms(features, _UNPRICED) if term.coefficient is None and term.bytes]
    off_line = ["attention_mask"] if features.get("has_attention_mask") else []
    off_line += [f"{term.name}={int(term.bytes)}" for term in on_top]
    shape = f"{features.get('image_tokens')}+{features.get('text_tokens')}tok"
    return Line(f"denoise_{dtype}", tokens, not off_line, "-".join(("denoise", dtype, shape, *off_line)))


def _terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    tokens = cast(int, features["tokens"])
    dtype = cast(str, features["dtype"])
    control_blocks = cast(int, features["control_blocks"])
    hint_bytes = 0
    if control_blocks:
        hint_bytes = (
            (control_blocks + _CONTROL_HINTS_BEYOND_BLOCKS) * tokens * Z_IMAGE_HIDDEN_SIZE * _ELEMENT_SIZES[dtype]
        )
    return (
        WorkingMemoryTerm.fitted("activations", f"k_denoise_{dtype}", tokens, values),
        WorkingMemoryTerm.fitted("base", f"b_denoise_{dtype}", 1, values),
        WorkingMemoryTerm("regional_attention_bias", cast(int, features["regional_bias_bytes"]), True),
        WorkingMemoryTerm("attention_scores", cast(int, features["score_matrix_bytes"]), True),
        WorkingMemoryTerm("control_hints", hint_bytes, True),
        WorkingMemoryTerm("lora_patches", cast(int, features["lora_count"]) * _LORA_PATCH_BYTES, True),
        WorkingMemoryTerm("dequant_transient", cast(int, features["dequant_transient_bytes"]), True),
        WorkingMemoryTerm("fp8_cast_transient", cast(int, features["fp8_cast_transient_bytes"]), True),
    )


Z_IMAGE_DENOISE_SPEC = register(
    EstimatorSpec(
        id="denoise.z_image",
        version=1,
        coefficients=_COEFFICIENTS,
        platform_keys=("conv_backend", "allocator"),
        terms=_terms,
        line=_line,
    )
)


def estimate_z_image_denoise_working_memory(
    transformer: torch.nn.Module,
    *,
    image_seq_len: int,
    text_seq_len: int,
    lora_count: int,
    control_blocks: int = 0,
    regional_attn_mask: Optional[torch.Tensor] = None,
    has_attention_mask: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> WorkingMemoryEstimate:
    """The working memory one forward of `transformer` needs on `device` in `dtype`.

    `transformer` is the model about to be locked, read unlocked for its dequantization and fp8 cast transients.
    `image_seq_len` and `text_seq_len` are the unpadded token counts, the longer caption of a CFG pair. `control_blocks`
    is the ControlNet adapter's block count, `regional_attn_mask` regional prompting's mask, and `has_attention_mask`
    whether the forward passes an attention mask at all.
    """
    dtype_name = _DTYPE_NAMES.get(dtype)
    if dtype_name is None:
        raise ValueError(
            f"Working-memory estimator '{Z_IMAGE_DENOISE_SPEC.id}' has no {dtype} line on {platform_for(device)}: the "
            "transformer runs in a dtype it is not measured for."
        )
    image_tokens = padded_tokens(image_seq_len)
    text_tokens = padded_tokens(text_seq_len)
    tokens = image_tokens + text_tokens
    features: dict[str, FeatureValue] = {
        "image_tokens": image_tokens,
        "text_tokens": text_tokens,
        "tokens": tokens,
        "dtype": dtype_name,
        "control_blocks": control_blocks,
        "has_attention_mask": has_attention_mask,
        "regional_bias_bytes": _regional_attention_bias_bytes(regional_attn_mask, image_seq_len, dtype),
        "score_matrix_bytes": sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=Z_IMAGE_ATTENTION_HEADS,
            head_dim=Z_IMAGE_HIDDEN_SIZE // Z_IMAGE_ATTENTION_HEADS,
            seq_len=tokens,
            has_attn_mask=has_attention_mask,
            # The attention goes through diffusers' `dispatch_attention_fn`, which can route around torch's SDPA.
            via_diffusers_dispatch=True,
        ),
        "lora_count": lora_count,
        "dequant_transient_bytes": peak_dequant_transient_bytes(transformer, dtype),
        "fp8_cast_transient_bytes": peak_fp8_cast_transient_bytes(transformer, dtype),
    }
    return Z_IMAGE_DENOISE_SPEC.estimate(features, device)
