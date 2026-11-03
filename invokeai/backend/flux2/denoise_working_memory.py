"""The working memory of one FLUX.2 transformer forward, as the denoise node reserves it.

The transformer attends over one joint sequence per sample: the caption's tokens, the image's and those of every
reference image, which `Flux2RefImageExtension` concatenates onto the image stream and repeats per sample. On a fused
SDPA kernel its activations scale linearly with that sequence and with the batch, which runs as that many sequences:
`k_denoise_bf16_h<width>` bytes per attended token per sample plus a base of `b_denoise_bf16_h<width>` bytes, shipped
per platform in `backend/working_memory/data/denoise.json`. The cost per token follows the transformer's width (Klein
4B 3072, Klein 9B 4096, [dev] 6144), so each width has a line of its own. The node runs the transformer in bf16 only.
Conditional and unconditional passes run one after the other, so only the longer caption counts. The platform keys
are those of the VAE estimators; for a transformer, `conv_backend` just names the CUDA (`cudnn`) or ROCm (`miopen`)
stack it runs on.

That line is `denoise_bf16_h<width>` over the attended tokens times the batch (`_line`), and only the plain forward
fits it: every analytic term below comes to zero, and no attention mask is passed. A batch fits too, because the line
prices exactly what a batch adds, its sequences, and a forward's base is paid once whatever the batch. `_line` reads
the fit off the terms themselves, so a term added here takes the forwards it prices out of the fit. The fit takes a
point's demand as measured, so a forward with an analytic term bounds its estimate but fits nothing: a materialized
score matrix's term is an upper bound, and a line fitted on the demand less that term would come out low by its slack.
The spec does not opt in to user calibration (`EstimatorSpec.user_calibrated`): the lines are the shipped fit's alone.

The lines are fitted per platform and width on fresh-process points of
`scripts/calibrate_flux2_denoise_working_memory.py` (their provenance is in the data): an RTX 4090 with the native
allocator and with `cudaMallocAsync`, and an RX 9060 XT under ROCm 10 with AOTriton's fused kernels, with expandable
segments and with the native allocator on Windows. Per width they run 512 to 2048px with a 512-token caption, 1024 and
1440px with a 77-token one, at 1024px one and three 1024px references and a batch of two, and long sequences: 2048px
with one 2024px reference (33025 tokens), 1536px with one (25857) and 1328px with three 1328px references, resized to
1024px each (19689). They run on a random-init transformer of the loader's geometry with 2 double-stream and 2
single-stream blocks, so the lines are measured up to 33025 tokens, except where a forward does not fit the card: [dev]
at 33025 tokens on the RTX 4090 under the native allocator, Klein 9B at 33025 on the RX 9060 XT under the native
allocator, and on the RX 9060 XT [dev] above 1440px and with three references, so its [dev] lines there come from 512 to
1440px, one reference included. Reference images and the batch lie on the line of the plain forwards. 4 blocks per
stream measured the same as 2 on the RTX 4090 and 5 to 7% more on the RX 9060 XT at 1024px (Klein 4B and 9B widths). The
real models in a running server demanded less than the 2-block point at the same tokens: 0.96 to 0.99 times it for an
fp8 Klein 4B and 0.96 to 0.97 for an fp8 Klein 9B on the RTX 4090 under the native allocator, 0.81 to 0.95 for a bf16
Klein 4B and 0.71 to 0.77 for an fp8 Klein 9B (after its 576 MiB fp8 cast term) on the RX 9060 XT with expandable
segments. On ROCm a process's first forward maps ~245 MiB more than its second, which the base carries.

The node caps neither the image's side nor the number of references (each of several is resized to 1024px, 4096 tokens),
so a longer sequence prices on its line extrapolated linearly. That is expected to hold: on fused attention kernels the
activations grow linearly with the sequence, and from 16896 to 33025 tokens the line reserves a flat 1.05 to 1.13 times
what each measured forward demands under the native allocator and expandable segments. Under `cudaMallocAsync` the
33025-token forwards decide the line (1.05) and the shorter ones sit up to 1.22 below it.

ROCm with the native allocator on every OS but Windows is not measured, and keeps the `legacy` constants the node
reserved by hand: 0.42 MiB per attended token at the Klein 9B width, scaled linearly by the width and truncated to
whole bytes per width (330300, 440401 and 660601), and a 1 GiB base. They lie above every measured line. Every other
platform nothing measures is priced as a measured one (`priced_as`), and the two fitted entries those borrow, the RTX
4090's and the RX 9060 XT's on Windows with the native allocator, floor every value at the legacy constants there
(`unmeasured_floor`): no unmeasured platform reserves less than the node did by hand. The legacy base also covers the
fp8 weights cast per forward, which an fp8-stored model reserves as a term of its own as well, so it pays that cast
twice wherever they price.

CUDA on Linux reuses the RTX 4090's Windows-measured lines as they are, without the unmeasured floor: its entries are
keyed for every OS (`platform.MEASURED`), as the Z-Image and VAE estimators' cuDNN entries are. That is a judgement,
not a measurement. A forward's activations are torch tensors from the same caching allocator over the same attention
and GEMM kernels on both systems; what made MIOpen's Windows lines differ, HIP's runtime and allocator under Windows, has
no counterpart on CUDA, where the Windows driver model, if anything, maps more per allocation than Linux does. A floor
there would charge the legacy 1 GiB base to the platform most FLUX.2 users run on, for a gap nothing has shown.

The analytic terms:

- The largest layer's stored weights, for a transformer the model cache may load partially: the weights it leaves in
  RAM are copied to the device per forward, one layer at a time, and the copy is alive during its layer's matmul
  alongside the activations. No point prices it: every point is a fully resident model. The size is read off the
  transformer, the largest module's own parameters and buffers as stored (`largest_module_bytes`, as the cache counts
  them): in bf16 the single-stream blocks' fused projection (`to_qkv_mlp_proj`, ``width x 9 width``), 679,477,248
  bytes for [dev]; for an fp8-stored or GGUF build its float8 or quantized bytes, whose widened copy is the fp8 cast or
  dequant transient's term. A real [dev] in bf16 with 15.0 of its 64.4 GB resident on the RTX 4090 (native
  allocator) demanded 5509 MB at 8704 tokens (1024px with one 1024px reference), 5459 MB at 8612 (1440px) and 3305 MB
  at 4608 (1024px), 535, 518 and 719 MB above the resident 2-block points; the line alone reserved 0.96, 0.96 and 0.85
  times that, with this term 1.08, 1.08 and 1.05 times. The node charges it on every forward: whether the cache loads
  the transformer whole is decided by the lock this estimate sizes, after it, so residency is unknown when the
  estimate is taken. A fully resident model over-reserves one layer. The calibration script, whose models are always
  resident, declares them so (`weights_resident`), and the term is zero.
- Regional prompting's additive attention bias, one ``S x S`` tensor in the compute dtype broadcast over the batch and
  the heads (`Flux2RegionalPromptingExtension.get_joint_attention_kwargs`). The node passes no mask with reference
  images, which the mask does not cover.
- With that bias, a second copy of it per sample, its last dimension padded to a multiple of 8: on CUDA a masked
  forward goes to the memory-efficient kernel, which materializes the bias in a layout of its own. On the RTX 4090
  (native allocator, 2-block Klein 9B width, 1440px with two 512-token regional captions, S = 9124, a 166.5 MB bias)
  the masked forward took 158 MB more of the driver's figure and 168 MB more reserved than the unmasked one, about one
  bias-sized transient, with allocated unchanged; its demand, 3761 MB with the bias, is 1.03 times below what the
  estimate reserved without this term. No fit point is masked, so the copy is priced on every platform, aligned or not.
- Where SDPA has no fused kernel for these shapes it materializes the score matrices, shaped (batch, heads, S, S):
  `sdpa_score_matrix_bytes` prices them only where they are really built. A mask narrows which fused kernels a build
  can use, so the call is told whether one is passed.
- The FLUX.2 [dev] Fun-ControlNet's hints. The branch keeps one hint per block alive until the last base block it feeds
  has consumed it, and a hint covers the image stream only, reference tokens included: its double-stream blocks carry a
  text stream of their own, but `add_hint` touches only `hidden_states`. `n + 4` hint-sized tensors (batch x image
  stream x width x element size) are charged for `n` blocks, the coefficient the Z-Image estimator measured for its
  adapter over the plain forward. It is not measured for FLUX.2.
- Half a GiB per LoRA and sample: a sidecar patch adds an activation branch per patched layer.
- The dequantization copy an int8_tensorwise, nvfp4 or GGUF build materializes per quantized linear and forward,
  alive alongside the activations and invisible to them (`peak_dequant_transient_bytes`); zero for every other build.
- The compute-dtype copies of a layer's float8-stored weights that FP8 Storage, or a kept fp8 checkpoint, widens per
  forward (`peak_fp8_cast_transient_bytes`); zero for every other build.
"""

from collections.abc import Mapping
from typing import Final, Optional, cast

import torch

from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.calc_tensor_size import largest_module_bytes
from invokeai.backend.util.fp8 import peak_fp8_cast_transient_bytes
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import Line, off_line_reasons
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.spec import EstimatorSpec, register

FLUX2_ATTENTION_HEAD_DIM: Final = 128
"""Every variant's head width; the head count follows the hidden size (Klein 4B 24, Klein 9B 32, [dev] 48)."""
FLUX2_HIDDEN_SIZES: Final = (3072, 4096, 6144)
"""The transformer widths with a line: Klein 4B, Klein 9B and [dev]."""

_BIAS_ALIGNMENT: Final = 8
"""The last-dimension alignment the memory-efficient attention kernel pads an attention bias to."""
_CONTROL_HINTS_BEYOND_BLOCKS: Final = 4
_LORA_PATCH_BYTES: Final = 2**29
_DTYPE_NAMES: Final[Mapping[torch.dtype, str]] = {torch.bfloat16: "bf16"}
"""The dtypes the node runs the transformer in."""
_ELEMENT_SIZES: Final[Mapping[str, int]] = {"bf16": 2}
_LINES: Final = tuple(f"denoise_{dtype}_h{width}" for dtype in _DTYPE_NAMES.values() for width in FLUX2_HIDDEN_SIZES)
_COEFFICIENTS: Final = frozenset(f"{part}_{line}" for part in ("k", "b") for line in _LINES)


def _suffix(features: Mapping[str, FeatureValue]) -> Optional[str]:
    """The line a forward's dtype and width price on, None for a pair that has none."""
    dtype, width = features.get("dtype"), features.get("hidden_size")
    suffix = f"denoise_{dtype}_h{width}"
    return suffix if suffix in _LINES else None


def _line(features: Mapping[str, FeatureValue]) -> Optional[Line]:
    """The forward's line, `denoise_<dtype>_h<width>` over the attended tokens of every sample. Only the plain forward
    fits it: no attention mask, and every analytic term of `_terms` zero (see the module docstring). The label names
    what puts a forward off the line: the mask, and each analytic term that is not zero, in bytes. A dtype or width
    without a line names none."""
    suffix = _suffix(features)
    tokens, batch = features.get("tokens"), features.get("batch_size")
    if suffix is None or not isinstance(tokens, int) or not isinstance(batch, int):
        return None
    off_line = off_line_reasons(_terms, features, _COEFFICIENTS)
    references = features.get("reference_tokens")
    shape = "+".join(
        (
            str(features.get("image_tokens")),
            *([f"{references}ref"] if references else []),
            f"{features.get('text_tokens')}tok",
        )
    )
    batched = [f"b{batch}"] if batch != 1 else []
    label = "-".join(("denoise", str(features["dtype"]), f"h{features['hidden_size']}", shape, *batched, *off_line))
    return Line(suffix, tokens * batch, not off_line, label)


def _terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    suffix = _suffix(features)
    if suffix is None:
        raise ValueError(f"no denoise line for {features.get('dtype')} at width {features.get('hidden_size')}")
    batch = cast(int, features["batch_size"])
    # Absent from the points measured before the control branch existed, all of them forwards without it.
    control_blocks = cast(int, features.get("control_blocks", 0))
    hint_bytes = 0
    if control_blocks:
        stream = cast(int, features["image_tokens"]) + cast(int, features["reference_tokens"])
        hint_bytes = (
            (control_blocks + _CONTROL_HINTS_BEYOND_BLOCKS)
            * stream
            * batch
            * cast(int, features["hidden_size"])
            * _ELEMENT_SIZES[cast(str, features["dtype"])]
        )
    # Both absent from the points measured before these terms existed, which are fully resident forwards without a
    # mask: zero is what such a point streams and pads, though a forward the node prices now always carries them.
    streamed_layer_bytes = cast(int, features.get("streamed_layer_bytes", 0))
    padded_bias_bytes = cast(int, features.get("padded_bias_bytes", 0))
    return (
        WorkingMemoryTerm.fitted("activations", f"k_{suffix}", cast(int, features["tokens"]) * batch, values),
        WorkingMemoryTerm.fitted("base", f"b_{suffix}", 1, values),
        WorkingMemoryTerm("regional_attention_bias", cast(int, features["regional_bias_bytes"]), True),
        WorkingMemoryTerm("padded_attention_bias", padded_bias_bytes, True),
        WorkingMemoryTerm("streamed_layer", streamed_layer_bytes, True),
        WorkingMemoryTerm("attention_scores", cast(int, features["score_matrix_bytes"]), True),
        WorkingMemoryTerm("control_hints", hint_bytes, True),
        WorkingMemoryTerm("lora_patches", cast(int, features["lora_count"]) * batch * _LORA_PATCH_BYTES, True),
        WorkingMemoryTerm("dequant_transient", cast(int, features["dequant_transient_bytes"]), True),
        WorkingMemoryTerm("fp8_cast_transient", cast(int, features["fp8_cast_transient_bytes"]), True),
    )


def _padded_bias_bytes(mask: Optional[torch.Tensor], dtype: torch.dtype) -> int:
    """One sample's copy of the additive bias in `dtype`, its last dimension padded to `_BIAS_ALIGNMENT`; 0 without one."""
    if mask is None:
        return 0
    columns = mask.shape[-1]
    padded = -(-columns // _BIAS_ALIGNMENT) * _BIAS_ALIGNMENT
    return mask.numel() // columns * padded * dtype.itemsize


FLUX2_DENOISE_SPEC = register(
    EstimatorSpec(
        id="denoise.flux2",
        version=2,
        coefficients=_COEFFICIENTS,
        platform_keys=("conv_backend", "allocator"),
        terms=_terms,
        line=_line,
    )
)


def estimate_flux2_denoise_working_memory(
    transformer: torch.nn.Module,
    *,
    hidden_size: int,
    image_seq_len: int,
    ref_image_seq_len: int,
    text_seq_len: int,
    batch_size: int,
    lora_count: int,
    control_blocks: int = 0,
    regional_attn_mask: Optional[torch.Tensor] = None,
    weights_resident: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> WorkingMemoryEstimate:
    """The working memory one forward of `transformer` needs on `device` in `dtype`.

    `transformer` is the model about to be locked, read unlocked for its dequantization and fp8 cast transients, and
    `hidden_size` its width. `image_seq_len`, `ref_image_seq_len` and `text_seq_len` are one sample's token counts, the
    longer caption of a CFG pair, and `batch_size` the samples the forward runs. `control_blocks` is the
    Fun-ControlNet branch's block count, 0 without one. `regional_attn_mask` is regional prompting's mask, which the
    forward passes as an additive bias. `weights_resident` is True only where the caller knows every weight stays on
    the device for the forward; otherwise the largest layer, which partial loading would stream, is reserved too.
    """
    dtype_name = _DTYPE_NAMES.get(dtype)
    if dtype_name is None or hidden_size not in FLUX2_HIDDEN_SIZES:
        raise ValueError(
            f"Working-memory estimator '{FLUX2_DENOISE_SPEC.id}' has no {dtype} line at width {hidden_size} on "
            f"{platform_for(device)}: the transformer runs in a dtype or at a width it is not measured for."
        )
    tokens = image_seq_len + ref_image_seq_len + text_seq_len
    has_attention_mask = regional_attn_mask is not None
    features: dict[str, FeatureValue] = {
        "hidden_size": hidden_size,
        "image_tokens": image_seq_len,
        "reference_tokens": ref_image_seq_len,
        "text_tokens": text_seq_len,
        "tokens": tokens,
        "batch_size": batch_size,
        "dtype": dtype_name,
        "has_attention_mask": has_attention_mask,
        "regional_bias_bytes": regional_attn_mask.numel() * dtype.itemsize if regional_attn_mask is not None else 0,
        "padded_bias_bytes": _padded_bias_bytes(regional_attn_mask, dtype) * batch_size,
        "streamed_layer_bytes": 0 if weights_resident else largest_module_bytes(transformer),
        "score_matrix_bytes": sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=(hidden_size // FLUX2_ATTENTION_HEAD_DIM) * batch_size,
            head_dim=FLUX2_ATTENTION_HEAD_DIM,
            seq_len=tokens,
            has_attn_mask=has_attention_mask,
            # The attention goes through diffusers' `dispatch_attention_fn`, which can route around torch's SDPA,
            # including to a forced `math` backend.
            via_diffusers_dispatch=True,
        ),
        "lora_count": lora_count,
        "control_blocks": control_blocks,
        "dequant_transient_bytes": peak_dequant_transient_bytes(transformer, dtype),
        "fp8_cast_transient_bytes": peak_fp8_cast_transient_bytes(transformer, dtype),
    }
    return FLUX2_DENOISE_SPEC.estimate(features, device)
