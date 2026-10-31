"""The working memory of one FLUX.3 transformer forward, as the denoise node reserves it.

The transformer attends over one joint sequence per sample: the caption's tokens, padded to a multiple of 80 (the
transformer attends to the padding, `flux3/text_encoding.py`), the image's at 1/16 of each side, and those of every
reference image (`flux3/ref_images.py`), a stream of their own that the joint blocks concatenate after the image's. The
per-mode blocks before them attend within one stream each, over a shorter sequence. On a fused SDPA kernel the
activations scale linearly with the joint sequence and with the batch, which runs as that many sequences:
`k_denoise_<dtype>_h<width>` bytes per attended token per sample plus a base of `b_denoise_<dtype>_h<width>` bytes,
shipped per platform in `backend/working_memory/data/denoise.json`. The cost per token follows the transformer's width,
read off the transformer itself (heads times head width). Every released FLUX.3 checkpoint is 3072 wide: Klein, the
step-distilled flow map, and Klein Base share one geometry and differ only by Klein's embedder of the timestep each step
jumps to (`ref_time_in`, one 256 x 3072 and one 3072 x 3072 linear), so both price on one line per dtype. The node runs
the transformer in the dtype its loaders build it in, `TorchDevice.choose_bfloat16_safe_dtype`: bf16, else fp16 on a
CUDA card without bf16, else fp32, each a line of its own. Klein Base's conditional and unconditional passes run one
after the other, so only the longer caption counts. The platform keys are those of the VAE estimators; for a
transformer, `conv_backend` just names the CUDA (`cudnn`) or ROCm (`miopen`) stack it runs on.

That line is `denoise_<dtype>_h<width>` over the attended tokens times the batch (`_line`), and only the plain forward
fits it: every analytic term below comes to zero. A batch fits too, because the line prices exactly what a batch adds,
its sequences, and a forward's base is paid once whatever the batch. Reference images fit it as well: they are tokens of
the joint sequence. `_line` reads the fit off the terms themselves, so a term added here takes the forwards it prices
out of the fit. The fit takes a point's demand as measured, so a forward with an analytic term bounds its estimate but
fits nothing: a materialized score matrix's term is an upper bound, and a line fitted on the demand less that term
would come out low by its slack. The spec does not opt in to user calibration (`EstimatorSpec.user_calibrated`): the
lines are the shipped fit's alone.

The lines are fitted per platform and dtype on fresh-process points of
`scripts/calibrate_flux3_denoise_working_memory.py` (their provenance is in the data): an RTX 4090 with the native
allocator and with `cudaMallocAsync`, an RX 9060 XT under ROCm 10 with AOTriton's fused kernels, with expandable
segments and with the native allocator on Windows, and a Radeon Pro W7900 under ROCm 10 on Linux, with fused SDPA and
the native allocator. In bf16, fp16 and fp32 the fit grid runs 512 to 2048px with a 560-token caption, 1024 and 1440px
with an 80-token one, at 1024px one, three and (bf16 and fp16) ten 1024px references and a batch of two, and Klein Base
at 1024px beside Klein; in bf16 the long sequences add 2048px with one 2048px reference (33328 tokens), 1536px with one
(26160), 2048px with three 1024px references (29232) and with ten (57904). They run on a random-init transformer of the
loader's geometry with 2 per-mode blocks per stream and 2 joint blocks. The RTX 4090 and the W7900 run them all; the RX
9060 XT cannot fit the ten-reference forwards (45616 and 57904 tokens) in 16 GB beside their weights, so its lines are
measured up to 33328 tokens.

The plain forwards from 512 to 2048px take, per token, 0.289 MiB in bf16 and fp16 and 0.476 in fp32 on the RTX 4090
under the native allocator, 0.184 and 0.369 under `cudaMallocAsync`; on the RX 9060 XT 0.291 and 0.440 under the native
allocator, 0.167 (over a 0.22 GiB intercept) and 0.333 with expandable segments; on the W7900 0.291 and 0.440 as well,
fp32's over a 0.48 GiB intercept. bf16 and fp16 measure within 34 MiB of each other, but for one forward under
`cudaMallocAsync` (144 MiB). Reference images cost more per token than the image and the caption: three 1024px
references sit 1.07x above the plain forwards' line under the native allocator (ten 1.08x on the RTX 4090 and the
W7900), and up to 1.12x under `cudaMallocAsync` and 1.17x with expandable segments. The fit is the upper envelope of
every in-fit point, so the reference forwards decide most lines and the plain ones sit below them: the RTX 4090's native
bf16 line, 0.328 MiB per token with no base, reserves 1.05x the ten-reference forward and 1.11 to 1.20x the plain ones
but its 1440px forward with an 80-token caption, which it holds at 1.31x. That forward in fp16, and under
`cudaMallocAsync` a batch of two in bf16 and two fp32 forwards, are held at 1.31 to 1.32x as well
(`bounds.UPPER_BOUND_EXCEPTIONS`); on the W7900 it sits at 1.26x, the most its lines reserve. A batch of two lies within
0.91 to 1.05x of the line of the plain forwards. Klein Base demanded what Klein does, byte for byte on the RTX 4090 and
the RX 9060 XT and within 20 KiB on the W7900, except under `cudaMallocAsync`, where it took 32 MiB more at 1024px,
inside the line. 4 blocks per stream measured the same as 2 on the RTX 4090 and the W7900 and 4.2% more on the RX 9060
XT with expandable segments (993 against 953 MiB at 1024px), which its line covers at 1.17x. On the RX 9060 XT a
process's first forward maps ~245 MiB more than its second; every point takes the larger of its cold and its warm
forward, so the envelope carries it. With SDPA forced onto the math kernel at 1024px the RTX 4090 took 13.69 bytes per
score element over the default kernel, against the 17 the score term charges; on ROCm the SDPA guard chunks the math
kernel, which took 129 MiB more on the RX 9060 XT and 122 MiB less on the W7900, against the 1 GiB chunk the term
charges. The released Klein in bf16, fully resident (`--mode real`), demanded what the 2-block point of the same tokens
does at 1024 and 2048px and at 1024px with one and three references: exactly on the RTX 4090 (native allocator), the
same or up to 21 MiB less on the W7900. On the RX 9060 XT (expandable segments) it demanded 1.08 and 1.06x it at 1024
and 2048px and 1.08x with one reference, but 1.006x with three: its extra depth costs a near-constant 20 to 180 MiB
there, not a share of the sequence. Fully resident the estimate reserves 1.07 to 1.15x those forwards on the RTX 4090,
1.04 to 1.16x on the RX 9060 XT and 1.05 to 1.13x on the W7900, and 1.08 to 1.24x with the streamed layer the node adds.
They ship as out-of-fit points, which hold every refit to the released model from below; the W7900's 1024px one is held
by its 4-block forward, an out-of-fit point of the same features that demanded 21 MiB more.

The node caps neither the image's side nor its references (ten of up to 1 MP each, or one of up to 4 MP), so a longer
sequence prices on its line extrapolated linearly, as the FLUX.2 lines do: on fused attention kernels the activations
grow linearly with the sequence. From 26160 to 57904 tokens the bf16 lines reserve 1.05 to 1.17x what each measured
forward demands under the native allocator, 1.11 to 1.18x under `cudaMallocAsync` and 1.12 to 1.18x with expandable
segments.

Every platform nothing measures is priced as a measured one (`priced_as`), and the three fitted entries those borrow,
each with the native allocator, the RTX 4090's, the RX 9060 XT's on Windows and the W7900's on every other OS, floor
every value there at the constants the node reserved by hand (`unmeasured_floor`): 0.33 MiB per attended token and
sample, truncated to whole bytes (346030), and a 1 GiB base, for every dtype. No unmeasured platform reserves less than
the node did by hand. The 0.33 bounded a real Klein forward on an RTX 4090 at the driver level: 0.286 MB per token from
1 to 4 MP without references, up to 0.320 with six 1 MP references. The constants lie above every measured bf16 and fp16
line at any size a card holds: the W7900's, 0.1% steeper, meets them only at 2.9 million tokens, 932 GiB. They lie below
the steeper fp32 lines past 2700 to 13000 tokens. The legacy base also covers the fp8 weights cast per forward, which an
fp8-stored model reserves as a term of its own as well, so it pays that cast twice wherever they price.

Two judgements, not measurements, extend an entry past the card it was measured on. CUDA on Linux reuses the RTX
4090's Windows-measured lines without the floor, as the FLUX.2 denoise does and for the same reason: the same allocator
and kernels on both systems. ROCm with the native allocator on every OS but Windows prices every card with the W7900's
lines, because two generations on two systems measured close: at every shape both measured, the RX 9060 XT (gfx1200,
Windows) demanded 35 to 45 MiB more than the W7900 (gfx1100, Linux) in bf16 and 74 to 84 MiB more in fp16, on the same
0.291 MiB per token, and 290 to 454 MiB less in fp32. A ROCm card without a fused kernel for these shapes prices the
score matrix on top, capped at the 1 GiB chunk the SDPA guard computes math attention in.

The analytic terms:

- The largest layer's stored weights, for a transformer the model cache may load partially: the weights it leaves in
  RAM are copied to the device per forward, one layer at a time, and the copy is alive during its layer's matmul
  alongside the activations. No point prices it: every point is a fully resident model. The size is read off the
  transformer, the largest module's own parameters and buffers as stored (`largest_module_bytes`, as the cache counts
  them): in bf16 the caption embedder (`context_embedder`, 20480 x 3072), 125,829,120 bytes; for an fp8-stored or GGUF
  build its float8 or quantized bytes, whose widened copy is the fp8 cast or dequant transient's term. The node charges
  it on every forward: whether the cache loads the transformer whole is decided by the lock this estimate sizes, after
  it, so residency is unknown when the estimate is taken. A fully resident model over-reserves one layer. The
  calibration script, whose models are always resident, declares them so (`weights_resident`), and the term is zero.
- Where SDPA has no fused kernel for these shapes it materializes the score matrices, shaped (batch, heads, S, S) over
  the joint sequence: `sdpa_score_matrix_bytes` prices them only where they are really built. No mask is passed.
- The dequantization copy a GGUF build materializes per quantized linear and forward, alive alongside the activations
  and invisible to them (`peak_dequant_transient_bytes`); zero for every other build.
- The compute-dtype copies of a layer's float8-stored weights that FP8 Storage widens per forward
  (`peak_fp8_cast_transient_bytes`); zero for every other build.
"""

from collections.abc import Mapping
from typing import Any, Final, cast

import torch

from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.calc_tensor_size import largest_module_bytes
from invokeai.backend.util.fp8 import peak_fp8_cast_transient_bytes
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import Line, off_line_reasons
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.spec import EstimatorSpec, register

FLUX3_HIDDEN_SIZES: Final = (3072,)
"""The transformer widths with a line: every released FLUX.3 checkpoint's, Klein's and Klein Base's."""

_DTYPE_NAMES: Final[Mapping[torch.dtype, str]] = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
"""The dtypes `TorchDevice.choose_bfloat16_safe_dtype` runs the transformer in."""
_LINES: Final = tuple(f"denoise_{dtype}_h{width}" for dtype in _DTYPE_NAMES.values() for width in FLUX3_HIDDEN_SIZES)
_COEFFICIENTS: Final = frozenset(f"{part}_{line}" for part in ("k", "b") for line in _LINES)


def _suffix(features: Mapping[str, FeatureValue]) -> str | None:
    """The line a forward's dtype and width price on, None for a pair that has none."""
    suffix = f"denoise_{features.get('dtype')}_h{features.get('hidden_size')}"
    return suffix if suffix in _LINES else None


def _line(features: Mapping[str, FeatureValue]) -> Line | None:
    """The forward's line, `denoise_<dtype>_h<width>` over the attended tokens of every sample. Only the plain forward
    fits it: every analytic term of `_terms` zero (see the module docstring). The label names what puts a forward off
    the line, each analytic term that is not zero, in bytes, and Klein Base by `-base`. A dtype or width without a line
    names none."""
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
    variant = [] if features.get("flow_map", True) else ["base"]
    label = "-".join(
        ("denoise", str(features["dtype"]), f"h{features['hidden_size']}", shape, *batched, *variant, *off_line)
    )
    return Line(suffix, tokens * batch, not off_line, label)


def _terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    suffix = _suffix(features)
    if suffix is None:
        raise ValueError(f"no denoise line for {features.get('dtype')} at width {features.get('hidden_size')}")
    batch = cast(int, features["batch_size"])
    return (
        WorkingMemoryTerm.fitted("activations", f"k_{suffix}", cast(int, features["tokens"]) * batch, values),
        WorkingMemoryTerm.fitted("base", f"b_{suffix}", 1, values),
        WorkingMemoryTerm("streamed_layer", cast(int, features["streamed_layer_bytes"]), True),
        WorkingMemoryTerm("attention_scores", cast(int, features["score_matrix_bytes"]), True),
        WorkingMemoryTerm("dequant_transient", cast(int, features["dequant_transient_bytes"]), True),
        WorkingMemoryTerm("fp8_cast_transient", cast(int, features["fp8_cast_transient_bytes"]), True),
    )


FLUX3_DENOISE_SPEC = register(
    EstimatorSpec(
        id="denoise.flux3",
        version=1,
        coefficients=_COEFFICIENTS,
        platform_keys=("conv_backend", "allocator"),
        terms=_terms,
        line=_line,
    )
)


def estimate_flux3_denoise_working_memory(
    transformer: torch.nn.Module,
    *,
    image_seq_len: int,
    ref_image_seq_len: int,
    text_seq_len: int,
    batch_size: int,
    weights_resident: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> WorkingMemoryEstimate:
    """The working memory one forward of `transformer` needs on `device` in `dtype`.

    `transformer` is the `Flux3Transformer2DModel` about to be locked, read unlocked: its width and head count from its
    config, its largest layer, and its dequantization and fp8 cast transients. `image_seq_len`, `ref_image_seq_len`
    and `text_seq_len` are one sample's token counts, the longer caption of a CFG pair, and `batch_size` the samples the
    forward runs. `weights_resident` is True only where the caller knows every weight stays on the device for the
    forward; otherwise the largest layer, which partial loading would stream, is reserved too.
    """
    config: Any = transformer.config  # a diffusers ConfigMixin's
    heads, head_dim = int(config.num_attention_heads), int(config.attention_head_dim)
    hidden_size = heads * head_dim
    dtype_name = _DTYPE_NAMES.get(dtype)
    if dtype_name is None or hidden_size not in FLUX3_HIDDEN_SIZES:
        raise ValueError(
            f"Working-memory estimator '{FLUX3_DENOISE_SPEC.id}' has no {dtype} line at width {hidden_size} on "
            f"{platform_for(device)}: the transformer runs in a dtype or at a width it is not measured for."
        )
    tokens = image_seq_len + ref_image_seq_len + text_seq_len
    features: dict[str, FeatureValue] = {
        "hidden_size": hidden_size,
        "flow_map": bool(config.flow_map),
        "image_tokens": image_seq_len,
        "reference_tokens": ref_image_seq_len,
        "text_tokens": text_seq_len,
        "tokens": tokens,
        "batch_size": batch_size,
        "dtype": dtype_name,
        "streamed_layer_bytes": 0 if weights_resident else largest_module_bytes(transformer),
        "score_matrix_bytes": sdpa_score_matrix_bytes(
            device=device,
            dtype=dtype,
            num_heads=heads * batch_size,
            head_dim=head_dim,
            seq_len=tokens,
            # diffusers' FLUX.3 transformer attends through `dispatch_attention_fn`, which can route around torch's
            # SDPA, including to a forced `math` backend.
            via_diffusers_dispatch=True,
        ),
        "dequant_transient_bytes": peak_dequant_transient_bytes(transformer, dtype),
        "fp8_cast_transient_bytes": peak_fp8_cast_transient_bytes(transformer, dtype),
    }
    return FLUX3_DENOISE_SPEC.estimate(features, device)
