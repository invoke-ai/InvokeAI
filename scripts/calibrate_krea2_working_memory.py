"""Measure the working memory of the Krea-2 transformer's forward, as shipped-data points for `denoise.krea2`.

What a point measures
---------------------
The demand of one transformer forward, measured as a fresh window (`invokeai/backend/working_memory/measurement.py`,
whose definitions the in-server probe's windows share). With the model and its inputs resident: synchronize, release the
allocator's cached blocks, read the baseline and reset the peaks; run the forward; the demand is then the largest growth
of three counters over the baseline: the driver's figure for this process, torch's reserved peak and its allocated peak.
A forward whose allocations failed, or only succeeded after the allocator released its cache and retried (a squeezed
peak), or after which Windows keeps more than `measurement.SPILL_TOLERANCE_BYTES` of the process in shared memory, is
invalid: it is recorded as such, not as a point.

Every point runs in a fresh process (`scripts/working_memory/fresh_points.py`, which the calibration scripts share),
which first applies the server's runtime (`apply_server_runtime` in `invokeai/app/run_app.py`), with the settings of a
server without a config file except the allocator configuration: `--alloc-conf`, or without one the platform default
the server uses (expandable segments for ROCm on Windows, torch's native allocator elsewhere). The forward then runs
twice, cold and warm, and the point takes the larger demand (`measure_cold_and_warm` and `measured_bytes_and_method`):
a `demand_peak` where both forwards read the driver's figure, else a `reserved_peak` of torch's counters alone.

The transformer is `Krea2Transformer2DModel` at its default geometry, the one the loaders build (hidden 6144: 48 query
and 12 K/V heads of 128, MLP 16384; text fusion over 12 tapped layers of 2560), with random weights in the measured
dtype, because memory depends on shapes, not on weight values. It has 2 blocks: a no-grad forward frees each block's
intermediates, which `--extra-blocks 4` re-checks at 1024px. It runs with the denoise node's attention processors
(`build_krea2_attention_processors`), whose memory-efficient SDPA call replaces the stock grouped-query one. The inputs
are what the node hands it: the packed image tokens (64 channels, one token per 16x16 pixels), a caption of tapped
encoder states, the timestep and the position ids; joint references go through `forward_with_references` as the node
runs them. A point carries the features of `estimate_krea2_denoise_working_memory` for a resident model, taken in the
measuring process after the runtime setup, so its `score_matrix_bytes` reflects the attention kernel this build really
runs.

Each point is priced before it runs: the estimate for it plus its weights in its dtype. It is skipped when that exceeds
free device memory, unless `--force`. An oversized allocation does not always fail: where the driver backs it with
system RAM (NVIDIA's sysmem fallback, AMD on Windows) it pages, which can stall the whole machine.

Modes
-----
- `fit` (default): the fit grid, in each of `--dtypes` (bfloat16, float16, float32). The `--sizes` (512, 768, 1024,
  1536, 2048 px) run with a `--text-tokens` caption (512). Those among `--short-text-sizes` (1024, 1536) also run with a
  `--short-text-tokens` caption (77), for the text slope. At 1024px, `--references` (1, 2 and 3) run with that many
  joint references of `--reference-px` (1024) square. `--extra-blocks N` adds one forward with N blocks at 1024px in
  the first dtype.
- `large`: long joint sequences in each of `--dtypes` (bfloat16): 2048px with one and with three 1024px references
  (20992 and 29184 tokens with a 512-token caption) and 1536px with three (22016). The node caps neither the image's
  side nor a reference's; longer sequences price on the line extrapolated.
- `kv`: reference K/V-cache mode in each of `--dtypes` (bfloat16): 1024px with one and with three 1024px references
  and 2048px with one. One window covers what the node's lock covers: the references' precompute, alone, into the
  cache, then one step's forward over the caption and the image, attending to the cached keys; the cache is cleared
  after it. These check the cache term and the larger-pass rule, not the line.
- `captions`: captions beyond the fit grid's 512 tokens, as regional prompting concatenates them, in each of `--dtypes`
  (bfloat16, float16, float32): `--long-captions` (1024 and 2048 tokens) at `--long-caption-sizes` (1024 and 1536 px).
  These check the caption excess the estimate adds beyond 512 tokens (`_CAPTION_EXCESS_BYTES_PER_TOKEN`), not the line.

The fit grid's and the large grid's points are fit points (`"fit": true`): the plain forwards, joint references
included, which the estimator's line admits (`denoise_working_memory._line`). The checks' points, the extra-blocks
forward, the K/V-cache windows and the long captions, are marked `"fit": false`, and their `in_fit` is false. A fit point is `in_fit` only
where the line admits its features: on a build whose SDPA kernel materializes the score matrix, the fit grid's forwards
carry a score term on top of the line, so they measure that term rather than the line, as a check does.

SageAttention
-------------
`--attention sage` measures every point with SageAttention serving the transformer blocks' attention, as a server with
`attention_backend: sage` runs the denoise: the measuring process installs it after the server's runtime, as that
setting does at the server's start (`apply_monkeypatches`), and each measured transformer step runs inside
`sage_attention_scope`, which the node enters once its reference K/V precompute is done, so the precompute stays on
PyTorch SDPA here too. The cold forward then includes SageAttention's check of its first result against SDPA, as the
first SageAttention generation after a server start does. It needs the `sageattention` package (2.2.0 or newer) and
Triton, refused before anything runs where either is missing, an NVIDIA GPU of compute capability 8.0 or newer, and
half precision: the dtypes default to bfloat16 and float16, and float32, which SageAttention never serves, is refused.
So is a Krea-2 SDPA override (`INVOKE_KREA2_SDPA_BACKEND`), under which the node enters no SageAttention scope, and
`--extra-blocks`: out of the fit, that forward would have the features of the fit grid's at 1024px, and the builder
keeps only the larger of two points with the same features.

A SageAttention point proves that SageAttention served it, from the scopes' usage counters (`SageAttentionUsage`):
at least one measured step, every transformer block's attention in each step on SageAttention, and no call on PyTorch
SDPA but those too short for SageAttention by design (the text fusion's, over the text layers and the caption), which
fall back in the server alike. A point where it fell back for any other reason (its out-of-memory fallback, an
ineligible shape, a failed kernel) is `invalid`, with the reasons, and stays out of the file. The counts are in
`raw.sage_usage`.

A SageAttention point carries the feature `attention_backend: sage` (`ATTENTION_BACKEND_FEATURE` in
`invokeai/backend/krea2/denoise_working_memory.py`), which no estimate and no SDPA point carries. The estimator's line
refuses it, so the point is never `in_fit` and the builder fits no SDPA line on it; and its features differ from those
of the SDPA point of the same forward, so the builder never takes one for the other. It ships as an out-of-fit point
the estimate has to cover, which is what such a run asks: whether the SDPA lines cover a SageAttention denoise. The
RTX 4090's points under both CUDA allocators ship so (see `denoise_working_memory`'s docstring for what they found).
The summary prints what the shipped estimate reserves against each point.

What this does not measure: GGUF, fp8-stored and int8 builds, LoRAs and CFG pairs run the same forward, plus analytic
terms the estimate adds (the dequantization and fp8 cast transients, the LoRA patches); CFG runs its passes one after
the other. A family E2E calibration run on real models
(`docs/src/content/docs/development/Guides/family-e2e.mdx`) records them as `probe_demand` points, which would bound the
estimate from below; none ship yet.

Output
------
Every measured point is appended to `--out` as one JSON object per line, the point file
`scripts/working_memory/build_shipped.py --points` reads. It carries the fields of a `MeasuredPoint`
(`invokeai/backend/working_memory/coefficients.py`): `estimator`, `version`, `features`, `platform` (OS included; the
builder names the OS only where the data needs it), `measured_bytes`, `method`, `device`, `in_fit` and `provenance`.
The other keys are the script's, which the builder ignores: `fit`, `point` (the grid entry) and `raw` (each forward's
counter deltas, the prediction and the runtime facts). An invalid, skipped or failed forward is no point: it shows in
the table on stdout, with its reasons, and stays out of the file.

The CUDA entries are fitted on an RTX 4090's points, the ROCm entries on Windows (expandable segments and the native
allocator) on an RX 9060 XT's. ROCm with the native allocator off Windows is `legacy` until points replace it. The
builder fits an entry once its points fit every dtype's line, so a run that is to replace one measures all three
dtypes. A point names the card as torch does (`NVIDIA GeForce RTX 4090`, `AMD Radeon RX 9060 XT (gfx1200)`), and so do
the shipped Krea-2 points.

Usage
-----
RTX 4090 (CUDA): the native allocator, then `cudaMallocAsync`; each in the fit, large and kv modes.

    python scripts/calibrate_krea2_working_memory.py --out 4090-native.jsonl --extra-blocks 4
    python scripts/calibrate_krea2_working_memory.py --out 4090-native.jsonl --mode large
    python scripts/calibrate_krea2_working_memory.py --out 4090-native.jsonl --mode kv
    python scripts/calibrate_krea2_working_memory.py --out 4090-native.jsonl --mode captions
    python scripts/calibrate_krea2_working_memory.py --out 4090-async.jsonl --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_krea2_working_memory.py --out 4090-async.jsonl --mode large \
        --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_krea2_working_memory.py --out 4090-async.jsonl --mode kv \
        --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_krea2_working_memory.py --out 4090-async.jsonl --mode captions \
        --alloc-conf backend:cudaMallocAsync

RX 9060 XT (ROCm on Windows): the server's default expandable segments, then the native allocator; each in the fit,
large (in every dtype), kv and captions modes, priced with the shipped Windows lines. On 16 GB the pricing skips the
long fp32 sequences that need more than the card has free: 2048px with three references with expandable segments,
all three under the native allocator. The card did not drive the desktop.

    python scripts/calibrate_krea2_working_memory.py --out 9060xt-expandable.jsonl --extra-blocks 4
    python scripts/calibrate_krea2_working_memory.py --out 9060xt-expandable.jsonl --mode large \
        --dtypes bfloat16,float16,float32
    python scripts/calibrate_krea2_working_memory.py --out 9060xt-expandable.jsonl --mode kv
    python scripts/calibrate_krea2_working_memory.py --out 9060xt-expandable.jsonl --mode captions
    python scripts/calibrate_krea2_working_memory.py --out 9060xt-native.jsonl --alloc-conf expandable_segments:False
    python scripts/calibrate_krea2_working_memory.py --out 9060xt-native.jsonl --mode large \
        --dtypes bfloat16,float16,float32 --alloc-conf expandable_segments:False
    python scripts/calibrate_krea2_working_memory.py --out 9060xt-native.jsonl --mode kv \
        --alloc-conf expandable_segments:False
    python scripts/calibrate_krea2_working_memory.py --out 9060xt-native.jsonl --mode captions \
        --alloc-conf expandable_segments:False

RTX 4090 with SageAttention: the native allocator, then `cudaMallocAsync`; each in the fit, large, kv and captions
modes, all in bfloat16 and float16.

    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-native.jsonl --attention sage
    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-native.jsonl --attention sage --mode large \
        --dtypes bfloat16,float16
    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-native.jsonl --attention sage --mode kv \
        --dtypes bfloat16,float16
    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-native.jsonl --attention sage --mode captions
    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-async.jsonl --attention sage \
        --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-async.jsonl --attention sage --mode large \
        --dtypes bfloat16,float16 --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-async.jsonl --attention sage --mode kv \
        --dtypes bfloat16,float16 --alloc-conf backend:cudaMallocAsync
    python scripts/calibrate_krea2_working_memory.py --out 4090-sage-async.jsonl --attention sage --mode captions \
        --alloc-conf backend:cudaMallocAsync

`--dry-run` prints the planned points without touching a device, each priced as a CPU prices it.
"""

import argparse
import dataclasses
import datetime
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# None of these imports torch: a measuring process has to configure the allocator before torch is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.working_memory import fresh_points  # noqa: E402

if TYPE_CHECKING:
    import torch

    from invokeai.backend.krea2.reference_edit import Krea2ReferenceKVCache
    from invokeai.backend.util.sage_attention import SageAttentionUsage
    from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate
    from invokeai.backend.working_memory.measurement import FreshWindow

GIB = 2**30
MIB = 2**20

PIXELS_PER_TOKEN_SIDE = 16
"""The VAE's 8x downsampling times the transformer's 2x2 patches."""
IN_CHANNELS = 64
"""A packed token's channels: the Qwen-Image VAE's 16 latent channels times the 2x2 patch."""
CHECK_PX = 1024
"""The size the reference forwards, the extra-blocks forward and the K/V-cache checks run at."""
REFERENCE_PX = 1024
LARGE_GRID = ((2048, 1), (2048, 3), (1536, 3))
"""The large grid's forwards: image side and the number of `--reference-px` references."""
KV_GRID = ((1024, 1), (1024, 3), (2048, 1))
"""The K/V-cache checks: image side and the number of `--reference-px` references."""
DTYPES = ("bfloat16", "float16", "float32")
SAGE_DTYPES = ("bfloat16", "float16")
"""The dtypes SageAttention serves; it leaves float32 attention to PyTorch SDPA."""
SAGE_PACKAGES = ("sageattention", "triton")
KREA2_SDPA_BACKEND_ENV_VAR = "INVOKE_KREA2_SDPA_BACKEND"
"""`krea2.attention.KREA2_SDPA_BACKEND_ENV_VAR`, under which the node enters no SageAttention scope. Not imported from
there: that imports torch, which the driving process must not before the run's runtime setup."""
TIMESTEP = 0.5
LOGGER_NAME = "calibrate_krea2_working_memory"

Kind = Literal["fit", "blocks", "kv", "caption"]
Mode = Literal["fit", "large", "kv", "captions"]
Attention = Literal["sdpa", "sage"]


@dataclass(frozen=True)
class Point:
    """One transformer forward (or, in K/V-cache mode, the precompute and one step) to measure, in its own process."""

    kind: Kind
    """`fit` for a point of the fit or large grid; otherwise the check it serves: another block count (`blocks`), the
    reference K/V cache (`kv`) or a caption beyond the fit grid's (`caption`)."""
    px: int
    text_tokens: int
    dtype: str
    """A torch dtype name, one of `DTYPES`."""
    blocks: int = 2
    references: int = 0
    """How many square references of `reference_px` the denoise conditions on."""
    reference_px: int = REFERENCE_PX
    attention: Attention = "sdpa"
    """`sage` to run the transformer steps with SageAttention, as `attention_backend: sage` does."""

    @property
    def fit(self) -> bool:
        return self.kind == "fit"

    @property
    def cache_references(self) -> bool:
        return self.kind == "kv"


def plan(
    mode: Mode,
    *,
    dtypes: Sequence[str],
    sizes: Sequence[int],
    text_tokens: int,
    short_text_tokens: int,
    short_text_sizes: Sequence[int],
    references: Sequence[int],
    reference_px: int,
    blocks: int,
    extra_blocks: int,
    long_captions: Sequence[int] = (),
    long_caption_sizes: Sequence[int] = (),
    attention: Attention = "sdpa",
) -> list[Point]:
    """The forwards a run measures, in order: per dtype, the fit grid, the large grid, the K/V-cache or the
    long-caption checks, each with `attention`."""
    points: list[Point] = []
    for dtype in dtypes:
        if mode == "captions":
            points += [
                Point("caption", px, caption, dtype, blocks) for px in long_caption_sizes for caption in long_captions
            ]
            continue
        if mode == "large":
            points += [Point("fit", px, text_tokens, dtype, blocks, n, reference_px) for px, n in LARGE_GRID]
            continue
        if mode == "kv":
            points += [Point("kv", px, text_tokens, dtype, blocks, n, reference_px) for px, n in KV_GRID]
            continue
        points += [Point("fit", px, text_tokens, dtype, blocks) for px in sizes]
        points += [Point("fit", px, short_text_tokens, dtype, blocks) for px in sizes if px in short_text_sizes]
        points += [Point("fit", CHECK_PX, text_tokens, dtype, blocks, n, reference_px) for n in references if n]
    if mode == "fit" and extra_blocks:
        points.append(Point("blocks", CHECK_PX, text_tokens, dtypes[0], extra_blocks))
    return [dataclasses.replace(point, attention=attention) for point in points]


def grid_side(px: int) -> int:
    """A square image's tokens per side."""
    return px // PIXELS_PER_TOKEN_SIDE


def image_tokens(px: int) -> int:
    return grid_side(px) ** 2


def reference_tokens(point: Point) -> int:
    return point.references * image_tokens(point.reference_px)


def provenance(point: Point, facts: Mapping[str, Any], date: str) -> str:
    """A measured point's `provenance`: where and how it was measured."""
    blocks = f"{point.blocks} block{'' if point.blocks == 1 else 's'}"
    regime = (
        ", reference K/V-cache mode: the references' precompute and one cached step in one window"
        if point.cache_references
        else ""
    )
    sage = (
        f" with SageAttention {facts['sageattention']} serving the steps (attention_backend: sage)"
        if point.attention == "sage"
        else ""
    )
    return (
        f"scripts/calibrate_krea2_working_memory.py ({date}): {fresh_points.runtime_provenance(facts)}. Random-init "
        f"Krea-2 transformer with the default geometry, {blocks}, the denoise node's attention processors"
        f"{sage}{regime}; "
        "memory does not depend on weight values. A fresh process with the server's runtime, the larger demand of a "
        "cold and a warm forward; no failed allocation, allocator retry or spill."
    )


# Everything below imports torch. In a measuring process it runs only after `fresh_points.apply_server_runtime`.


def _transformer(blocks: int) -> "torch.nn.Module":
    """The Krea-2 transformer at its default geometry with `blocks` blocks, on the meta device."""
    import torch
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

    with torch.device("meta"):
        return Krea2Transformer2DModel(num_layers=blocks)


def _estimate(point: Point, transformer: "torch.nn.Module", device: "torch.device") -> "WorkingMemoryEstimate":
    """The estimate for the point's denoise on `device`, priced in the kernel regime the caller holds."""
    import torch

    from invokeai.backend.krea2.denoise_working_memory import estimate_krea2_denoise_working_memory

    return estimate_krea2_denoise_working_memory(
        transformer,
        image_seq_len=image_tokens(point.px),
        text_seq_len=point.text_tokens,
        reference_seq_len=reference_tokens(point),
        cache_references=point.cache_references,
        # Measured whole on the device: nothing is streamed per forward.
        weights_resident=True,
        device=device,
        dtype=getattr(torch, point.dtype),
    )


def recorded_features(point: Point, estimate: "WorkingMemoryEstimate") -> dict[str, Any]:
    """The features a point records: the estimate's, plus the attention backend where it is not PyTorch SDPA, which
    keeps a SageAttention point off the SDPA line and apart from the SDPA point of the same forward."""
    from invokeai.backend.krea2.denoise_working_memory import ATTENTION_BACKEND_FEATURE

    sage = {ATTENTION_BACKEND_FEATURE: point.attention} if point.attention == "sage" else {}
    return dict(estimate.features) | sage


class SageRecorder:
    """The SageAttention scopes of a point's measured transformer steps, one per step, and whether they prove that
    SageAttention served the point. The node enters one scope around its whole loop; one per step tells the same, as a
    fallback in any of them voids the point."""

    def __init__(self) -> None:
        self.usages: list[SageAttentionUsage] = []

    @contextmanager
    def scope(self) -> Iterator[None]:
        from invokeai.backend.util.sage_attention import sage_attention_scope

        with sage_attention_scope() as usage:
            self.usages.append(usage)
            yield

    def reasons(self, blocks: int) -> list[str]:
        """Why the steps measured no SageAttention denoise of `blocks` blocks: no step ran in a scope, calls fell back
        to PyTorch SDPA, by reason, or a step had fewer calls served than its blocks make. Only `short` calls fall back
        by design: the text fusion's, over the text layers and the caption. Within a step, a block's call is never
        `short` while a caption long enough for SageAttention makes up the count, since the block attends over that
        caption and the image."""
        if not self.usages:
            return ["sage scoped no transformer step"]
        fell_back: Counter[str] = Counter()
        for usage in self.usages:
            fell_back.update(usage.fell_back)
        reasons = [
            f"sage fell back: {reason}={count}" for reason, count in fell_back.most_common() if reason != "short"
        ]
        unserved = sum(usage.served < blocks for usage in self.usages)
        if unserved:
            reasons.append(
                f"sage served fewer than {blocks} block attentions in {unserved} of {len(self.usages)} steps"
            )
        return reasons


def _install_sage() -> str:
    """Install SageAttention as `attention_backend: sage` does after the rest of the server's runtime; its version.
    Leaves where it does not install."""
    from importlib.metadata import version

    from invokeai.backend.util.sage_attention import install_sage_attention

    if not install_sage_attention():
        raise SystemExit(
            "--attention sage: SageAttention did not install (it needs CUDA, a GPU of compute capability 8.0 or newer "
            "and sageattention 2.2.0 or newer with Triton; its warning on stderr names what is missing)"
        )
    return version("sageattention")


def _weight_bytes(transformer: "torch.nn.Module", dtype: "torch.dtype") -> int:
    return sum(param.numel() for param in transformer.parameters()) * dtype.itemsize


def install_processors(transformer: "torch.nn.Module", point: Point) -> "Krea2ReferenceKVCache | None":
    """The denoise node's attention processors, with the reference K/V cache in K/V-cache mode, which they return."""
    from invokeai.backend.krea2.attention import Krea2RegionalPromptingState, build_krea2_attention_processors
    from invokeai.backend.krea2.reference_edit import Krea2ReferenceKVCache

    cache = Krea2ReferenceKVCache() if point.cache_references else None
    transformer.set_attn_processor(
        build_krea2_attention_processors(transformer, Krea2RegionalPromptingState(), reference_kv_cache=cache)
    )
    return cache


def forward(
    point: Point,
    transformer: "torch.nn.Module",
    cache: "Krea2ReferenceKVCache | None",
    device: "torch.device",
    step_scope: Callable[[], AbstractContextManager[None]] = nullcontext,
) -> Callable[[], object]:
    """The operation a point measures, with the inputs the node hands the transformer, allocated now. Each transformer
    step runs inside `step_scope`, as the node runs its loop; the references' K/V precompute runs before it."""
    import torch

    from invokeai.backend.krea2.reference_edit import forward_with_references, precompute_reference_kv
    from invokeai.backend.krea2.sampling_utils import (
        prepare_position_ids,
        prepare_position_ids_with_references,
        prepare_reference_position_ids,
    )

    dtype = getattr(torch, point.dtype)
    side = grid_side(point.px)
    config = transformer.config
    hidden_states = torch.randn(1, image_tokens(point.px), IN_CHANNELS, device=device, dtype=dtype)
    caption = torch.randn(
        1, point.text_tokens, config.num_text_layers, config.text_hidden_dim, device=device, dtype=dtype
    )
    timestep = torch.full((1,), TIMESTEP, device=device, dtype=dtype)
    grids = [(grid_side(point.reference_px),) * 2] * point.references
    references = torch.randn(1, reference_tokens(point), IN_CHANNELS, device=device, dtype=dtype)

    def step(position_ids: "torch.Tensor") -> object:
        with step_scope():
            return transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=caption,
                encoder_attention_mask=None,
                timestep=timestep,
                position_ids=position_ids,
                return_dict=False,
            )

    position_ids = prepare_position_ids(point.text_tokens, side, side, device)
    if not point.references:
        return lambda: step(position_ids)
    if cache is None:
        joint_ids = prepare_position_ids_with_references(point.text_tokens, side, side, grids, device)

        def joint_step() -> object:
            with step_scope():
                return forward_with_references(transformer, hidden_states, caption, timestep, joint_ids, references)

        return joint_step

    reference_ids = prepare_reference_position_ids(grids, device)

    def cached_step() -> object:
        precompute_reference_kv(transformer, references, reference_ids, cache)
        try:
            return step(position_ids)
        finally:
            cache.clear()

    return cached_step


def _measure(point: Point, force: bool) -> dict[str, Any]:
    """The output line of `point`, measured in this process after `fresh_points.apply_server_runtime`."""
    import torch

    from invokeai.backend.krea2 import denoise_working_memory
    from invokeai.backend.util.devices import TorchDevice
    from invokeai.backend.working_memory.measurement import measure_cold_and_warm
    from invokeai.backend.working_memory.platform import platform_for

    # The device the node itself takes.
    device = TorchDevice.choose_torch_device()
    if device.type != "cuda":
        raise SystemExit(f"Needs a CUDA or ROCm device; this process would run on {device}.")
    dtype = getattr(torch, point.dtype)
    recorder = SageRecorder() if point.attention == "sage" else None
    sage_version = _install_sage() if recorder is not None else None

    with torch.inference_mode():
        transformer = _transformer(point.blocks)
        estimate = _estimate(point, transformer, device)
        platform = platform_for(device)
        facts = fresh_points.runtime_facts(device, platform.os, denoise_working_memory.__file__)
        if sage_version is not None:
            facts["sageattention"] = sage_version
        common = {
            "estimator": estimate.estimator,
            "version": estimate.version,
            "platform": dataclasses.asdict(platform),
            "features": recorded_features(point, estimate),
            "device": facts["device"],
        }
        weights = _weight_bytes(transformer, dtype)
        free, _ = TorchDevice.cuda_mem_get_info(device)
        raw = {
            "reservation_bytes": estimate.reservation_bytes,
            "weight_bytes": weights,
            "predicted_bytes": estimate.reservation_bytes + weights,
            "free_bytes": free,
            "runtime": facts,
        }
        if raw["predicted_bytes"] > free and not force:
            return fresh_points.skipped_line(point, common, raw)

        # Allocated directly in the measured dtype on the device, randomly initialized.
        transformer = transformer.to(dtype).to_empty(device=device)
        for param in transformer.parameters():
            torch.nn.init.normal_(param, std=0.02)
        transformer.eval()
        cache = install_processors(transformer, point)
        step_scope = recorder.scope if recorder is not None else nullcontext
        windows = measure_cold_and_warm(device, forward(point, transformer, cache, device, step_scope))
    date = datetime.date.today().isoformat()
    return measured_line(
        point, recorder, common=common, provenance=provenance(point, facts, date), windows=windows, raw=raw
    )


def measured_line(
    point: Point,
    recorder: SageRecorder | None,
    *,
    common: Mapping[str, Any],
    provenance: str,
    windows: Sequence["FreshWindow"],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """The output line of a measured point (`fresh_points.point_line`). With SageAttention it is `invalid` where the
    steps `recorder` scoped prove no SageAttention denoise, and `raw.sage_usage` holds each step's counts."""
    reasons: Sequence[str] = ()
    if recorder is not None:
        usage = [
            {"served": step.served, "fell_back": dict(step.fell_back), "memory_short": step.memory_short}
            for step in recorder.usages
        ]
        raw = {**raw, "sage_usage": usage}
        reasons = recorder.reasons(point.blocks)
    return fresh_points.point_line(
        point, common=common, provenance=provenance, windows=windows, raw=raw, reasons=reasons
    )


_COLUMNS = (
    f"{'kind':>7} {'attn':>4} {'dtype':>8} {'blocks':>6} {'px':>5} {'text':>5} {'refs':>4} {'ref px':>6} {'tokens':>7}"
)


def _grid(point: Mapping[str, Any]) -> str:
    refs = point["references"]
    return (
        f"{point['kind']:>7} {point['attention']:>4} {point['dtype']:>8} {point['blocks']:>6} {point['px']:>5} "
        f"{point['text_tokens']:>5} {refs:>4} {point['reference_px'] if refs else '':>6}"
    )


def _head(line: Mapping[str, Any]) -> str:
    """A line's grid columns in the table, with its size on the line."""
    return f"{_grid(line['point'])} {line['features']['tokens'] if 'features' in line else '':>7}"


def _summarize(lines: Sequence[Mapping[str, Any]], text_tokens: int) -> None:
    """Per dtype, the slope of the plain forwards with the long caption from the smallest to the largest, and where
    every other point lies against it; per point out of the fit (a check, a SageAttention forward), what the estimate
    reserves for it."""
    measured = [line for line in lines if line["status"] == "point"]
    fit = [line for line in measured if line["in_fit"]]
    for dtype in dict.fromkeys(line["point"]["dtype"] for line in fit):
        of_dtype = [line for line in fit if line["point"]["dtype"] == dtype]
        main = sorted(
            (
                line
                for line in of_dtype
                if not line["point"]["references"] and line["point"]["text_tokens"] == text_tokens
            ),
            key=lambda line: line["features"]["tokens"],
        )
        if len(main) < 2:
            continue
        lo, hi = main[0], main[-1]
        slope = (hi["measured_bytes"] - lo["measured_bytes"]) / (hi["features"]["tokens"] - lo["features"]["tokens"])
        base = lo["measured_bytes"] - slope * lo["features"]["tokens"]
        print(f"{dtype}: {slope / MIB:.4f} MiB per token, intercept {base / GIB:.3f} GiB")
        for line in of_dtype:
            tokens = line["features"]["tokens"]
            print(
                f"{dtype}, {line['point']['px']}px, {line['point']['text_tokens']} text, {line['point']['references']} "
                f"references: {line['measured_bytes'] / GIB:.3f} GiB at {tokens} tokens, "
                f"{line['measured_bytes'] / (base + slope * tokens):.3f}x the line"
            )
    for line in (line for line in measured if not line["in_fit"]):
        point = line["point"]
        reserved = line["raw"]["reservation_bytes"]
        print(
            f"{point['kind']} {point['attention']} {point['dtype']}, {point['px']}px, {point['text_tokens']} text, "
            f"{point['references']} references, {point['blocks']} blocks: "
            f"{line['measured_bytes'] / GIB:.3f} GiB; the shipped "
            f"estimate reserves {reserved / GIB:.3f} GiB ({reserved / line['measured_bytes']:.3f}x)"
        )


def _print_plan(points: Sequence[Point]) -> None:
    """The planned points, each priced as a CPU prices it, plus the weights."""
    import torch

    print(f"{_COLUMNS} {'estimate GiB':>12} {'weights GiB':>11} {'total GiB':>9}")
    cpu = torch.device("cpu")
    transformers: dict[int, torch.nn.Module] = {}
    for point in points:
        transformer = transformers.setdefault(point.blocks, _transformer(point.blocks))
        estimate = _estimate(point, transformer, cpu)
        weights = _weight_bytes(transformer, getattr(torch, point.dtype))
        print(
            f"{_grid(dataclasses.asdict(point))} {estimate.features['tokens']:>7} "
            f"{estimate.reservation_bytes / GIB:>12.2f} {weights / GIB:>11.2f} "
            f"{(estimate.reservation_bytes + weights) / GIB:>9.2f}"
        )
    print(f"{len(points)} forwards, one process each")


def _ints(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {text!r}")


def _sizes(text: str) -> list[int]:
    sizes = _ints(text)
    if any(px <= 0 or px % PIXELS_PER_TOKEN_SIDE for px in sizes):
        raise argparse.ArgumentTypeError(
            f"sizes must be positive multiples of {PIXELS_PER_TOKEN_SIDE} px, got {text!r}"
        )
    return sizes


def _side(text: str) -> int:
    sizes = _sizes(text)
    if len(sizes) != 1:
        raise argparse.ArgumentTypeError(f"not one side length in px: {text!r}")
    return sizes[0]


def _counts(text: str) -> list[int]:
    counts = _ints(text)
    if any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError(f"counts must not be negative, got {text!r}")
    return counts


def _dtypes(text: str) -> list[str]:
    dtypes = [part.strip() for part in text.split(",") if part.strip()]
    if not dtypes or not set(dtypes) <= set(DTYPES):
        raise argparse.ArgumentTypeError(f"dtypes must be among {', '.join(DTYPES)}, got {text!r}")
    return dtypes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fit", "large", "kv", "captions"), default="fit", help="Default: fit.")
    parser.add_argument("--out", type=Path, help="The JSON-lines file the lines are appended to. Required to measure.")
    parser.add_argument(
        "--dtypes",
        type=_dtypes,
        help="Default: every dtype in the fit and captions modes, bfloat16 otherwise. SageAttention: no float32.",
    )
    parser.add_argument("--sizes", type=_sizes, default="512,768,1024,1536,2048", help="The fit grid's sizes, px.")
    parser.add_argument("--text-tokens", type=int, default=512, help="The caption at every size. Default 512.")
    parser.add_argument("--short-text-tokens", type=int, default=77, help="The text-slope caption. Default 77.")
    parser.add_argument(
        "--short-text-sizes", type=_sizes, default="1024,1536", help="Fit sizes that also run the text-slope caption."
    )
    parser.add_argument(
        "--references", type=_counts, default="1,2,3", help="Fit mode: joint references at 1024px. Default 1,2,3."
    )
    parser.add_argument(
        "--reference-px", type=_side, default=REFERENCE_PX, help="A reference's side, px. Default 1024."
    )
    parser.add_argument(
        "--long-captions", type=_counts, default="1024,2048", help="Captions mode: caption tokens. Default 1024,2048."
    )
    parser.add_argument(
        "--long-caption-sizes", type=_sizes, default="1024,1536", help="Captions mode: sizes, px. Default 1024,1536."
    )
    parser.add_argument("--blocks", type=int, default=2, help="Transformer blocks. Default 2.")
    parser.add_argument("--extra-blocks", type=int, default=0, help="Fit mode: this block count at 1024px as well.")
    parser.add_argument(
        "--alloc-conf",
        default="",
        help="The allocator configuration, as the server's pytorch_cuda_alloc_conf. Empty: the platform's default.",
    )
    parser.add_argument(
        "--attention",
        choices=("sdpa", "sage"),
        default="sdpa",
        help="sage: SageAttention serves the transformer steps, as attention_backend: sage does. Default: sdpa.",
    )
    parser.add_argument("--force", action="store_true", help="Run points predicted not to fit free device memory.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned points without touching a device.")
    parser.add_argument("--single", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single is not None:
        fresh_points.measure_in_this_process(args.single, point_type=Point, measure=_measure, logger_name=LOGGER_NAME)
    every_dtype = SAGE_DTYPES if args.attention == "sage" else DTYPES
    if args.dtypes and not set(args.dtypes) <= set(every_dtype):
        parser.error(f"--attention sage measures half precision only: {', '.join(SAGE_DTYPES)}")
    if args.attention == "sage" and args.extra_blocks:
        parser.error(
            "--attention sage takes no --extra-blocks: out of the fit, that forward has the features of the fit grid's "
            "at 1024px, and the builder would keep only the larger of the two"
        )
    points = plan(
        args.mode,
        dtypes=args.dtypes or (list(every_dtype) if args.mode in ("fit", "captions") else ["bfloat16"]),
        sizes=args.sizes,
        text_tokens=args.text_tokens,
        short_text_tokens=args.short_text_tokens,
        short_text_sizes=args.short_text_sizes,
        references=args.references,
        reference_px=args.reference_px,
        blocks=args.blocks,
        extra_blocks=args.extra_blocks,
        long_captions=args.long_captions,
        long_caption_sizes=args.long_caption_sizes,
        attention=args.attention,
    )
    if args.dry_run:
        _print_plan(points)
        return
    if args.out is None:
        parser.error("--out is required to measure")
    fresh_points.check_alloc_conf(parser, args.alloc_conf)
    if args.attention == "sage":
        missing = [name for name in SAGE_PACKAGES if find_spec(name) is None]
        if missing:
            parser.error(f"--attention sage needs {' and '.join(missing)}, which this environment does not have")
        if os.environ.get(KREA2_SDPA_BACKEND_ENV_VAR, "").strip():
            parser.error(f"--attention sage: the denoise node runs no SageAttention under {KREA2_SDPA_BACKEND_ENV_VAR}")
    lines = fresh_points.measure_points(
        points,
        script=Path(__file__).resolve(),
        out=args.out,
        alloc_conf=args.alloc_conf,
        force=args.force,
        logger_name=LOGGER_NAME,
        columns=_COLUMNS,
        head=_head,
    )

    print()
    _summarize(lines, args.text_tokens)
    fresh_points.print_counts(lines, args.out)


if __name__ == "__main__":
    main()
