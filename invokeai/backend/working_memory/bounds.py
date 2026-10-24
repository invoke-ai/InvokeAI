"""How a measured point bounds the estimate for its features.

One rule, read by the data test (`tests/backend/working_memory/test_shipped_coefficients.py`), which holds the shipped
data to it, and by `scripts/working_memory/build_shipped.py`, which refuses to write data that breaks it:

- Lower bound. The model cache consumes an estimate as ``free >= estimate``, so the estimate must be at least the point.
  Where only analytic terms decide it (a materialized score matrix), the cache's `device_working_mem_gb` floor counts.
- Upper bound, for points that hold one (`MeasuredPoint.holds_upper_bound`) of features that cap their line
  (`fit.Line.caps`; a VAE operation on a tile or an image below `fit.SMALLEST_APP_SIDE` does not): where a fitted term
  decides an in-fit point, the estimate is at most `MAX_OVER_RESERVATION` times the point, its listed
  `UPPER_BOUND_EXCEPTIONS` ratio, or its tighter `max_ratio`, which applies whichever term decides. A reservation the
  operation never uses is model weights evicted for nothing.
"""

import json
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.working_memory.coefficients import MeasuredPoint, WorkingMemoryData, resolve_coefficients
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate
from invokeai.backend.working_memory.platform import Platform

if TYPE_CHECKING:
    from invokeai.backend.working_memory.spec import EstimatorSpec

MAX_OVER_RESERVATION: Final = 1.3
"""How far above a measured point a reservation may sit where a fitted coefficient decides it, unless the point is
listed in `UPPER_BOUND_EXCEPTIONS` or carries a tighter `max_ratio`."""

WORKING_MEMORY_FLOOR: Final = int(InvokeAIAppConfig.model_fields["device_working_mem_gb"].default * 2**30)
"""What `ModelCache._working_mem_reserve` floors every reservation at. The field default, not a config instance an
environment variable could move."""

UPPER_BOUND_EXCEPTIONS: Final[Mapping[str, float]] = MappingProxyType(
    {
        # Untiled demand, concave under MIOpen's native allocator on Windows.
        "vae.flux-miopen,native,windows-decode-bf16-512x512-RX9060XT-demand_peak": 1.45,
        "vae.flux2-miopen,native,windows-decode-bf16-512x512-RX9060XT-demand_peak": 1.45,
        "vae.sd-miopen,native,windows-encode-fp16-1024x1024-RX9060XT-demand_peak": 1.49,
        "vae.sd-miopen,native,windows-encode-fp16-1280x1280-RX9060XT-demand_peak": 1.65,
        # The untiled line's headroom at the tile, under MIOpen's native allocator on Windows.
        "vae.flux-miopen,native,windows-decode-bf16-1024x1024-tile512-RX9060XT-demand_peak": 1.42,
        "vae.flux-miopen,native,windows-decode-bf16-1536x1536-tile512-RX9060XT-demand_peak": 1.44,
        "vae.flux-miopen,native,windows-decode-bf16-2048x2048-tile512-RX9060XT-demand_peak": 1.45,
        "vae.flux-miopen,native,windows-decode-bf16-3072x3072-tile512-RX9060XT-demand_peak": 1.48,
        "vae.flux-miopen,native,windows-decode-bf16-4096x4096-tile512-RX9060XT-demand_peak": 1.55,
        "vae.flux-miopen,native,windows-decode-bf16-4096x4096-tile768-RX9060XT-demand_peak": 1.32,
        "vae.flux-miopen,native,windows-decode-bf16-6144x6144-tile512-RX9060XT-demand_peak": 1.56,
        "vae.flux-miopen,native,windows-encode-bf16-8192x8192-tile768-RX9060XT-demand_peak": 1.32,
        "vae.flux2-miopen,native,windows-decode-bf16-1024x1024-tile512-RX9060XT-demand_peak": 1.43,
        "vae.flux2-miopen,native,windows-decode-bf16-12288x12288-tile512-RX9060XT-demand_peak": 1.42,
        "vae.flux2-miopen,native,windows-decode-bf16-1536x1536-tile512-RX9060XT-demand_peak": 1.43,
        "vae.flux2-miopen,native,windows-decode-bf16-2048x2048-tile512-RX9060XT-demand_peak": 1.44,
        "vae.flux2-miopen,native,windows-decode-bf16-3072x3072-tile512-RX9060XT-demand_peak": 1.49,
        "vae.flux2-miopen,native,windows-decode-bf16-4096x4096-tile512-RX9060XT-demand_peak": 1.55,
        "vae.flux2-miopen,native,windows-decode-bf16-4096x4096-tile768-RX9060XT-demand_peak": 1.32,
        "vae.flux2-miopen,native,windows-decode-bf16-6144x6144-tile512-RX9060XT-demand_peak": 1.55,
        "vae.flux2-miopen,native,windows-decode-bf16-8192x8192-tile512-RX9060XT-demand_peak": 1.38,
        "vae.flux2-miopen,native,windows-encode-bf16-8192x8192-tile768-RX9060XT-demand_peak": 1.41,
        "vae.sd-miopen,native,windows-decode-fp16-1536x1536-tile1024-RX9060XT-demand_peak": 1.55,
        "vae.sd-miopen,native,windows-decode-fp16-2048x2048-tile1024-RX9060XT-demand_peak": 1.48,
        "vae.sd-miopen,native,windows-decode-fp16-3072x3072-tile1024-RX9060XT-demand_peak": 1.48,
        "vae.sd-miopen,native,windows-decode-fp16-4096x4096-tile1024-RX9060XT-demand_peak": 1.43,
        "vae.sd-miopen,native,windows-decode-fp16-6144x6144-tile1024-RX9060XT-demand_peak": 1.48,
        "vae.sd-miopen,native,windows-encode-fp16-12288x12288-tile1024-RX9060XT-demand_peak": 1.47,
        "vae.sd-miopen,native,windows-encode-fp16-1536x1536-tile1024-RX9060XT-demand_peak": 1.62,
        "vae.sd-miopen,native,windows-encode-fp16-1536x1536-tile512-RX9060XT-demand_peak": 1.38,
        "vae.sd-miopen,native,windows-encode-fp16-2048x2048-tile1024-RX9060XT-demand_peak": 1.72,
        "vae.sd-miopen,native,windows-encode-fp16-2048x2048-tile512-RX9060XT-demand_peak": 1.34,
        "vae.sd-miopen,native,windows-encode-fp16-3072x3072-tile1024-RX9060XT-demand_peak": 1.66,
        "vae.sd-miopen,native,windows-encode-fp16-3072x3072-tile512-RX9060XT-demand_peak": 1.37,
        "vae.sd-miopen,native,windows-encode-fp16-4096x4096-tile1024-RX9060XT-demand_peak": 1.66,
        "vae.sd-miopen,native,windows-encode-fp16-6144x6144-tile1024-RX9060XT-demand_peak": 1.62,
        "vae.sd-miopen,native,windows-encode-fp16-6144x6144-tile512-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,native,windows-encode-fp16-8192x8192-tile1024-RX9060XT-demand_peak": 1.63,
        # One tiled line serves every tile.
        "vae.flux-cudnn,cudaMallocAsync-decode-bf16-12288x12288-tile1024-RTX4090-demand_peak": 1.42,
        "vae.flux-cudnn,cudaMallocAsync-decode-bf16-6144x6144-tile512-RTX4090-demand_peak": 1.42,
        "vae.flux-cudnn,cudaMallocAsync-decode-bf16-8192x8192-tile512-RTX4090-demand_peak": 1.31,
        "vae.flux-cudnn,cudaMallocAsync-decode-bf16-8192x8192-tile768-RTX4090-demand_peak": 1.44,
        "vae.flux-miopen,native,windows-decode-bf16-12288x12288-tile1024-RX9060XT-demand_peak": 1.39,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-1024x1024-tile512-RTX4090-demand_peak": 2.36,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-1536x1536-tile1024-RTX4090-demand_peak": 1.32,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-1536x1536-tile512-RTX4090-demand_peak": 1.81,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-2048x2048-tile1024-RTX4090-demand_peak": 1.33,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-2048x2048-tile512-RTX4090-demand_peak": 1.83,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-2048x2048-tile768-RTX4090-demand_peak": 1.38,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-3072x3072-tile512-RTX4090-demand_peak": 1.90,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-4096x4096-tile512-RTX4090-demand_peak": 1.99,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-4096x4096-tile768-RTX4090-demand_peak": 1.48,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-6144x6144-tile512-RTX4090-demand_peak": 1.78,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-8192x8192-tile512-RTX4090-demand_peak": 1.70,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-8192x8192-tile768-RTX4090-demand_peak": 1.38,
        "vae.flux2-cudnn,cudaMallocAsync-encode-bf16-4096x4096-tile512-RTX4090-demand_peak": 1.31,
        "vae.flux2-cudnn,native-encode-bf16-12288x12288-tile1024-RTX4090-demand_peak": 1.52,
        "vae.flux2-cudnn,native-encode-bf16-12288x12288-tile1536-RTX4090-demand_peak": 1.31,
        "vae.flux2-cudnn,native-encode-bf16-2024x2024-tile512-RTX4090-reserved_peak": 1.36,
        "vae.flux2-cudnn,native-encode-bf16-6144x6144-tile512-RTX4090-demand_peak": 1.33,
        "vae.flux2-cudnn,native-encode-bf16-8192x8192-tile512-RTX4090-demand_peak": 1.40,
        "vae.flux2-cudnn,native-encode-bf16-8192x8192-tile768-RTX4090-demand_peak": 1.43,
        "vae.flux2-miopen,native,windows-decode-bf16-12288x12288-tile1024-RX9060XT-demand_peak": 1.38,
        "vae.flux2-miopen,native,windows-encode-bf16-12288x12288-tile1024-RX9060XT-demand_peak": 1.32,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-1024x1024-tile512-RTX4090-demand_peak": 1.71,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-1536x1536-tile512-RTX4090-demand_peak": 1.33,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-2048x2048-tile512-RTX4090-demand_peak": 1.36,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-3072x3072-tile512-RTX4090-demand_peak": 1.43,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-4096x4096-tile512-RTX4090-demand_peak": 1.52,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-4096x4096-tile768-RTX4090-demand_peak": 1.38,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-6144x6144-tile1024-RTX4090-demand_peak": 1.31,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-6144x6144-tile512-RTX4090-demand_peak": 1.46,
        "vae.sd-cudnn,cudaMallocAsync-decode-fp16-8192x8192-tile768-RTX4090-demand_peak": 1.34,
        "vae.sd-cudnn,cudaMallocAsync-encode-fp32-12288x12288-tile1024-RTX4090-demand_peak": 1.41,
        "vae.sd-cudnn,cudaMallocAsync-encode-fp32-12288x12288-tile512-RTX4090-demand_peak": 1.91,
        "vae.sd-cudnn,cudaMallocAsync-encode-fp32-4096x4096-tile512-RTX4090-demand_peak": 1.37,
        "vae.sd-cudnn,cudaMallocAsync-encode-fp32-6144x6144-tile512-RTX4090-demand_peak": 1.45,
        "vae.sd-cudnn,cudaMallocAsync-encode-fp32-8192x8192-tile512-RTX4090-demand_peak": 1.63,
        "vae.sd-cudnn,cudaMallocAsync-encode-fp32-8192x8192-tile768-RTX4090-demand_peak": 1.43,
        "vae.sd-miopen,native,windows-decode-fp16-1024x1024-tile512-RX9060XT-demand_peak": 1.72,
        "vae.sd-miopen,native,windows-decode-fp16-1536x1536-tile512-RX9060XT-demand_peak": 1.60,
        "vae.sd-miopen,native,windows-decode-fp16-2048x2048-tile512-RX9060XT-demand_peak": 1.69,
        "vae.sd-miopen,native,windows-decode-fp16-3072x3072-tile512-RX9060XT-demand_peak": 1.66,
        "vae.sd-miopen,native,windows-decode-fp16-4096x4096-tile512-RX9060XT-demand_peak": 1.56,
        "vae.sd-miopen,native,windows-decode-fp32-1024x1024-tile512-RX9060XT-demand_peak": 1.53,
        "vae.sd-miopen,native,windows-decode-fp32-1536x1536-tile512-RX9060XT-demand_peak": 1.53,
        "vae.sd-miopen,native,windows-decode-fp32-2048x2048-tile512-RX9060XT-demand_peak": 1.52,
        "vae.sd-miopen,native,windows-decode-fp32-4096x4096-tile512-RX9060XT-demand_peak": 1.53,
        "vae.sd-miopen,native,windows-encode-fp32-2048x2048-tile768-RX9060XT-demand_peak": 1.31,
        # The assembly lines of the largest decodes.
        "vae.flux-cudnn,cudaMallocAsync-decode-bf16-1024x1024-tile512-RTX4090-demand_peak": 1.35,
        "vae.sd-cudnn,native-decode-fp32-12288x12288-tile512-RTX4090-demand_peak": 1.37,
        "vae.sd-cudnn,native-decode-fp32-4096x4096-tile512-RTX4090-demand_peak": 1.43,
        "vae.sd-cudnn,native-decode-fp32-8192x8192-tile512-RTX4090-demand_peak": 1.35,
        "vae.sd-miopen,expandable,windows-decode-fp16-12288x12288-tile1024-RX9060XT-demand_peak": 1.31,
        "vae.sd-miopen,expandable,windows-decode-fp16-12288x12288-tile512-RX9060XT-demand_peak": 1.31,
        "vae.sd-miopen,expandable,windows-decode-fp16-8192x8192-tile512-RX9060XT-demand_peak": 1.31,
        "vae.sd-miopen,expandable,windows-decode-fp32-12288x12288-tile512-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,expandable,windows-decode-fp32-6144x6144-tile512-RX9060XT-demand_peak": 1.49,
        "vae.sd-miopen,expandable,windows-decode-fp32-8192x8192-tile512-RX9060XT-demand_peak": 1.56,
        "vae.sd-miopen,native,windows-decode-fp16-12288x12288-tile1024-RX9060XT-demand_peak": 1.31,
        "vae.sd-miopen,native,windows-decode-fp16-12288x12288-tile512-RX9060XT-demand_peak": 1.63,
        "vae.sd-miopen,native,windows-decode-fp16-6144x6144-tile512-RX9060XT-demand_peak": 1.44,
        "vae.sd-miopen,native,windows-decode-fp16-8192x8192-tile512-RX9060XT-demand_peak": 1.52,
        "vae.sd-miopen,native,windows-decode-fp32-8192x8192-tile512-RX9060XT-demand_peak": 1.42,
        # The FLUX.2 denoise's short caption.
        "denoise.flux2-cudnn,native-denoise-bf16-h3072-8100+77tok-RTX4090-demand_peak": 1.42,
        "denoise.flux2-miopen,native,windows-denoise-bf16-h3072-8100+77tok-RX9060XT-demand_peak": 1.40,
    }
)
"""In-fit points, by `point_id`, whose envelope sits more than 1.3x above them, with the ratio they are held to: their
own, rounded up to two decimals, so a refit that moves one shows in the data test. Grouped by cause; with the default
`device_working_mem_gb` the cache reserves 3 GiB anyway for the 40 of them that estimate less.

- Untiled demand, concave under MIOpen's native allocator on Windows: first-run driver spikes at small sizes and a
  second run reserving up to 1.6x the first at large ones, so a line that covers every point cannot stay within 1.3x
  of all of them. Windows ROCm defaults to expandable segments; this is the override's price: the SD fp16 encodes at
  1024 and 1280 px are held at 1.48x and 1.64x. The 256 px references the tiling runs measure cap nothing
  (`fit.SMALLEST_APP_SIDE`) but are covered, and the bf16 decode's lies on the hull: the line that covers it, 10684
  bytes per pixel from 327 MiB, holds the 512 px decode at 1.44x.
- The untiled line's headroom at the tile there. A tiled line covers what its points measured above the untiled demand
  at their tile (`fit.py`), and the estimate prices the tile on the shipped untiled line, whose headroom over that
  demand comes on top: 1.44x over the bf16 decode at 512 px and 1.29x at 768 px, 1.26x over the bf16 encode at 768 px,
  1.27x over the SD fp16 decode at 1024 px, and 1.48x and 1.23x over the SD fp16 encode at 1024 and 512 px. The
  tiled points on those tiles are held at 1.32-1.72x.
- One tiled line serves every tile, so the largest jump any tile takes above its untiled demand sets its base for all,
  and the fastest growth per pixel its slope:
  - Under `cudaMallocAsync` FLUX.2's 1536 px tile decoded 4096 px in 8.59 GiB against 7.58 untiled: the tiled line,
    14 bytes per pixel from 880 MiB, holds the 512 px tile at 1.70-2.35x, the 768 px one at 1.38-1.47x and the 1024 px
    one at 1.32-1.33x. SD fp16's 1024 px tile took 3.95 GiB at 3072 px against 3.46 (the 0 sentinel's cold run) and
    its 512 px tile 1.08 GiB from 1536 px against 0.83: the line, 15 bytes per pixel from 408 MiB, holds the 512 px
    tile at 1.33-1.71x, the 768 px one at 1.33-1.37x and the 1024 px one at 6144 px at 1.31x. FLUX.1's line, 25 bytes
    per pixel through the 512 px tile's 4.17 GiB at 12288 px, holds that tile at 6144-8192 px at 1.30-1.42x, the
    768 px one at 8192 px at 1.44x and the 1024 px one at 12288 px at 1.41x.
  - On MIOpen with the native allocator SD fp16's 768 px tile decoded 2048 px in 2.95 GiB against 2.18 untiled, and
    SD fp32's 512 px tile 3072 px in 3.44 GiB against 2.50 at every other side to 4096 px: bases of 819 and 1013 MiB
    hold the 512 px tiles at 1.52-1.72x. FLUX.1's and FLUX.2's 512 px tiles at 12288 px (5.35 and 4.58 GiB against
    2.03 untiled) set their decode lines' 22 bytes per pixel, which hold the 1024 px tile there at 1.38x. FLUX.2's
    512 px tile encoded 12288 px 2.19 GiB above its untiled demand, 16 bytes per pixel, which hold the 1024 px tile
    there at 1.32x; SD fp32's 512 px tile encoded 1024 px 0.48 GiB above its own, which holds the 768 px tile at
    2048 px at 1.31x.
  - Under `cudaMallocAsync` SD fp32's 1536 px tile encoded 8192 px 1.80 GiB above its untiled demand, 29 bytes per
    pixel, its 512 px tile 12288 px 1.88 GiB above, 13: the line through the former holds the 512 px tile at
    4096-12288 px at 1.37-1.91x, the 768 px one at 8192 px at 1.43x and the 1024 px one at 12288 px at 1.40x; FLUX.2's
    holds the 512 px tile at 4096 px at 1.31x. On cuDNN with the native allocator FLUX.2's 512 px tile encoded
    12288 px 2.04 GiB above its untiled demand, 14.5 bytes per pixel, its 1024 px tile 0.84 GiB: the line holds the
    latter at 1.52x, the 1536 px tile there at 1.31x, the 512 px tile at 6144-8192 px and the 768 px one at 8192 px at
    1.32-1.43x, and an earlier `reserved_peak` point at 2024 px at 1.35x.
- The assembly lines of the largest decodes, at 12288 px up to 7x the tile's own demand: SD fp32's at a 512 px tile on
  cuDNN with the native allocator (92 bytes per pixel from 903 MiB) holds 4096-12288 px at 1.34-1.43x; on MIOpen with
  expandable segments SD fp32's (51 from 1607 MiB) holds the 512 px tile at 6144-12288 px at 1.33-1.55x and SD fp16's
  (30 from 474 MiB) the 512 and 1024 px tiles at 8192-12288 px at 1.30-1.31x; on MIOpen with the native allocator SD
  fp16's (60 from 1042 MiB) holds the 512 px tile at 6144-12288 px at 1.44-1.62x and the 1024 px one at 12288 px at
  1.30x, SD fp32's (80 from 1402 MiB) the 512 px tile at 8192 px at 1.42x. Under `cudaMallocAsync` FLUX.1's
  (15 from 1127 MiB) covers its 512 px tile's jump and holds 1024 px at 1.35x.
- The FLUX.2 denoise points are the Klein 4B width at 1440px with a 77-token caption, 8177 tokens, on both cards: the
  native allocator served them from 1676 MiB (RTX 4090) and 1879 MiB (RX 9060 XT), while the 512-token caption at
  8612 tokens took 2382 and 2577 MiB. The line has to cover the latter. Under `cudaMallocAsync` and expandable
  segments the two lie on one line."""


def point_id(point: MeasuredPoint, spec: "EstimatorSpec") -> str:
    """A readable id of a point: estimator, platform, its features as its estimator's line labels them (`fit.Line`:
    operation, dtype, size, tile and batch of a VAE operation), card and method. Features that name no line show as
    they are."""
    found = spec.line(point.features)
    label = found.label if found is not None else json.dumps(point.features, sort_keys=True, separators=(",", ":"))
    out_of_fit = "-outfit" if not point.in_fit else ""
    platform = ",".join(point.platform.values())
    card = point.device.split(" (")[0].replace(" ", "")
    return f"{point.estimator}-{platform}-{label}{out_of_fit}-{card}-{point.method}"


def platform_of(point: MeasuredPoint) -> Platform:
    """The platform a point is evaluated on. One that names no OS measured what holds on every OS."""
    return Platform(**({"os": "other"} | point.platform))  # type: ignore[arg-type]


def evaluate_point(
    point: MeasuredPoint, spec: "EstimatorSpec", data: Iterable[WorkingMemoryData]
) -> WorkingMemoryEstimate:
    """The estimate for the point's features, priced with the coefficients `data` gives its platform."""
    return spec.evaluate(point.features, resolve_coefficients(data, spec, platform_of(point), origin="data"))


def fitted_decides(estimate: WorkingMemoryEstimate) -> bool:
    return any(term.included and term.coefficient is not None for term in estimate.terms)


def bound_violation(point: MeasuredPoint, spec: "EstimatorSpec", estimate: WorkingMemoryEstimate) -> Optional[str]:
    """Why `estimate`, the estimate `spec` gives `point`'s features, breaks a bound `point` sets for it, or None."""
    fitted = fitted_decides(estimate)
    reserved = estimate.reservation_bytes if fitted else max(estimate.reservation_bytes, WORKING_MEMORY_FLOOR)
    if reserved < point.measured_bytes:
        return f"reserves {reserved / 2**30:.3f} GiB for a measured {point.measured_bytes / 2**30:.3f} GiB"
    # An out-of-fit point checks another term, a lower-bound-only point is no cold first run, and features the app
    # does not run on their own cap no line: none of them is held from above, but by a tighter `max_ratio`.
    line = spec.line(point.features)
    caps = point.holds_upper_bound and (line is None or line.caps)
    if point.max_ratio is not None or (caps and fitted and point.in_fit):
        default = UPPER_BOUND_EXCEPTIONS.get(point_id(point, spec), MAX_OVER_RESERVATION)
        max_ratio = point.max_ratio if point.max_ratio is not None else default
        ratio = estimate.reservation_bytes / point.measured_bytes
        if ratio > max_ratio:
            return f"reserves {ratio:.2f}x the measured peak, over {max_ratio}x"
    return None
