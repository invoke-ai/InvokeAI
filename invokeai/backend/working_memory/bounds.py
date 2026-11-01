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

The fit keeps the upper bounds while it raises a line for points above it: `line_bounds` states them as constraints on
the line, which `fit.raised_pair` satisfies.
"""

import json
from collections.abc import Iterable, Mapping
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.working_memory.coefficients import MeasuredPoint, WorkingMemoryData, resolve_coefficients
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate
from invokeai.backend.working_memory.fit import LineBound, LineKey, carried_lines, carrier_of, entry_key
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
        "vae.flux-miopen,native,windows-decode-bf16-512x512-scores-RX9060XT-demand_peak": 1.45,
        "vae.flux2-miopen,native,windows-decode-bf16-512x512-scores-RX9060XT-demand_peak": 1.45,
        "vae.sd-miopen,native,windows-encode-fp16-1024x1024-scores-RX9060XT-demand_peak": 1.49,
        "vae.sd-miopen,native,windows-encode-fp16-1280x1280-scores-RX9060XT-demand_peak": 1.65,
        # The untiled line's headroom at the tile, under MIOpen's native allocator on Windows.
        "vae.flux-miopen,native,windows-decode-bf16-1024x1024-tile512-scores-RX9060XT-demand_peak": 1.42,
        "vae.flux-miopen,native,windows-decode-bf16-1536x1536-tile512-scores-RX9060XT-demand_peak": 1.44,
        "vae.flux-miopen,native,windows-decode-bf16-2048x2048-tile512-scores-RX9060XT-demand_peak": 1.45,
        "vae.flux-miopen,native,windows-decode-bf16-3072x3072-tile512-scores-RX9060XT-demand_peak": 1.48,
        "vae.flux-miopen,native,windows-decode-bf16-4096x4096-tile512-scores-RX9060XT-demand_peak": 1.55,
        "vae.flux-miopen,native,windows-decode-bf16-4096x4096-tile768-scores-RX9060XT-demand_peak": 1.32,
        "vae.flux-miopen,native,windows-decode-bf16-6144x6144-tile512-scores-RX9060XT-demand_peak": 1.56,
        "vae.flux-miopen,native,windows-encode-bf16-8192x8192-tile768-scores-RX9060XT-demand_peak": 1.32,
        "vae.flux2-miopen,native,windows-decode-bf16-1024x1024-tile512-scores-RX9060XT-demand_peak": 1.43,
        "vae.flux2-miopen,native,windows-decode-bf16-12288x12288-tile512-scores-RX9060XT-demand_peak": 1.42,
        "vae.flux2-miopen,native,windows-decode-bf16-1536x1536-tile512-scores-RX9060XT-demand_peak": 1.43,
        "vae.flux2-miopen,native,windows-decode-bf16-2048x2048-tile512-scores-RX9060XT-demand_peak": 1.44,
        "vae.flux2-miopen,native,windows-decode-bf16-3072x3072-tile512-scores-RX9060XT-demand_peak": 1.49,
        "vae.flux2-miopen,native,windows-decode-bf16-4096x4096-tile512-scores-RX9060XT-demand_peak": 1.55,
        "vae.flux2-miopen,native,windows-decode-bf16-4096x4096-tile768-scores-RX9060XT-demand_peak": 1.32,
        "vae.flux2-miopen,native,windows-decode-bf16-6144x6144-tile512-scores-RX9060XT-demand_peak": 1.55,
        "vae.flux2-miopen,native,windows-decode-bf16-8192x8192-tile512-scores-RX9060XT-demand_peak": 1.38,
        "vae.flux2-miopen,native,windows-encode-bf16-8192x8192-tile768-scores-RX9060XT-demand_peak": 1.41,
        "vae.sd-miopen,native,windows-decode-fp16-1536x1536-tile1024-scores-RX9060XT-demand_peak": 1.55,
        "vae.sd-miopen,native,windows-decode-fp16-2048x2048-tile1024-scores-RX9060XT-demand_peak": 1.48,
        "vae.sd-miopen,native,windows-decode-fp16-3072x3072-tile1024-scores-RX9060XT-demand_peak": 1.48,
        "vae.sd-miopen,native,windows-decode-fp16-4096x4096-tile1024-scores-RX9060XT-demand_peak": 1.43,
        "vae.sd-miopen,native,windows-decode-fp16-6144x6144-tile1024-scores-RX9060XT-demand_peak": 1.48,
        "vae.sd-miopen,native,windows-encode-fp16-12288x12288-tile1024-scores-RX9060XT-demand_peak": 1.47,
        "vae.sd-miopen,native,windows-encode-fp16-1536x1536-tile1024-scores-RX9060XT-demand_peak": 1.62,
        "vae.sd-miopen,native,windows-encode-fp16-1536x1536-tile512-scores-RX9060XT-demand_peak": 1.38,
        "vae.sd-miopen,native,windows-encode-fp16-2048x2048-tile1024-scores-RX9060XT-demand_peak": 1.72,
        "vae.sd-miopen,native,windows-encode-fp16-2048x2048-tile512-scores-RX9060XT-demand_peak": 1.34,
        "vae.sd-miopen,native,windows-encode-fp16-3072x3072-tile1024-scores-RX9060XT-demand_peak": 1.66,
        "vae.sd-miopen,native,windows-encode-fp16-3072x3072-tile512-scores-RX9060XT-demand_peak": 1.37,
        "vae.sd-miopen,native,windows-encode-fp16-4096x4096-tile1024-scores-RX9060XT-demand_peak": 1.66,
        "vae.sd-miopen,native,windows-encode-fp16-6144x6144-tile1024-scores-RX9060XT-demand_peak": 1.62,
        "vae.sd-miopen,native,windows-encode-fp16-6144x6144-tile512-scores-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,native,windows-encode-fp16-8192x8192-tile1024-scores-RX9060XT-demand_peak": 1.63,
        # One tiled line serves every tile.
        "vae.flux-miopen,native,windows-decode-bf16-12288x12288-tile1024-scores-RX9060XT-demand_peak": 1.39,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-1024x1024-tile512-RTX4090-demand_peak": 2.39,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-1536x1536-tile1024-RTX4090-demand_peak": 1.33,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-1536x1536-tile512-RTX4090-demand_peak": 1.83,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-2048x2048-tile1024-RTX4090-demand_peak": 1.34,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-2048x2048-tile512-RTX4090-demand_peak": 1.85,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-2048x2048-tile768-RTX4090-demand_peak": 1.39,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-3072x3072-tile512-RTX4090-demand_peak": 1.91,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-4096x4096-tile512-RTX4090-demand_peak": 1.99,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-4096x4096-tile768-RTX4090-demand_peak": 1.47,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-6144x6144-tile512-RTX4090-demand_peak": 1.74,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-8192x8192-tile512-RTX4090-demand_peak": 1.64,
        "vae.flux2-cudnn,cudaMallocAsync-decode-bf16-8192x8192-tile768-RTX4090-demand_peak": 1.35,
        "vae.flux2-cudnn,native-encode-bf16-12288x12288-tile1024-RTX4090-demand_peak": 1.56,
        "vae.flux2-cudnn,native-encode-bf16-12288x12288-tile1536-RTX4090-demand_peak": 1.34,
        "vae.flux2-cudnn,native-encode-bf16-2024x2024-tile512-RTX4090-reserved_peak": 1.34,
        "vae.flux2-cudnn,native-encode-bf16-6144x6144-tile512-RTX4090-demand_peak": 1.35,
        "vae.flux2-cudnn,native-encode-bf16-8192x8192-tile512-RTX4090-demand_peak": 1.44,
        "vae.flux2-cudnn,native-encode-bf16-8192x8192-tile768-RTX4090-demand_peak": 1.46,
        "vae.flux2-cudnn,native-encode-bf16-8192x8192-tile1024-RTX4090-demand_peak": 1.32,
        "vae.flux2-cudnn,native-encode-bf16-16384x16384-tile1024-RTX4090-demand_peak": 1.57,
        "vae.flux2-cudnn,native-encode-bf16-20480x20480-tile1024-RTX4090-demand_peak": 1.41,
        "vae.flux2-cudnn,native-decode-bf16-8192x8192-tile768-RTX4090-demand_peak": 1.41,
        "vae.flux2-miopen,native,windows-decode-bf16-12288x12288-tile1024-scores-RX9060XT-demand_peak": 1.38,
        "vae.flux2-miopen,native,windows-encode-bf16-12288x12288-tile1024-scores-RX9060XT-demand_peak": 1.32,
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
        "vae.sd-miopen,native,windows-decode-fp16-1024x1024-tile512-scores-RX9060XT-demand_peak": 1.72,
        "vae.sd-miopen,native,windows-decode-fp16-1536x1536-tile512-scores-RX9060XT-demand_peak": 1.60,
        "vae.sd-miopen,native,windows-decode-fp16-2048x2048-tile512-scores-RX9060XT-demand_peak": 1.69,
        "vae.sd-miopen,native,windows-decode-fp16-3072x3072-tile512-scores-RX9060XT-demand_peak": 1.66,
        "vae.sd-miopen,native,windows-decode-fp16-4096x4096-tile512-scores-RX9060XT-demand_peak": 1.56,
        "vae.sd-miopen,native,windows-decode-fp32-1024x1024-tile512-scores-RX9060XT-demand_peak": 1.53,
        "vae.sd-miopen,native,windows-decode-fp32-1536x1536-tile512-scores-RX9060XT-demand_peak": 1.53,
        "vae.sd-miopen,native,windows-decode-fp32-2048x2048-tile512-scores-RX9060XT-demand_peak": 1.52,
        "vae.sd-miopen,native,windows-decode-fp32-4096x4096-tile512-scores-RX9060XT-demand_peak": 1.53,
        "vae.sd-miopen,native,windows-encode-fp32-2048x2048-tile768-scores-RX9060XT-demand_peak": 1.31,
        # The assembly lines of the largest decodes.
        "vae.flux-cudnn,cudaMallocAsync-decode-bf16-1024x1024-tile512-RTX4090-demand_peak": 1.35,
        "vae.sd-cudnn,native-decode-fp32-12288x12288-tile512-RTX4090-demand_peak": 1.37,
        "vae.sd-cudnn,native-decode-fp32-4096x4096-tile512-RTX4090-demand_peak": 1.43,
        "vae.sd-cudnn,native-decode-fp32-8192x8192-tile512-RTX4090-demand_peak": 1.35,
        "vae.sd-miopen,expandable,windows-decode-fp32-12288x12288-tile512-scores-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,expandable,windows-decode-fp32-6144x6144-tile512-scores-RX9060XT-demand_peak": 1.49,
        "vae.sd-miopen,expandable,windows-decode-fp32-8192x8192-tile512-scores-RX9060XT-demand_peak": 1.56,
        "vae.sd-miopen,native,windows-decode-fp16-12288x12288-tile1024-scores-RX9060XT-demand_peak": 1.31,
        "vae.sd-miopen,native,windows-decode-fp16-12288x12288-tile512-scores-RX9060XT-demand_peak": 1.63,
        "vae.sd-miopen,native,windows-decode-fp16-6144x6144-tile512-scores-RX9060XT-demand_peak": 1.44,
        "vae.sd-miopen,native,windows-decode-fp16-8192x8192-tile512-scores-RX9060XT-demand_peak": 1.52,
        "vae.sd-miopen,native,windows-decode-fp32-8192x8192-tile512-scores-RX9060XT-demand_peak": 1.42,
        "vae.sd-miopen,native,windows-decode-fp16-12288x12288-tile768-scores-RX9060XT-demand_peak": 1.35,
        # The floor's conservative counts, with the margin on them.
        "vae.flux-cudnn,cudaMallocAsync-encode-bf16-16384x16384-tile512-RTX4090-demand_peak": 1.32,
        "vae.flux-cudnn,cudaMallocAsync-encode-bf16-20480x20480-tile512-RTX4090-demand_peak": 1.32,
        "vae.flux-miopen,expandable,windows-decode-bf16-12288x12288-tile512-scores-RX9060XT-demand_peak": 1.34,
        "vae.flux-miopen,expandable,windows-decode-bf16-16384x16384-tile512-scores-RX9060XT-demand_peak": 1.33,
        "vae.flux-miopen,expandable,windows-decode-bf16-20480x20480-tile512-scores-RX9060XT-demand_peak": 1.33,
        "vae.flux-miopen,expandable,windows-decode-bf16-24576x24576-tile512-scores-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,expandable,windows-decode-fp16-12288x12288-tile1024-scores-RX9060XT-demand_peak": 1.32,
        "vae.sd-miopen,expandable,windows-decode-fp16-12288x12288-tile512-scores-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,expandable,windows-decode-fp16-6144x6144-tile512-scores-RX9060XT-demand_peak": 1.32,
        "vae.sd-miopen,expandable,windows-decode-fp16-8192x8192-tile512-scores-RX9060XT-demand_peak": 1.33,
        # The FLUX.2 denoise's short caption.
        "denoise.flux2-cudnn,native-denoise-bf16-h3072-8100+77tok-RTX4090-demand_peak": 1.42,
        "denoise.flux2-miopen,native,windows-denoise-bf16-h3072-8100+77tok-RX9060XT-demand_peak": 1.40,
        # The FLUX.3 denoise's forwards without reference images.
        "denoise.flux3-cudnn,native-denoise-bf16-h3072-8100+80tok-RTX4090-demand_peak": 1.31,
        "denoise.flux3-cudnn,native-denoise-fp16-h3072-8100+80tok-RTX4090-demand_peak": 1.31,
        "denoise.flux3-cudnn,cudaMallocAsync-denoise-bf16-h3072-4096+560tok-b2-RTX4090-demand_peak": 1.31,
        "denoise.flux3-cudnn,cudaMallocAsync-denoise-fp32-h3072-8100+80tok-RTX4090-demand_peak": 1.31,
        "denoise.flux3-cudnn,cudaMallocAsync-denoise-fp32-h3072-1024+560tok-RTX4090-demand_peak": 1.32,
        # The Krea-2 denoise's short caption.
        "denoise.krea2-cudnn,native-denoise-bf16-4096+77tok-NVIDIAGeForceRTX4090-demand_peak": 1.35,
        "denoise.krea2-cudnn,native-denoise-fp16-4096+77tok-NVIDIAGeForceRTX4090-demand_peak": 1.35,
        "denoise.krea2-cudnn,cudaMallocAsync-denoise-bf16-9216+77tok-NVIDIAGeForceRTX4090-demand_peak": 1.33,
        "denoise.krea2-cudnn,cudaMallocAsync-denoise-fp16-9216+77tok-NVIDIAGeForceRTX4090-demand_peak": 1.33,
    }
)
"""In-fit points, by `point_id`, whose envelope sits more than 1.3x above them, with the ratio they are held to: their
own, rounded up to two decimals, so a refit that moves one shows in the data test. Grouped by cause; with the default
`device_working_mem_gb` the cache reserves 3 GiB anyway for the 46 of them that estimate less.

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
  - Under `cudaMallocAsync` FLUX.2's 1536 px tile decoded 4096 px in 8.59 GiB against 7.58 untiled: the tiled line, 12
    bytes per pixel from 906 MiB, holds the 512 px tile at 1.64-2.39x, the 768 px one at 1.35-1.47x and the 1024 px one
    at 1.33-1.34x; its 512 px decodes from 12288 px on are the floor's. SD fp16's 1024 px tile took 3.95 GiB at 3072 px
    against 3.46 (the 0 sentinel's cold run) and its 512 px tile 1.08 GiB from 1536 px against 0.83: the line, 15 bytes
    per pixel from 408 MiB, holds the 512 px tile at 1.33-1.71x, the 768 px one at 1.33-1.37x and the 1024 px one at
    6144 px at 1.31x. FLUX.1's line, 18 bytes per pixel, with its 512 px decodes from 12288 px left to the floor, holds
    every tile within 1.3x.
  - On MIOpen with the native allocator SD fp16's 768 px tile decoded 2048 px in 2.95 GiB against 2.18 untiled, and
    SD fp32's 512 px tile 3072 px in 3.44 GiB against 2.50 at every other side to 4096 px: bases of 819 and 1013 MiB
    hold the 512 px tiles at 1.52-1.72x. FLUX.1's and FLUX.2's 512 px tiles at 12288 px (5.35 and 4.58 GiB against
    2.03 untiled) set their decode lines' 22 bytes per pixel, which hold the 1024 px tile there at 1.38x. FLUX.2's
    512 px tile encoded 12288 px 2.19 GiB above its untiled demand, 16 bytes per pixel, which hold the 1024 px tile
    there at 1.32x; SD fp32's 512 px tile encoded 1024 px 0.48 GiB above its own, which holds the 768 px tile at
    2048 px at 1.31x.
  - Under `cudaMallocAsync` SD fp32's 1536 px tile encoded 8192 px 1.80 GiB above its untiled demand, 29 bytes per
    pixel, its 512 px tile 12288 px 1.88 GiB above, 13: the line through the former holds the 512 px tile at
    4096-12288 px at 1.37-1.91x, the 768 px one at 8192 px at 1.43x and the 1024 px one at 12288 px at 1.40x. On
    cuDNN with the native allocator FLUX.2's 512 px tile encoded 12288 px 2.04 GiB above its untiled demand, its 1024
    px tile 0.84 GiB, and at 26624 and 28672 px its encodes lie 0.6-0.7% above the floor, so the line covers them: 17
    bytes per pixel from 15 MiB, it holds the 1024 px tile at 12288 and 16384 px at 1.56x, at 20480 px at 1.41x
    and at 8192 px at 1.32x, the 1536 px tile at 12288 px at 1.34x, the 512 px tile at 6144-8192 px and the 768 px
    one at 8192 px at 1.35-1.46x, and an earlier `reserved_peak` point at 2024 px at 1.34x. Its 1024 px tile decoded
    16384 px 3.76 GiB above its untiled demand, 15 bytes per pixel: the decode line, 16 from 0, holds the 768 px
    tile at 8192 px at 1.41x. That leaves the decode lines there no headroom. A FLUX.1 or FLUX.2 bf16 decode probe
    that raises the untiled line they share, any above about 4.63 GB at 1024 px, prices that point's tile higher
    and lifts it past its ratio, and no pair of tiled and assembly lines then holds every point within its bound:
    the builder refuses the raise until these exceptions are adjusted.
- The assembly lines of the largest decodes, at 12288 px up to 7x the tile's own demand: SD fp32's at a 512 px tile on
  cuDNN with the native allocator (92 bytes per pixel from 903 MiB) holds 4096-12288 px at 1.34-1.43x; on MIOpen with
  expandable segments SD fp32's (51 from 1607 MiB) holds the 512 px tile at 6144-12288 px at 1.33-1.55x; on MIOpen
  with the native allocator SD fp16's (60 from 1042 MiB) holds the 512 px tile at 6144-12288 px at 1.44-1.62x and the
  768 and 1024 px ones at 12288 px at 1.35x and 1.30x, SD fp32's (80 from 1402 MiB) the 512 px tile at 8192 px at
  1.42x. Under `cudaMallocAsync` FLUX.1's (16 from 1125 MiB) covers its 512 px tile's jump and holds 1024 px at 1.35x.
- The floor's conservative counts, with the shipped margin on them (`vae_working_memory._tiled_convolutions`): an
  image a decode node post-processes inside its window, and the distribution an encode node samples, counted as if
  the allocator kept every block the assembly freed. On MIOpen with expandable segments it reuses them: the floor
  decides FLUX.1's decodes at a 512 px tile from 12288 to 24576 px at 1.32-1.34x and SD fp16's at 6144-12288 px at
  1.32-1.33x, where the counted floor alone lies 1.25-1.27x above them. Under `cudaMallocAsync` FLUX.1's encodes at
  16384 and 20480 px, its six latent-wide tensors on top, at 1.32x.
- The FLUX.2 denoise points are the Klein 4B width at 1440px with a 77-token caption, 8177 tokens, on both cards: the
  native allocator served them from 1676 MiB (RTX 4090) and 1879 MiB (RX 9060 XT), while the 512-token caption at
  8612 tokens took 2382 and 2577 MiB. The line has to cover the latter. Under `cudaMallocAsync` and expandable
  segments the two lie on one line.
- The FLUX.3 denoise points are forwards without reference images on the RTX 4090, one of them a batch of two, that
  the reference points put 1.302 to 1.320x under their line: references cost more per token than the image and the
  caption (up to 8% under the native allocator, 12% under `cudaMallocAsync`; see `flux3/denoise_working_memory.py`), and
  the line has to cover the reference forwards. Under the native allocator
  the 1440px forward with an 80-token caption (8180 tokens, 2060 MiB, against 2516 MiB for the 560-token caption at 8660
  tokens) in bf16 and fp16, under `cudaMallocAsync` a batch of two at 1024px in bf16 and, in fp32, the 512px forward
  and the 1440px one with an 80-token caption.
- The Krea-2 denoise points are a 77-token caption on the RTX 4090, bf16 and fp16 alike. A caption token reserves more
  than an image token. Allocated memory grows 0.14 to 0.15 MiB per token for caption and image tokens alike, reserved
  memory 0.30 MiB per image token and 0.73 to 0.95 MiB per caption token under the native allocator (1404 against
  1086 MiB at 4096 image tokens with 512 and 77 caption tokens, 2916 against 2502 MiB at 9216): the excess is
  allocator reserve, not live tensors, consistent with the text fusion, which runs each caption token as 12 tapped
  encoder layers of 2560 before the blocks, leaving segments the blocks' larger tensors do not reuse. One slope prices
  both, and the line has to cover the 512-token captions, so the
  77-token forward at 4096 image tokens sits 1.35x below it. Under `cudaMallocAsync` a caption token reserves about
  twice an image token (1856 against 1680 MiB at 9216 image tokens), and the line is held up by a forward whose warm
  run read 18% above its cold one (one 1024px joint reference at 1024px: 1984 against 1680 MiB), which puts the
  77-token forward at 9216 image tokens 1.32x below it."""


def point_id(point: MeasuredPoint, spec: "EstimatorSpec") -> str:
    """A readable id of a point: estimator, platform, its features as its estimator's line labels them (`fit.Line`:
    operation, dtype, size, tile, batch and a materialized score matrix of a VAE operation; size and prompt tokens of a
    MOSS-Image denoise), card and method. Features that name no line show as they are. An estimator keyed by no
    platform trait records none."""
    found = spec.line(point.features)
    label = found.label if found is not None else json.dumps(point.features, sort_keys=True, separators=(",", ":"))
    platform = ",".join(point.platform.values())
    out_of_fit = "-outfit" if not point.in_fit else ""
    card = point.device.split(" (")[0].replace(" ", "")
    head = f"{point.estimator}-{platform}" if platform else point.estimator
    return f"{head}-{label}{out_of_fit}-{card}-{point.method}"


_ANY_PLATFORM: Final = Platform("cudnn", "native", "linux")
"""Where a point of an estimator keyed by no platform trait is evaluated. Its one entry resolves alike on every
platform, so any will do; this one is measured."""


def _platform_of(point: MeasuredPoint, spec: "EstimatorSpec") -> Platform:
    """The platform a point is evaluated on. One that names no OS measured what holds on every OS. One of an estimator
    keyed by no platform trait names nothing (`_ANY_PLATFORM`)."""
    if not spec.platform_keys:
        return _ANY_PLATFORM
    return Platform(**({"os": "other"} | point.platform))  # type: ignore[arg-type]


def evaluate_point(
    point: MeasuredPoint, spec: "EstimatorSpec", data: Iterable[WorkingMemoryData]
) -> WorkingMemoryEstimate:
    """The estimate for the point's features, priced with the coefficients `data` gives its platform."""
    return spec.evaluate(point.features, resolve_coefficients(data, spec, _platform_of(point, spec), origin="data"))


def fitted_decides(estimate: WorkingMemoryEstimate) -> bool:
    """Whether a fitted term is among those the estimate reserves. A tiled VAE operation's floor carries the untiled
    line's base (`floor_base`, a fitted value), so a point its floor decides is held to the upper bound below as the
    fitted lines' points are; were the base analytic, those points would drop out of it unseen."""
    return any(term.included and term.coefficient is not None for term in estimate.terms)


def upper_bound_ratio(point: MeasuredPoint, spec: "EstimatorSpec", fitted: bool) -> Optional[float]:
    """How many times the point its estimate may reserve at most, as a fitted term decides it or not (`fitted`); None
    where the point sets it no upper bound. An out-of-fit point checks another term, a lower-bound-only point is no
    cold first run, and features the app does not run on their own cap no line (`fit.Line.caps`): none of them caps a
    fitted term, but by a tighter `max_ratio`."""
    if not point.holds_upper_bound:
        return None
    if point.max_ratio is not None:
        return point.max_ratio
    line = spec.line(point.features)
    if fitted and point.in_fit and (line is None or line.caps):
        return UPPER_BOUND_EXCEPTIONS.get(point_id(point, spec), MAX_OVER_RESERVATION)
    return None


def bound_violation(point: MeasuredPoint, spec: "EstimatorSpec", estimate: WorkingMemoryEstimate) -> Optional[str]:
    """Why `estimate`, the estimate `spec` gives `point`'s features, breaks a bound `point` sets for it, or None."""
    fitted = fitted_decides(estimate)
    reserved = estimate.reservation_bytes if fitted else max(estimate.reservation_bytes, WORKING_MEMORY_FLOOR)
    if reserved < point.measured_bytes:
        return f"reserves {reserved / 2**30:.3f} GiB for a measured {point.measured_bytes / 2**30:.3f} GiB"
    max_ratio = upper_bound_ratio(point, spec, fitted)
    if max_ratio is not None:
        ratio = estimate.reservation_bytes / point.measured_bytes
        if ratio > max_ratio:
            return f"reserves {ratio:.2f}x the measured peak, over {max_ratio}x"
    return None


_MIB: Final = 2**20

_DECIDING: Final = float(2**60)
"""A coefficient value large enough that the terms priced with it decide any estimate they take part in. A line's
terms are read where it decides, since only there can raising it break an upper bound."""


def line_bounds(
    data: Iterable[WorkingMemoryData], specs: Mapping[str, "EstimatorSpec"]
) -> Mapping[LineKey, list[LineBound]]:
    """Per line, the upper bounds its points set on it, which `fit.raised_pair` keeps when it raises the line.

    A point bounds the line its estimator's `EstimatorSpec.line` places its features on, on the entry that carries its
    values (`fit.carrier_of`), where it holds an upper bound with a fitted term deciding (`upper_bound_ratio`). That
    includes a point the line does not admit to its fit (`fit.Line.fits`): a batched or upcast VAE operation, a denoise
    forward with an analytic term. Its terms are read where the line's decide (`_DECIDING`), the other included terms at
    the bytes `data` prices them at, so the bound reads ``k * size + b * bases + others <= ratio * measured``, `size` in
    the units the terms price at the slope (pixels, tokens, a batch's samples). Where another term decides at the raised
    line, this holds the line to more than `bound_violation` asks: the line still lies within the point's ratio.

    A point on a line that sits on another (`fit.Line.sits_on`, a tiled VAE operation) bounds neither line. That line is
    not raised (`fit.sitting_pair`), and it is fitted after the line beneath, on what that one prices the point's tile
    at (`fit.base_values`): a raise beneath moves the tile's cost off the line above rather than adding to it, which a
    bound holding the line above at its carried values would forbid. The result is checked against the point's bound
    all the same (`bound_violation`, in the data test and the builder).

    The other terms keep the bytes of the values `data` carries, before any refit: a term on a second line that the
    same refit moves would be priced at its old value here. No shipped estimator prices a point that bounds a line on a
    second line that is refitted with it."""
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    bounds: dict[LineKey, list[LineBound]] = {}
    for point in (point for file in files for point in file.points):
        spec = specs[point.estimator]
        found = spec.line(point.features)
        ratio = upper_bound_ratio(point, spec, fitted=True)
        if found is None or found.sits_on is not None or ratio is None:
            continue
        carrier = carrier_of(entries, point, found.suffix)
        if found.suffix not in carried_lines(carrier):
            continue
        k_name, b_name = f"k_{found.suffix}", f"b_{found.suffix}"
        values = dict(resolve_coefficients(files, spec, _platform_of(point, spec), origin="data").values)
        terms = spec.terms(point.features, values | {k_name: _DECIDING, b_name: _DECIDING})
        on_line = [term for term in terms if term.included and term.coefficient in (k_name, b_name)]
        if not on_line:
            continue
        others = sum(term.bytes for term in terms if term.included and term.coefficient not in (k_name, b_name))
        allowed = Fraction(ratio) * point.measured_bytes
        # Two points can share an id (repeated measurements); their bytes tell them apart.
        measured = f"measured {point.measured_bytes / _MIB:.0f} MiB, at most {allowed / _MIB:.0f} MiB"
        label = f"{point_id(point, spec)} ({measured})"
        bound = LineBound(
            size=sum((Fraction(term.scale or 0) for term in on_line if term.coefficient == k_name), Fraction(0)),
            bases=sum((Fraction(term.scale or 0) for term in on_line if term.coefficient == b_name), Fraction(0)),
            ceiling=allowed - Fraction(others),
            point=label,
        )
        bounds.setdefault((entry_key(carrier), found.suffix), []).append(bound)
    return bounds
