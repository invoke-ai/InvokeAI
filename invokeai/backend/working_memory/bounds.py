"""How a measured point bounds the estimate for its features.

One rule, read by the data test (`tests/backend/working_memory/test_shipped_coefficients.py`), which holds the shipped
data to it, and by `scripts/working_memory/build_shipped.py`, which refuses to write data that breaks it:

- Lower bound. The model cache consumes an estimate as ``free >= estimate``, so the estimate must be at least the point.
  Where only analytic terms decide it (a materialized score matrix), the cache's `device_working_mem_gb` floor counts.
- Upper bound, for points that hold one (`MeasuredPoint.holds_upper_bound`): where a fitted term decides an in-fit
  point, the estimate is at most `MAX_OVER_RESERVATION` times the point, its listed `UPPER_BOUND_EXCEPTIONS` ratio, or
  its tighter `max_ratio`, which applies whichever term decides. A reservation the operation never uses is model
  weights evicted for nothing.
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
        "vae.sd-miopen,native,windows-decode-fp16-1280x1280-scores-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,native,windows-encode-fp16-1024x1024-scores-RX9060XT-demand_peak": 1.57,
        "vae.sd-miopen,native,windows-encode-fp16-1280x1280-scores-RX9060XT-demand_peak": 1.78,
        "vae.flux-miopen,native,windows-decode-bf16-512x512-scores-RX9060XT-demand_peak": 1.33,
        "vae.flux2-miopen,native,windows-decode-bf16-512x512-scores-RX9060XT-demand_peak": 1.33,
        "denoise.flux2-cudnn,native-denoise-bf16-h3072-8100+77tok-RTX4090-demand_peak": 1.42,
        "denoise.flux2-miopen,native,windows-denoise-bf16-h3072-8100+77tok-RX9060XT-demand_peak": 1.40,
    }
)
"""In-fit points, by `point_id`, whose envelope sits more than 1.3x above them, with the ratio they are held to
instead. All are the native allocator, never expandable segments or `cudaMallocAsync`:

- The VAE points are MIOpen on Windows, where the demand is concave in the size: first-run driver spikes at small sizes
  and a second run reserving up to 1.6x the first at large ones, so a line that covers every point cannot stay within
  1.3x of all of them. Windows ROCm defaults to expandable segments; this is the override's price.
- The FLUX.2 denoise points are the Klein 4B width at 1440px with a 77-token caption, 8177 tokens, on both cards: the
  native allocator served them from 1676 MiB (RTX 4090) and 1879 MiB (RX 9060 XT), while the 512-token caption at
  8612 tokens took 2382 and 2577 MiB. The line has to cover the latter. Under `cudaMallocAsync` and expandable
  segments the two lie on one line."""


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
    return any(term.included and term.coefficient is not None for term in estimate.terms)


def bound_violation(point: MeasuredPoint, spec: "EstimatorSpec", estimate: WorkingMemoryEstimate) -> Optional[str]:
    """Why `estimate`, the estimate `spec` gives `point`'s features, breaks a bound `point` sets for it, or None."""
    fitted = fitted_decides(estimate)
    reserved = estimate.reservation_bytes if fitted else max(estimate.reservation_bytes, WORKING_MEMORY_FLOOR)
    if reserved < point.measured_bytes:
        return f"reserves {reserved / 2**30:.3f} GiB for a measured {point.measured_bytes / 2**30:.3f} GiB"
    # An out-of-fit point checks another term, and a lower-bound-only point is no cold first run: neither caps.
    if point.holds_upper_bound and (point.max_ratio is not None or (fitted and point.in_fit)):
        default = UPPER_BOUND_EXCEPTIONS.get(point_id(point, spec), MAX_OVER_RESERVATION)
        max_ratio = point.max_ratio if point.max_ratio is not None else default
        ratio = estimate.reservation_bytes / point.measured_bytes
        if ratio > max_ratio:
            return f"reserves {ratio:.2f}x the measured peak, over {max_ratio}x"
    return None
