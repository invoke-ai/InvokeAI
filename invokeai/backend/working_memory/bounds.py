"""How a measured point bounds the estimate for its features.

One rule, read by the data test (`tests/backend/working_memory/test_shipped_coefficients.py`), which holds the shipped
data to it, and by `scripts/working_memory/build_shipped.py`, which refuses to write data that breaks it:

- Lower bound. The model cache consumes an estimate as ``free >= estimate``, so the estimate must be at least the point.
  Where only analytic terms decide it (a materialized score matrix), the cache's `device_working_mem_gb` floor counts.
- Upper bound, for points that hold one (`MeasuredPoint.holds_upper_bound`): where a fitted term decides an in-fit
  point, the estimate is at most `MAX_OVER_RESERVATION` times the point, its listed `UPPER_BOUND_EXCEPTIONS` ratio, or
  its tighter `max_ratio`, which applies whichever term decides. A reservation the operation never uses is model
  weights evicted for nothing.

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
        "vae.sd-miopen,native,windows-decode-fp16-1280x1280-scores-RX9060XT-demand_peak": 1.33,
        "vae.sd-miopen,native,windows-encode-fp16-1024x1024-scores-RX9060XT-demand_peak": 1.57,
        "vae.sd-miopen,native,windows-encode-fp16-1280x1280-scores-RX9060XT-demand_peak": 1.78,
        "vae.flux-miopen,native,windows-decode-bf16-512x512-scores-RX9060XT-demand_peak": 1.33,
        "vae.flux2-miopen,native,windows-decode-bf16-512x512-scores-RX9060XT-demand_peak": 1.33,
    }
)
"""In-fit points, by `point_id`, whose envelope sits more than 1.3x above them, with the ratio they are held to
instead. All are MIOpen with the native allocator on Windows, where the demand is concave in the size: first-run driver
spikes at small sizes and a second run reserving up to 1.6x the first at large ones, so a line that covers every point
cannot stay within 1.3x of all of them. Windows ROCm defaults to expandable segments; this is the override's price."""


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


def upper_bound_ratio(point: MeasuredPoint, spec: "EstimatorSpec", fitted: bool) -> Optional[float]:
    """How many times the point its estimate may reserve at most, as a fitted term decides it or not (`fitted`); None
    where the point sets it no upper bound. An out-of-fit point checks another term, and a lower-bound-only point is no
    cold first run: neither caps a fitted term."""
    if not point.holds_upper_bound:
        return None
    if point.max_ratio is not None:
        return point.max_ratio
    if fitted and point.in_fit:
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
    includes a point the line does not admit to its fit (`fit.Line.fits`): a tiled, batched or upcast VAE operation, a
    denoise forward with an analytic term. Its terms are read where the line's decide (`_DECIDING`), the other included
    terms at the bytes `data` prices them at, so the bound reads ``k * size + b * bases + others <= ratio * measured``,
    `size` in the units the terms price at the slope (pixels, tokens, a batch's samples). Where another term decides at
    the raised line, this holds the line to more than `bound_violation` asks: the line still lies within the point's
    ratio.

    The other terms keep the bytes of the values `data` carries, before any refit: a term on a second line that the
    same refit moves would be priced at its old value here. No shipped estimator prices one point on two lines."""
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    bounds: dict[LineKey, list[LineBound]] = {}
    for point in (point for file in files for point in file.points):
        spec = specs[point.estimator]
        found = spec.line(point.features)
        ratio = upper_bound_ratio(point, spec, fitted=True)
        if found is None or ratio is None:
            continue
        carrier = carrier_of(entries, point)
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
