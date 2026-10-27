"""The coefficients estimators are evaluated with, and the measured points that bound them.

Both ship as JSON under `data/`, read through `importlib.resources` so a wheel install finds them. A coefficient entry
belongs to one estimator version on one measured platform, and either carries its values or borrows another entry's
(`values_from`); a point records the features of an operation that was measured and its measured peak. A carried value
is the fit of the points it is resolved for (`fit.py`), a declared stand-in (`stand_ins`), or a hand-set constant of an
entry marked `legacy`. `tests/backend/working_memory/test_shipped_coefficients.py` resolves every entry against its
`EstimatorSpec`, holds every point to the estimate the shipped coefficients give it, and every fitted value to the fit
of its points.
"""

import functools
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from invokeai.backend.working_memory.estimate import FeatureValue
from invokeai.backend.working_memory.platform import Platform, priced_as

if TYPE_CHECKING:
    from invokeai.backend.working_memory.spec import EstimatorSpec


class _DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValuesFrom(_DataModel):
    """The entry whose values another entry takes: same platform, this estimator id and version. It may carry more
    values than the borrower reads -- a family member that runs fewer dtypes reads fewer of them -- never fewer."""

    estimator: str
    version: int


class CoefficientEntry(_DataModel):
    """The coefficient values of one estimator version on one platform: its own, or another entry's."""

    estimator: str
    version: int
    platform: dict[str, str]
    values: dict[str, float] | None = None
    values_from: ValuesFrom | None = None
    """Borrow the values of another estimator's entry on the same platform, so that its measured points bound this
    estimator too, and this estimator's points join the fit of those values. One hop: the entry borrowed from carries
    its own values."""
    stand_ins: list[str] = []
    """Carried values that are declared conservative stand-ins rather than fits: what this platform prices a line no
    in-fit point measures at. `provenance` says what they stand for; every other carried value is a fit, unless the
    entry is `legacy`."""
    unmeasured_floor: dict[str, float] = {}
    """Declared lower bounds, not fits: what carried values are raised to where this entry prices a platform nothing
    is measured on (see `priced_as`). A fit describes the platform its points were measured on, not the one it stands
    in for. `provenance` says where the floors come from."""
    legacy: bool = False
    """True for an entry whose carried values are hand-set constants, not a fit of measured points: no point decides
    them and `fit.py` does not reproduce them, though the points of its estimator still bound its estimates. The data
    test lists the entries that may ship such values; `scripts/working_memory/build_shipped.py` replaces them with the
    fit of measured points once those fit every line and a fresh-process point measures each."""
    provenance: str
    """Card, torch build, method and script the values come from, and which points they were fitted on."""

    @model_validator(mode="after")
    def _values_or_values_from(self) -> Self:
        if (self.values is None) == (self.values_from is None):
            raise ValueError("a coefficient entry carries exactly one of `values` and `values_from`")
        for field, names in (("stand-ins", self.stand_ins), ("unmeasured floors", self.unmeasured_floor)):
            unknown = sorted(set(names) - set(self.values or {}))
            if unknown:
                raise ValueError(f"{field} {unknown} are not values this entry carries")
        return self


LOWER_BOUND_ONLY_METHODS: Final = frozenset({"probe_demand"})
"""Point methods that bound an estimate from below only (see `MeasuredPoint.holds_upper_bound`)."""


class MeasuredPoint(_DataModel):
    """One measured operation. The estimate for its features must be at least `measured_bytes`, and where a fitted
    term decides an in-fit point that `holds_upper_bound`, at most 1.3 times that (or `max_ratio`). Measured with the
    model's weights already in place, so `measured_bytes` is the operation's working memory alone."""

    estimator: str
    version: int
    platform: dict[str, str]
    features: dict[str, FeatureValue]
    measured_bytes: int
    method: Literal["reserved_peak", "driver_peak", "demand_peak", "probe_demand", "device_peak"]
    """How `measured_bytes` was read, each as a peak over its value just before the operation:

    - ``reserved_peak``: the caching allocator's reserved bytes, one fresh process per point. Reserved rather than
      allocated, because reserved is what the model cache's reservation has to cover.
    - ``driver_peak``: the process's device usage as the driver reports it (Windows ``D3DKMT`` ``CurrentUsage``), the
      larger of two runs, one fresh process per point.
    - ``demand_peak``: the largest of the driver-level, the reserved and the allocated peak, the larger of two runs,
      one fresh process per point. The driver-level figure is what reserved misses: on HIP the first operation of a
      process maps ~300 MiB beyond ``max_memory_reserved``.
    - ``device_peak``: the drop in the driver's free memory (`torch.cuda.mem_get_info`, polled during the operation)
      below its value just before it, the larger over the forwards measured. It counts the allocator's reserve and
      whatever is allocated around it, so it is at least the reserved peak.
    - ``probe_demand``: the working-memory probe's ``demand_bytes`` (`probe.MeasuredWindow`), the largest over the
      valid, fit-eligible windows of calibration runs in a running server, warm-ups included.
      `scripts/working_memory/build_shipped.py` builds these points from calibration store exports. A lower bound
      only: a warm window in a running server can read well below the cold first run after a server start, which is
      the worst case the reservation must cover, so it may raise a line but never caps one."""
    device: str
    provenance: str
    max_ratio: float | None = None
    """A tighter upper bound than the default 1.3, as a multiple of `measured_bytes`, for a point the estimate has to
    stay closer to. It applies whichever term decides the estimate."""
    in_fit: bool = True
    """False for a point whose demand is not its line's alone, kept to check another term: one measured in a regime its
    platform does not run (CUDA with the math attention kernel forced, whose score matrix shares the forward with the
    convolutions), or one whose features carry a term on top of the line (`fit.Line.fits`), such as the score matrix a
    build's own SDPA kernel materializes. The fit leaves it out, and it gets the lower bound only."""

    @model_validator(mode="after")
    def _max_ratio_needs_an_upper_bound(self) -> Self:
        if self.max_ratio is not None and not self.holds_upper_bound:
            raise ValueError(f"a {self.method} point bounds its estimate from below only; it takes no max_ratio")
        return self

    @property
    def holds_upper_bound(self) -> bool:
        """Whether the point also bounds its estimate from above: a fresh-process peak, not a `probe_demand` window."""
        return self.method not in LOWER_BOUND_ONLY_METHODS


class WorkingMemoryData(_DataModel):
    """One file under `data/`."""

    schema_version: Literal[1]
    coefficients: list[CoefficientEntry]
    points: list[MeasuredPoint]


@dataclass(frozen=True)
class ResolvedCoefficients:
    """The coefficient values an estimate is evaluated with, and where they came from."""

    values: Mapping[str, float]
    source: str


def _parse(entry: Traversable) -> WorkingMemoryData:
    try:
        return WorkingMemoryData.model_validate_json(entry.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise ValueError(f"Malformed working-memory data file {entry}: {e}") from e


def load_data(directory: Traversable) -> Mapping[str, WorkingMemoryData]:
    """Every JSON data file in `directory`, parsed, by file name.

    A directory with none raises, naming it: every estimator resolves its coefficients from there, so an install that
    lost the files would otherwise fail later as a missing entry for whichever estimator ran first.
    """
    entries = (
        sorted((entry for entry in directory.iterdir() if entry.name.endswith(".json")), key=lambda entry: entry.name)
        if directory.is_dir()
        else []
    )
    if not entries:
        raise FileNotFoundError(f"No working-memory data files in {directory}; every EstimatorSpec reads them there.")
    return MappingProxyType({entry.name: _parse(entry) for entry in entries})


@functools.cache
def shipped_data() -> Mapping[str, WorkingMemoryData]:
    """The data files shipped under `data/`. Read once per process."""
    return load_data(files("invokeai.backend.working_memory") / "data")


def _where(keys: Mapping[str, str]) -> str:
    # An estimator keyed by no platform trait (`platform_keys=()`) has one entry for every platform.
    return ",".join(f"{key}={value}" for key, value in keys.items()) or "any platform"


def find_entry(
    entries: Iterable[CoefficientEntry], estimator: str, version: int, platform: Mapping[str, str], where: str
) -> CoefficientEntry:
    """The entry of `estimator` `version` that applies on `platform`: of those whose every platform key it matches,
    the most specific one. An entry that also names the OS thereby takes precedence on that OS over the one that does
    not (see `Platform`). Raises, naming `where`, if none applies or the most specific is declared twice."""
    candidates = [
        entry
        for entry in entries
        if entry.estimator == estimator and all(platform.get(key) == value for key, value in entry.platform.items())
    ]
    matches = [entry for entry in candidates if entry.version == version]
    if not matches:
        found = f" (found only version {sorted({entry.version for entry in candidates})})" if candidates else ""
        raise LookupError(
            f"No working-memory coefficients for estimator '{estimator}' version {version} on {where}{found}."
        )
    most_specific = max(len(entry.platform) for entry in matches)
    matches = [entry for entry in matches if len(entry.platform) == most_specific]
    if len(matches) > 1:
        raise LookupError(
            f"Working-memory coefficients for estimator '{estimator}' version {version} on {where} are declared "
            f"{len(matches)} times."
        )
    return matches[0]


def resolve_coefficients(
    data: Iterable[WorkingMemoryData], spec: "EstimatorSpec", platform: Optional[Platform], origin: str
) -> ResolvedCoefficients:
    """The values `spec` takes on `platform`, and a `source` that says how they were found.

    A platform nothing is measured on is priced as the measured platform `priced_as` names. The entry for the spec's
    id and version that applies there (`find_entry`) either carries the values or borrows them, one hop, from the
    entry of another estimator that applies on the same platform. No other fallback: an estimate priced with
    coefficients fitted for something else is the shortfall this data exists to prevent, so a missing, ambiguous,
    chained or mismatched entry raises, naming what is wrong. `origin` prefixes the `source`.

    An estimator keyed by no platform trait (`platform_keys=()`) is priced alike everywhere: nothing is priced as
    anything else, and its points, which record no platform, are priced with `platform=None`. Any other estimator
    needs a platform.
    """
    if platform is None and spec.platform_keys:
        raise ValueError(f"Estimator '{spec.id}' is keyed by {list(spec.platform_keys)}: price it on a platform.")
    measured = priced_as(platform) if platform is not None else None
    full = {key: str(value) for key, value in asdict(measured).items()} if measured is not None else {}
    keys = {key: full[key] for key in spec.platform_keys}
    requested = {key: getattr(platform, key) for key in spec.platform_keys}
    where = _where(keys) if keys == requested else f"{_where(keys)} (priced for {_where(requested)})"
    entries = [entry for file in data for entry in file.coefficients]

    entry = find_entry(entries, spec.id, spec.version, full, where)
    carrier, lender = entry, None
    if entry.values_from is not None:
        lender = entry.values_from.estimator
        carrier = find_entry(entries, lender, entry.values_from.version, full, where)
        if carrier.values is None:
            raise LookupError(
                f"Working-memory coefficients for estimator '{spec.id}' on {where} borrow from '{lender}', whose "
                "entry borrows in turn; borrow from the entry that carries the values."
            )
    values = carrier.values
    assert values is not None, "a validated entry carries values or values_from"
    # An entry's own values are exactly what its spec reads; a lender's may be more (see `ValuesFrom`).
    matches = set(values) == spec.coefficients if lender is None else spec.coefficients <= set(values)
    if not matches:
        raise LookupError(
            f"Working-memory coefficients for estimator '{spec.id}' on {where} provide {sorted(values)}"
            f"{f' (from {lender})' if lender else ''}; the estimator reads {sorted(spec.coefficients)}."
        )

    # The source names the spec's keys, and a refining key (the OS) where an entry it read names one.
    shown = [*spec.platform_keys, *(key for key in full if key in {*entry.platform, *carrier.platform})]
    shown = list(dict.fromkeys(shown))
    read = {name: values[name] for name in sorted(spec.coefficients)}
    notes = [f"values of {lender}"] if lender else []
    if measured != platform and spec.platform_keys:
        floors = {name: floor for name, floor in carrier.unmeasured_floor.items() if name in read}
        read |= {name: max(read[name], floor) for name, floor in floors.items()}
        notes.insert(0, f"priced as {_where({key: full[key] for key in shown})}{', floored' if floors else ''}")
    note = f" ({', '.join(notes)})" if notes else ""
    named = asdict(platform) if platform is not None else {}
    source = f"{origin}:{_where({key: str(named[key]) for key in shown})}{note}"
    return ResolvedCoefficients(values=MappingProxyType(read), source=source)


@functools.cache
def shipped_coefficients(spec: "EstimatorSpec", platform: Platform) -> ResolvedCoefficients:
    """The shipped values `spec` takes on `platform`. Resolved once per spec and platform."""
    return resolve_coefficients(shipped_data().values(), spec, platform, origin="shipped")
