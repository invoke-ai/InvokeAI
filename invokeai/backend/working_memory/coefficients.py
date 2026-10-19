"""The coefficients estimators are evaluated with, and the measured points that bound them.

Both ship as JSON under `data/`, read through `importlib.resources` so a wheel install finds them. A coefficient entry
belongs to one estimator version on one measured platform value, and either carries its values or borrows another
entry's (`values_from`); a point records the features of an operation that was measured and its measured peak.
`tests/backend/working_memory/test_shipped_coefficients.py` resolves every entry against its `EstimatorSpec` and holds
every point to the estimate the shipped coefficients give it.
"""

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from invokeai.backend.working_memory.estimate import FeatureValue
from invokeai.backend.working_memory.platform import UNMEASURED_PRICED_AS, Platform

if TYPE_CHECKING:
    from invokeai.backend.working_memory.spec import EstimatorSpec


class _DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValuesFrom(_DataModel):
    """The entry whose values another entry takes: same platform, this estimator id and version."""

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
    estimator too. One hop: the entry borrowed from carries its own values."""
    provenance: str
    """Card, torch build, method and script the values come from, and which points they were fitted on."""

    @model_validator(mode="after")
    def _values_or_values_from(self) -> Self:
        if (self.values is None) == (self.values_from is None):
            raise ValueError("a coefficient entry carries exactly one of `values` and `values_from`")
        return self


class MeasuredPoint(_DataModel):
    """One measured operation. The estimate for its features must be at least `measured_bytes`, and where a fitted
    term decides it, at most 1.3 times that (or `max_ratio`)."""

    estimator: str
    version: int
    platform: dict[str, str]
    features: dict[str, FeatureValue]
    measured_bytes: int
    method: Literal["reserved_peak"]
    """``reserved_peak``: the caching allocator's peak reserved bytes over the operation, one fresh process per
    point. Reserved rather than allocated, because reserved is what the model cache's reservation has to cover."""
    device: str
    provenance: str
    max_ratio: float | None = None
    """A tighter upper bound than the default 1.3, as a multiple of `measured_bytes`, for a point the estimate has to
    stay closer to. It applies whichever term decides the estimate."""


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
    return ",".join(f"{key}={value}" for key, value in keys.items())


def _entry(
    entries: list[CoefficientEntry], estimator: str, version: int, keys: Mapping[str, str], where: str
) -> CoefficientEntry:
    candidates = [entry for entry in entries if entry.estimator == estimator and entry.platform == keys]
    matches = [entry for entry in candidates if entry.version == version]
    if len(matches) > 1:
        raise LookupError(
            f"Working-memory coefficients for estimator '{estimator}' version {version} on {where} are declared "
            f"{len(matches)} times."
        )
    if not matches:
        found = f" (found only version {sorted({entry.version for entry in candidates})})" if candidates else ""
        raise LookupError(
            f"No working-memory coefficients for estimator '{estimator}' version {version} on {where}{found}."
        )
    return matches[0]


def resolve_coefficients(
    data: Iterable[WorkingMemoryData], spec: "EstimatorSpec", platform: Platform, origin: str
) -> ResolvedCoefficients:
    """The values `spec` takes on `platform`, and a `source` that says how they were found.

    A platform value nothing is measured for is priced as the measured value `UNMEASURED_PRICED_AS` names. The entry
    for the spec's id, version and those platform values either carries the values or borrows them, one hop, from
    another estimator's entry on the same platform. No other fallback: an estimate priced with coefficients fitted for
    something else is the shortfall this data exists to prevent, so a missing, ambiguous, chained or mismatched entry
    raises, naming what is wrong. `origin` prefixes the `source`.
    """
    requested = {key: getattr(platform, key) for key in spec.platform_keys}
    keys = {key: UNMEASURED_PRICED_AS.get(key, {}).get(value, value) for key, value in requested.items()}
    where = _where(keys) if keys == requested else f"{_where(keys)} (priced for {_where(requested)})"
    entries = [entry for file in data for entry in file.coefficients]

    entry = _entry(entries, spec.id, spec.version, keys, where)
    values, lender = entry.values, None
    if entry.values_from is not None:
        lender = entry.values_from.estimator
        lent = _entry(entries, lender, entry.values_from.version, keys, where)
        if lent.values is None:
            raise LookupError(
                f"Working-memory coefficients for estimator '{spec.id}' on {where} borrow from '{lender}', whose "
                "entry borrows in turn; borrow from the entry that carries the values."
            )
        values = lent.values
    assert values is not None, "a validated entry carries values or values_from"
    if set(values) != spec.coefficients:
        raise LookupError(
            f"Working-memory coefficients for estimator '{spec.id}' on {where} provide {sorted(values)}"
            f"{f' (from {lender})' if lender else ''}; the estimator reads {sorted(spec.coefficients)}."
        )

    priced = ",".join(value for key, value in keys.items() if value != requested[key])
    if priced and lender:
        note = f" ({priced} values of {lender})"
    elif priced:
        note = f" ({priced} values)"
    else:
        note = f" (values of {lender})" if lender else ""
    return ResolvedCoefficients(values=MappingProxyType(dict(values)), source=f"{origin}:{_where(requested)}{note}")


@functools.cache
def shipped_coefficients(spec: "EstimatorSpec", platform: Platform) -> ResolvedCoefficients:
    """The shipped values `spec` takes on `platform`. Resolved once per spec and platform."""
    return resolve_coefficients(shipped_data().values(), spec, platform, origin="shipped")
