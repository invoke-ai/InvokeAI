"""Shipped working-memory points and fitted values from calibration store exports and calibration point files.

    python -m scripts.working_memory.build_shipped [EXPORT ...] [--points FILE ...] [--data-dir DIR] [--write]

An export is a device class's calibration store as `GET /api/v2/models/working_memory/export` returns it; a family E2E
calibration run saves one per server lifetime under `runs/<run_id>/calibration/`. Every valid, fit-eligible window
becomes a `probe_demand` point: per card (device class), estimator, version, platform and features, the largest
`demand_bytes` of the windows, so repetitions and several exports of one card merge. The platform is what
`platform_for` gives on the machine the export came from (`platform.platform_for_environment`).

A point file is what a calibration script such as `scripts/calibrate_z_image_working_memory.py` writes: one JSON
object per line, a `MeasuredPoint`'s fields and any other keys, which are ignored. Its points are fresh-process
measurements, so they hold the upper bound as well. A check's point, which tests an analytic term or an assumption
rather than the line (a forced math kernel, a ControlNet, more blocks), says `in_fit: false`: it ships as an
out-of-fit point that bounds its estimate from below only.

A point joins the data file that declares its estimator's coefficients, keeping the larger of itself and a shipped
point of the same measurement (`point_identity`), and says which it kept; the carried values are then refitted with
`fit.py`. A stand-in stays one until fresh-process points at two sizes far enough apart measure its line:
`probe_demand` points can raise it, never lower it. A `legacy` entry is replaced by its fit where the points fit every line it carries and each line has a point
that holds the upper bound, a fresh-process one: probe windows alone never replace hand-set values. It keeps them
otherwise.

Without `--write` the changes are printed only. An export or point of another estimator version than the registered
spec, or features the spec's terms cannot evaluate, is refused. So is a result any point's bounds
(`working_memory/bounds.py`, the data test's rule) reject, and a line raised on a point flagged as fragmentation unless
`--accept-fragmentation` is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

import invokeai.backend.working_memory as working_memory_package
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationEnvironment,
    CalibrationPoint,
    CalibrationStoreFile,
)
from invokeai.backend.working_memory.bounds import bound_violation, evaluate_point, point_id
from invokeai.backend.working_memory.coefficients import (
    CoefficientEntry,
    MeasuredPoint,
    WorkingMemoryData,
    load_data,
)
from invokeai.backend.working_memory.estimate import FeatureValue
from invokeai.backend.working_memory.fit import (
    EntryKey,
    FitPoint,
    capping_spans,
    entry_fit,
    entry_key,
    fit_points,
    fitted_values,
    shipped_pair,
)
from invokeai.backend.working_memory.fragmentation import Measurement, fragmented
from invokeai.backend.working_memory.platform import (
    Allocator,
    ConvBackend,
    OperatingSystem,
    Platform,
    is_measured,
    platform_for_environment,
)
from invokeai.backend.working_memory.spec import EstimatorSpec, import_estimator_modules

METHOD = "probe_demand"
DATA_DIR = Path(working_memory_package.__file__).parent / "data"
EXPORT_NAME = re.compile(r"-(?:(?P<lifetime>\d+)-)?[0-9a-f]{16}\.json$")
"""A store export as a family E2E run names it, `<run_id>-<lifetime>-<device class id>.json`."""
REFIT_NOTE = "Refitted on its points by scripts/working_memory/build_shipped.py."
RAISE_NOTE = (
    "Stand-ins raised to cover probe_demand points by scripts/working_memory/build_shipped.py; still stand-ins until "
    "fresh-process points at two sizes at least twice apart measure their line."
)
_RUNTIME_FACTS = ("cuda", "hip", "cudnn", "miopen", "xpu")

PointIdentity = tuple[str, int, str, str, str, str, bool]
"""What makes two points one measurement (`point_identity`)."""


class BuildError(ValueError):
    pass


@dataclass(frozen=True)
class _Window:
    """A window that becomes, or competes to become, a point."""

    point: CalibrationPoint
    env: CalibrationEnvironment
    device: str
    export: Path
    attempt: str
    """The calibration attempt the window ran in (`attempt_id`), or its run without one."""


@dataclass(frozen=True)
class Spread:
    """The windows a point is the largest of, in the order they ran: export order, then queue item."""

    demand: tuple[int, ...]
    reserved: tuple[int | None, ...]
    """Each window's reserved delta."""
    allocated: tuple[int | None, ...]
    fragmented: bool = False
    """Whether the attempt its largest window ran in measured fragmentation at its features
    (`fragmentation.fragmented`): the allocator's cache grew around the same tensors, as a long-running server
    fragments it, rather than the operation needing more."""

    @property
    def first_to_max(self) -> float:
        return max(self.demand) / self.demand[0]


@dataclass
class Build:
    points: list[MeasuredPoint]
    skipped: Counter[str] = field(default_factory=Counter)
    """Windows or records left out, by reason."""
    spreads: dict[PointIdentity, Spread] = field(default_factory=dict)
    """By `point_identity`; the points built from export windows only."""


def _canonical(features: Mapping[str, Any]) -> str:
    return json.dumps(features, sort_keys=True, separators=(",", ":"))


def point_identity(point: MeasuredPoint) -> PointIdentity:
    """What a newer point of the same measurement replaces: estimator, version, platform, card, method and features,
    and whether it is in the fit. A check can share its features with a fit point, as a forward with more blocks does,
    and must not take its place."""
    return (
        point.estimator,
        point.version,
        _canonical(point.platform),
        point.device,
        point.method,
        _canonical(point.features),
        point.in_fit,
    )


def point_platform(platform: Platform, entries: Sequence[CoefficientEntry]) -> dict[str, str]:
    """The platform keys a point names: all of `platform`, the OS only where an entry that applies there names it."""
    full = {key: str(value) for key, value in asdict(platform).items()}
    os_specific = any(
        "os" in entry.platform and all(full.get(key) == value for key, value in entry.platform.items())
        for entry in entries
    )
    return full if os_specific else {key: value for key, value in full.items() if key != "os"}


def _load_export(path: Path) -> CalibrationStoreFile:
    try:
        return CalibrationStoreFile.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as e:
        raise BuildError(f"{path}: not a calibration store this version reads: {e}") from e


def _registered_spec(where: str, estimator: str, version: int, specs: Mapping[str, EstimatorSpec]) -> EstimatorSpec:
    spec = specs.get(estimator)
    if spec is None:
        raise BuildError(f"{where}: no registered estimator is called {estimator!r}")
    if version != spec.version:
        raise BuildError(
            f"{where}: {estimator} version {version}, the registered estimator is version {spec.version}; measure "
            "again with this version"
        )
    return spec


def _check_features(where: str, features: Mapping[str, FeatureValue], spec: EstimatorSpec) -> None:
    try:
        spec.terms(features, dict.fromkeys(spec.coefficients, 1.0))
    except Exception as e:
        raise BuildError(f"{where}: the estimator cannot evaluate its features {dict(features)}: {e!r}") from e


def _provenance(best: _Window, count: int) -> str:
    hard, soft = best.env.hard, best.env.soft
    runtime = ", ".join(f"{name} {hard[name]}" for name in _RUNTIME_FACTS if hard.get(name))
    window = best.point
    return (
        f"probe_demand, the largest of {count} valid fit-eligible window{'s' if count != 1 else ''}; the largest from "
        f"calibration run {window.run_id}, item {window.queue_item_id}, node {window.node_type} {window.node_id}, "
        f"model {window.model_key}. {best.device}, torch {soft.get('torch_build') or hard.get('torch')}"
        f"{', ' + runtime if runtime else ''}. Export {best.export.name}."
    )


def build_points(
    exports: Sequence[Path], specs: Mapping[str, EstimatorSpec], entries: Sequence[CoefficientEntry]
) -> Build:
    """The `probe_demand` points of the exports' valid, fit-eligible windows, and what was left out."""
    groups: dict[tuple[str, str, int, str, str], list[_Window]] = {}
    platforms: dict[tuple[str, str, int, str, str], dict[str, str]] = {}
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for path in exports:
        store = _load_export(path)
        envs = {env.id: env for env in store.envs}
        attempts = {run.run_id: run.attempt_id or run.run_id for run in store.runs}
        # A point's card is its device class. One name can stand for several (another memory size, say); the id
        # tells them apart, in the groups and in the points written.
        device = f"{store.device_class.name} ({store.device_class.id})"
        for window in store.points:
            # The same store saved twice (an export and a copy of its file, one file named twice) counts once.
            identity = window.model_dump_json()
            if identity in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(identity)
            spec = _registered_spec(str(path), window.estimator, window.version, specs)
            if not window.valid:
                skipped["invalid"] += 1
                continue
            if not window.fit_eligible:
                skipped["not fit-eligible"] += 1
                continue
            if window.demand_bytes is None or window.demand_bytes <= 0:
                # Co-locks that moved more weights off the device than the operation took; nothing to bound.
                skipped["no positive demand"] += 1
                continue
            where = (
                f"{path}: {window.estimator} window of node {window.node_type} {window.node_id} "
                f"(item {window.queue_item_id})"
            )
            _check_features(where, window.features, spec)
            env = envs.get(window.env_id)
            if env is None:
                raise BuildError(f"{path}: a window names environment {window.env_id}, which the store does not hold")
            try:
                platform = platform_for_environment(store.device_class.backend, env.hard)
            except ValueError:
                # Recorded by a server that does not record every fact the platform is read from.
                skipped["no platform facts"] += 1
                continue
            if not is_measured(platform):
                raise BuildError(
                    f"{path}: measured on {platform}, which `platform.MEASURED` does not list; ship coefficient "
                    "entries for it first"
                )
            keys = point_platform(platform, entries)
            group = (device, window.estimator, window.version, _canonical(keys), _canonical(window.features))
            platforms[group] = keys
            attempt = attempts.get(window.run_id, window.run_id)
            groups.setdefault(group, []).append(_Window(window, env, device, path, attempt))

    order = {path: index for index, path in enumerate(exports)}
    ordered = {
        group: sorted(windows, key=lambda w: (order[w.export], w.point.queue_item_id))
        for group, windows in sorted(groups.items())
    }
    flagged = _fragmented_attempts(ordered, specs)
    points = []
    spreads = {}
    for group, windows in ordered.items():
        best = max(windows, key=lambda w: w.point.demand_bytes or 0)
        assert best.point.demand_bytes is not None, "windows without a positive demand are left out"
        points.append(
            MeasuredPoint(
                estimator=best.point.estimator,
                version=best.point.version,
                platform=platforms[group],
                features=dict(best.point.features),
                measured_bytes=best.point.demand_bytes,
                method=METHOD,
                device=best.device,
                provenance=_provenance(best, len(windows)),
            )
        )
        spreads[point_identity(points[-1])] = Spread(
            demand=tuple(w.point.demand_bytes or 0 for w in windows),
            reserved=tuple(w.point.reserved.delta if w.point.reserved else None for w in windows),
            allocated=tuple(w.point.allocated.delta if w.point.allocated else None for w in windows),
            fragmented=(group, best.export, best.attempt) in flagged,
        )
    return Build(points=points, skipped=skipped, spreads=spreads)


_Group = tuple[str, str, int, str, str]
"""A point's windows: card, estimator, version, platform keys and features."""


def _fragmented_attempts(
    groups: Mapping[_Group, Sequence[_Window]], specs: Mapping[str, EstimatorSpec]
) -> set[tuple[_Group, Path, str]]:
    """The attempts whose windows of a point measured fragmentation (`fragmentation.fragmented`), as (group, export,
    attempt): each attempt in one server lifetime is a measurement of its own, compared with the smaller sizes of
    its card's line."""
    keys: list[tuple[_Group, Path, str]] = []
    measurements: list[Measurement] = []
    for group, windows in groups.items():
        card, estimator, version, platform, _ = group
        found = specs[estimator].line(windows[0].point.features)
        # Features that name no line compare with nothing: the growth rule alone judges them.
        line: object = group if found is None else (card, estimator, version, platform, found.suffix)
        size = 0 if found is None else found.size
        by_attempt: dict[tuple[Path, str], list[_Window]] = {}
        for window in windows:
            by_attempt.setdefault((window.export, window.attempt), []).append(window)
        for (export, attempt), members in by_attempt.items():
            keys.append((group, export, attempt))
            measurements.append(
                Measurement(
                    line=line,
                    size=size,
                    demand=[w.point.demand_bytes or 0 for w in members],
                    reserved=[w.point.reserved.delta if w.point.reserved else None for w in members],
                    allocated=[w.point.allocated.delta if w.point.allocated else None for w in members],
                )
            )
    return {key for key, flag in zip(keys, fragmented(measurements), strict=True) if flag}


def _record_platform(where: str, platform: Mapping[str, str], entries: Sequence[CoefficientEntry]) -> dict[str, str]:
    """A point file record's platform as the data names it (`point_platform`). A record names the OS where it measured
    an OS-specific entry's platform, and may name it anywhere; one that does not measured what holds on every OS."""
    named = {"os": "other"} | dict(platform)
    choices = {"conv_backend": ConvBackend, "allocator": Allocator, "os": OperatingSystem}
    problems = [f"no {key}" for key in sorted(choices.keys() - named.keys())] + [
        f"{key}={value!r}" for key, value in named.items() if key not in choices or value not in get_args(choices[key])
    ]
    if problems:
        raise BuildError(f"{where}: platform {dict(platform)} is no platform the data keys points by: {problems}")
    measured = Platform(**named)  # type: ignore[arg-type]
    if not is_measured(measured):
        raise BuildError(
            f"{where}: measured on {measured}, which `platform.MEASURED` does not list; ship coefficient entries for "
            "it first"
        )
    return point_platform(measured, entries)


def read_point_files(
    paths: Sequence[Path], specs: Mapping[str, EstimatorSpec], entries: Sequence[CoefficientEntry]
) -> Build:
    """The fresh-process points of calibration point files: per measurement (`point_identity`) the largest, and the
    records left out.

    A record is a JSON object on a line of its own: a `MeasuredPoint`'s fields (`estimator`, `version`, `platform`,
    `features`, `measured_bytes`, `method`, `device`, `provenance`, optionally `in_fit` and `max_ratio`) and any other
    keys, which are ignored. A fresh-process point holds the upper bound, so a `probe_demand` record is refused: probe
    windows come from exports.
    """
    best: dict[PointIdentity, MeasuredPoint] = {}
    skipped: Counter[str] = Counter()
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise BuildError(f"{path}: not readable: {e}") from e
        for number, text in enumerate(lines, start=1):
            if not text.strip():
                continue
            where = f"{path}:{number}"
            try:
                record = json.loads(text)
            except json.JSONDecodeError as e:
                raise BuildError(f"{where}: not a JSON object: {e}") from e
            if not isinstance(record, dict):
                raise BuildError(f"{where}: not a JSON object")
            try:
                point = MeasuredPoint.model_validate(
                    {key: value for key, value in record.items() if key in MeasuredPoint.model_fields}
                )
            except ValidationError as e:
                raise BuildError(f"{where}: not a measured point: {e}") from e
            if not point.holds_upper_bound:
                raise BuildError(
                    f"{where}: a {point.method} point bounds its estimate from below only; a point file holds "
                    "fresh-process measurements, and probe windows come from exports"
                )
            spec = _registered_spec(where, point.estimator, point.version, specs)
            _check_features(where, point.features, spec)
            point = point.model_copy(update={"platform": _record_platform(where, point.platform, entries)})
            identity = point_identity(point)
            kept = best.get(identity)
            if kept is not None:
                skipped["remeasured, the larger kept"] += 1
            if kept is None or point.measured_bytes > kept.measured_bytes:
                best[identity] = point
    return Build(points=list(best.values()), skipped=skipped)


def merge_points(
    data: Mapping[str, WorkingMemoryData], points: Sequence[MeasuredPoint]
) -> dict[str, WorkingMemoryData]:
    """`data` with `points` in the file that declares their estimator's coefficients. A point of the same measurement
    already there keeps its place and the larger `measured_bytes` of the two; any other point is appended."""
    owners: dict[str, str] = {}
    for name, file in data.items():
        for entry in file.coefficients:
            if owners.setdefault(entry.estimator, name) != name:
                raise BuildError(f"{entry.estimator} has coefficients in both {owners[entry.estimator]} and {name}")
    new: dict[str, dict[PointIdentity, MeasuredPoint]] = {name: {} for name in data}
    for point in points:
        owner = owners.get(point.estimator)
        if owner is None:
            raise BuildError(f"no data file declares coefficients for {point.estimator}; add its entries first")
        new[owner][point_identity(point)] = point
    merged = {}
    for name, file in data.items():
        replacing = new[name]
        kept = []
        for point in file.points:
            newer = replacing.pop(point_identity(point), None)
            kept.append(point if newer is None or newer.measured_bytes <= point.measured_bytes else newer)
        merged[name] = file.model_copy(update={"points": kept + list(replacing.values())})
    return merged


FITTED_LEGACY_PROVENANCE = (
    "The fit of this platform's in-fit points by scripts/working_memory/build_shipped.py, in place of hand-set legacy "
    "values. Each point's provenance names what measured it."
)
"""The provenance of a `legacy` entry its points now fit: its hand-set provenance describes values it no longer
carries."""


def legacy_fit(
    entry: CoefficientEntry, groups: Mapping[tuple[EntryKey, str], Sequence[FitPoint]]
) -> tuple[dict[str, int], dict[str, str]]:
    """The fit that would replace a `legacy` entry's hand-set values (`fit.entry_fit` on `groups`), and per line why
    its points cannot: they do not fit it, as its fresh-process points do not span (`fit.capping_spans`), or none of
    them holds the upper bound (`MeasuredPoint.holds_upper_bound`). A probe window raises a line but never caps one,
    so probe windows alone never replace hand-set values."""
    values, problems = entry_fit(entry, groups)
    for suffix in problems:
        line = groups.get((entry_key(entry), suffix), [])
        if line and not any(point.caps for point in line):
            problems[suffix] = f"only {METHOD} points measure it, which bound it from below only"
    return values, dict(sorted(problems.items()))


def refit(data: Mapping[str, WorkingMemoryData], specs: Mapping[str, EstimatorSpec]) -> dict[str, WorkingMemoryData]:
    """`data` with every carried value that is not a stand-in set to its fit. A stand-in line whose in-fit
    fresh-process points now span the sizes a fit needs (`fit.capping_spans`) is fitted from then on, as the data test
    requires. Any other stays a stand-in, raised where the fit of its points lies above it: `probe_demand` points bound
    it from below only, so they never lower it. A `legacy` entry whose points fit every line it carries, each line
    with a point that holds the upper bound (`legacy_fit`), is fitted from then on too; one they do not keeps its
    hand-set values."""
    groups = fit_points(data.values(), specs)
    replaced: set[EntryKey] = set()

    def raised_stand_ins(entry: CoefficientEntry) -> dict[str, float]:
        values = entry.values or {}
        raised = {}
        for suffix in sorted({name[2:] for name in entry.stand_ins}):
            line = groups.get((entry_key(entry), suffix))
            if not line:
                continue
            try:
                k, b = shipped_pair([(point.size, point.bytes) for point in line])
            except ValueError:
                # Too few sizes for a line; `bound_violations` refuses a stand-in below one of them.
                continue
            raised[f"k_{suffix}"] = max(values[f"k_{suffix}"], k)
            raised[f"b_{suffix}"] = max(values[f"b_{suffix}"], b)
        return raised

    def staged(entry: CoefficientEntry) -> CoefficientEntry:
        stand_ins = [
            name for name in entry.stand_ins if not capping_spans(groups.get((entry_key(entry), name[2:]), []))
        ]
        entry = entry.model_copy(update={"stand_ins": stand_ins})
        if not entry.legacy:
            return entry
        values, problems = legacy_fit(entry, groups)
        if not values or problems:
            return entry
        replaced.add(entry_key(entry))
        return entry.model_copy(update={"legacy": False, "provenance": FITTED_LEGACY_PROVENANCE})

    staged_data = {
        name: file.model_copy(update={"coefficients": [staged(entry) for entry in file.coefficients]})
        for name, file in data.items()
    }
    try:
        fitted = fitted_values(staged_data.values(), specs)
    except ValueError as e:
        raise BuildError(f"the points cannot fit a line: {e}") from e

    def refitted(entry: CoefficientEntry) -> CoefficientEntry:
        if entry.values is None:
            return entry
        line = fitted.get(entry_key(entry), {})
        raised = raised_stand_ins(entry)
        values = {
            name: float(raised.get(name, value)) if name in entry.stand_ins else float(line.get(name, value))
            for name, value in entry.values.items()
        }
        if values == entry.values:
            return entry
        provenance = entry.provenance
        for note, changed in (
            (REFIT_NOTE, any(values[n] != entry.values[n] for n in values if n not in entry.stand_ins)),
            (RAISE_NOTE, any(values[n] != entry.values[n] for n in raised)),
        ):
            if changed and note not in provenance and not (note == REFIT_NOTE and entry_key(entry) in replaced):
                provenance = f"{provenance} {note}"
        return entry.model_copy(update={"values": values, "provenance": provenance})

    return {
        name: file.model_copy(update={"coefficients": [refitted(entry) for entry in file.coefficients]})
        for name, file in staged_data.items()
    }


def legacy_notes(
    data: Mapping[str, WorkingMemoryData], new_points: Sequence[MeasuredPoint], specs: Mapping[str, EstimatorSpec]
) -> list[str]:
    """Per `legacy` entry of an estimator the new points measured, why it keeps its hand-set values: no in-fit point
    measures its platform, or the lines its points cannot replace (`legacy_fit`)."""
    measured = {point.estimator for point in new_points}
    groups = fit_points(data.values(), specs)
    notes = []
    for entry in (entry for file in data.values() for entry in file.coefficients):
        if not entry.legacy or entry.estimator not in measured:
            continue
        if not any(key == entry_key(entry) for key, _ in groups):
            reasons = "no in-fit point measures this platform"
        else:
            _, problems = legacy_fit(entry, groups)
            reasons = "; ".join(f"{suffix}: {problem}" for suffix, problem in problems.items())
        notes.append(f"note: {entry.estimator} {entry.platform} stays legacy: {reasons}")
    return notes


def bound_violations(data: Mapping[str, WorkingMemoryData], specs: Mapping[str, EstimatorSpec]) -> list[str]:
    """Every point of `data` whose estimate under `data` breaks a bound it sets (`bounds.bound_violation`)."""
    violations = []
    for point in (point for file in data.values() for point in file.points):
        spec = specs[point.estimator]
        violation = bound_violation(point, spec, evaluate_point(point, spec, data.values()))
        if violation is not None:
            violations.append(f"{point_id(point, spec)}: {violation}")
    return violations


@dataclass(frozen=True)
class LineChange:
    """A carried line the refit moves, with the new points it is fitted on."""

    file: str
    entry: CoefficientEntry
    line: str
    old: tuple[float, float]
    new: tuple[float, float]
    points: list[MeasuredPoint]
    raised: bool
    """Whether the new line lies above the old one at any size its points measure."""

    def fragmented(self, spreads: Mapping[PointIdentity, Spread]) -> list[MeasuredPoint]:
        """The new points on a raised line whose windows are flagged as fragmentation (`Spread.fragmented`). A point
        of a point file ran in a fresh process and has no windows to flag."""
        if not self.raised:
            return []
        flagged = (spreads.get(point_identity(point)) for point in self.points)
        return [point for point, spread in zip(self.points, flagged, strict=True) if spread and spread.fragmented]


def review_lines(
    before: Mapping[str, WorkingMemoryData],
    after: Mapping[str, WorkingMemoryData],
    new_points: Sequence[MeasuredPoint],
    specs: Mapping[str, EstimatorSpec],
) -> list[LineChange]:
    """Every line the refit moves."""
    entries = [entry for file in after.values() for entry in file.coefficients]
    lines: dict[tuple[Any, str], list[MeasuredPoint]] = {}
    for point in new_points:
        single = WorkingMemoryData(schema_version=1, coefficients=entries, points=[point])
        for line in fit_points([single], specs):
            lines.setdefault(line, []).append(point)
    fitted_on = fit_points(after.values(), specs)
    changes = []
    for name, file in after.items():
        old_entries = {entry_key(entry): entry for entry in before[name].coefficients}
        for entry in file.coefficients:
            old_values, values = old_entries[entry_key(entry)].values or {}, entry.values or {}
            for suffix in sorted(key[2:] for key in values if key.startswith("k_")):
                old = (old_values[f"k_{suffix}"], old_values[f"b_{suffix}"])
                new = (values[f"k_{suffix}"], values[f"b_{suffix}"])
                if old == new:
                    continue
                sizes = {p.size for p in fitted_on.get((entry_key(entry), suffix), [])}
                raised = any(new[0] * x + new[1] > old[0] * x + old[1] for x in sizes)
                on_line = lines.get((entry_key(entry), suffix), [])
                changes.append(LineChange(name, entry, suffix, old, new, on_line, raised))
    return changes


def _plain(value: Any) -> Any:
    """Whole floats as integers, as the data files write them."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def write_data(path: Path, data: WorkingMemoryData) -> None:
    """Writes a data file as the shipped ones are written, keeping the file's line endings."""
    newline = "\r\n" if path.is_file() and b"\r\n" in path.read_bytes() else "\n"
    text = json.dumps(_plain(data.model_dump(mode="json", exclude_defaults=True)), indent=2, ensure_ascii=False)
    path.write_bytes((text + "\n").replace("\n", newline).encode("utf-8"))


_MIB = 2**20


def _mib(value: float | None) -> str:
    return "-" if value is None else f"{value / _MIB:.0f}"


def _spread_text(spread: Spread | None) -> str:
    if spread is None or len(spread.demand) < 2:
        return ""
    demand = ", ".join(map(_mib, spread.demand))
    return f" [{len(spread.demand)} windows, MiB {demand}; max/first {spread.first_to_max:.2f}]"


def describe_changes(
    before: Mapping[str, WorkingMemoryData],
    after: Mapping[str, WorkingMemoryData],
    new_points: Sequence[MeasuredPoint],
    spreads: Mapping[PointIdentity, Spread],
) -> list[str]:
    """One line per point added or remeasured, with the spread of its windows, carried value refitted, stand-in
    now fitted, and legacy entry now fitted. A remeasurement below the shipped point says that the shipped one stays."""
    new = {point_identity(point): point for point in new_points}
    lines = []
    for name, file in after.items():
        old_points = {point_identity(point): point for point in before[name].points if point_identity(point) in new}
        for point in (point for point in file.points if point_identity(point) in new):
            old = old_points.get(point_identity(point))
            remeasured = new[point_identity(point)]
            where = f"{name}: {point.estimator} {point.platform} {point.device} {_canonical(point.features)}"
            spread = _spread_text(spreads.get(point_identity(point)))
            if old is None:
                lines.append(f"{where} + {point.measured_bytes}{spread}")
            elif remeasured.measured_bytes < old.measured_bytes:
                lines.append(
                    f"{where} kept shipped {old.measured_bytes} over remeasured {remeasured.measured_bytes}{spread}"
                )
            elif old.measured_bytes != point.measured_bytes:
                lines.append(f"{where} {old.measured_bytes} -> {point.measured_bytes}{spread}")
        old_entries = {entry_key(entry): entry for entry in before[name].coefficients}
        for entry in file.coefficients:
            old_entry = old_entries[entry_key(entry)]
            for value_name, value in (entry.values or {}).items():
                previous = (old_entry.values or {}).get(value_name)
                if previous != value:
                    lines.append(
                        f"{name}: {entry.estimator} {entry.platform} {value_name} {_plain(previous)} -> {_plain(value)}"
                    )
            for value_name in sorted(set(old_entry.stand_ins) - set(entry.stand_ins)):
                lines.append(f"{name}: {entry.estimator} {entry.platform} {value_name} is fitted now, no stand-in")
            if old_entry.legacy and not entry.legacy:
                lines.append(f"{name}: {entry.estimator} {entry.platform} is fitted now, no longer legacy")
    return lines


def _export_order(path: Path) -> tuple[str, int]:
    """Exports of one run in the order of its server lifetimes, `-10-` after `-9-`."""
    match = EXPORT_NAME.search(path.name)
    assert match is not None, "only exports are ordered"
    return path.name[: match.start()], int(match["lifetime"] or 0)


def export_paths(arguments: Sequence[Path]) -> list[Path]:
    """The export files named, a directory standing for the exports a family E2E run saved in it (`<name>-<device
    class id>.json`, not the status files next to them)."""
    paths: list[Path] = []
    for argument in arguments:
        if argument.is_dir():
            found = sorted(
                (path for path in argument.glob("*.json") if EXPORT_NAME.search(path.name)), key=_export_order
            )
            if not found:
                raise BuildError(f"{argument} holds no calibration store exports")
            paths += found
        else:
            paths.append(argument)
    return paths


def _left_out(skipped: Counter[str]) -> str:
    return ", ".join(f"{count} {reason}" for reason, count in sorted(skipped.items())) or "none"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.working_memory.build_shipped", description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "exports", nargs="*", type=Path, help="calibration store exports, or directories of them (a run's calibration/)"
    )
    parser.add_argument(
        "--points",
        nargs="+",
        action="extend",
        type=Path,
        default=[],
        metavar="FILE",
        help="point files a calibration script wrote: fresh-process points, one JSON object per line",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help=f"the working-memory data files (default {DATA_DIR})"
    )
    parser.add_argument("--write", action="store_true", help="write the merged points and refitted values")
    parser.add_argument(
        "--accept-fragmentation",
        action="store_true",
        help="let --write raise a line on points flagged as fragmentation (working_memory/fragmentation.py)",
    )
    args = parser.parse_args(argv)
    if not args.exports and not args.points:
        parser.error("name calibration store exports, --points files, or both")

    try:
        specs = import_estimator_modules()
        try:
            before = dict(load_data(args.data_dir))
        except (FileNotFoundError, ValueError) as e:
            raise BuildError(str(e)) from e
        exports = export_paths(args.exports)
        entries = [entry for file in before.values() for entry in file.coefficients]
        build = build_points(exports, specs, entries)
        fresh = read_point_files(args.points, specs, entries)
        new_points = [*build.points, *fresh.points]
        after = refit(merge_points(before, new_points), specs)
        violations = bound_violations(after, specs)
        reviews = review_lines(before, after, new_points, specs)
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if exports:
        print(
            f"{len(build.points)} probe_demand points from {len(exports)} exports; windows left out: "
            f"{_left_out(build.skipped)}"
        )
    if args.points:
        print(
            f"{len(fresh.points)} fresh-process points from {len(args.points)} point files; records left out: "
            f"{_left_out(fresh.skipped)}"
        )
    changes = describe_changes(before, after, new_points, build.spreads)
    print("\n".join(changes) if changes else "no changes")
    for point in build.points:
        spread = build.spreads[point_identity(point)]
        if spread.fragmented:
            print(
                f"warning: fragmentation? {point.estimator} {point.platform} {point.device} {_canonical(point.features)}"
                f": reserved MiB {', '.join(map(_mib, spread.reserved))} while allocated MiB "
                f"{', '.join(map(_mib, spread.allocated))}; max/first {spread.first_to_max:.2f}"
            )
    for change in reviews:
        (old_k, old_b), (new_k, new_b) = change.old, change.new
        flagged = change.fragmented(build.spreads)
        print(
            f"line {change.file}: {change.entry.estimator} {change.entry.platform} {change.line}: "
            f"k {_plain(old_k)} -> {_plain(new_k)}, b {_mib(old_b)} -> {_mib(new_b)} MiB; "
            f"{'raised' if change.raised else 'lowered'}, {len(change.points)} new points on it"
            + (f", {len(flagged)} flagged as fragmentation" if flagged else "")
        )
    for note in legacy_notes(after, new_points, specs):
        print(note)
    if violations:
        print(
            "error: not written: the bounds of these points reject the result (working_memory/bounds.py):\n"
            + "\n".join(violations),
            file=sys.stderr,
        )
        return 1
    if args.write and not args.accept_fragmentation and any(change.fragmented(build.spreads) for change in reviews):
        print(
            "error: not written: a line is raised on points flagged as fragmentation; pass --accept-fragmentation "
            "to ship them",
            file=sys.stderr,
        )
        return 1
    if args.write and changes:
        for name, file in after.items():
            if file != before[name]:
                write_data(args.data_dir / name, file)
                print(f"wrote {args.data_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
