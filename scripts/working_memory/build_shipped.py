"""Shipped working-memory points and fitted values from calibration store exports.

    python -m scripts.working_memory.build_shipped EXPORT [EXPORT ...] [--data-dir DIR] [--write]

An export is a device class's calibration store as `GET /api/v2/models/working_memory/export` returns it; a family E2E
calibration run saves one per server lifetime under `runs/<run_id>/calibration/`. Every valid, fit-eligible window
becomes a `probe_demand` point: per card, estimator, version, platform and features, the largest `demand_bytes` of
the windows, so repetitions and several exports of one card merge. The platform is what `platform_for` gives on the
machine the export came from (`platform.platform_for_environment`). A point joins the data file that declares its
estimator's coefficients, keeping the larger of itself and a shipped `probe_demand` point of the same card, platform
and features; the carried values are then refitted with `fit.py`. A stand-in stays one until fresh-process points at
two sizes far enough apart measure its line: `probe_demand` points can raise it, never lower it.

Without `--write` the changes are printed only. An export of another estimator version than the registered spec, or a
window whose features the spec's terms cannot evaluate, is refused. So is a result any point's bounds
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
from typing import Any

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
from invokeai.backend.working_memory.fit import capping_spans, entry_key, fit_points, fitted_values, shipped_pair
from invokeai.backend.working_memory.platform import Platform, is_measured, platform_for_environment
from invokeai.backend.working_memory.spec import EstimatorSpec, import_estimator_modules

METHOD = "probe_demand"
DATA_DIR = Path(working_memory_package.__file__).parent / "data"
EXPORT_NAME = re.compile(r"-(?:(?P<lifetime>\d+)-)?[0-9a-f]{16}\.json$")
"""A store export as a family E2E run names it, `<run_id>-<lifetime>-<device class id>.json`."""
REFIT_NOTE = "Refitted with probe_demand points by scripts/working_memory/build_shipped.py."
RAISE_NOTE = (
    "Stand-ins raised to cover probe_demand points by scripts/working_memory/build_shipped.py; still stand-ins until "
    "fresh-process points at two sizes at least twice apart in pixels measure their line."
)
_RUNTIME_FACTS = ("cuda", "hip", "cudnn", "miopen", "xpu")
FRAGMENTATION_GROWTH = 1.10
"""Reserved growth from a point's first window to its largest that, with flat allocated bytes, flags fragmentation."""
ALLOCATED_FLAT = 0.02
"""How far allocated bytes may spread across a point's windows and still count as flat."""


class BuildError(ValueError):
    pass


@dataclass(frozen=True)
class _Window:
    """A window that becomes, or competes to become, a point."""

    point: CalibrationPoint
    env: CalibrationEnvironment
    device: str
    export: Path


@dataclass(frozen=True)
class Spread:
    """The windows a point is the largest of, in the order they ran: export order, then queue item."""

    demand: tuple[int, ...]
    reserved: tuple[int | None, ...]
    """Each window's reserved delta."""
    allocated: tuple[int | None, ...]

    @property
    def first_to_max(self) -> float:
        return max(self.demand) / self.demand[0]

    @property
    def fragmented(self) -> bool:
        """Reserved grew from the first window to the largest while allocated stayed flat: the allocator's cache grew
        around the same tensors, as a long-running server fragments it, rather than the operation needing more."""
        largest = self.demand.index(max(self.demand))
        first, peak = self.reserved[0], self.reserved[largest]
        if largest == 0 or first is None or peak is None or None in self.allocated:
            return False
        allocated = [value for value in self.allocated if value is not None]
        return peak >= FRAGMENTATION_GROWTH * first and max(allocated) <= (1 + ALLOCATED_FLAT) * min(allocated)


@dataclass
class Build:
    points: list[MeasuredPoint]
    skipped: Counter[str] = field(default_factory=Counter)
    """Windows left out, by reason."""
    spreads: dict[tuple[str, int, str, str, str, str], Spread] = field(default_factory=dict)
    """By `point_identity`."""


def _canonical(features: Mapping[str, Any]) -> str:
    return json.dumps(features, sort_keys=True, separators=(",", ":"))


def point_identity(point: MeasuredPoint) -> tuple[str, int, str, str, str, str]:
    """What a newer point of the same measurement replaces."""
    return (
        point.estimator,
        point.version,
        _canonical(point.platform),
        point.device,
        point.method,
        _canonical(point.features),
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


def _check_window(path: Path, window: CalibrationPoint, spec: EstimatorSpec) -> None:
    where = (
        f"{path}: {window.estimator} window of node {window.node_type} {window.node_id} (item {window.queue_item_id})"
    )
    try:
        spec.terms(window.features, dict.fromkeys(spec.coefficients, 1.0))
    except Exception as e:
        raise BuildError(f"{where}: the estimator cannot evaluate its features {window.features}: {e!r}") from e


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
        for window in store.points:
            # The same store saved twice (an export and a copy of its file, one file named twice) counts once.
            identity = window.model_dump_json()
            if identity in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(identity)
            spec = specs.get(window.estimator)
            if spec is None:
                raise BuildError(f"{path}: no registered estimator is called {window.estimator!r}")
            if window.version != spec.version:
                raise BuildError(
                    f"{path}: {window.estimator} windows of version {window.version}, the registered estimator is "
                    f"version {spec.version}; measure again with this version"
                )
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
            _check_window(path, window, spec)
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
            group = (
                store.device_class.name,
                window.estimator,
                window.version,
                _canonical(keys),
                _canonical(window.features),
            )
            platforms[group] = keys
            groups.setdefault(group, []).append(_Window(window, env, store.device_class.name, path))

    points = []
    spreads = {}
    order = {path: index for index, path in enumerate(exports)}
    for group, windows in sorted(groups.items()):
        windows = sorted(windows, key=lambda w: (order[w.export], w.point.queue_item_id))
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
        )
    return Build(points=points, skipped=skipped, spreads=spreads)


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
    new: dict[str, dict[tuple[str, int, str, str, str, str], MeasuredPoint]] = {name: {} for name in data}
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


def refit(data: Mapping[str, WorkingMemoryData]) -> dict[str, WorkingMemoryData]:
    """`data` with every carried value that is not a stand-in set to its fit. A stand-in line whose in-fit
    fresh-process points now span the sizes a fit needs (`fit.capping_spans`) is fitted from then on, as the data test
    requires. Any other stays a stand-in, raised where the fit of its points lies above it: `probe_demand` points bound
    it from below only, so they never lower it."""
    groups = fit_points(data.values())

    def raised_stand_ins(entry: CoefficientEntry) -> dict[str, float]:
        values = entry.values or {}
        raised = {}
        for suffix in sorted({name[2:] for name in entry.stand_ins}):
            line = groups.get((entry_key(entry), suffix))
            if not line:
                continue
            try:
                k, b = shipped_pair([(point.pixels, point.bytes) for point in line])
            except ValueError:
                # Too few sizes for a line; `bound_violations` refuses a stand-in below one of them.
                continue
            raised[f"k_{suffix}"] = max(values[f"k_{suffix}"], k)
            raised[f"b_{suffix}"] = max(values[f"b_{suffix}"], b)
        return raised

    def measured_lines_fitted(entry: CoefficientEntry) -> CoefficientEntry:
        stand_ins = [
            name for name in entry.stand_ins if not capping_spans(groups.get((entry_key(entry), name[2:]), []))
        ]
        return entry.model_copy(update={"stand_ins": stand_ins})

    staged = {
        name: file.model_copy(update={"coefficients": [measured_lines_fitted(entry) for entry in file.coefficients]})
        for name, file in data.items()
    }
    try:
        fitted = fitted_values(staged.values())
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
            if changed and note not in provenance:
                provenance = f"{provenance} {note}"
        return entry.model_copy(update={"values": values, "provenance": provenance})

    return {
        name: file.model_copy(update={"coefficients": [refitted(entry) for entry in file.coefficients]})
        for name, file in staged.items()
    }


def bound_violations(data: Mapping[str, WorkingMemoryData], specs: Mapping[str, EstimatorSpec]) -> list[str]:
    """Every point of `data` whose estimate under `data` breaks a bound it sets (`bounds.bound_violation`)."""
    violations = []
    for point in (point for file in data.values() for point in file.points):
        violation = bound_violation(point, evaluate_point(point, specs[point.estimator], data.values()))
        if violation is not None:
            violations.append(f"{point_id(point)}: {violation}")
    return violations


@dataclass(frozen=True)
class LineChange:
    """A carried convolution line the refit moves, with the new points it is fitted on."""

    file: str
    entry: CoefficientEntry
    line: str
    old: tuple[float, float]
    new: tuple[float, float]
    points: list[MeasuredPoint]
    raised: bool
    """Whether the new line lies above the old one at any size its points measure."""

    def fragmented(self, spreads: Mapping[tuple[str, int, str, str, str, str], Spread]) -> list[MeasuredPoint]:
        """The new points on a raised line whose windows are flagged as fragmentation (`Spread.fragmented`)."""
        if not self.raised:
            return []
        return [point for point in self.points if spreads[point_identity(point)].fragmented]


def review_lines(
    before: Mapping[str, WorkingMemoryData], after: Mapping[str, WorkingMemoryData], new_points: Sequence[MeasuredPoint]
) -> list[LineChange]:
    """Every line the refit moves."""
    entries = [entry for file in after.values() for entry in file.coefficients]
    lines: dict[tuple[Any, str], list[MeasuredPoint]] = {}
    for point in new_points:
        single = WorkingMemoryData(schema_version=1, coefficients=entries, points=[point])
        for line in fit_points([single]):
            lines.setdefault(line, []).append(point)
    fitted_on = fit_points(after.values())
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
                sizes = {p.pixels for p in fitted_on.get((entry_key(entry), suffix), [])}
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
    spreads: Mapping[tuple[str, int, str, str, str, str], Spread],
) -> list[str]:
    """One line per point added or remeasured, with the spread of its windows, carried value refitted, and stand-in
    now fitted."""
    new = {point_identity(point) for point in new_points}
    lines = []
    for name, file in after.items():
        old_points = {point_identity(point): point for point in before[name].points if point_identity(point) in new}
        for point in (point for point in file.points if point_identity(point) in new):
            old = old_points.get(point_identity(point))
            where = f"{name}: {point.estimator} {point.platform} {point.device} {_canonical(point.features)}"
            spread = _spread_text(spreads.get(point_identity(point)))
            if old is None:
                lines.append(f"{where} + {point.measured_bytes}{spread}")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.working_memory.build_shipped", description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "exports", nargs="+", type=Path, help="calibration store exports, or directories of them (a run's calibration/)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help=f"the working-memory data files (default {DATA_DIR})"
    )
    parser.add_argument("--write", action="store_true", help="write the merged points and refitted values")
    parser.add_argument(
        "--accept-fragmentation",
        action="store_true",
        help="let --write raise a line on points whose reserved bytes grew across repetitions at flat allocated bytes",
    )
    args = parser.parse_args(argv)

    try:
        specs = import_estimator_modules()
        try:
            before = dict(load_data(args.data_dir))
        except (FileNotFoundError, ValueError) as e:
            raise BuildError(str(e)) from e
        exports = export_paths(args.exports)
        entries = [entry for file in before.values() for entry in file.coefficients]
        build = build_points(exports, specs, entries)
        after = refit(merge_points(before, build.points))
        violations = bound_violations(after, specs)
        reviews = review_lines(before, after, build.points)
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    skipped = ", ".join(f"{count} {reason}" for reason, count in sorted(build.skipped.items())) or "none"
    print(f"{len(build.points)} probe_demand points from {len(exports)} exports; windows left out: {skipped}")
    changes = describe_changes(before, after, build.points, build.spreads)
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
