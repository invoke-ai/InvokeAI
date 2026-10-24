"""User lines from a calibration store file, which models its windows count for, and what the windows of one model's
lines measured.

The fitting and resolution rules are the backend's (`backend/working_memory/user_fit.py`); this module picks the
windows a fit uses: those of one environment whose hard facts are the current ones, valid, fit-eligible and not
fragmented (`exclusions`).
"""

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Optional

from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationEnvironment,
    CalibrationHoldBack,
    CalibrationLineState,
    CalibrationPoint,
    CalibrationRun,
    CalibrationSizeExclusion,
    CalibrationStoreFile,
    WorkingMemoryCalibrationLineContributor,
    WorkingMemoryCalibrationLineSize,
    WorkingMemoryCalibrationModelLine,
    WorkingMemoryCalibrationPixelRange,
)
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.working_memory import fragmentation, user_fit
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.fingerprint import Environment
from invokeai.backend.working_memory.fit import EntryKey, fits_a_line, line_of
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.spec import EstimatorSpec
from invokeai.backend.working_memory.user_fit import UserFit, UserLine


def fit_label(device_class_id: str, env_id: str) -> str:
    return f"user:{device_class_id}:{env_id}"


def _measures_a_line(point: CalibrationPoint) -> bool:
    """Whether a window measured a line: valid, a positive demand, and features a line admits."""
    return point.valid and point.demand_bytes is not None and point.demand_bytes > 0 and fits_a_line(point.features)


def _suffix(point: CalibrationPoint) -> str:
    """The ``<operation>_<dtype>`` of the line a window measured."""
    found = line_of(point.features)
    assert found is not None, "a window that measured a line names it"
    return found[0]


def _attempts(file: CalibrationStoreFile) -> dict[str, str]:
    """Per run id, the attempt its windows are judged with: its `attempt_id`, or the run itself without one."""
    return {run.run_id: run.attempt_id or run.run_id for run in file.runs}


HeldBack = Mapping[str, CalibrationHoldBack]
"""Why each run whose windows no fit takes is held back (`CalibrationHoldBack`), by run id; the service derives it
from the queue and the attempts' runs."""

_NONE_HELD_BACK: Final[HeldBack] = MappingProxyType({})


def exclusions(
    file: CalibrationStoreFile, held_back: HeldBack = _NONE_HELD_BACK
) -> dict[int, CalibrationSizeExclusion]:
    """Why each window that measured a line does not count toward its fit, by position in `file.points`; a window
    that measured a line and is absent counts.

    - aborted_attempt: a window of a run `held_back` names so: its attempt was stopped, cleared or cut short.
    - partial_residency: valid but not fit-eligible, its model not fully on the device.
    - fragmented: the fit-eligible windows of one measurement -- one attempt (`attempt_id`, else the run) at one
      environment, estimator, version and features, in the order they ran -- are what `fragmentation.fragmented`
      flags: they grew around a flat allocation, or their largest window is a ratio outlier against the smaller
      sizes of the same line in the environment that count, from any attempt. The measurement drops out whole: its
      windows ran back to back on one allocator state. Another attempt's windows of the same size ran on another and
      are judged on their own, so a recalibration can bring a size back and a later fragmented attempt drops no clean
      one. Windows of runs still running (`held_back`) are judged, but set no reference for larger sizes, as they
      join no user line yet.
    """
    points = file.points
    attempt_of = _attempts(file)
    excluded: dict[int, CalibrationSizeExclusion] = {}
    groups: dict[tuple[str, str, int, str, str], list[int]] = {}
    for index, point in enumerate(points):
        if not _measures_a_line(point):
            continue
        if held_back.get(point.run_id) == "aborted_attempt":
            excluded[index] = "aborted_attempt"
            continue
        if not point.fit_eligible:
            excluded[index] = "partial_residency"
            continue
        features = json.dumps(point.features, sort_keys=True, separators=(",", ":"))
        attempt = attempt_of.get(point.run_id, point.run_id)
        groups.setdefault((point.env_id, point.estimator, point.version, features, attempt), []).append(index)

    members = list(groups.values())
    measurements = []
    for (env_id, estimator, version, _, _), indices in groups.items():
        windows = [points[index] for index in indices]
        measurements.append(
            fragmentation.Measurement(
                line=(env_id, estimator, version, _suffix(windows[0])),
                pixels=_pixels(windows[0]),
                demand=[window.demand_bytes or 0 for window in windows],
                reserved=[window.reserved.delta if window.reserved else None for window in windows],
                allocated=[window.allocated.delta if window.allocated else None for window in windows],
                reference=not any(window.run_id in held_back for window in windows),
            )
        )
    for indices, flagged in zip(members, fragmentation.fragmented(measurements), strict=True):
        if flagged:
            excluded.update(dict.fromkeys(indices, "fragmented"))
    return excluded


def _pixels(point: CalibrationPoint) -> int:
    found = line_of(point.features)
    assert found is not None, "a window that measured a line names it"
    return found[1]


def counted_windows(file: CalibrationStoreFile, held_back: HeldBack = _NONE_HELD_BACK) -> list[CalibrationPoint]:
    """The windows that count toward a fit: they measured a line, and `exclusions` names no reason against them.
    Those of runs still running count, though no fit takes them yet."""
    excluded = exclusions(file, held_back)
    return [point for index, point in enumerate(file.points) if _measures_a_line(point) and index not in excluded]


def recorded_outside_torch_bytes(file: CalibrationStoreFile, env_id: str) -> Optional[int]:
    """The most the measuring process held outside torch's allocator when a valid window of environment `env_id`
    opened: the driver's figure for the process less torch's reserve, at the window's baseline. None without such a
    window, or where the driver had no figure for the process (`torch-only`, and XPU's device-wide reading)."""
    found = [
        point.driver.baseline - point.reserved.baseline
        for point in file.points
        if point.env_id == env_id and point.valid and point.driver is not None and point.reserved is not None
    ]
    return max(found, default=None)


def user_fit_of(
    file: CalibrationStoreFile, env_id: str, platform: Platform, *, held_back: HeldBack = _NONE_HELD_BACK
) -> tuple[Optional[UserFit], Mapping[user_fit.LineKey, str]]:
    """The user fit of the file's counted windows (`counted_windows`) taken under environment `env_id`, measured on
    `platform`, or None when they fit no line; and why the lines they measure but cannot fit cannot be. The windows of
    the runs `held_back` names take no part: those of runs still running join once their attempt finished, those of
    an attempt that ended early never."""
    points = [
        user_fit.UserPoint(point.estimator, point.version, point.features, point.demand_bytes or 0)
        for point in counted_windows(file, held_back)
        if point.env_id == env_id and point.run_id not in held_back
    ]
    lines, problems = user_fit.fit_user_lines(points, platform)
    if not lines:
        return None, problems
    return UserFit(label=fit_label(file.device_class.id, env_id), platform=platform, lines=lines), problems


def _changed(old: Mapping[str, Optional[str]], new: Mapping[str, Optional[str]]) -> list[str]:
    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


_SUBMODEL_TYPES: Final = frozenset(submodel.value for submodel in SubModelType)


def measured_models(point: CalibrationPoint, run_models: Mapping[str, str]) -> frozenset[str]:
    """The models a window is attributed to: the model its run calibrated (`run_models`, by run id), and the model it
    measured, where a submodel's window (`<key>:vae`) counts for its model. The summary shows a model's windows by
    this, and resetting the model removes exactly these."""
    head, separator, tail = point.model_key.rpartition(":")
    measured = head if separator and tail in _SUBMODEL_TYPES else point.model_key
    run_model = run_models.get(point.run_id)
    return frozenset({measured} if run_model is None else {measured, run_model})


def run_models(file: CalibrationStoreFile) -> dict[str, str]:
    return {run.run_id: run.model_key for run in file.runs}


@dataclass(frozen=True)
class _EnvFit:
    env: Optional[CalibrationEnvironment]
    platform: Optional[Platform]
    fit: Optional[UserFit]
    problems: Mapping[user_fit.LineKey, str]


LineId = tuple[str, str, int, str]
"""A summary line: environment id, estimator, version and ``<operation>_<dtype>``."""


def model_lines(
    file: CalibrationStoreFile,
    model_key: str,
    current_env: Optional[Environment],
    platform_of: Callable[[CalibrationEnvironment], Optional[Platform]],
    specs: Mapping[str, EstimatorSpec],
    priced_on: Iterable[tuple[str, str]],
    name_of: Callable[[str], Optional[str]],
    held_back: HeldBack = _NONE_HELD_BACK,
) -> tuple[list[WorkingMemoryCalibrationModelLine], int]:
    """Per environment and line, what the windows that feed it measured, with the shipped and the user estimate at
    each size; and how many of the model's own windows fit no line.

    The lines are those the model's own windows (`measured_models`) measured, and, in `current_env`, each line of
    `priced_on` (estimator, ``<operation>_<dtype>``: what the model's operations price on) that other models' windows
    measured. A line is fed by every model's counted windows (`exclusions`) of its environment that join the same user
    line, as estimates read it: those whose estimator resolves to the same carrying entry on the environment's
    platform. Its sizes also show, as not counted, the sizes only excluded windows measured.

    The sizes, window counts and contributors show every such window, those of runs still running (`held_back`)
    included, so a running calibration shows what it measured so far; the windows of an aborted attempt show as not
    counted. The user line, its range, the user estimates and the state are what estimates use: the fit without
    the windows of either.

    The state compares the environment with `current_env` (None: no GPU of the file's class generates here): the same
    hard facts use the line, stale when the soft facts of the newest finished run that fed it differ from the current
    ones. A run stored without soft facts stands for the environment's, those first recorded with it. Older windows of
    the same hard environment keep feeding the line whatever their soft facts: soft facts do not change what a window
    measured enough to drop it.
    """
    envs = {env.id: env for env in file.envs}
    runs = {run.run_id: run for run in file.runs}
    models = run_models(file)
    reasons = exclusions(file, held_back)
    # By identity: the windows below are the file's own objects.
    reason_of: dict[int, CalibrationSizeExclusion] = {
        id(file.points[index]): reason for index, reason in reasons.items()
    }
    excluded = 0
    own: dict[LineId, list[CalibrationPoint]] = {}
    measured_by_env: dict[str, list[CalibrationPoint]] = {}
    for point in file.points:
        measured = _measures_a_line(point)
        if measured:
            measured_by_env.setdefault(point.env_id, []).append(point)
        if model_key not in measured_models(point, models):
            continue
        if not measured or id(point) in reason_of:
            excluded += 1
        if measured:
            own.setdefault((point.env_id, point.estimator, point.version, _suffix(point)), []).append(point)

    def counts(point: CalibrationPoint) -> bool:
        return id(point) not in reason_of

    env_fits: dict[str, _EnvFit] = {}

    def env_fit(env_id: str) -> _EnvFit:
        if env_id not in env_fits:
            env = envs.get(env_id)
            platform = platform_of(env) if env is not None else None
            fit, problems = (
                user_fit_of(file, env_id, platform, held_back=held_back) if platform is not None else (None, {})
            )
            env_fits[env_id] = _EnvFit(env, platform, fit, problems)
        return env_fits[env_id]

    def carrier_of(estimator: str, version: int, platform: Optional[Platform]) -> Optional[EntryKey]:
        """The entry whose user line a window of this estimator version joins; None where it is not priced here."""
        spec = specs.get(estimator)
        if spec is None or spec.version != version or platform is None:
            return None
        return user_fit.carrier_key(estimator, version, platform)

    def feeding(line_id: LineId) -> list[CalibrationPoint]:
        """The windows of the line's environment that measured its user line, counted or not; without a carrying
        entry, those of the same estimator version."""
        env_id, estimator, version, suffix = line_id
        platform = env_fit(env_id).platform
        carrier = carrier_of(estimator, version, platform)
        found = []
        for point in measured_by_env.get(env_id, []):
            if _suffix(point) != suffix:
                continue
            if carrier is None:
                same = (point.estimator, point.version) == (estimator, version)
            else:
                same = carrier_of(point.estimator, point.version, platform) == carrier
            if same:
                found.append(point)
        return found

    line_ids = set(own)
    if current_env is not None:
        for estimator, suffix in priced_on:
            spec = specs.get(estimator)
            if spec is not None:
                line_ids.add((current_env.id, estimator, spec.version, suffix))

    names: dict[str, Optional[str]] = {}
    lines: list[WorkingMemoryCalibrationModelLine] = []
    for line_id in sorted(line_ids):
        env_id, estimator, version, suffix = line_id
        measured = feeding(line_id)
        if not measured:
            # A line the model prices on that no window measured in this environment.
            continue
        points = [point for point in measured if counts(point)]
        own_points = [point for point in own.get(line_id, []) if counts(point)]
        found_fit = env_fit(env_id)
        env, platform, fit = found_fit.env, found_fit.platform, found_fit.fit
        spec = specs.get(estimator)
        priced = spec is not None and spec.version == version
        carrier = carrier_of(estimator, version, platform)
        line: Optional[UserLine] = None
        if fit is not None and carrier is not None:
            line = fit.lines.get((carrier, suffix))

        changed: list[str] = []
        problem: Optional[str] = None
        state: CalibrationLineState
        if env is None or current_env is None:
            state, problem = "unusable", "No generation device of this class is present."
        elif env.id != current_env.id:
            state, changed = "unusable", _changed(env.hard, current_env.hard)
        elif not priced:
            state = "unusable"
            problem = f"Measured with {estimator} version {version}, which this version does not price with."
        elif line is None:
            state = "insufficient"
            known = found_fit.problems.get((carrier, suffix)) if carrier is not None else None
            if not points and all(reason_of.get(id(point)) == "aborted_attempt" for point in measured):
                problem = (
                    "A stopped or interrupted calibration measured this line, and its measurements were discarded. "
                    "Calibrate again to measure it."
                )
            elif not points:
                problem = "None of this line's windows count: see each size's excluded_reason."
            elif any(held_back.get(point.run_id) == "running" for point in points):
                problem = "A calibration run is still measuring this line; its windows count once it finished."
                if known:
                    problem += f" So far: {known}"
            else:
                problem = known or "The shipped data has no such line for this estimator on this platform."
        else:
            finished = [point for point in points if point.run_id not in held_back]
            changed = _changed(_newest_soft(finished, runs, env), current_env.soft)
            state = "stale" if changed else "in_use"

        windows: Counter[str] = Counter()
        for point in points:
            windows.update(measured_models(point, models))
        for key in windows:
            if key not in names:
                names[key] = name_of(key)
        measured_by = [
            WorkingMemoryCalibrationLineContributor(model_key=key, name=names[key], windows=count)
            for key, count in windows.items()
        ]
        measured_by.sort(
            key=lambda contributor: (contributor.name is None, contributor.name or "", contributor.model_key)
        )

        lines.append(
            WorkingMemoryCalibrationModelLine(
                estimator=estimator,
                version=version,
                line=suffix,
                carrier=carrier[0] if carrier else None,
                env_id=env_id,
                state=state,
                changed_facts=changed,
                problem=problem,
                measured_range=(
                    WorkingMemoryCalibrationPixelRange(min_pixels=line.x_min, max_pixels=line.x_max) if line else None
                ),
                slope_bytes_per_pixel=line.k if line else None,
                base_bytes=line.b if line else None,
                sizes=_sizes(measured, reason_of, estimator, specs if priced else {}, platform, fit if line else None),
                own_windows=len(own_points),
                measured_by=measured_by,
            )
        )
    return lines, excluded


def _newest_soft(
    points: Iterable[CalibrationPoint], runs: Mapping[str, CalibrationRun], env: CalibrationEnvironment
) -> Mapping[str, Optional[str]]:
    """The soft facts of the newest run among those that measured `points`; the environment's when it recorded none.
    `runs` is in the file's order, which is the order they were created in: it decides a tie of the clock."""
    order = {run_id: index for index, run_id in enumerate(runs)}
    fed = {point.run_id for point in points if point.run_id in runs}
    newest = max((runs[run_id] for run_id in fed), key=lambda run: (run.created_at, order[run.run_id]), default=None)
    return newest.soft if newest is not None and newest.soft is not None else env.soft


def _sizes(
    points: Iterable[CalibrationPoint],
    reason_of: Mapping[int, CalibrationSizeExclusion],
    estimator: str,
    specs: Mapping[str, EstimatorSpec],
    platform: Optional[Platform],
    fit: Optional[UserFit],
) -> list[WorkingMemoryCalibrationLineSize]:
    """Per measured size, the windows and their largest demand, and the shipped and the user estimate there.

    A size with counted windows shows those; a size whose windows all are excluded (`reason_of`, by window identity)
    shows them as not counted: aborted_attempt where that is the only reason, else fragmented where any of them is.
    The estimates price the largest window of the line's
    own estimator among those shown, or, where only other estimators that share the line measured the size, the largest
    window with its own estimator: their features are their own."""
    by_size: dict[tuple[int, int], list[CalibrationPoint]] = {}
    for point in points:
        by_size.setdefault((int(point.features["width"]), int(point.features["height"])), []).append(point)  # type: ignore[arg-type]
    sizes = []
    for (width, height), measured in sorted(by_size.items(), key=lambda item: (item[0][0] * item[0][1], item[0])):
        group = [point for point in measured if id(point) not in reason_of]
        counted = bool(group)
        excluded_reason: Optional[CalibrationSizeExclusion] = None
        if not counted:
            group = measured
            found = {reason_of[id(point)] for point in measured}
            if found == {"aborted_attempt"}:
                excluded_reason = "aborted_attempt"
            else:
                excluded_reason = "fragmented" if "fragmented" in found else "partial_residency"
        largest = max(group, key=lambda point: point.demand_bytes or 0)
        mine = [point for point in group if point.estimator == estimator]
        priced = max(mine, key=lambda point: point.demand_bytes or 0) if mine else largest
        spec = specs.get(priced.estimator) if estimator in specs else None
        shipped_bytes = user_bytes = None
        if spec is not None and spec.version == priced.version and platform is not None:
            shipped = shipped_coefficients(spec, platform)
            shipped_bytes = spec.evaluate(priced.features, shipped).reservation_bytes
            if fit is not None:
                user_bytes = user_fit.resolve(spec, priced.features, shipped, fit).reservation_bytes
        sizes.append(
            WorkingMemoryCalibrationLineSize(
                width=width,
                height=height,
                pixels=width * height,
                windows=len(group),
                measured_bytes=largest.demand_bytes or 0,
                counted=counted,
                excluded_reason=excluded_reason,
                shipped_bytes=shipped_bytes,
                user_bytes=user_bytes,
            )
        )
    return sizes
