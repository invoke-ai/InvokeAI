"""User lines from a calibration store file, which models its windows count for, and what the windows of one model's
lines measured.

The fitting and resolution rules are the backend's (`backend/working_memory/user_fit.py`); this module picks the
windows a fit uses: those of one environment whose hard facts are the current ones, valid and fit-eligible.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Final, Optional

from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationEnvironment,
    CalibrationLineState,
    CalibrationPoint,
    CalibrationRun,
    CalibrationStoreFile,
    WorkingMemoryCalibrationLineContributor,
    WorkingMemoryCalibrationLineSize,
    WorkingMemoryCalibrationModelLine,
    WorkingMemoryCalibrationPixelRange,
)
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.working_memory import user_fit
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.fingerprint import Environment
from invokeai.backend.working_memory.fit import EntryKey, fits_a_line, line_of
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.spec import EstimatorSpec
from invokeai.backend.working_memory.user_fit import UserFit, UserLine


def fit_label(device_class_id: str, env_id: str) -> str:
    return f"user:{device_class_id}:{env_id}"


def _usable(point: CalibrationPoint) -> bool:
    """Whether a window can join a fit: valid, fit-eligible, a positive demand, and features a line admits."""
    return (
        point.valid
        and point.fit_eligible
        and point.demand_bytes is not None
        and point.demand_bytes > 0
        and fits_a_line(point.features)
    )


def _suffix(point: CalibrationPoint) -> str:
    """The ``<operation>_<dtype>`` of a usable window's line."""
    found = line_of(point.features)
    assert found is not None, "a usable window names its line"
    return found[0]


def user_fit_of(
    file: CalibrationStoreFile, env_id: str, platform: Platform, *, exclude_runs: AbstractSet[str] = frozenset()
) -> tuple[Optional[UserFit], Mapping[user_fit.LineKey, str]]:
    """The user fit of the file's windows taken under environment `env_id`, measured on `platform`, or None when they
    fit no line; and why the lines they measure but cannot fit cannot be. The windows of `exclude_runs` (runs still
    running) take no part: they join once their run finished."""
    points = [
        user_fit.UserPoint(point.estimator, point.version, point.features, point.demand_bytes or 0)
        for point in file.points
        if point.env_id == env_id and point.run_id not in exclude_runs and _usable(point)
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
    live_runs: AbstractSet[str] = frozenset(),
) -> tuple[list[WorkingMemoryCalibrationModelLine], int]:
    """Per environment and line, what the windows that feed it measured, with the shipped and the user estimate at
    each size; and how many of the model's own windows fit no line.

    The lines are those the model's own windows (`measured_models`) measured, and, in `current_env`, each line of
    `priced_on` (estimator, ``<operation>_<dtype>``: what the model's operations price on) that other models' windows
    measured. A line is fed by every model's usable windows of its environment that join the same user line, as
    estimates read it: those whose estimator resolves to the same carrying entry on the environment's platform.

    The sizes, window counts and contributors show every such window, those of `live_runs` (runs still running)
    included, so a running calibration shows what it measured so far. The user line, its range, the user estimates and
    the state are what estimates use: the fit without the windows of `live_runs`.

    The state compares the environment with `current_env` (None: no GPU of the file's class generates here): the same
    hard facts use the line, stale when the soft facts of the newest finished run that fed it differ from the current
    ones. A run stored without soft facts stands for the environment's, those first recorded with it. Older windows of
    the same hard environment keep feeding the line whatever their soft facts: soft facts do not change what a window
    measured enough to drop it.
    """
    envs = {env.id: env for env in file.envs}
    runs = {run.run_id: run for run in file.runs}
    models = run_models(file)
    excluded = 0
    own: dict[LineId, list[CalibrationPoint]] = {}
    usable_by_env: dict[str, list[CalibrationPoint]] = {}
    for point in file.points:
        usable = _usable(point)
        if usable:
            usable_by_env.setdefault(point.env_id, []).append(point)
        if model_key not in measured_models(point, models):
            continue
        if not usable:
            excluded += 1
            continue
        own.setdefault((point.env_id, point.estimator, point.version, _suffix(point)), []).append(point)

    env_fits: dict[str, _EnvFit] = {}

    def env_fit(env_id: str) -> _EnvFit:
        if env_id not in env_fits:
            env = envs.get(env_id)
            platform = platform_of(env) if env is not None else None
            fit, problems = (
                user_fit_of(file, env_id, platform, exclude_runs=live_runs) if platform is not None else (None, {})
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
        """The usable windows of the line's environment that join its user line; without a carrying entry, those of
        the same estimator version."""
        env_id, estimator, version, suffix = line_id
        platform = env_fit(env_id).platform
        carrier = carrier_of(estimator, version, platform)
        found = []
        for point in usable_by_env.get(env_id, []):
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
        points = feeding(line_id)
        own_points = own.get(line_id, [])
        if not points:
            # A line the model prices on that no window measured in this environment.
            continue
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
            if any(point.run_id in live_runs for point in points):
                problem = "A calibration run is still measuring this line; its windows count once it finished."
                if known:
                    problem += f" So far: {known}"
            else:
                problem = known or "The shipped data has no such line for this estimator on this platform."
        else:
            finished = [point for point in points if point.run_id not in live_runs]
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
                sizes=_sizes(points, estimator, specs if priced else {}, platform, fit if line else None),
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
    estimator: str,
    specs: Mapping[str, EstimatorSpec],
    platform: Optional[Platform],
    fit: Optional[UserFit],
) -> list[WorkingMemoryCalibrationLineSize]:
    """Per measured size, the windows and their largest demand, and the shipped and the user estimate there. The
    estimates price the largest window of the line's own estimator at the size, or, where only other estimators that
    share the line measured it, the largest window with its own estimator: their features are their own."""
    by_size: dict[tuple[int, int], list[CalibrationPoint]] = {}
    for point in points:
        by_size.setdefault((int(point.features["width"]), int(point.features["height"])), []).append(point)  # type: ignore[arg-type]
    sizes = []
    for (width, height), group in sorted(by_size.items(), key=lambda item: (item[0][0] * item[0][1], item[0])):
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
                shipped_bytes=shipped_bytes,
                user_bytes=user_bytes,
            )
        )
    return sizes
