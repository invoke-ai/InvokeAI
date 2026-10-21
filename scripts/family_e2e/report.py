"""Static HTML report and Markdown summary of one run against the accepted references and the previous run.

The report only reads records and references. `write_report` writes report.html, summary.md and diff/ inside the run
directory; `render_report` renders them for another place, such as a bundle.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import urllib.parse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from scripts.family_e2e.manifest import DEFAULT_ARM
from scripts.family_e2e.metrics import (
    DIFF_GAIN,
    MB,
    Delta,
    compare_peak,
    compare_time,
    load_rgb,
    measured_values,
    median,
    psnr,
    write_diff_image,
)
from scripts.family_e2e.records import (
    Environment,
    ModelIdentity,
    Record,
    Reference,
    RunInfo,
    Thresholds,
    load_records,
    load_run_info,
    run_id_profile,
    write_bytes_atomic,
)
from scripts.family_e2e.references import load_reference

REPORT_FILE = "report.html"
SUMMARY_FILE = "summary.md"
DIFF_DIR = "diff"
# invokeai.yaml settings that change speed, memory peaks or (sage) the image itself; SageAttention is
# attention_backend=sage, and rocm_aotriton_experimental picks the attention kernels on AMD GPUs.
RUNTIME_CONFIG_KEYS = (
    "pytorch_cuda_alloc_conf",
    "enable_partial_loading",
    "device_working_mem_gb",
    "attention_type",
    "attention_slice_size",
    "attention_backend",
    "rocm_aotriton_experimental",
)

# CHANGED: a case expected to change differs from its reference. BASELINE: its reference comes from this run, so
# there is nothing to compare against. Neither is a failure.
Status = Literal["PASS", "CHANGED", "BASELINE", "FAIL", "ERROR", "INVALID", "NO-REFERENCE"]
STATUSES: tuple[Status, ...] = ("PASS", "CHANGED", "BASELINE", "FAIL", "ERROR", "INVALID", "NO-REFERENCE")
Unit = Literal["s", "bytes"]


@dataclass(frozen=True)
class RenderedReport:
    html: str
    summary: str
    failures: int
    """Cases with status FAIL or ERROR."""
    outliers: int
    """Cases with at least one outlier against the reference or the previous run."""
    invalid: int
    """Cases with status INVALID or with measured runs invalid for time, VRAM or PSNR (e.g. partial residency)."""
    changed: int
    """Cases with status CHANGED."""
    baseline: int
    """Cases with status BASELINE."""


@dataclass(frozen=True)
class ReportResult:
    """Where `write_report` wrote the report, with the counts of its RenderedReport."""

    report_html: Path
    summary_md: Path
    failures: int
    outliers: int
    invalid: int
    changed: int
    baseline: int


def write_report(data_dir: Path, run_dir: Path) -> ReportResult:
    report = render_report(
        data_dir,
        run_dir,
        load_run_info(run_dir),
        load_records(run_dir),
        diff_dir=run_dir / DIFF_DIR,
        href=lambda path: _href(path, run_dir),
    )
    report_path = run_dir / REPORT_FILE
    summary_path = run_dir / SUMMARY_FILE
    write_bytes_atomic(report_path, report.html.encode("utf-8"))
    write_bytes_atomic(summary_path, report.summary.encode("utf-8"))
    return ReportResult(
        report_path, summary_path, report.failures, report.outliers, report.invalid, report.changed, report.baseline
    )


class Scrubber(Protocol):
    """What a rendered report must not show, for a report that leaves this machine."""

    def text(self, value: str) -> str:
        """Cleans the text of an error the report describes itself, e.g. an unreadable earlier run or image."""
        ...

    def environment(self, environment: Environment) -> Environment:
        """Cleans an environment the report compares; it treats the run's own and the earlier ones alike."""
        ...


class _Unchanged:
    def text(self, value: str) -> str:
        return value

    def environment(self, environment: Environment) -> Environment:
        return environment


_UNCHANGED = _Unchanged()


def render_report(
    data_dir: Path,
    run_dir: Path,
    info: RunInfo,
    records: list[Record],
    *,
    diff_dir: Path,
    href: Callable[[Path], str],
    scrubber: Scrubber = _UNCHANGED,
    status: str | None = None,
) -> RenderedReport:
    """Evaluates the run's cases and renders the report; writes nothing but the difference images into `diff_dir`.

    `href` links each image file the report shows. `status` names the run's state when its recorded status is out
    of date, e.g. for a run whose runner died.
    """
    context = _Context(
        data_dir, run_dir, info, _History(data_dir, info), _ImageCache(scrubber.text), diff_dir, scrubber
    )
    cases = [
        _evaluate_safely(context, arm, case_key, case_records)
        for (arm, case_key), case_records in _group(info, records).items()
    ]
    notes = [scrubber.text(note) for note in context.history.notes]
    run_status = status or info.status
    return RenderedReport(
        html=_render_html(context, cases, notes, href, run_status),
        summary=_render_summary(info, cases, notes, run_status),
        failures=sum(case.status in ("FAIL", "ERROR") for case in cases),
        outliers=sum(case.has_outliers for case in cases),
        invalid=sum(case.status == "INVALID" or bool(case.invalid_metrics) for case in cases),
        changed=sum(case.status == "CHANGED" for case in cases),
        baseline=sum(case.status == "BASELINE" for case in cases),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Inputs


def _group(info: RunInfo, records: list[Record]) -> dict[tuple[str, str], list[Record]]:
    """Records by (arm, case_key) in plan order, default arm first; planned cases without records stay visible."""
    keys = [(step.arm, step.case_key) for step in info.plan] + [(r.arm, r.case_key) for r in records]
    ordered = sorted(dict.fromkeys(keys), key=lambda key: key[0] != DEFAULT_ARM)
    groups: dict[tuple[str, str], list[Record]] = {key: [] for key in ordered}
    for record in records:
        groups[(record.arm, record.case_key)].append(record)
    return groups


@dataclass(frozen=True)
class _Run:
    run_dir: Path
    info: RunInfo
    records: list[Record]


@dataclass(frozen=True)
class _Previous:
    run: _Run
    records: list[Record]
    """Completed measured records of the case."""
    image_path: Path | None


class _History:
    """Earlier runs of the same profile, newest first, read only when a case needs them.

    The profile comes from the directory name, so runs of other profiles are never read: one that another version
    of the tool wrote would not load, and is no previous run anyway.
    """

    def __init__(self, data_dir: Path, current: RunInfo) -> None:
        self._profile = current.profile
        runs_dir = data_dir / "runs"
        names = sorted((d.name for d in runs_dir.iterdir() if d.is_dir()), reverse=True) if runs_dir.is_dir() else []
        self._candidates = [
            runs_dir / name for name in names if name < current.run_id and run_id_profile(name) == current.profile
        ]
        self._loaded: dict[Path, _Run | None] = {}
        self.notes: list[str] = []

    def previous(self, case_key: str) -> _Previous | None:
        """The latest earlier normal run with completed measured records of `case_key`."""
        for run_dir in self._candidates:
            run = self._load(run_dir)
            if run is None:
                continue
            case_records = [r for r in run.records if r.case_key == case_key and r.arm == DEFAULT_ARM]
            records = [r for r in _final_records(case_records) if r.kind == "measure" and r.status == "completed"]
            if records:
                source = next((r for r in records if r.valid_psnr and r.image_path), None)
                image_path = run_dir / source.image_path if source and source.image_path else None
                return _Previous(run, records, image_path)
        return None

    def _load(self, run_dir: Path) -> _Run | None:
        if run_dir not in self._loaded:
            run: _Run | None = None
            try:
                info = load_run_info(run_dir)
                if info.profile == self._profile:
                    run = _Run(run_dir, info, load_records(run_dir))
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as e:
                self.notes.append(f"Earlier run {run_dir.name} skipped, unreadable: {_first_line(e)}")
            self._loaded[run_dir] = run
        return self._loaded[run_dir]


class _ImageCache:
    def __init__(self, scrub: Callable[[str], str]) -> None:
        self._arrays: dict[Path, np.ndarray | str] = {}
        self._scrub = scrub

    def get(self, path: Path) -> np.ndarray | str:
        """The decoded image, or why it is unavailable."""
        if path not in self._arrays:
            try:
                self._arrays[path] = load_rgb(path)
            except FileNotFoundError:
                self._arrays[path] = "image missing"
            except OSError as e:  # PIL names the file in its message
                self._arrays[path] = f"image unreadable: {self._scrub(_first_line(e))}"
        return self._arrays[path]


@dataclass(frozen=True)
class _Context:
    data_dir: Path
    run_dir: Path
    info: RunInfo
    history: _History
    images: _ImageCache
    diff_dir: Path
    scrubber: Scrubber


def _node_types(run_dir: Path, case_key: str) -> dict[str, str]:
    """node_id -> node type from graphs/<case_key>.json (a graph, or a batch holding one under `graph`)."""
    try:
        data = json.loads((run_dir / "graphs" / f"{case_key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    graph = data.get("graph", data) if isinstance(data, dict) else None
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict):
        return {}
    return {node_id: str(node["type"]) for node_id, node in nodes.items() if isinstance(node, dict) and "type" in node}


def _environment_facts(env: Environment) -> dict[str, str | None]:
    """The parts of an environment that change images, times or memory peaks."""
    deps = {name.lower(): version for name, version in env.app_deps.items()}
    config = env.runtime_config.get("config", env.runtime_config)
    if not isinstance(config, dict):
        config = {}

    def text(value: object) -> str | None:
        return None if value is None else str(value)

    return {
        "app version": text(env.app_version.get("version")),
        "torch": text(deps.get("torch")),
        "CUDA": text(deps.get("cuda")),
        "GPU": env.gpu_description,
        "GPU driver": env.gpu_driver,
        **{key: text(config.get(key)) for key in RUNTIME_CONFIG_KEYS},
    }


def _environment_changes(before: Environment, after: Environment | None) -> list[str]:
    if after is None:
        return ["this run has no recorded environment"]
    old, new = _environment_facts(before), _environment_facts(after)
    return [f"{key} {_dash(old[key])} → {_dash(new[key])}" for key in old if old[key] != new[key]]


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation


@dataclass(frozen=True)
class _Image:
    title: str
    path: Path | None
    caption: str


@dataclass
class _Row:
    record: Record
    psnr_reference: float | str | None = None
    psnr_previous: float | str | None = None
    final: bool = True
    """The last attempt of its step; earlier attempts were retried and do not count."""

    @property
    def rated(self) -> bool:
        """Counts for the quality verdict."""
        r = self.record
        return self.final and r.kind == "measure" and r.status == "completed" and r.valid_psnr and bool(r.image_path)


@dataclass(frozen=True)
class _Metric:
    label: str
    unit: Unit
    values: list[float]
    reference: float | None
    previous: float | None
    vs_reference: Delta | None
    vs_previous: Delta | None
    invalid: str | None = None
    """Why there are no values although measured runs completed."""

    @property
    def median_text(self) -> str:
        return "invalid" if self.invalid else _value_text(median(self.values), self.unit)


@dataclass(frozen=True)
class _Node:
    label: str
    duration_s: float | None
    baseline_bytes: float | None
    peak: _Metric


@dataclass
class _Case:
    arm: str
    case_key: str
    family: str
    status: Status = "PASS"
    size: tuple[int, int] | None = None
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    outliers: list[str] = field(default_factory=list)
    node_outliers: list[str] = field(default_factory=list)
    invalid_metrics: list[str] = field(default_factory=list)
    min_psnr: float | None = None
    psnr_min_db: float | None = None
    rows: list[_Row] = field(default_factory=list)
    metrics: list[_Metric] = field(default_factory=list)
    nodes: list[_Node] = field(default_factory=list)
    images: list[_Image] = field(default_factory=list)
    reference: Reference | None = None
    models: list[ModelIdentity] = field(default_factory=list)

    @property
    def has_outliers(self) -> bool:
        return bool(self.outliers or self.node_outliers)

    @property
    def label(self) -> str:
        return self.case_key if self.arm == DEFAULT_ARM else f"{self.case_key} [{self.arm}]"

    @property
    def anchor(self) -> str:
        return "case-" + "".join(c if c.isalnum() else "-" for c in self.label)


def _evaluate_safely(context: _Context, arm: str, case_key: str, records: list[Record]) -> _Case:
    family = records[0].family if records else next((s.family for s in context.info.plan if s.case_key == case_key), "")
    case = _Case(arm, case_key, family, models=context.info.case_models.get(case_key, []))
    try:
        _evaluate(context, case, records)
    except Exception as e:  # one broken case must not hide the others
        case.status = "ERROR"
        case.reasons.append(f"the report could not evaluate this case: {context.scrubber.text(repr(e))}")
    return case


def _evaluate(context: _Context, case: _Case, records: list[Record]) -> None:
    info, images = context.info, context.images
    if records:
        case.size = (records[0].width, records[0].height)
    thresholds = info.case_thresholds.get(case.case_key)
    case.psnr_min_db = thresholds.psnr_min_db if thresholds else None

    reference_image: np.ndarray | str | None = None
    reference_problem: str | None = None
    try:
        loaded = load_reference(context.data_dir, info.profile, case.case_key)
    except ValueError as e:
        loaded, reference_problem = None, f"reference.json unreadable: {_first_line(e)}"
    if loaded is not None:
        case.reference, reference_path = loaded
        reference_image = images.get(reference_path)
        if isinstance(reference_image, str):
            reference_problem = f"reference {reference_image}"
        elif hashlib.sha256(reference_path.read_bytes()).hexdigest() != case.reference.image_sha256:
            case.flags.append("reference image does not match reference.json")
        source = case.reference.source_run_id
        caption = "accepted from this run" if source == info.run_id else f"accepted from {source}"
        case.images.append(_Image("Reference", reference_path, caption))
    else:
        case.images.append(_Image("Reference", None, reference_problem or "no accepted reference"))

    previous = context.history.previous(case.case_key)
    previous_image = images.get(previous.image_path) if previous and previous.image_path else None

    final = _final_records(records)
    final_ids = {r.record_id for r in final}
    for record in records:
        row = _Row(record, final=record.record_id in final_ids)
        if record.status == "completed" and record.image_path:
            image = images.get(context.run_dir / record.image_path)
            row.psnr_reference = _psnr_or_problem(image, reference_image)
            row.psnr_previous = _psnr_or_problem(image, previous_image)
        case.rows.append(row)

    _set_status(case, final, thresholds, reference_problem, info.run_id)
    _compare_metrics(case, context.run_dir, final, previous, thresholds)
    _set_change_flags(context, case, previous)
    _add_images(case, context, reference_image, previous, previous_image)


def _psnr_or_problem(image: np.ndarray | str, other: np.ndarray | str | None) -> float | str | None:
    if other is None or isinstance(other, str):
        return None
    if isinstance(image, str):
        return image
    try:
        return psnr(image, other)
    except ValueError as e:
        return str(e)


def _set_status(
    case: _Case, final: list[Record], thresholds: Thresholds | None, reference_problem: str | None, run_id: str
) -> None:
    """Judges each step by its final attempt; a failure that a later attempt recovered from stays visible as a flag.

    Errors, invalid runs and a missing reference come first. A reference set from this run gives BASELINE. Then the
    minimum PSNR against the reference decides: below the threshold is FAIL, or CHANGED for a case expected to change,
    which fails when it matches instead.
    """
    rated = [row for row in case.rows if row.rated]
    values = [row.psnr_reference for row in rated if isinstance(row.psnr_reference, float)]
    case.min_psnr = min(values) if values else None
    case.flags += [
        f"earlier attempt {row.record.record_id} {row.record.status}: {row.record.error or 'no error message'}"
        for row in case.rows
        if not row.final and row.record.status in ("failed", "error")
    ]
    errored = [r for r in final if r.status in ("failed", "error")]
    wrong_size = [
        r
        for r in final
        if r.status == "completed"
        and r.image_width is not None
        and (r.image_width, r.image_height) != (r.width, r.height)
    ]
    unrated = [f"{row.record.record_id}: {row.psnr_reference}" for row in rated if isinstance(row.psnr_reference, str)]

    if errored:
        case.status = "ERROR"
        case.reasons += [f"{r.record_id}: {r.status}: {r.error or 'no error message'}" for r in errored]
    elif wrong_size:
        case.status = "FAIL"
        case.reasons += [
            f"{r.record_id}: image is {r.image_width}x{r.image_height}, requested {r.width}x{r.height}"
            for r in wrong_size
        ]
    elif not rated:
        case.status = "INVALID"
        case.reasons += _invalid_reasons([r for r in final if r.kind == "measure"])
    elif case.reference is None or reference_problem is not None:
        case.status = "NO-REFERENCE"
        case.reasons.append(reference_problem or "no accepted reference; set one with `accept`")
    elif unrated:
        case.status = "FAIL"
        case.reasons += unrated
    elif case.reference.source_run_id == run_id:
        case.status = "BASELINE"
        case.reasons.append("reference set from this run")
    elif thresholds is None:
        case.status = "INVALID"
        case.reasons.append("run.json has no thresholds for this case")
    else:
        differs = case.min_psnr is not None and case.min_psnr < thresholds.psnr_min_db
        min_psnr = f"min PSNR {_psnr_text(case.min_psnr)}"
        if thresholds.expect == "changed":
            case.status = "CHANGED" if differs else "FAIL"
            case.reasons.append(
                f"differs from reference as expected ({min_psnr})"
                if differs
                else f"expected a change, but the output matches the reference ({min_psnr})"
            )
        elif differs:
            case.status = "FAIL"
            case.reasons.append(f"{min_psnr} is below {thresholds.psnr_min_db:g} dB")


def _invalid_reasons(measured: list[Record]) -> list[str]:
    """Why no measured image is rated; the per-metric reasons are in `invalid_metrics`."""
    if not measured:
        return ["no measured records"]
    reasons = Counter(r.status if r.status != "completed" else "no image" for r in measured if not r.image_path)
    lines = [f"{reason} ({count} of {len(measured)} measured)" for reason, count in reasons.items()]
    return lines or ["no measured image is valid for PSNR"]


def _final_records(records: list[Record]) -> list[Record]:
    """The last attempt of each step of one case and arm, in record order.

    The runner retries a step whose attempt foreign load spoiled, and `resume` reruns a step that failed.
    """
    final: dict[tuple[str, int], Record] = {}
    for record in records:
        step = (record.kind, record.rep)
        if step not in final or record.attempt > final[step].attempt:
            final[step] = record
    return [record for record in records if final[(record.kind, record.rep)] is record]


_VALIDITY: tuple[tuple[str, Callable[[Record], bool]], ...] = (
    ("time", lambda r: r.valid_timing),
    ("VRAM", lambda r: r.valid_memory),
    ("PSNR", lambda r: r.valid_psnr),
)


def _set_invalid_metrics(case: _Case, final: list[Record]) -> dict[str, str]:
    """Lists which metrics the final measured runs could not deliver and why; returns metric -> reasons."""
    notes: Counter[tuple[tuple[str, ...], str]] = Counter()
    reasons_by_metric: dict[str, dict[str, None]] = {name: {} for name, _ in _VALIDITY}
    for record in final:
        if record.status != "completed":
            continue
        affected = tuple(name for name, valid in _VALIDITY if not valid(record))
        for reason in dict.fromkeys(record.invalid_reasons or ["no reason recorded"]) if affected else []:
            notes[(affected, reason)] += 1
            for name in affected:
                reasons_by_metric[name][reason] = None
    case.invalid_metrics = [
        f"{_join(affected)} invalid: {reason} ({count} of {len(final)} measured)"
        for (affected, reason), count in notes.items()
    ]
    return {name: "; ".join(reasons) for name, reasons in reasons_by_metric.items() if reasons}


def _join(names: tuple[str, ...]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _compare_metrics(
    case: _Case, run_dir: Path, final: list[Record], previous: _Previous | None, thresholds: Thresholds | None
) -> None:
    current = measured_values(final)
    before = measured_values(previous.records if previous else [])
    # A reference from this run holds this run's own medians; comparing against them says nothing.
    reference = None if case.status == "BASELINE" else case.reference
    invalid = _set_invalid_metrics(case, [r for r in final if r.kind == "measure"])

    def metric(
        label: str,
        unit: Unit,
        values: list[float],
        ref: float | None,
        prev: float | None,
        outliers: list[str],
        invalid_reason: str | None = None,
    ) -> _Metric:
        def compare(baseline: float | None) -> Delta | None:
            if baseline is None or thresholds is None:
                return None
            if unit == "s":
                return compare_time(values, baseline, thresholds.time_tolerance)
            return compare_peak(values, baseline, thresholds.peak_tolerance_mb)

        reason = invalid_reason if not values else None
        result = _Metric(label, unit, values, ref, prev, compare(ref), compare(prev), reason)
        for target, delta in (("reference", result.vs_reference), ("previous run", result.vs_previous)):
            if delta is not None and delta.outlier:
                outliers.append(f"{label} vs {target}: {_delta_text(delta, unit)}")
        return result

    case.metrics = [
        metric(
            "Total time",
            "s",
            current.total_s,
            reference.total_s_median if reference else None,
            median(before.total_s),
            case.outliers,
            invalid.get("time"),
        ),
        metric(
            "Time per step",
            "s",
            current.per_step_s,
            reference.per_step_s_median if reference else None,
            median(before.per_step_s),
            case.outliers,
            invalid.get("time"),
        ),
        metric(
            "VRAM peak",
            "bytes",
            [float(v) for v in current.process_peak_bytes],
            reference.process_peak_bytes_median if reference else None,
            median(before.process_peak_bytes),
            case.outliers,
            invalid.get("VRAM"),
        ),
    ]

    types = _node_types(run_dir, case.case_key)
    for record in final:
        for node in record.timing.nodes if record.timing else []:
            types.setdefault(node.node_id, node.type)
    for node_id in dict.fromkeys([*current.node_duration_s, *current.node_peak_bytes]):
        label = f"{types[node_id]} · {node_id}" if node_id in types else node_id
        peak = metric(
            f"{label} peak",
            "bytes",
            [float(v) for v in current.node_peak_bytes.get(node_id, [])],
            reference.node_peak_bytes_median.get(node_id) if reference else None,
            median(before.node_peak_bytes.get(node_id, [])),
            case.node_outliers,
        )
        duration = median(current.node_duration_s.get(node_id, []))
        case.nodes.append(_Node(label, duration, median(current.node_baseline_bytes.get(node_id, [])), peak))


def _set_change_flags(context: _Context, case: _Case, previous: _Previous | None) -> None:
    info, clean = context.info, context.scrubber.environment
    graph_hash = info.graph_hashes.get(case.case_key) or next((row.record.graph_hash for row in case.rows), None)
    environment = info.environments.get(case.arm)
    environment = None if environment is None else clean(environment)
    if case.reference is not None:
        if graph_hash != case.reference.graph_hash:
            case.flags.append("graph changed vs reference")
        changes = _environment_changes(clean(case.reference.environment), environment)
        if changes:
            case.flags.append("environment changed vs reference: " + "; ".join(changes))
    if previous is not None:
        previous_hash = previous.run.info.graph_hashes.get(case.case_key) or previous.records[0].graph_hash
        if graph_hash != previous_hash:
            case.flags.append("graph changed vs previous run")
        previous_environment = previous.run.info.environments.get(DEFAULT_ARM)
        if previous_environment is not None:
            changes = _environment_changes(clean(previous_environment), environment)
            if changes:
                case.flags.append("environment changed vs previous run: " + "; ".join(changes))
    shas = {row.record.image_sha256 for row in case.rows if row.rated}
    if len(shas) > 1:
        case.flags.append(f"measured images differ from each other ({len(shas)} distinct)")


def _add_images(
    case: _Case,
    context: _Context,
    reference_image: np.ndarray | str | None,
    previous: _Previous | None,
    previous_image: np.ndarray | str | None,
) -> None:
    candidates = [row for row in case.rows if row.rated] or [
        row for row in case.rows if row.record.status == "completed" and row.record.image_path
    ]
    # The worst image tells most about a quality regression.
    shown = min(candidates, key=_psnr_sort_key, default=None)
    shown_path = context.run_dir / shown.record.image_path if shown and shown.record.image_path else None
    if shown is None:
        case.images.append(_Image("This run", None, "no image"))
    else:
        case.images.append(_Image("This run", shown_path, shown.record.record_id))

    if previous is None:
        case.images.append(_Image("Previous run", None, "no earlier run of this case"))
    elif previous.image_path is None:
        case.images.append(_Image("Previous run", None, f"{previous.run.info.run_id} has no image valid for PSNR"))
    else:
        case.images.append(_Image("Previous run", previous.image_path, previous.run.info.run_id))

    base, base_name = (
        (reference_image, "reference") if isinstance(reference_image, np.ndarray) else (previous_image, "previous run")
    )
    title = f"Difference ×{DIFF_GAIN}"
    image = context.images.get(shown_path) if shown_path else None
    if not isinstance(image, np.ndarray) or not isinstance(base, np.ndarray):
        case.images.append(_Image(title, None, "needs this run's image and a reference or previous image"))
        return
    suffix = "" if case.arm == DEFAULT_ARM else f".{case.arm}"
    diff_path = context.diff_dir / f"{case.case_key}{suffix}.png"
    try:
        write_diff_image(image, base, diff_path)
        identical = np.array_equal(image, base)
    except ValueError as e:
        case.images.append(_Image(title, None, str(e)))
    else:
        case.images.append(_Image(title, diff_path, f"this run vs {base_name}" + (" · identical" if identical else "")))


def _psnr_sort_key(row: _Row) -> float:
    if isinstance(row.psnr_reference, str):  # a missing or mismatched image is the worst
        return -math.inf
    return row.psnr_reference if row.psnr_reference is not None else math.inf


# ---------------------------------------------------------------------------------------------------------------------
# Formatting


def _first_line(error: BaseException) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__


def _dash(value: object) -> str:
    return "—" if value is None else str(value)


def _psnr_text(value: float | str | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return "∞" if math.isinf(value) else f"{value:.2f} dB"


def _value_text(value: float | None, unit: Unit) -> str:
    if value is None:
        return "—"
    if unit == "bytes":
        return f"{value / MB:,.0f} MB"
    return f"{value:.3f} s" if abs(value) < 10 else f"{value:.2f} s"


def _delta_text(delta: Delta, unit: Unit) -> str:
    sign = "+" if delta.delta >= 0 else "-"
    amount = _value_text(abs(delta.delta), unit)
    return f"{sign}{amount}" if delta.relative is None else f"{sign}{amount} ({delta.relative:+.1%})"


def _relative_text(delta: Delta | None) -> str:
    if delta is None or delta.relative is None:
        return "—"
    return f"{delta.relative:+.1%}"


def _short_hash(value: str | None) -> str:
    """`blake3:<64 hex>` as `blake3:<12 hex>`: enough to tell models apart."""
    if not value:
        return "—"
    algorithm, _, digest = value.rpartition(":")
    return f"{algorithm}:{digest[:12]}" if algorithm else digest[:12]


def _model_text(model: ModelIdentity) -> str:
    return f"{model.name} ({model.type}, {_short_hash(model.hash)})"


def _href(path: Path, run_dir: Path) -> str:
    return urllib.parse.quote(Path(os.path.relpath(path, run_dir)).as_posix())


def _e(value: object) -> str:
    return html.escape(str(value))


# ---------------------------------------------------------------------------------------------------------------------
# HTML

_CSS = """
:root { color-scheme: light dark;
  --bg: #f6f7f9; --card: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0; --stripe: #f3f5f7;
  --pass-bg: #dafbe1; --pass-fg: #116329; --fail-bg: #ffebe9; --fail-fg: #a40e26;
  --invalid-bg: #fff1c2; --invalid-fg: #6f4400; --noref-bg: #ddf4ff; --noref-fg: #0a3069;
  --changed-bg: #fbefff; --changed-fg: #6e40c9; --baseline-bg: #eaeef2; --baseline-fg: #424a53;
  --warn: #9a6700; --link: #0969da; }
@media (prefers-color-scheme: dark) { :root {
  --bg: #0d1117; --card: #161b22; --fg: #e6edf3; --muted: #9198a1; --border: #30363d; --stripe: #1b2129;
  --pass-bg: #12361f; --pass-fg: #7ee2a8; --fail-bg: #4a1a1f; --fail-fg: #ffa198;
  --invalid-bg: #3b2e0a; --invalid-fg: #e3b341; --noref-bg: #0f2a4a; --noref-fg: #79c0ff;
  --changed-bg: #2e1f4d; --changed-fg: #d2a8ff; --baseline-bg: #262c36; --baseline-fg: #b7bdc8;
  --warn: #e3b341; --link: #4493f8; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1440px; margin: 0 auto; padding: 24px 32px 48px; }
a { color: var(--link); }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 12px; }
h3 { font-size: 16px; margin: 0; display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
h4 { font-size: 13px; margin: 20px 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.muted { color: var(--muted); font-weight: normal; }
.meta { color: var(--muted); margin: 0; }
.counts { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.badge { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 12px; font-weight: 600;
  white-space: nowrap; }
.s-PASS { background: var(--pass-bg); color: var(--pass-fg); }
.s-FAIL, .s-ERROR { background: var(--fail-bg); color: var(--fail-fg); }
.s-INVALID { background: var(--invalid-bg); color: var(--invalid-fg); }
.s-NO-REFERENCE { background: var(--noref-bg); color: var(--noref-fg); }
.s-CHANGED { background: var(--changed-bg); color: var(--changed-fg); }
.s-BASELINE { background: var(--baseline-bg); color: var(--baseline-fg); }
.s-outlier { background: var(--invalid-bg); color: var(--invalid-fg); }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 12px; }
tbody tr:nth-child(even) { background: var(--stripe); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
.table-wrap table { border: 0; }
.case { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px;
  margin: 18px 0; }
.case ul { margin: 8px 0 0; padding-left: 20px; }
.case ul.models { list-style: none; padding: 0; margin: 6px 0 0; display: flex; flex-wrap: wrap; gap: 2px 20px;
  font-size: 13px; }
.reasons li { color: var(--fail-fg); }
.case.s-INVALID-case .reasons li, .case.s-NO-REFERENCE-case .reasons li, .case.s-CHANGED-case .reasons li,
.case.s-BASELINE-case .reasons li { color: var(--fg); }
.flags li { color: var(--warn); }
.outlier { color: var(--warn); font-weight: 600; }
td.invalid { color: var(--invalid-fg); font-weight: 600; cursor: help; text-decoration: underline dotted; }
.images { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }
figure { margin: 0; }
figcaption { font-size: 12px; margin-bottom: 4px; }
figcaption strong { display: block; font-size: 13px; }
figure img { display: block; width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px;
  background: var(--stripe); }
.placeholder { display: flex; align-items: center; justify-content: center; text-align: center; padding: 12px;
  border: 1px dashed var(--border); border-radius: 6px; color: var(--muted); }
tr.warmup td { color: var(--muted); font-style: italic; }
table.generations td:first-child { white-space: nowrap; }
table.compact { width: auto; min-width: 50%; }
.notes li { color: var(--warn); }
@media (max-width: 900px) { .images { grid-template-columns: repeat(2, minmax(0, 1fr)); } main { padding: 16px; } }
"""


def _render_html(
    context: _Context, cases: list[_Case], notes: list[str], href: Callable[[Path], str], run_status: str
) -> str:
    info = context.info
    counts = Counter(case.status for case in cases)
    outliers = sum(case.has_outliers for case in cases)
    count_badges = "".join(
        f'<span class="badge s-{status}">{counts[status]} {status}</span>' for status in STATUSES if counts[status]
    )
    if outliers:
        count_badges += f'<span class="badge s-outlier">{outliers} with outliers</span>'
    invalid_metrics = sum(bool(case.invalid_metrics) for case in cases)
    if invalid_metrics:
        count_badges += f'<span class="badge s-INVALID">{invalid_metrics} with invalid metrics</span>'
    finished = f" · finished {_e(info.finished_at)}" if info.finished_at else ""
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Family E2E {_e(info.run_id)}</title><style>{_CSS}</style></head><body><main>",
        f'<header><h1>Family E2E report</h1><p class="meta">{_e(info.run_id)} · profile {_e(info.profile)} · '
        f"run {_e(run_status)} · started {_e(info.created_at)}{finished}</p>",
        f'<div class="counts">{count_badges or "<span class=muted>no cases</span>"}</div></header>',
    ]
    if run_status != "completed":
        notes = [f"The run is {run_status}; cases it did not reach show as INVALID.", *notes]
    if notes:
        parts.append('<ul class="notes">' + "".join(f"<li>{_e(note)}</li>" for note in notes) + "</ul>")
    parts.append(_environment_html(info))
    parts.append(_overview_html(cases))
    parts += [_case_html(case, href) for case in cases]
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _environment_html(info: RunInfo) -> str:
    arms = list(info.environments)
    if not arms:
        return "<h2>Environment</h2><p class=muted>No environment recorded.</p>"
    facts = {arm: _environment_facts(env) for arm, env in info.environments.items()}
    rows: list[tuple[str, list[str]]] = []
    rows.append(("code", [env.code_dir for env in info.environments.values()]))
    rows.append(
        (
            "commit",
            [
                f"{_dash(env.git_commit)[:12]} {_dash(env.git_branch)}" + (" (dirty)" if env.git_dirty else "")
                for env in info.environments.values()
            ],
        )
    )
    for key in next(iter(facts.values())):
        rows.append((key, [_dash(facts[arm][key]) for arm in arms]))
    header = "".join(f"<th>{_e(arm)}</th>" for arm in arms)
    body = "".join(
        f"<tr><th scope=row>{_e(key)}</th>" + "".join(f"<td>{_e(v)}</td>" for v in values) + "</tr>"
        for key, values in rows
    )
    manifest = f"manifest {_e(info.manifest_path)} · sha256 {_e(info.manifest_sha256[:12])}"
    tool = f" · tool commit {_e(info.tool_commit[:12])}" if info.tool_commit else ""
    return (
        f'<h2>Environment</h2><div class="table-wrap"><table class="compact"><thead><tr><th></th>{header}</tr></thead>'
        f'<tbody>{body}</tbody></table></div><p class="meta">{manifest}{tool}</p>'
    )


def _overview_html(cases: list[_Case]) -> str:
    rows = []
    for case in cases:
        time, peak = (case.metrics[0], case.metrics[2]) if case.metrics else (None, None)
        threshold = f" / {case.psnr_min_db:g}" if case.psnr_min_db is not None else ""
        notes = len(_flag_lines(case))
        rows.append(
            f'<tr><td><a href="#{case.anchor}">{_e(case.label)}</a></td><td>{_e(case.family)}</td>'
            f'<td><span class="badge s-{case.status}">{case.status}</span></td>'
            f'<td class="num">{_e(_psnr_text(case.min_psnr))}{_e(threshold)}</td>'
            + _overview_metric_cells(time)
            + _overview_metric_cells(peak)
            + f'<td class="num">{notes or ""}</td></tr>'
        )
    return (
        '<h2>Cases</h2><div class="table-wrap"><table><thead><tr><th>Case</th><th>Family</th><th>Status</th>'
        '<th class="num">min PSNR / threshold (dB)</th><th class="num">Time</th><th class="num">vs ref</th>'
        '<th class="num">vs prev</th><th class="num">VRAM peak</th><th class="num">vs ref</th>'
        '<th class="num">vs prev</th><th class="num">Flags</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _overview_metric_cells(metric: _Metric | None) -> str:
    if metric is None:
        return '<td class="num">—</td>' * 3
    if metric.invalid:
        cells = [f'<td class="num invalid" title="{_e(metric.invalid)}">invalid</td>']
    else:
        cells = [f'<td class="num">{_e(metric.median_text)}</td>']
    for delta in (metric.vs_reference, metric.vs_previous):
        css = "num outlier" if delta is not None and delta.outlier else "num"
        cells.append(f'<td class="{css}">{_e(_relative_text(delta))}</td>')
    return "".join(cells)


def _case_html(case: _Case, href: Callable[[Path], str]) -> str:
    size = f" · {case.size[0]}×{case.size[1]}" if case.size else ""
    parts = [
        f'<article class="case s-{case.status}-case" id="{case.anchor}">',
        f'<h3><span class="badge s-{case.status}">{case.status}</span>{_e(case.label)}'
        f'<span class="muted">{_e(case.family)}{size}</span></h3>',
    ]
    if case.models:
        parts.append(_models_html(case.models))
    if case.reasons:
        parts.append('<ul class="reasons">' + "".join(f"<li>{_e(r)}</li>" for r in case.reasons) + "</ul>")
    flags = _flag_lines(case)
    if flags:
        parts.append('<ul class="flags">' + "".join(f"<li>{_e(f)}</li>" for f in flags) + "</ul>")
    if case.images:
        parts.append('<div class="images">' + "".join(_figure_html(i, case, href) for i in case.images) + "</div>")
    if case.metrics:
        parts.append("<h4>Measurements (measured runs valid for the metric)</h4>" + _metrics_table(case.metrics))
    if case.nodes:
        parts.append("<h4>Per node</h4>" + _nodes_table(case.nodes))
    if case.rows:
        parts.append("<h4>Generations</h4>" + _rows_table(case.rows))
    parts.append("</article>")
    return "\n".join(parts)


def _flag_lines(case: _Case) -> list[str]:
    lines = [f"invalid metrics: {note}" for note in case.invalid_metrics]
    lines += case.flags + [f"outlier: {o}" for o in case.outliers]
    if case.node_outliers:
        lines.append(f"outlier: {len(case.node_outliers)} per-node peak comparisons, marked under Per node")
    return lines


def _models_html(models: list[ModelIdentity]) -> str:
    items = "".join(
        f'<li>{_e(m.name)} <span class="muted">{_e(m.type)} · <span title="{_e(_dash(m.hash))}">'
        f"{_e(_short_hash(m.hash))}</span></span></li>"
        for m in models
    )
    return f'<ul class="models" aria-label="Models">{items}</ul>'


def _figure_html(image: _Image, case: _Case, href: Callable[[Path], str]) -> str:
    ratio = f"{case.size[0]} / {case.size[1]}" if case.size else "1 / 1"
    caption = f"<figcaption><strong>{_e(image.title)}</strong>{_e(image.caption)}</figcaption>"
    if image.path is None or not image.path.is_file():
        reason = image.caption if image.path is None else f"missing: {image.path.name}"
        return f'<figure>{caption}<div class="placeholder" style="aspect-ratio: {ratio}">{_e(reason)}</div></figure>'
    link = href(image.path)
    alt = f"{image.title} for {case.label}"
    return (
        f'<figure>{caption}<a href="{link}"><img src="{link}" alt="{_e(alt)}" loading="lazy" '
        f'style="aspect-ratio: {ratio}"></a></figure>'
    )


def _delta_cell(delta: Delta | None, unit: Unit) -> str:
    if delta is None:
        return '<td class="num">—</td>'
    if delta.outlier:
        return f'<td class="num outlier">{_e(_delta_text(delta, unit))} · outlier</td>'
    return f'<td class="num">{_e(_delta_text(delta, unit))}</td>'


def _metric_cells(metric: _Metric) -> str:
    return (
        f'<td class="num">{_e(metric.median_text)}</td>'
        f'<td class="num">{_e(_value_text(metric.reference, metric.unit))}</td>'
        + _delta_cell(metric.vs_reference, metric.unit)
        + f'<td class="num">{_e(_value_text(metric.previous, metric.unit))}</td>'
        + _delta_cell(metric.vs_previous, metric.unit)
    )


def _runs_text(metric: _Metric) -> str:
    if metric.invalid:
        return f"invalid: {metric.invalid}"
    return ", ".join(_value_text(v, metric.unit) for v in metric.values) or "no values"


def _metrics_table(metrics: list[_Metric]) -> str:
    rows = "".join(
        f"<tr><th scope=row>{_e(m.label)}</th>" + _metric_cells(m) + f'<td class="num">{_e(_runs_text(m))}</td></tr>'
        for m in metrics
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Metric</th><th class="num">Median</th>'
        '<th class="num">Reference</th><th class="num">vs reference</th><th class="num">Previous run</th>'
        '<th class="num">vs previous</th><th class="num">Runs</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _nodes_table(nodes: list[_Node]) -> str:
    rows = "".join(
        f"<tr><th scope=row>{_e(n.label)}</th>"
        f'<td class="num">{_e(_value_text(n.duration_s, "s"))}</td>'
        f'<td class="num">{_e(_value_text(n.baseline_bytes, "bytes"))}</td>' + _metric_cells(n.peak) + "</tr>"
        for n in nodes
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Node</th><th class="num">Time</th>'
        '<th class="num">Baseline</th><th class="num">Peak</th><th class="num">Reference</th>'
        '<th class="num">vs reference</th><th class="num">Previous run</th><th class="num">vs previous</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _residency_text(record: Record) -> str:
    if not record.residency:
        return "unknown"
    lowest = min(record.residency, key=lambda r: r.percent)
    if lowest.percent >= 100:
        return "100 %"
    model = f"{lowest.model}/{lowest.submodel}" if lowest.submodel else lowest.model
    return f"{lowest.percent:.0f} % ({model})"


def _generation_text(record: Record) -> str:
    kind = "warm-up" if record.kind == "warmup" else "measured"
    return f"{kind} · rep {record.rep} · attempt {record.attempt}"


def _rows_table(rows: list[_Row]) -> str:
    body = []
    last_attempt: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row.record.kind, row.record.rep)
        last_attempt[key] = max(last_attempt.get(key, row.record.attempt), row.record.attempt)
    for row in sorted(rows, key=lambda row: row.record.kind != "warmup"):
        r = row.record
        invalid = [name for name, valid in _VALIDITY if not valid(r)]
        notes = [*([r.error] if r.error else []), *r.invalid_reasons]
        if r.attempt < last_attempt[(r.kind, r.rep)]:
            notes.append(f"retried as attempt {last_attempt[(r.kind, r.rep)]}")
        peak = r.memory.process_peak_bytes if r.memory else None
        body.append(
            f'<tr class="{r.kind}"><td>{_e(_generation_text(r))}</td><td>{_e(r.status)}</td>'
            f"<td>{_e('valid' if not invalid else 'invalid: ' + ', '.join(invalid))}</td>"
            f'<td class="num">{_e(_value_text(r.timing.total_s if r.timing else None, "s"))}</td>'
            f'<td class="num">{_e(_value_text(r.timing.per_step_s if r.timing else None, "s"))}</td>'
            f'<td class="num">{_e(_value_text(peak, "bytes"))}</td>'
            f'<td class="num">{_e(_psnr_text(row.psnr_reference))}</td>'
            f'<td class="num">{_e(_psnr_text(row.psnr_previous))}</td>'
            f"<td>{_e(_residency_text(r))}</td><td>{_e('; '.join(notes))}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table class="generations"><thead><tr><th>Generation</th><th>Status</th><th>Validity</th>'
        '<th class="num">Total</th><th class="num">Per step</th><th class="num">VRAM peak</th>'
        '<th class="num">PSNR vs reference</th><th class="num">PSNR vs previous</th><th>Residency</th>'
        f"<th>Notes</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Markdown


def _render_summary(info: RunInfo, cases: list[_Case], notes: list[str], run_status: str) -> str:
    counts = Counter(case.status for case in cases)
    outliers = sum(case.has_outliers for case in cases)
    invalid_metrics = sum(bool(case.invalid_metrics) for case in cases)
    tally = ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s]) or "no cases"
    lines = [
        f"# Family E2E {info.run_id}",
        "",
        f"Profile {info.profile}, run {run_status}: {tally}, {outliers} with outliers, "
        f"{invalid_metrics} with invalid metrics.",
        "",
        "| Case | Status | min PSNR | Time | vs ref | vs prev | VRAM peak | vs ref | vs prev |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        cells = [case.label.replace("|", "\\|"), case.status, _psnr_text(case.min_psnr)]
        for metric in (case.metrics[0], case.metrics[2]) if case.metrics else (None, None):
            if metric is None:
                cells += ["—"] * 3
                continue
            cells.append(metric.median_text)
            for delta in (metric.vs_reference, metric.vs_previous):
                text = _relative_text(delta)
                cells.append(f"**{text}**" if delta is not None and delta.outlier else text)
        lines.append("| " + " | ".join(cells) + " |")

    # The sizes of a case share its models; each distinct list once.
    models: dict[tuple[str, ...], list[str]] = {}
    for case in cases:
        if case.models:
            models.setdefault(tuple(_model_text(m) for m in case.models), []).append(case.label)
    if models:
        lines += ["", "## Models", ""]
        lines += [f"- {', '.join(labels)}: {'; '.join(texts)}" for texts, labels in models.items()]

    attention = []
    for case in cases:
        if case.status != "PASS":
            attention += [f"- {case.status} {case.label}: {reason}" for reason in case.reasons or [""]]
        attention += [f"- INVALID METRICS {case.label}: {note}" for note in case.invalid_metrics]
        attention += [f"- OUTLIER {case.label}: {outlier}" for outlier in case.outliers]
        if case.node_outliers:
            attention.append(f"- OUTLIER {case.label}: {len(case.node_outliers)} per-node peak comparisons, see report")
    # One environment change usually affects every case; list each distinct flag once.
    flagged: dict[str, list[str]] = {}
    for case in cases:
        for flag in case.flags:
            flagged.setdefault(flag, []).append(case.label)
    attention += [f"- FLAG {', '.join(labels)}: {flag}" for flag, labels in flagged.items()]
    attention += [f"- NOTE {note}" for note in notes]
    if attention:
        lines += ["", "## Attention", "", *attention]
    return "\n".join(lines) + "\n"
