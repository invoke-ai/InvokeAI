"""Persistent records of a family E2E run: one JSON line per generation, plus the run info and references.

Times are seconds; `t_*` fields are relative to the enqueue instant (client `time.perf_counter()`). Paths stored in
records are relative to the run directory.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

RECORDS_FILE = "records.jsonl"
RUN_INFO_FILE = "run.json"
RUN_ID = re.compile(r"^\d{8}-\d{6}-(?P<profile>.+)$")
"""`YYYYMMDD-HHMMSS-<profile>`, also the name of the run's directory."""

# Windows refuses os.replace while another process (e.g. `status`) holds the target open; such reads are brief.
REPLACE_ATTEMPTS = 20
REPLACE_RETRY_DELAY_S = 0.05

Kind = Literal["warmup", "measure"]
Expect = Literal["unchanged", "changed"]
"""What a case's image should do against its reference: stay the same, or change (a PR meant to change it)."""


class _Model(BaseModel):
    # Records are written by one module and read by others; a misspelled field must fail loudly.
    model_config = ConfigDict(extra="forbid")


class NodeTiming(_Model):
    node_id: str
    type: str
    t_start: float
    t_end: float | None


class ModelLoad(_Model):
    model: str
    submodel: str | None
    t_start: float
    t_end: float | None


class Timing(_Model):
    total_s: float
    """Enqueue to completed."""
    nodes: list[NodeTiming]
    model_loads: list[ModelLoad]
    per_step_s: float | None
    """Slope over the denoise node's invocation_progress events."""
    queue_started_at: str | None
    queue_completed_at: str | None


class NodeMemory(_Model):
    node_id: str
    baseline_bytes: int | None
    peak_bytes: int | None


class ForeignProcess(_Model):
    pid: int
    name: str | None
    dedicated_bytes: int


class Memory(_Model):
    method: str
    interval_ms: float
    process_baseline_bytes: int | None
    """Just before enqueue."""
    process_peak_bytes: int | None
    nodes: list[NodeMemory]
    shared_max_delta_bytes: int | None
    """max(shared) - shared at enqueue."""
    foreign: list[ForeignProcess]
    """Other processes on the adapter above the threshold during the window."""
    trace: str | None
    """traces/<record_id>.csv.gz with columns series,t,bytes (series = dedicated|shared, t relative to enqueue)."""


class Residency(_Model):
    model: str
    submodel: str | None
    vram_mb: float
    total_mb: float | None = None
    """The model's size as the cache logs it; absent in records written before the field existed."""
    percent: float


class CalibrationSubmission(_Model):
    """How a calibration run's step reached the server: the run the calibrate route created for it."""

    run_id: str
    device_class_id: str


class Record(_Model):
    record_id: str
    """`<case_key>.<arm>.<kind>.<rep>.<attempt>`."""
    run_id: str
    arm: str
    """`default` for normal runs, `A`/`B` for A/B runs."""
    case_id: str
    case_key: str
    """`<case_id>@<W>x<H>`."""
    family: str
    width: int
    height: int
    steps: int
    seed: int
    kind: Kind
    rep: int
    attempt: int
    queue_item_id: int | None
    graph_hash: str
    status: Literal["completed", "failed", "canceled", "error"]
    error: str | None
    image_path: str | None
    image_sha256: str | None
    image_width: int | None
    image_height: int | None
    timing: Timing | None
    memory: Memory | None
    residency: list[Residency]
    uses_fp8: bool
    valid_timing: bool
    valid_memory: bool
    valid_psnr: bool
    invalid_reasons: list[str]
    calibration: CalibrationSubmission | None = None
    """Set for a step of a calibration run that the server accepted."""


class Environment(_Model):
    code_dir: str
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None
    python: str
    invokeai_file: str
    """`invokeai.__file__` as resolved by the server interpreter."""
    app_version: dict[str, Any]
    app_deps: dict[str, Any]
    runtime_config: dict[str, Any]
    gpu_description: str
    gpu_luid: int | None
    """None on Linux. NVML's device UUID is not recorded: a hardware serial that bundles would carry along."""
    gpu_driver: str | None = None
    """Absent in run.json and reference.json files written before the field existed."""
    server_cmd: list[str]
    extra_env: dict[str, str]


class PlannedStep(_Model):
    case_key: str
    case_id: str
    family: str
    arm: str
    kind: Kind
    rep: int


class Thresholds(_Model):
    psnr_min_db: float
    time_tolerance: float
    """Relative, e.g. 0.10 for 10 %."""
    peak_tolerance_mb: float
    expect: Expect = "unchanged"
    """Absent in run.json files written before the field existed."""


class ModelIdentity(_Model):
    """A model as another machine can recognise it; `key` is local to this machine's database."""

    key: str
    name: str
    base: str
    type: str
    format: str | None
    variant: str | None
    hash: str | None
    source: str | None
    """An HF repo id or a public http(s) URL (`graphs.shareable_source`); None for a local path, a host of the local
    network, or a URL that may carry a token."""


class CalibrationExport(_Model):
    """A calibration store saved from one server lifetime of a calibration run."""

    lifetime: int
    """Counts the run's server lifetimes that saved a store, from 1. Each lifetime has its own scratch store."""
    device_class_id: str
    export: str
    """calibration/<run_id>-<lifetime>-<device_class_id>.json: the store file byte for byte."""
    status: str | None
    """calibration/<run_id>-<lifetime>-status.json, the server's calibration status; None for a `store_file` copy."""
    source: Literal["api", "store_file"]
    """`api`: GET working_memory/export at the end of the lifetime. `store_file`: copied from the scratch root after the
    server died or was stopped, which leaves no server to ask."""
    saved_at: str


class RunInfo(_Model):
    run_id: str
    profile: str
    created_at: str
    finished_at: str | None
    status: Literal["running", "completed", "failed", "stopped"]
    manifest_path: str
    manifest_sha256: str
    tool_commit: str | None
    environments: dict[str, Environment]
    """Keyed by arm."""
    plan: list[PlannedStep]
    case_thresholds: dict[str, Thresholds]
    """Keyed by case_key."""
    graph_hashes: dict[str, str]
    """Keyed by case_key."""
    case_models: dict[str, list[ModelIdentity]] = {}
    """Keyed by case_key: the models its graph uses, main model first. Absent in runs written before the field."""
    calibrate: bool = False
    """Steps go through the working-memory calibrate route instead of enqueue_batch."""
    calibration_exports: list[CalibrationExport] = []


class Reference(_Model):
    profile: str
    case_key: str
    source_run_id: str
    accepted_at: str
    image_sha256: str
    graph_hash: str
    environment: Environment
    total_s_median: float | None
    per_step_s_median: float | None
    process_peak_bytes_median: int | None
    node_peak_bytes_median: dict[str, int]
    """node_id -> bytes."""


def append_record(run_dir: Path, record: Record) -> None:
    """Appends one line to records.jsonl and makes it durable before returning.

    An incomplete trailing line left by a killed writer is dropped first, so a resumed run keeps the file parseable.
    """
    path = run_dir / RECORDS_FILE
    line = (record.model_dump_json() + "\n").encode("utf-8")
    with open(path, "a+b") as f:
        _drop_incomplete_tail(f)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _drop_incomplete_tail(f: Any) -> None:
    size = f.seek(0, os.SEEK_END)
    if size == 0:
        return
    f.seek(size - 1)
    if f.read(1) == b"\n":
        return
    f.seek(0)
    content = f.read()
    keep = content.rfind(b"\n") + 1
    tail = content[keep:]
    try:
        Record.model_validate_json(tail)
    except ValidationError:
        f.truncate(keep)
    else:
        f.write(b"\n")
    f.seek(0, os.SEEK_END)


def load_records(run_dir: Path) -> list[Record]:
    """Loads all records. A run without records yields an empty list.

    An unparseable last line without a line terminator is the trace of a killed writer and is skipped; any other
    malformed line raises ValueError.
    """
    path = run_dir / RECORDS_FILE
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return []
    lines = content.split(b"\n")
    # A complete file ends with "\n", so the final element is empty; otherwise it is an unterminated tail.
    tail = lines.pop()
    records: list[Record] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(Record.model_validate_json(line))
        except ValidationError as e:
            raise ValueError(f"{path}: line {number} is not a valid record: {e}") from e
    if tail.strip():
        try:
            records.append(Record.model_validate_json(tail))
        except ValidationError:
            pass
    return records


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Writes `data` to `path` so that readers see either the old or the new file, never a partial one."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(REPLACE_RETRY_DELAY_S)
    finally:
        tmp.unlink(missing_ok=True)


def write_run_info(run_dir: Path, info: RunInfo) -> None:
    write_bytes_atomic(run_dir / RUN_INFO_FILE, info.model_dump_json(indent=2).encode("utf-8"))


def load_run_info(run_dir: Path) -> RunInfo:
    return RunInfo.model_validate_json((run_dir / RUN_INFO_FILE).read_bytes())


def run_id_profile(run_id: str) -> str | None:
    """The profile part of a run id, or None for a name that is no run id."""
    match = RUN_ID.match(run_id)
    return match["profile"] if match else None
