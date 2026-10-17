"""The runner: executes a run's plan against a server it launches, and writes records, traces and the report.

One server lifetime: preflight -> scratch root and server -> environment -> models and capabilities -> input uploads
-> graph export -> per family: empty the model cache, then per case key the warm-ups and measures -> stop the server.
The GPU lock is held throughout and the report is written at the end. Every new server lifetime (a resumed run, or a
restart after the server crashed in a step) skips steps that already have a final record and repeats the last
warm-up of every case key with measures left, because the new server starts with a cold model cache.

A calibration run also checks that the server can calibrate, submits each step through the working-memory calibrate
route, and saves the lifetime's calibration store before the scratch root goes (`calibration.py`).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import psutil
from PIL import Image

from scripts.family_e2e import calibration, gpu_memory
from scripts.family_e2e.client import (
    ApiError,
    EventRecorder,
    InvokeClient,
    build_timing,
    wait_for_idle_queue,
    wait_for_item,
)
from scripts.family_e2e.graphs import (
    PreparedGraph,
    build_export_request,
    output_image_name,
    prepare_batch,
    run_pnpm_exporter,
)
from scripts.family_e2e.manifest import (
    DEFAULT_ARM,
    TOOL_CHECKOUT,
    Manifest,
    ResolvedCase,
    load_manifest,
    plan_steps,
    resolve_cases,
)
from scripts.family_e2e.records import (
    Environment,
    ForeignProcess,
    Memory,
    NodeMemory,
    PlannedStep,
    Record,
    RunInfo,
    Timing,
    append_record,
    load_records,
    load_run_info,
    write_bytes_atomic,
    write_run_info,
)
from scripts.family_e2e.report import write_report
from scripts.family_e2e.residency import ResidencyTracker, parse_log
from scripts.family_e2e.server import (
    ServerError,
    ServerProcess,
    build_scratch_root,
    free_port,
    git_state,
    launch_server,
    pending_installs,
    read_template_settings,
    remove_tree,
    template_server_running,
)

logger = logging.getLogger("family_e2e")

HEARTBEAT_FILE = "heartbeat.json"
HEARTBEAT_INTERVAL_S = 30.0
# Machine-wide, not per data directory: two data directories can share a GPU.
LOCK_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "invokeai-family-e2e" / "locks"
MAX_ATTEMPTS = 3
# A case whose steps crashed the server this often is given up; its remaining steps count as final.
MAX_SERVER_CRASHES = 2
QUEUE_DRAIN_TIMEOUT_S = 600.0
KILL_WAIT_S = 30.0
MIB = 2**20
GIB = 2**30
# Thresholds on PDH "Dedicated Usage" (process_usage); never mixed with the sampler's D3DKMT values.
FOREIGN_BYTES = GIB
FOREIGN_GROWTH_BYTES = 256 * MIB
SHARED_GROWTH_BYTES = 256 * MIB
CANCEL_WAIT_S = 120.0
# Reasons that make a run repeat (up to MAX_ATTEMPTS records per step); the generation itself was fine.
FOREIGN_GPU_LOAD = "foreign GPU load"
FOREIGN_QUEUE_ITEM = "foreign queue item"
SERVER_CRASHED = "server crashed"


class RunnerError(RuntimeError):
    pass


class ServerCrashed(Exception):
    """The server died during a step; the step has its error record and the session restarts the server."""


class Sampler(Protocol):
    def start(self) -> None: ...
    def stop(self) -> gpu_memory.MemoryTrace: ...


@dataclass(frozen=True)
class Gpu:
    """The adapter under test and how to read memory on it; tests substitute fakes for the WDDM implementation."""

    adapter: gpu_memory.Adapter
    process_usage: Callable[[], dict[int, gpu_memory.ProcessUsage]]
    sampler: Callable[[int], Sampler]


def open_gpu(description: str) -> Gpu:
    try:
        adapter = gpu_memory.find_adapter(description)
    except LookupError as e:
        raise RunnerError(str(e)) from e
    return Gpu(
        adapter=adapter,
        process_usage=lambda: gpu_memory.process_usage_on_adapter(adapter),
        sampler=lambda pid: gpu_memory.ProcessMemorySampler(pid, adapter),
    )


Exporter = Callable[[dict[str, Any], Path], dict[str, Any]]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step_id(step: PlannedStep | Record) -> tuple[str, str, str, int]:
    return (step.case_key, step.arm, step.kind, step.rep)


def needs_retry(record: Record) -> bool:
    """Whether foreign load spoiled a completed run. A calibration step is not repeated: the server keeps every window
    it measured, and the builder of shipped data takes the largest."""
    return (
        record.status == "completed"
        and record.calibration is None
        and any(reason.startswith((FOREIGN_GPU_LOAD, FOREIGN_QUEUE_ITEM)) for reason in record.invalid_reasons)
    )


def crashed_server(record: Record) -> bool:
    return any(reason.startswith(SERVER_CRASHED) for reason in record.invalid_reasons)


def remaining_steps(plan: list[PlannedStep], records: list[Record]) -> list[PlannedStep]:
    """Planned steps without a final record, in plan order.

    A step is final when its last record completed and needs no retry, or when its retries are used up. Every step of
    a case key whose steps crashed the server MAX_SERVER_CRASHES times is final too, so a crashing case cannot keep a
    run (or its resumes) restarting the server forever.
    """
    attempts: dict[tuple[str, str, str, int], list[Record]] = {}
    crashes: dict[str, int] = {}
    for record in records:
        attempts.setdefault(_step_id(record), []).append(record)
        crashes[record.case_key] = crashes.get(record.case_key, 0) + crashed_server(record)
    remaining = []
    for step in plan:
        own = sorted(attempts.get(_step_id(step), []), key=lambda r: r.attempt)
        last = own[-1] if own else None
        done = last is not None and last.status == "completed" and (not needs_retry(last) or len(own) >= MAX_ATTEMPTS)
        if not done and crashes.get(step.case_key, 0) < MAX_SERVER_CRASHES:
            remaining.append(step)
    return remaining


@contextmanager
def gpu_lock(adapter: gpu_memory.Adapter) -> Iterator[None]:
    """Holds `LOCK_DIR/<luid>.lock` so a second runner on the same GPU fails fast."""
    path = LOCK_DIR / f"{adapter.luid:x}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        try:
            _lock_file(f, lock=True)
        except OSError:
            raise RunnerError(f"another runner is using {adapter.description} (lock {path})") from None
        try:
            yield
        finally:
            _lock_file(f, lock=False)


def _lock_file(f: BinaryIO, lock: bool) -> None:
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f, (fcntl.LOCK_EX | fcntl.LOCK_NB) if lock else fcntl.LOCK_UN)


def process_identity(pid: int) -> list[float] | None:
    """[pid, create time]: tells a live process from a later one that reuses its pid."""
    try:
        return [pid, psutil.Process(pid).create_time()]
    except psutil.Error:
        return None


def is_alive(identity: list[float] | None) -> bool:
    if not identity:
        return False
    try:
        return abs(psutil.Process(int(identity[0])).create_time() - identity[1]) < 1.0
    except psutil.Error:
        return False


def kill_identified(identity: list[float] | None) -> bool:
    """Kills the process named by [pid, create time] and its descendants, if that process still runs.

    Descendants come from psutil, which drops processes older than their supposed parent; a raw parent-pid walk
    (taskkill /T) can follow a reused pid into an unrelated tree.
    """
    if not identity:
        return False
    try:
        root = psutil.Process(int(identity[0]))
        if abs(root.create_time() - identity[1]) >= 1.0:
            return False
        tree = [root, *root.children(recursive=True)]
    except psutil.Error:
        return False
    for process in tree:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(tree, timeout=KILL_WAIT_S)
    return True


def scratch_root(data_dir: Path, run_id: str, arm: str) -> Path:
    return data_dir / "scratch-roots" / f"{run_id}-{arm}"


def remove_scratch_roots(data_dir: Path, info: RunInfo) -> list[str]:
    """Deletes the scratch roots of the run's arms; returns the paths that could not be deleted."""
    leftovers: list[str] = []
    for arm in sorted({DEFAULT_ARM, *info.environments}):
        leftovers.extend(remove_tree(scratch_root(data_dir, info.run_id, arm)))
    return leftovers


def read_heartbeat(run_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((run_dir / HEARTBEAT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _process_name(pid: int) -> str | None:
    try:
        return psutil.Process(pid).name()
    except psutil.Error:
        return None


def foreign_processes(
    before: dict[int, gpu_memory.ProcessUsage], after: dict[int, gpu_memory.ProcessUsage], own: set[int]
) -> list[ForeignProcess]:
    """Other processes above FOREIGN_BYTES at either end of a window, or grown by more than FOREIGN_GROWTH_BYTES."""
    foreign = []
    for pid in sorted((set(before) | set(after)) - own):
        start = before[pid].dedicated_bytes if pid in before else 0
        end = after[pid].dedicated_bytes if pid in after else 0
        if max(start, end) > FOREIGN_BYTES or end - start > FOREIGN_GROWTH_BYTES:
            foreign.append(ForeignProcess(pid=pid, name=_process_name(pid), dedicated_bytes=max(start, end)))
    return foreign


def _describe(processes: list[ForeignProcess]) -> str:
    return ", ".join(f"pid {p.pid} ({p.name or '?'}) {p.dedicated_bytes / MIB:.0f} MiB" for p in processes)


def preflight(manifest: Manifest, gpu: Gpu) -> None:
    """Refuses to run next to the template's dev server or next to other GPU load."""
    spec = manifest.server
    for path in (spec.python, spec.code_dir / "scripts" / "invokeai-web.py"):
        if not path.is_file():
            raise RunnerError(f"{path} does not exist")
    template_root = spec.template_root
    settings = read_template_settings(template_root)
    pending = pending_installs(template_root, settings)
    if pending:
        raise RunnerError(
            f"the models directory of {template_root} holds unfinished installs "
            f"({', '.join(p.name for p in pending)}); finish or cancel the pending model install in the dev server first"
        )
    if template_server_running(settings):
        raise RunnerError(
            f"something listens on port {settings.get('port', 9090)}, the port of the template root {template_root}; "
            "stop its server first (the tool copies its database)"
        )
    usage = gpu.process_usage()
    # A server that just crashed can linger in the counters for a moment.
    busy = [p for p in foreign_processes(usage, usage, own=set()) if psutil.pid_exists(p.pid)]
    if busy:
        raise RunnerError(f"other processes use more than 1 GiB on {gpu.adapter.description}: {_describe(busy)}")


def uses_fp8(model_keys: frozenset[str], configs: dict[str, dict[str, Any]], fp8_compute: bool) -> bool:
    """Whether fp8 weights may be in play, which makes a partially resident run irreproducible.

    `fp8_storage` in a model's default settings keeps its weights in fp8. With the runtime setting `fp8_compute`,
    the server keeps any scaled-fp8 checkpoint packed and runs fp8 matmuls only for layers already in VRAM; model
    configs do not record whether a checkpoint is scaled fp8, so every graph counts then.
    """
    return fp8_compute or any(
        ((configs.get(key) or {}).get("default_settings") or {}).get("fp8_storage") for key in model_keys
    )


def write_trace(path: Path, trace: gpu_memory.MemoryTrace, t_enqueue: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "t", "bytes"])
        for series, samples in (("dedicated", trace.dedicated), ("shared", trace.shared)):
            writer.writerows((series, f"{t - t_enqueue:.6f}", value) for t, value in samples)


def _sample_at(samples: tuple[tuple[float, int], ...], t: float) -> int | None:
    before = [value for ts, value in samples if ts <= t]
    return before[-1] if before else None


def build_memory(
    trace: gpu_memory.MemoryTrace,
    timing: Timing,
    t_enqueue: float,
    t_done: float,
    foreign: list[ForeignProcess],
    trace_path: str,
) -> Memory:
    nodes = []
    for node in timing.nodes:
        start = t_enqueue + node.t_start
        end = t_enqueue + (node.t_end if node.t_end is not None else timing.total_s)
        nodes.append(
            NodeMemory(
                node_id=node.node_id, baseline_bytes=trace.at(start), peak_bytes=trace.peak(start, max(start, end))
            )
        )
    shared_start = _sample_at(trace.shared, t_enqueue)
    shared_window = [value for ts, value in trace.shared if t_enqueue <= ts <= t_done]
    return Memory(
        method=trace.method,
        interval_ms=trace.interval_ms,
        process_baseline_bytes=trace.at(t_enqueue),
        process_peak_bytes=trace.peak(t_enqueue, t_done),
        nodes=nodes,
        shared_max_delta_bytes=None if shared_start is None else max([shared_start, *shared_window]) - shared_start,
        foreign=foreign,
        trace=trace_path,
    )


def new_run(manifest_path: Path, families: list[str], case_ids: list[str], calibrate: bool = False) -> Path:
    """Creates the run directory and its run.json with the expanded plan. A calibration run submits its steps through
    the working-memory calibrate route (`calibration.py`)."""
    manifest = load_manifest(manifest_path)
    cases = resolve_cases(manifest, families, case_ids)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{manifest.profile}"
    run_dir = manifest.data_dir / "runs" / run_id
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError:
        raise RunnerError(f"run {run_id} exists already; start again in a second") from None
    info = RunInfo(
        run_id=run_id,
        profile=manifest.profile,
        created_at=now_iso(),
        finished_at=None,
        status="running",
        manifest_path=str(manifest_path.resolve()),
        manifest_sha256=file_sha256(manifest_path),
        tool_commit=git_state(TOOL_CHECKOUT)[0],
        environments={},
        plan=plan_steps(cases, manifest.defaults.warmup, manifest.defaults.repeats),
        case_thresholds={case.case_key: case.thresholds for case in cases},
        graph_hashes={},
        calibrate=calibrate,
    )
    write_run_info(run_dir, info)
    return run_dir


def load_run_manifest(info: RunInfo) -> Manifest:
    path = Path(info.manifest_path)
    if file_sha256(path) != info.manifest_sha256:
        raise RunnerError(f"{path} changed since run {info.run_id} started; start a new run")
    return load_manifest(path)


def _set_status(run_dir: Path, status: str) -> None:
    info = load_run_info(run_dir)
    write_run_info(run_dir, info.model_copy(update={"status": status, "finished_at": now_iso()}))


def run(run_dir: Path, gpu_factory: Callable[[str], Gpu] = open_gpu, export: Exporter = run_pnpm_exporter) -> int:
    """Executes the remaining steps of a run, then writes the report. Returns 1 when the report has failures."""
    info = load_run_info(run_dir)
    manifest = load_run_manifest(info)
    try:
        gpu = gpu_factory(manifest.gpu)
        with gpu_lock(gpu.adapter):
            _Session(run_dir, manifest, gpu, export).run()
    except BaseException as e:
        _set_status(run_dir, "stopped" if isinstance(e, KeyboardInterrupt) else "failed")
        try:
            write_report(manifest.data_dir, run_dir)
        except Exception:
            logger.exception("writing the report of the aborted run failed")
        raise
    _set_status(run_dir, "completed")
    result = write_report(manifest.data_dir, run_dir)
    logger.info(
        "report %s: %d failures, %d outliers, %d invalid",
        result.report_html,
        result.failures,
        result.outliers,
        result.invalid,
    )
    return 1 if result.failures else 0


class _Session:
    """One server lifetime of a run."""

    def __init__(self, run_dir: Path, manifest: Manifest, gpu: Gpu, export: Exporter) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.gpu = gpu
        self.export = export
        self.info = load_run_info(run_dir)
        self.arm = DEFAULT_ARM
        self.records = load_records(run_dir)
        self.tracker = ResidencyTracker()
        self.server: ServerProcess | None = None
        self.step_label = "starting"
        self.last_beat = 0.0
        # Per server lifetime: the device classes its calibration runs were created for, and whether their stores
        # were saved through the export route.
        self.device_class_ids: set[str] = set()
        self.store_saved = False
        all_cases = {case.case_key: case for case in resolve_cases(manifest)}
        self.cases = {step.case_key: all_cases[step.case_key] for step in self.info.plan}

    def heartbeat(self, step_label: str | None = None) -> None:
        if step_label is not None:
            self.step_label = step_label
        data = {
            "runner": process_identity(os.getpid()),
            "server": process_identity(self.server.pid) if self.server else None,
            "server_launcher": process_identity(self.server.process.pid) if self.server else None,
            "step": self.step_label,
            "updated_at": now_iso(),
        }
        write_bytes_atomic(self.run_dir / HEARTBEAT_FILE, json.dumps(data, indent=2).encode("utf-8"))
        self.last_beat = time.perf_counter()

    def tick(self) -> None:
        if time.perf_counter() - self.last_beat >= HEARTBEAT_INTERVAL_S:
            self.heartbeat()

    def run(self) -> None:
        self.heartbeat("preflight")
        todo = remaining_steps(self.info.plan, self.records)
        if not todo:
            logger.info("every planned step has a final record")
        # Each crash adds a crash record to a case, and MAX_SERVER_CRASHES of them make the case final: this ends.
        while todo:
            preflight(self.manifest, self.gpu)
            if not self.serve_once(todo):
                return
            todo = remaining_steps(self.info.plan, self.records)

    def serve_once(self, todo: list[PlannedStep]) -> bool:
        """One server lifetime over `todo`. Returns True when the server crashed in a step."""
        root = scratch_root(self.manifest.data_dir, self.info.run_id, self.arm)
        log_path = self.run_dir / f"server-{self.arm}.log"
        self.device_class_ids, self.store_saved = set(), False
        try:
            build_scratch_root(self.manifest.server.template_root, root, log_path, free_port())
            self.heartbeat("starting the server")
            self.server = launch_server(
                self.manifest.server, root, log_path, self.run_dir / f"server-{self.arm}.console.log"
            )
            try:
                self.heartbeat("server ready")
                self.serve(self.server, todo)
            except ServerCrashed as e:
                logger.error("%s; starting a new server", e)
                return True
            finally:
                self.server.stop()
                self.server = None
        finally:
            # stop() returns once the server tree has exited, so its database files are closed by now.
            if self.info.calibrate and not self.store_saved:
                try:
                    calibration.save_store_files(self.run_dir, self.info, root)
                except OSError:
                    logger.exception("could not save the calibration store of %s", root)
            leftovers = remove_tree(root)
            if leftovers:
                logger.warning("could not remove the scratch root %s: %s", root, ", ".join(leftovers))
        return False

    def serve(self, server: ServerProcess, todo: list[PlannedStep]) -> None:
        client = InvokeClient(server.base_url)
        recorder = EventRecorder(server.base_url)
        recorder.connect()
        try:
            runtime_config = self.record_environment(server, client)
            if self.info.calibrate:
                calibration.check_available(client)
            models = client.get("/api/v2/models/")["models"]
            capabilities = client.get("/api/v2/models/capabilities")
            configs = {model["key"]: model for model in models}
            todo_cases = [self.cases[key] for key in dict.fromkeys(step.case_key for step in todo)]
            image_names, input_ids = self.upload_inputs(client, todo_cases)
            graphs, errors = self.export_graphs(todo_cases, models, capabilities, image_names, input_ids)
            fp8_compute = bool(runtime_config.get("config", {}).get("fp8_compute"))

            adapter_checked = False
            for family in dict.fromkeys(step.family for step in todo):
                client.post("/api/v2/models/empty_model_cache")
                self.tracker.reset()
                family_steps = [step for step in todo if step.family == family]
                for case_key in dict.fromkeys(step.case_key for step in family_steps):
                    steps = [step for step in family_steps if step.case_key == case_key]
                    if case_key in errors:
                        for step in steps:
                            self.store(self.error_record(step, f"graph export failed: {errors[case_key]}"))
                        continue
                    if steps[0].kind == "measure":
                        steps = [self.last_warmup(case_key), *steps]
                    graph = graphs[case_key]
                    fp8 = uses_fp8(graph.model_keys, configs, fp8_compute)
                    for step in steps:
                        record = self.execute_step(step, graph, fp8, client, recorder, server)
                        if not adapter_checked and record.status == "completed":
                            self.check_server_on_adapter(server)
                            adapter_checked = True
            if self.device_class_ids:
                calibration.save_exports(client, self.run_dir, self.info, self.device_class_ids)
                self.store_saved = True
        finally:
            recorder.close()

    def record_environment(self, server: ServerProcess, client: InvokeClient) -> dict[str, Any]:
        try:
            runtime_config = client.get("/api/v1/app/runtime_config")
        except ApiError as e:
            raise RunnerError(f"cannot read the runtime config ({e}); the scratch server must run single-user") from e
        if runtime_config.get("config", {}).get("multiuser"):
            raise RunnerError("the template root enables multiuser; the tool needs a single-user server")
        spec = self.manifest.server
        commit, branch, dirty = git_state(spec.code_dir)
        environment = Environment(
            code_dir=str(spec.code_dir),
            git_commit=commit,
            git_branch=branch,
            git_dirty=dirty,
            python=str(spec.python),
            invokeai_file=server.invokeai_file,
            app_version=client.get("/api/v1/app/version"),
            app_deps=client.get("/api/v1/app/app_deps"),
            runtime_config=runtime_config,
            gpu_description=self.gpu.adapter.description,
            gpu_luid=self.gpu.adapter.luid,
            gpu_driver=self.gpu.adapter.driver_version,
            server_cmd=server.cmd,
            extra_env=dict(spec.env),
        )
        previous = self.info.environments.get(self.arm)
        # Edits to an already dirty checkout cannot be told apart; a commit or a dirty flag that changed can.
        if previous is not None and (previous.git_commit, previous.git_dirty, previous.app_version) != (
            environment.git_commit,
            environment.git_dirty,
            environment.app_version,
        ):
            raise RunnerError(
                f"the server code changed since run {self.info.run_id} started (commit {previous.git_commit}, dirty "
                f"{previous.git_dirty} -> commit {environment.git_commit}, dirty {environment.git_dirty}); "
                "start a new run"
            )
        self.info.environments[self.arm] = environment
        write_run_info(self.run_dir, self.info)
        return runtime_config

    def upload_inputs(
        self, client: InvokeClient, cases: list[ResolvedCase]
    ) -> tuple[dict[str, dict[tuple[str, bool], str]], dict[str, str]]:
        """Uploads every input once per size it is needed at.

        Returns case key -> (manifest path, used at the case size) -> image name, and image name -> a stable identity
        of the input. Init and control images enter the canvas compiler as finished layers, which the UI always
        composites at the generation size; reference images are used as they are.
        """
        uploaded: dict[tuple[str, tuple[int, int] | None], str] = {}
        input_ids: dict[str, str] = {}
        names_by_case: dict[str, dict[tuple[str, bool], str]] = {}
        for case in cases:
            spec = case.spec
            usages = [(path, False) for path in spec.reference_images] + [(c.image, True) for c in spec.control]
            if spec.init_image is not None:
                usages.append((spec.init_image.path, True))
            names: dict[tuple[str, bool], str] = {}
            for relative, at_case_size in usages:
                key = (relative, (case.width, case.height) if at_case_size else None)
                if key not in uploaded:
                    path, identity = self.input_at_size(relative, key[1])
                    uploaded[key] = client.upload_image(path)
                    input_ids[uploaded[key]] = identity
                names[(relative, at_case_size)] = uploaded[key]
            names_by_case[case.case_key] = names
        return names_by_case, input_ids

    def input_at_size(self, relative: str, size: tuple[int, int] | None) -> tuple[Path, str]:
        """The input file to upload and its identity; a LANCZOS copy in the run directory when the size differs."""
        source = self.manifest.input_file(relative)
        identity = f"input:{file_sha256(source)}"
        with Image.open(source) as image:
            if size is None or image.size == size:
                return source, identity
            original = image.size
            resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        target = (
            self.run_dir
            / "inputs"
            / f"{Path(relative).with_suffix('').as_posix().replace('/', '_')}@{size[0]}x{size[1]}.png"
        )
        target.parent.mkdir(exist_ok=True)
        resized.save(target)
        logger.info("input %s is %dx%d; uploading a resized copy %s", relative, *original, target)
        return target, f"{identity}@{size[0]}x{size[1]}"

    def export_graphs(
        self,
        cases: list[ResolvedCase],
        models: list[dict[str, Any]],
        capabilities: Any,
        image_names: dict[str, dict[tuple[str, bool], str]],
        input_ids: dict[str, str],
    ) -> tuple[dict[str, PreparedGraph], dict[str, str]]:
        self.heartbeat("exporting graphs")
        response = self.export(build_export_request(cases, models, capabilities, image_names), self.run_dir / "export")
        entries = {entry["id"]: entry for entry in response["cases"]}
        graphs: dict[str, PreparedGraph] = {}
        errors: dict[str, str] = {}
        graph_dir = self.run_dir / "graphs"
        graph_dir.mkdir(exist_ok=True)
        for case in cases:
            entry = entries.get(case.case_key)
            if entry is None or "error" in entry:
                errors[case.case_key] = entry["error"] if entry else "the exporter returned no entry"
                logger.error("%s: %s", case.case_key, errors[case.case_key])
                continue
            graph = prepare_batch(entry["batch"], entry.get("components", {}), input_ids)
            graphs[case.case_key] = graph
            (graph_dir / f"{case.case_key}.json").write_text(json.dumps(graph.batch, indent=2), encoding="utf-8")
        changed = [
            key
            for key, graph in graphs.items()
            if self.info.graph_hashes.get(key, graph.graph_hash) != graph.graph_hash
        ]
        if changed:
            raise RunnerError(f"graphs changed since run {self.info.run_id} started: {', '.join(changed)}")
        self.info.graph_hashes.update({key: graph.graph_hash for key, graph in graphs.items()})
        write_run_info(self.run_dir, self.info)
        return graphs, errors

    def last_warmup(self, case_key: str) -> PlannedStep:
        return [s for s in self.info.plan if s.case_key == case_key and s.kind == "warmup"][-1]

    def check_server_on_adapter(self, server: ServerProcess) -> None:
        usage = self.gpu.process_usage().get(server.pid)
        if usage is None or usage.dedicated_bytes == 0:
            raise RunnerError(
                f"the server (pid {server.pid}) has no memory on {self.gpu.adapter.description}: it runs on another "
                "device; check CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES in the manifest's server env"
            )

    def store(self, record: Record) -> None:
        append_record(self.run_dir, record)
        self.records.append(record)
        peak = record.memory.process_peak_bytes if record.memory else None
        logger.info(
            "%s: %s%s%s%s",
            record.record_id,
            record.status,
            f" {record.timing.total_s:.2f}s" if record.timing else "",
            f" peak {peak / MIB:.0f} MiB" if peak is not None else "",
            f" invalid: {'; '.join(record.invalid_reasons)}" if record.invalid_reasons else "",
        )

    def execute_step(
        self,
        step: PlannedStep,
        graph: PreparedGraph,
        fp8: bool,
        client: InvokeClient,
        recorder: EventRecorder,
        server: ServerProcess,
    ) -> Record:
        """Runs a step, repeating it while foreign load spoiled the measurement and attempts are left.

        Raises ServerCrashed, after storing the step's error record, when the server died during the step.
        """
        attempts = [r for r in self.records if _step_id(r) == _step_id(step)]
        while True:
            attempt = max((r.attempt for r in attempts), default=0) + 1
            self.heartbeat(f"{step.case_key} {step.kind} {step.rep} attempt {attempt}")
            try:
                record = self.run_once(step, attempt, graph, fp8, client, recorder, server)
            except Exception as e:
                if server.alive():
                    raise
                reason = f"{SERVER_CRASHED} (exit code {server.process.poll()}): {type(e).__name__}: {e}"
                fields = self.base_record(step, attempt, graph.graph_hash, fp8)
                fields.update(error=reason, invalid_reasons=[reason])
                self.store(Record(**fields))
                raise ServerCrashed(f"{step.case_key} {step.kind} {step.rep}: {reason}") from e
            self.store(record)
            attempts.append(record)
            if not needs_retry(record) or len(attempts) >= MAX_ATTEMPTS:
                return record

    def base_record(self, step: PlannedStep, attempt: int, graph_hash: str, fp8: bool) -> dict[str, Any]:
        case = self.cases[step.case_key]
        return {
            "record_id": f"{step.case_key}.{step.arm}.{step.kind}.{step.rep}.{attempt}",
            "run_id": self.info.run_id,
            "arm": step.arm,
            "case_id": step.case_id,
            "case_key": step.case_key,
            "family": step.family,
            "width": case.width,
            "height": case.height,
            "steps": case.spec.steps,
            "seed": case.seed,
            "kind": step.kind,
            "rep": step.rep,
            "attempt": attempt,
            "queue_item_id": None,
            "graph_hash": graph_hash,
            "status": "error",
            "error": None,
            "image_path": None,
            "image_sha256": None,
            "image_width": None,
            "image_height": None,
            "timing": None,
            "memory": None,
            "residency": [],
            "uses_fp8": fp8,
            "valid_timing": False,
            "valid_memory": False,
            "valid_psnr": False,
            "invalid_reasons": [],
        }

    def error_record(self, step: PlannedStep, error: str) -> Record:
        attempt = max((r.attempt for r in self.records if _step_id(r) == _step_id(step)), default=0) + 1
        fields = self.base_record(step, attempt, graph_hash="", fp8=False)
        fields.update(error=error, invalid_reasons=[error])
        return Record(**fields)

    def run_once(
        self,
        step: PlannedStep,
        attempt: int,
        graph: PreparedGraph,
        fp8: bool,
        client: InvokeClient,
        recorder: EventRecorder,
        server: ServerProcess,
    ) -> Record:
        fields = self.base_record(step, attempt, graph.graph_hash, fp8)
        reasons: list[str] = []

        def watch() -> None:
            # Checked before every HTTP poll: a dead server's port refuses connections only after seconds on Windows.
            if not server.alive():
                raise ServerError(f"the server process exited with code {server.process.poll()}")
            self.tick()

        if not wait_for_idle_queue(client, QUEUE_DRAIN_TIMEOUT_S, watch):
            reasons.append(f"{FOREIGN_QUEUE_ITEM}: the queue was still busy after {QUEUE_DRAIN_TIMEOUT_S:.0f}s")
        usage_before = self.gpu.process_usage()
        sampler = self.gpu.sampler(server.pid)
        sampler.start()
        try:
            log_offset = server.log_size()
            mark = recorder.mark()
            t_enqueue = time.perf_counter()
            if self.info.calibrate:
                try:
                    item_id, fields["calibration"] = calibration.submit(
                        client, self.cases[step.case_key].spec.model, graph.batch
                    )
                except calibration.CaseRefused as e:
                    fields.update(error=str(e), invalid_reasons=[str(e)])
                    return Record(**fields)
                self.device_class_ids.add(fields["calibration"].device_class_id)
            else:
                item_id = client.enqueue(graph.batch)
            fields["queue_item_id"] = item_id
            try:
                item, t_done = wait_for_item(
                    client, recorder, mark, item_id, self.manifest.defaults.run_timeout_s, watch
                )
            except TimeoutError as e:
                client.cancel(item_id)
                try:
                    wait_for_item(client, recorder, mark, item_id, CANCEL_WAIT_S, watch)
                except TimeoutError:
                    logger.warning("queue item %d is still running after its cancel", item_id)
                fields.update(error=str(e), invalid_reasons=[str(e)])
                return Record(**fields)
        finally:
            trace = sampler.stop()
        usage_after = self.gpu.process_usage()

        status = item["status"]
        fields["status"] = status
        # Only img2img starts inside the schedule, at a step count each family derives differently.
        case = self.cases[step.case_key]
        expected_steps = None if case.spec.mode == "img2img" else case.spec.steps
        timing = build_timing(recorder.since(mark), item_id, t_enqueue, t_done, item, expected_steps)
        residency, residency_reasons = self.tracker.observe(parse_log(server.read_log(log_offset)), graph.model_keys)
        foreign = foreign_processes(usage_before, usage_after, own={server.pid, server.process.pid})
        trace_path = f"traces/{fields['record_id']}.csv.gz"
        write_trace(self.run_dir / trace_path, trace, t_enqueue)
        memory = build_memory(trace, timing, t_enqueue, t_done, foreign, trace_path)
        fields.update(timing=timing, memory=memory, residency=residency)

        completed = status == "completed"
        valid_timing = valid_memory = valid_psnr = completed
        if completed:
            try:
                fields.update(self.save_image(fields["record_id"], graph, item, client))
            except (ValueError, ApiError) as e:
                fields["error"] = f"no output image: {e}"
                reasons.append(fields["error"])
                valid_psnr = False
        else:
            fields["error"] = f"{item.get('error_type')}: {item.get('error_message')}"
            reasons.append(f"generation {status}")
        if residency_reasons:
            reasons.extend(residency_reasons)
            valid_timing = valid_memory = False
            if fp8:
                reasons.append("fp8 weights with incomplete residency do not reproduce")
                valid_psnr = False
        if memory.shared_max_delta_bytes is not None and memory.shared_max_delta_bytes > SHARED_GROWTH_BYTES:
            reasons.append(f"paged to shared memory: +{memory.shared_max_delta_bytes / MIB:.0f} MiB")
            valid_timing = valid_memory = False
        if foreign:
            reasons.append(f"{FOREIGN_GPU_LOAD}: {_describe(foreign)}")
        if any(r.startswith((FOREIGN_GPU_LOAD, FOREIGN_QUEUE_ITEM)) for r in reasons):
            valid_timing = valid_memory = False
        if self.info.calibrate:
            reasons.append(calibration.PROBE_REASON)
            valid_timing = valid_memory = False
        fields.update(
            valid_timing=valid_timing, valid_memory=valid_memory, valid_psnr=valid_psnr, invalid_reasons=reasons
        )
        return Record(**fields)

    def save_image(
        self, record_id: str, graph: PreparedGraph, item: dict[str, Any], client: InvokeClient
    ) -> dict[str, Any]:
        data = client.image_bytes(output_image_name(graph.batch["graph"], item["session"]))
        relative = f"images/{record_id}.png"
        path = self.run_dir / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        return {
            "image_path": relative,
            "image_sha256": hashlib.sha256(data).hexdigest(),
            "image_width": width,
            "image_height": height,
        }
