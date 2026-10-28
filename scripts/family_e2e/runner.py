"""The runner: executes a run's plan against a server it launches, and writes records, traces and the report.

One server lifetime: preflight -> scratch root and server -> environment -> models and capabilities -> input uploads
-> graph export -> per family: empty the model cache, then per case key the warm-ups and measures -> stop the server.
The GPU lock is held throughout and the report is written at the end. A runner first stops a server that an earlier
runner of the run left running (`stop_surviving_server`). Every new server lifetime (a resumed run, or a restart
after the server crashed in a step) skips steps that already have a final record and repeats the last warm-up of
every case key with measures left, because the new server starts with a cold model cache.

`stop` and the runner agree through run.json under the run lock (`run_lock`): the runner names itself in the heartbeat
before anything else (`register_runner`), and `stop` marks the run stopped before it reads the heartbeat. So `stop`
either finds the runner and kills it, or the runner finds the run stopped and leaves it untouched. The runner checks
again before every server and every step (`ensure_running`), and never overwrites a status other than `running`.
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

from scripts.family_e2e import gpu_memory, nvml_memory
from scripts.family_e2e.client import (
    ApiError,
    EventRecorder,
    InvokeClient,
    build_timing,
    wait_for_idle_queue,
    wait_for_item,
)
from scripts.family_e2e.detach import RUNNER_LOG
from scripts.family_e2e.graphs import (
    PreparedGraph,
    build_export_request,
    output_image_name,
    prepare_batch,
    run_pnpm_exporter,
    shareable_source,
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
    ModelIdentity,
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
    kill_group,
    launch_server,
    living_members,
    pending_installs,
    read_template_settings,
    remove_tree,
    template_server_running,
)

logger = logging.getLogger("family_e2e")

HEARTBEAT_FILE = "heartbeat.json"
RUN_LOCK_FILE = "run.lock"
# Holders keep the run lock for one read and one write of a small JSON file.
RUN_LOCK_TIMEOUT_S = 30.0
RUN_LOCK_POLL_S = 0.01
# A started runner registers once its interpreter has imported the tool, which takes seconds on a busy machine.
REGISTER_TIMEOUT_S = 120.0
REGISTER_POLL_S = 0.05
# The heartbeat entries that identify the server's processes. The launcher (the process the runner started, which
# leads the server's process group on POSIX) comes first: its tree and group take the serving process along.
SERVER_ROLES = ("server_launcher", "server")
EXPORT_DIR = "export"
"""The graph exporter's work directory in the run directory."""
HEARTBEAT_INTERVAL_S = 30.0
# Machine-wide, not per data directory: two data directories can share a GPU.
LOCK_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "invokeai-family-e2e" / "locks"
MAX_ATTEMPTS = 3
# A case whose steps crashed the server this often is given up; its remaining steps count as final.
MAX_SERVER_CRASHES = 2
# Windows Subsystem for Linux names its kernel so; NVML there runs on the Windows driver (WDDM).
WSL_KERNEL_RELEASE = Path("/proc/sys/kernel/osrelease")
QUEUE_DRAIN_TIMEOUT_S = 600.0
KILL_WAIT_S = 30.0
KILL_POLL_S = 0.02
MIB = 2**20
GIB = 2**30
# Thresholds on the per-process dedicated usage of process_usage (PDH on Windows, NVML on Linux); never mixed with
# the sampler's trace.
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


class RunStopped(Exception):
    """The run is no longer `running`, normally because `stop` marked it: the runner leaves it as it is."""


class ServerCrashed(Exception):
    """The server died during a step; the step has its error record and the session restarts the server."""


class Sampler(Protocol):
    def start(self) -> None: ...
    def stop(self) -> gpu_memory.MemoryTrace: ...


@dataclass(frozen=True)
class Gpu:
    """The adapter under test and how to read memory on it: `gpu_memory` (WDDM) on Windows, `nvml_memory` on Linux.

    Tests substitute fakes for both."""

    adapter: gpu_memory.Adapter
    process_usage: Callable[[], dict[int, gpu_memory.ProcessUsage]]
    sampler: Callable[[int], Sampler]


def open_gpu(description: str) -> Gpu:
    if sys.platform == "win32":
        backend = gpu_memory
    elif sys.platform == "linux":
        if _under_wsl():
            raise gpu_memory.UnsupportedPlatformError(
                "this Linux runs under WSL, where NVML reports no per-process memory (the Windows driver owns the GPU); "
                "run the tool on Windows on this machine"
            )
        backend = nvml_memory
    else:
        raise gpu_memory.UnsupportedPlatformError(
            f"GPU memory sampling is implemented for Windows (WDDM) and for NVIDIA GPUs on Linux (NVML), not for "
            f"{sys.platform}"
        )
    try:
        adapter = backend.find_adapter(description)
    except (LookupError, OSError) as e:
        raise RunnerError(str(e)) from e
    return Gpu(
        adapter=adapter,
        process_usage=lambda: backend.process_usage_on_adapter(adapter),
        sampler=lambda pid: backend.ProcessMemorySampler(pid, adapter),
    )


def _under_wsl() -> bool:
    try:
        return "microsoft" in WSL_KERNEL_RELEASE.read_text(encoding="utf-8").lower()
    except OSError:
        return False


def device_env(adapter: gpu_memory.Adapter, server_env: dict[str, str]) -> dict[str, str]:
    """What pins the server to the adapter. On Linux CUDA_VISIBLE_DEVICES names the NVML UUID, so a server on a machine
    with several GPUs neither spreads its workers over them nor borrows an idle one; a manifest that pins it to another
    device is refused. WDDM adapters have no such selector: the manifest's env pins the server there.
    """
    if adapter.uuid is None:
        return {}
    pinned = server_env.get("CUDA_VISIBLE_DEVICES")
    if pinned is not None and pinned.strip().casefold() != adapter.uuid.casefold():
        raise RunnerError(
            f"server.env sets CUDA_VISIBLE_DEVICES={pinned}, but gpu selects {adapter.description} ({adapter.uuid}); "
            "drop it from server.env, the runner pins the server to the selected GPU"
        )
    return {"CUDA_VISIBLE_DEVICES": adapter.uuid}


Exporter = Callable[[dict[str, Any], Path], dict[str, Any]]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step_id(step: PlannedStep | Record) -> tuple[str, str, str, int]:
    return (step.case_key, step.arm, step.kind, step.rep)


def needs_retry(record: Record) -> bool:
    return record.status == "completed" and any(
        reason.startswith((FOREIGN_GPU_LOAD, FOREIGN_QUEUE_ITEM)) for reason in record.invalid_reasons
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
    """Holds `LOCK_DIR/<adapter key>.lock` so a second runner on the same GPU fails fast."""
    path = LOCK_DIR / f"{adapter.key}.lock"
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


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Holds `<run_dir>/run.lock`, which orders status changes of run.json and the runner's registration between the
    runner, `stop` and `resume`. Waits while another process holds it; the operating system releases the lock of a
    killed holder."""
    with open(run_dir / RUN_LOCK_FILE, "a+b") as f:
        deadline = time.monotonic() + RUN_LOCK_TIMEOUT_S
        while True:
            try:
                _lock_file(f, lock=True)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RunnerError(f"{run_dir / RUN_LOCK_FILE} stayed locked for {RUN_LOCK_TIMEOUT_S:g}s") from None
                time.sleep(RUN_LOCK_POLL_S)
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


def _identified(identity: list[float] | None) -> psutil.Process | None:
    """The process named by [pid, create time], also one that exited and is not reaped yet (a zombie)."""
    if not identity:
        return None
    try:
        process = psutil.Process(int(identity[0]))
        return process if abs(process.create_time() - identity[1]) < 1.0 else None
    except psutil.Error:
        return None


def _running(process: psutil.Process) -> bool:
    """Whether a process still runs: a zombie has exited and holds no memory, files or port any more."""
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def is_alive(identity: list[float] | None) -> bool:
    process = _identified(identity)
    return process is not None and _running(process)


def _led_group(process: psutil.Process) -> list[int]:
    """The living members of the process group that a captured process leads (POSIX); none when it leads none.

    Until it is reaped a process holds its pid, so a group of that id can only be one it created. That holds for a
    zombie too, whose group can still have living members (macOS does not even tell a zombie's group id).
    is_running() checks the create time, so a pid reused since the process was captured is left alone.
    """
    if sys.platform == "win32" or not process.is_running():
        return []
    return living_members(process.pid)


def _still_running(identity: list[float] | None) -> list[int]:
    """The pids that still run of the process named by `identity`: itself unless it is a zombie, and the living
    members of the group it leads."""
    process = _identified(identity)
    if process is None:
        return []
    return sorted({*([process.pid] if _running(process) else []), *_led_group(process)})


def kill_identified(identity: list[float] | None) -> bool:
    """Kills what still runs of the process named by [pid, create time] and its descendants, and waits until it has
    exited. Returns whether anything ran.

    Descendants come from psutil, which drops processes older than their supposed parent; a raw parent-pid walk
    (taskkill /T) can follow a reused pid into an unrelated tree. On POSIX the groups that processes of the tree lead
    are killed too (the runner and the server launcher each lead one): a descendant whose parent died is in no tree
    any more, but still in its group. A zombie still matches its identity and still leads its group, so orphaned
    members of a zombie leader's group are killed; the zombie itself has nothing left to kill.
    """
    root = _identified(identity)
    if root is None:
        return False
    try:
        tree = [root, *root.children(recursive=True)]
    except psutil.Error:
        return False
    groups = [process.pid for process in tree if _led_group(process)]
    processes = [process for process in tree if _running(process)]
    if not groups and not processes:
        return False
    for pgid in groups:
        try:
            kill_group(pgid)
        except PermissionError:
            pass  # a living member refused the signal; it shows as still running
    for process in processes:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    # Polled rather than psutil.wait_procs, which takes a zombie for running until its parent reaps it.
    deadline = time.monotonic() + KILL_WAIT_S
    while time.monotonic() < deadline and (
        any(_running(process) for process in processes) or any(living_members(pgid) for pgid in groups)
    ):
        time.sleep(KILL_POLL_S)
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


def ensure_running(run_dir: Path) -> None:
    """Raises RunStopped unless run.json still says `running`.

    Under the run lock: Windows refuses to open run.json for a moment while `stop` replaces it, which would end the
    runner as failed instead of stopped.
    """
    with run_lock(run_dir):
        _raise_unless_running(run_dir)


def _raise_unless_running(run_dir: Path) -> None:
    """ensure_running for a caller that holds the run lock: the lock belongs to a file handle, so taking it again would
    wait for itself."""
    status = load_run_info(run_dir).status
    if status != "running":
        raise RunStopped(f"run {run_dir.name} is {status}; the runner leaves it as it is")


def finish_run(run_dir: Path, status: str) -> bool:
    """Ends a running run with `status` and returns True. Any other status stays: the runner does not overwrite a
    stop, and `stop` does not overwrite a finished run."""
    with run_lock(run_dir):
        info = load_run_info(run_dir)
        if info.status != "running":
            return False
        write_run_info(run_dir, info.model_copy(update={"status": status, "finished_at": now_iso()}))
        return True


def register_runner(run_dir: Path) -> None:
    """Names this process as the run's runner in heartbeat.json, before the runner does anything else.

    Under the run lock, as `stop` marks the run stopped: a runner registered first is in the heartbeat that `stop`
    reads next, and one that comes later finds the run stopped (RunStopped). The server entries an earlier runner left
    stay for stop_surviving_server. While another runner of the run is alive this refuses.
    """
    with run_lock(run_dir):
        _raise_unless_running(run_dir)
        heartbeat = read_heartbeat(run_dir)
        other = heartbeat.get("runner")
        if other and int(other[0]) != os.getpid() and is_alive(other):
            raise RunnerError(f"runner pid {int(other[0])} of run {run_dir.name} is still running")
        heartbeat.update(runner=process_identity(os.getpid()), step="starting", updated_at=now_iso())
        write_bytes_atomic(run_dir / HEARTBEAT_FILE, json.dumps(heartbeat, indent=2).encode("utf-8"))


def wait_for_runner(run_dir: Path, spawned_pid: int, previous: list[float] | None) -> list[float]:
    """Waits until a runner other than `previous` registered for the run, and returns its identity.

    `spawned_pid` is the process `start` or `resume` launched: the runner, or a venv launcher whose child it is. Once
    that has exited, a registration that has not appeared will not appear any more.
    """
    try:
        spawned: psutil.Process | None = psutil.Process(spawned_pid)
    except psutil.NoSuchProcess:
        spawned = None
    deadline = time.monotonic() + REGISTER_TIMEOUT_S
    while True:
        # Before the heartbeat is read: a runner that registered and exited at once still counts.
        exited = spawned is None or not _running(spawned)
        registered = read_heartbeat(run_dir).get("runner")
        if registered and registered != previous:
            return registered
        if exited:
            if load_run_info(run_dir).status == "stopped":
                raise RunnerError(f"run {run_dir.name} was stopped before its runner registered")
            raise RunnerError(f"the runner exited before it registered; see {run_dir / RUNNER_LOG}")
        if time.monotonic() >= deadline:
            raise RunnerError(
                f"the runner (pid {spawned_pid}) did not register within {REGISTER_TIMEOUT_S:g}s; see "
                f"{run_dir / RUNNER_LOG}. `python -m scripts.family_e2e stop {run_dir}` keeps it from starting"
            )
        time.sleep(REGISTER_POLL_S)


def stop_surviving_server(run_dir: Path) -> None:
    """Stops a server that an earlier runner of the run left running, before anything replaces what identifies it.

    On POSIX the server runs in a session of its own and survives a runner killed with SIGKILL; on Windows the
    runner's job object ends it with the runner. The heartbeat's identities are all `stop` finds that server by, so
    while a recorded server process cannot be stopped this refuses, before the heartbeat or a scratch root changes.
    Call it under the GPU lock: a live runner of the run would hold it, so the recorded server is no live runner's.
    """
    heartbeat = read_heartbeat(run_dir)
    survivors = {
        int(heartbeat[role][0]): heartbeat[role] for role in SERVER_ROLES if _still_running(heartbeat.get(role))
    }
    refused: dict[int, str] = {}
    for pid, identity in survivors.items():
        running = ", ".join(str(p) for p in _still_running(identity))
        logger.warning("an earlier runner of this run left its server running (pid %d: %s); stopping it", pid, running)
        try:
            kill_identified(identity)
        except psutil.Error as e:
            refused[pid] = f"the kill was refused: {type(e).__name__} {e}"
    left = {pid: running for pid, identity in survivors.items() if (running := _still_running(identity))}
    if left:
        waited = f"{KILL_WAIT_S:g}s after the kill"
        details = ", ".join(
            f"pid {pid} ({refused.get(pid, waited)}; still running: {', '.join(str(p) for p in running)})"
            for pid, running in left.items()
        )
        raise RunnerError(
            f"the server an earlier runner left running cannot be stopped: {details}. heartbeat.json and the scratch "
            f"roots stay as they are; end it, e.g. with `python -m scripts.family_e2e stop {run_dir}`, then resume"
        )


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
    device_env(gpu.adapter, spec.env)
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
    try:
        usage = gpu.process_usage()
    except OSError as e:
        raise RunnerError(f"cannot list the processes on {gpu.adapter.description}: {e}") from e
    # A server that just crashed can linger in the counters for a moment.
    busy = [p for p in foreign_processes(usage, usage, own=set()) if psutil.pid_exists(p.pid)]
    if busy:
        raise RunnerError(
            f"other processes use more than 1 GiB on {gpu.adapter.description}: {_describe(busy)}. One may be the "
            "server of an earlier run whose runner was killed; `python -m scripts.family_e2e stop RUN` ends it"
        )


def uses_fp8(model_keys: frozenset[str], configs: dict[str, dict[str, Any]], fp8_compute: bool) -> bool:
    """Whether fp8 weights may be in play, which makes a partially resident run irreproducible.

    `fp8_storage` in a model's default settings keeps its weights in fp8. With the runtime setting `fp8_compute`,
    the server keeps any scaled-fp8 checkpoint packed and runs fp8 matmuls only for layers already in VRAM; model
    configs do not record whether a checkpoint is scaled fp8, so every graph counts then.
    """
    return fp8_compute or any(
        ((configs.get(key) or {}).get("default_settings") or {}).get("fp8_storage") for key in model_keys
    )


def model_identities(
    model_keys: frozenset[str], main_key: str, configs: dict[str, dict[str, Any]]
) -> list[ModelIdentity]:
    """What identifies a graph's models on another machine, the case's main model first; never their path here.

    Not by type: a graph can hold several main models, e.g. Z-Image's text encoder source.
    """
    identities = []
    for key in model_keys:
        config = configs[key]
        identities.append(
            ModelIdentity(
                key=key,
                name=config["name"],
                base=config["base"],
                type=config["type"],
                format=config.get("format"),
                variant=config.get("variant"),
                hash=config.get("hash"),
                source=shareable_source(config),
            )
        )
    return sorted(identities, key=lambda m: (m.key != main_key, m.type, m.name))


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


def new_run(manifest_path: Path, families: list[str], case_ids: list[str]) -> Path:
    """Creates the run directory and its run.json with the expanded plan."""
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
    )
    write_run_info(run_dir, info)
    return run_dir


def load_run_manifest(info: RunInfo) -> Manifest:
    path = Path(info.manifest_path)
    if file_sha256(path) != info.manifest_sha256:
        raise RunnerError(f"{path} changed since run {info.run_id} started; start a new run")
    return load_manifest(path)


def run(run_dir: Path, gpu_factory: Callable[[str], Gpu] = open_gpu, export: Exporter = run_pnpm_exporter) -> int:
    """Executes the remaining steps of a run, then writes the report. Returns 1 when the report has failures.

    Raises RunStopped, with run.json and the report left as they are, once the run is no longer `running`.
    """
    register_runner(run_dir)
    info = load_run_info(run_dir)
    manifest = load_run_manifest(info)
    try:
        gpu = gpu_factory(manifest.gpu)
        with gpu_lock(gpu.adapter):
            ensure_running(run_dir)
            stop_surviving_server(run_dir)
            _Session(run_dir, manifest, gpu, export).run()
    except RunStopped:
        raise
    except BaseException as e:
        finish_run(run_dir, "stopped" if isinstance(e, KeyboardInterrupt) else "failed")
        try:
            write_report(manifest.data_dir, run_dir)
        except Exception:
            logger.exception("writing the report of the aborted run failed")
        raise
    finish_run(run_dir, "completed")
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
            ensure_running(self.run_dir)
            preflight(self.manifest, self.gpu)
            if not self.serve_once(todo):
                return
            todo = remaining_steps(self.info.plan, self.records)

    def serve_once(self, todo: list[PlannedStep]) -> bool:
        """One server lifetime over `todo`. Returns True when the server crashed in a step."""
        root = scratch_root(self.manifest.data_dir, self.info.run_id, self.arm)
        log_path = self.run_dir / f"server-{self.arm}.log"
        try:
            pin = device_env(self.gpu.adapter, self.manifest.server.env)
            build_scratch_root(self.manifest.server.template_root, root, log_path, free_port(), bool(pin))
            self.heartbeat("starting the server")
            self.server = launch_server(
                self.manifest.server,
                root,
                log_path,
                self.run_dir / f"server-{self.arm}.console.log",
                pin,
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
            models = client.get("/api/v2/models/")["models"]
            capabilities = client.get("/api/v2/models/capabilities")
            configs = {model["key"]: model for model in models}
            todo_cases = [self.cases[key] for key in dict.fromkeys(step.case_key for step in todo)]
            image_names, input_ids = self.upload_inputs(client, todo_cases)
            graphs, errors = self.export_graphs(todo_cases, configs, capabilities, image_names, input_ids)
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
        self.save_info()
        return runtime_config

    def save_info(self) -> None:
        """Writes the session's run.json while the run is still running; `stop` owns it otherwise."""
        with run_lock(self.run_dir):
            _raise_unless_running(self.run_dir)
            write_run_info(self.run_dir, self.info)

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
        configs: dict[str, dict[str, Any]],
        capabilities: Any,
        image_names: dict[str, dict[tuple[str, bool], str]],
        input_ids: dict[str, str],
    ) -> tuple[dict[str, PreparedGraph], dict[str, str]]:
        self.heartbeat("exporting graphs")
        request = build_export_request(cases, list(configs.values()), capabilities, image_names)
        response = self.export(request, self.run_dir / EXPORT_DIR)
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
        self.info.case_models.update(
            {
                key: model_identities(graph.model_keys, self.cases[key].spec.model, configs)
                for key, graph in graphs.items()
            }
        )
        self.save_info()
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
            ensure_running(self.run_dir)
            attempt = max((r.attempt for r in attempts), default=0) + 1
            self.heartbeat(f"{step.case_key} {step.kind} {step.rep} attempt {attempt}")
            try:
                record = self.run_once(step, attempt, graph, fp8, client, recorder, server)
            except Exception as e:
                if server.alive():
                    raise
                reason = f"{SERVER_CRASHED} ({server.end_status()}): {type(e).__name__}: {e}"
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
                raise ServerError(f"the server process exited ({server.end_status()})")
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
