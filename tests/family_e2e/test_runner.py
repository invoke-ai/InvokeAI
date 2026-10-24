import errno
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest
import yaml

from scripts.family_e2e import runner, server
from scripts.family_e2e.__main__ import main
from scripts.family_e2e.gpu_memory import Adapter, MemoryTrace, ProcessUsage, UnsupportedPlatformError
from scripts.family_e2e.manifest import DEFAULT_ARM, Manifest
from scripts.family_e2e.records import (
    NodeTiming,
    PlannedStep,
    Record,
    RunInfo,
    Timing,
    append_record,
    load_records,
    write_run_info,
)
from scripts.family_e2e.runner import (
    FOREIGN_GPU_LOAD,
    HEARTBEAT_FILE,
    SERVER_CRASHED,
    SERVER_ROLES,
    Gpu,
    RunnerError,
    build_memory,
    device_env,
    foreign_processes,
    gpu_lock,
    model_identities,
    preflight,
    remaining_steps,
    uses_fp8,
)

MIB = 2**20
GIB = 2**30
# Waiting for a helper process to report that it is ready.
CHILD_POLL_S = 0.01
CHILD_TIMEOUT_S = 30.0


def _plan() -> list[PlannedStep]:
    return [
        PlannedStep(case_key="a@64x64", case_id="a", family="f", arm="default", kind=kind, rep=rep)
        for kind, rep in (("warmup", 0), ("measure", 0), ("measure", 1), ("measure", 2))
    ]


def _record(step: PlannedStep, attempt: int, status: str = "completed", reasons: tuple[str, ...] = ()) -> Record:
    return Record(
        record_id=f"{step.case_key}.{step.arm}.{step.kind}.{step.rep}.{attempt}",
        run_id="r",
        arm=step.arm,
        case_id=step.case_id,
        case_key=step.case_key,
        family=step.family,
        width=64,
        height=64,
        steps=4,
        seed=1,
        kind=step.kind,
        rep=step.rep,
        attempt=attempt,
        queue_item_id=attempt,
        graph_hash="h",
        status=status,
        error=None,
        image_path=None,
        image_sha256=None,
        image_width=None,
        image_height=None,
        timing=None,
        memory=None,
        residency=[],
        uses_fp8=False,
        valid_timing=not reasons,
        valid_memory=not reasons,
        valid_psnr=True,
        invalid_reasons=list(reasons),
    )


def test_resume_skips_final_steps_and_tolerates_a_truncated_last_line(tmp_path: Path) -> None:
    warmup, m0, m1, m2 = _plan()
    foreign = (f"{FOREIGN_GPU_LOAD}: pid 4 (chrome.exe) 2048 MiB",)
    append_record(tmp_path, _record(warmup, 1))
    append_record(tmp_path, _record(m0, 1, reasons=foreign))  # a retry was due
    append_record(tmp_path, _record(m1, 1, status="failed"))
    with open(tmp_path / "records.jsonl", "ab") as f:
        f.write(_record(m2, 1).model_dump_json().encode()[:40])  # the runner died while writing
    assert [(s.kind, s.rep) for s in remaining_steps(_plan(), load_records(tmp_path))] == [
        ("measure", 0),
        ("measure", 1),
        ("measure", 2),
    ]

    append_record(tmp_path, _record(m0, 2, reasons=foreign))
    append_record(tmp_path, _record(m0, 3, reasons=foreign))  # retries used up: final, though invalid
    append_record(tmp_path, _record(m1, 2))
    assert [(s.kind, s.rep) for s in remaining_steps(_plan(), load_records(tmp_path))] == [("measure", 2)]


def test_a_case_that_keeps_crashing_the_server_is_given_up() -> None:
    warmup, m0, m1, m2 = _plan()
    crash = (f"{SERVER_CRASHED} (exit code 3): ConnectionError: refused",)
    records = [_record(warmup, 1), _record(m0, 1, status="error", reasons=crash)]
    assert [(s.kind, s.rep) for s in remaining_steps(_plan(), records)] == [
        ("measure", 0),
        ("measure", 1),
        ("measure", 2),
    ]
    # The restarted server repeats the warm-up, which crashes it again: the case stops counting as remaining.
    records.append(_record(warmup, 2, status="error", reasons=crash))
    assert remaining_steps(_plan(), records) == []


def test_foreign_processes_are_large_or_growing_others() -> None:
    before = {
        1: ProcessUsage(dedicated_bytes=8 * GIB, shared_bytes=0),  # the server itself
        2: ProcessUsage(dedicated_bytes=2 * GIB, shared_bytes=0),  # large from the start
        3: ProcessUsage(dedicated_bytes=300 * MIB, shared_bytes=0),  # grows by 400 MiB
        4: ProcessUsage(dedicated_bytes=500 * MIB, shared_bytes=0),  # stable and small
    }
    after = dict(before)
    after[3] = ProcessUsage(dedicated_bytes=700 * MIB, shared_bytes=0)
    after[5] = ProcessUsage(dedicated_bytes=200 * MIB, shared_bytes=0)  # new but small
    found = foreign_processes(before, after, own={1})
    assert [(p.pid, p.dedicated_bytes) for p in found] == [(2, 2 * GIB), (3, 700 * MIB)]


def test_fp8_comes_from_storage_settings_or_runtime_compute() -> None:
    configs = {
        "plain": {"default_settings": {"fp8_storage": False}},
        "fp8": {"default_settings": {"fp8_storage": True}},
    }
    assert not uses_fp8(frozenset({"plain", "lora-without-config"}), configs, fp8_compute=False)
    assert uses_fp8(frozenset({"plain", "fp8"}), configs, fp8_compute=False)
    assert uses_fp8(frozenset({"plain"}), configs, fp8_compute=True)


def test_a_cases_models_start_with_its_main_model_even_beside_another_main_model() -> None:
    # Z-Image takes its text encoder from a second diffusers main model.
    configs = {
        key: {"key": key, "name": name, "base": "z-image", "type": model_type, "path": f"S:/{name}"}
        for key, name, model_type in (
            ("vae", "ae", "vae"),
            ("source", "A Z-Image source", "main"),
            ("turbo", "Z-Image Turbo", "main"),
        )
    }

    identities = model_identities(frozenset(configs), "turbo", configs)

    assert [m.name for m in identities] == ["Z-Image Turbo", "A Z-Image source", "ae"]


def test_memory_is_split_per_node_and_shared_growth_is_measured_from_enqueue() -> None:
    t0 = 100.0
    trace = MemoryTrace(
        method="d3dkmt",
        interval_ms=3.0,
        dedicated=((t0 - 0.01, GIB), (t0 + 0.2, 4 * GIB), (t0 + 0.4, 2 * GIB), (t0 + 0.6, 6 * GIB), (t0 + 0.9, GIB)),
        shared=((t0 - 0.05, 100 * MIB), (t0 + 0.5, 500 * MIB), (t0 + 2.0, 900 * MIB)),
    )
    nodes = [
        NodeTiming(node_id="encode", type="e", t_start=0.1, t_end=0.3),
        NodeTiming(node_id="denoise", type="d", t_start=0.5, t_end=None),
    ]
    timing = Timing(
        total_s=1.0, nodes=nodes, model_loads=[], per_step_s=None, queue_started_at=None, queue_completed_at=None
    )
    memory = build_memory(trace, timing, t0, t0 + 1.0, [], "traces/x.csv.gz")
    assert (memory.process_baseline_bytes, memory.process_peak_bytes) == (GIB, 6 * GIB)
    assert [(n.node_id, n.baseline_bytes, n.peak_bytes) for n in memory.nodes] == [
        ("encode", GIB, 4 * GIB),
        ("denoise", 2 * GIB, 6 * GIB),
    ]
    # The sample after the window does not count.
    assert memory.shared_max_delta_bytes == 400 * MIB


@pytest.mark.parametrize(
    ("adapter", "lock_file"),
    [
        (Adapter(luid=0xABC, description="Fake GPU"), "abc.lock"),
        (Adapter(luid=None, uuid="GPU-5f1c2e3a-1111", description="Fake GPU"), "GPU-5f1c2e3a-1111.lock"),
    ],
    ids=["wddm-luid", "nvml-uuid"],
)
def test_second_runner_on_the_same_gpu_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter: Adapter, lock_file: str
) -> None:
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")
    with gpu_lock(adapter):
        assert [path.name for path in (tmp_path / "locks").iterdir()] == [lock_file]
        with pytest.raises(RunnerError, match="another runner"):
            with gpu_lock(adapter):
                pass
    with gpu_lock(adapter):
        pass


def test_runners_on_two_gpus_of_one_machine_do_not_block_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")
    name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    with gpu_lock(Adapter(luid=None, uuid="GPU-aaaaaaaa", description=name)):
        with gpu_lock(Adapter(luid=None, uuid="GPU-bbbbbbbb", description=name)):
            pass


def test_the_server_is_pinned_to_the_selected_nvml_gpu_and_a_conflicting_pin_is_refused() -> None:
    uuid = "GPU-aaaaaaaa-1111-4000-8000-000000000001"
    nvml = Adapter(luid=None, uuid=uuid, description="NVIDIA RTX PRO 6000 Blackwell Server Edition")

    assert device_env(nvml, {}) == {"CUDA_VISIBLE_DEVICES": uuid}
    assert device_env(nvml, {"CUDA_VISIBLE_DEVICES": uuid.upper()}) == {"CUDA_VISIBLE_DEVICES": uuid}
    with pytest.raises(RunnerError, match="CUDA_VISIBLE_DEVICES=1, but gpu selects .*GPU-aaaaaaaa"):
        device_env(nvml, {"CUDA_VISIBLE_DEVICES": "1"})
    # WDDM adapters have no device selector; the manifest's env pins the server there.
    assert device_env(Adapter(luid=0x22A61, description="NVIDIA GeForce RTX 4090"), {"CUDA_VISIBLE_DEVICES": "1"}) == {}


def test_open_gpu_refuses_a_platform_without_a_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(UnsupportedPlatformError, match="not for darwin"):
        runner.open_gpu("M3")


def test_preflight_refuses_a_template_with_an_unfinished_model_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "PORT_PROBE_TIMEOUT_S", 0.01)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    template = tmp_path / "template"
    (template / "my-models").mkdir(parents=True)
    (template / "invokeai.yaml").write_text(
        yaml.safe_dump({"schema_version": "4.0.3", "port": free, "models_dir": "my-models"})
    )
    code_dir = tmp_path / "code"
    (code_dir / "scripts").mkdir(parents=True)
    (code_dir / "scripts" / "invokeai-web.py").write_text("")
    manifest = Manifest.model_validate(
        {
            "profile": "box",
            "data_dir": str(tmp_path / "data"),
            "gpu": "Fake",
            "server": {"template_root": str(template), "python": sys.executable, "code_dir": str(code_dir)},
            "defaults": {"prompt": "a fox"},
            "families": [{"family": "f", "cases": [{"id": "a", "model": "m", "steps": 1, "sizes": [[8, 8]]}]}],
        }
    )
    gpu = Gpu(adapter=Adapter(luid=1, description="Fake"), process_usage=dict, sampler=lambda pid: None)

    (template / "my-models" / "tmpinstall_k2j3").mkdir()
    with pytest.raises(RunnerError, match="tmpinstall_k2j3.*finish or cancel the pending model install"):
        preflight(manifest, gpu)
    (template / "my-models" / "tmpinstall_k2j3").rmdir()
    preflight(manifest, gpu)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups; on Windows the job object holds the tree")
def test_stop_kills_a_descendant_whose_parent_died(tmp_path: Path) -> None:
    # The launcher's child starts a grandchild and exits: the grandchild is in no tree any more, but in the group
    # the launcher leads, as with a server launched by spawn_in_job.
    grandchild_file, ready = tmp_path / "grandchild.txt", tmp_path / "ready"
    middle = (
        "import subprocess, sys\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"with open({str(grandchild_file)!r}, 'w') as f:\n"
        "    f.write(str(grandchild.pid))\n"
    )
    script = tmp_path / "launcher.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.run([sys.executable, '-c', {middle!r}], check=True)\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(60)\n"
    )
    process, _ = server.spawn_in_job([sys.executable, str(script)])
    grandchild: psutil.Process | None = None
    try:
        deadline = time.monotonic() + CHILD_TIMEOUT_S
        while not ready.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(CHILD_POLL_S)
        grandchild = psutil.Process(int(grandchild_file.read_text()))
        assert grandchild.pid not in {p.pid for p in psutil.Process(process.pid).children(recursive=True)}

        assert runner.kill_identified(runner.process_identity(process.pid))

        _, alive = psutil.wait_procs([grandchild], timeout=CHILD_TIMEOUT_S)
        assert alive == []
    finally:
        if grandchild is not None and grandchild.is_running():
            grandchild.kill()
        process.kill()
        process.wait()


RUN_ID = "20261025-000000-box"


def _fake_gpu(description: str) -> Gpu:
    return Gpu(adapter=Adapter(luid=1, description=description), process_usage=dict, sampler=lambda pid: None)


@pytest.fixture
def killed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The run directory of a runner that was killed: a step is left, and its scratch root is still there.

    The template's port is free, so a new session of the run would pass preflight and go on to rebuild that root.
    """
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    (tmp_path / "template").mkdir()
    (tmp_path / "template" / "invokeai.yaml").write_text(yaml.safe_dump({"schema_version": "4.0.3", "port": free}))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "profile": "box",
                "data_dir": "data",
                "gpu": "Fake",
                "server": {"template_root": "template", "python": sys.executable},
                "defaults": {"prompt": "a fox"},
                "families": [{"family": "f", "cases": [{"id": "a", "model": "m", "steps": 1, "sizes": [[8, 8]]}]}],
            }
        )
    )
    run_dir = tmp_path / "data" / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    step = PlannedStep(case_key="a@8x8", case_id="a", family="f", arm=DEFAULT_ARM, kind="warmup", rep=0)
    write_run_info(
        run_dir,
        RunInfo(
            run_id=RUN_ID,
            profile="box",
            created_at="2026-10-25T00:00:00+00:00",
            finished_at=None,
            status="running",
            manifest_path=str(manifest),
            manifest_sha256=runner.file_sha256(manifest),
            tool_commit=None,
            environments={},
            plan=[step],
            case_thresholds={},
            graph_hashes={},
        ),
    )
    scratch = runner.scratch_root(tmp_path / "data", RUN_ID, DEFAULT_ARM)
    scratch.mkdir(parents=True)
    (scratch / "invokeai.yaml").write_text("left by the killed runner")
    return run_dir


def _scratch_file(run_dir: Path) -> Path:
    return runner.scratch_root(run_dir.parent.parent, RUN_ID, DEFAULT_ARM) / "invokeai.yaml"


def _write_heartbeat(run_dir: Path, **identities: list[float]) -> bytes:
    data = {"runner": None, "server": None, "server_launcher": None, "step": "a@8x8 warmup 0 attempt 1"}
    encoded = json.dumps({**data, **identities, "updated_at": runner.now_iso()}).encode()
    (run_dir / HEARTBEAT_FILE).write_bytes(encoded)
    return encoded


@pytest.fixture
def surviving_server() -> Iterator[list[float]]:
    """The identity of a server whose runner was killed: a process in a group (on Windows a job) of its own."""
    process, job = server.spawn_in_job([sys.executable, "-c", "import time; time.sleep(60)"])
    identity = runner.process_identity(process.pid)
    assert identity is not None
    try:
        yield identity
    finally:
        if job is not None:
            job.close()
        runner.kill_identified(identity)
        process.wait()


class _FirstHeartbeat(Exception):
    pass


@pytest.mark.parametrize("role", SERVER_ROLES)
def test_a_runner_stops_the_server_a_killed_runner_left_before_it_replaces_the_heartbeat(
    killed_run: Path, surviving_server: list[float], monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    # The other entry's pid now belongs to an unrelated process: its start time gives it away.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        reused = [unrelated.pid, psutil.Process(unrelated.pid).create_time() - 100.0]
        other = next(r for r in SERVER_ROLES if r != role)
        _write_heartbeat(killed_run, **{role: surviving_server, other: reused})
        seen: list[tuple[str, bool, bool]] = []

        def first_write(path: Path, data: bytes) -> None:
            # The runner's first write; the scratch root is rebuilt only after it.
            seen.append((path.name, runner.is_alive(surviving_server), _scratch_file(killed_run).is_file()))
            raise _FirstHeartbeat

        monkeypatch.setattr(runner, "write_bytes_atomic", first_write)
        with pytest.raises(_FirstHeartbeat):
            runner.run(killed_run, gpu_factory=_fake_gpu)

        assert seen == [(HEARTBEAT_FILE, False, True)]
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


@pytest.mark.parametrize("failure", ["survives the kill", "kill refused"])
def test_a_runner_refuses_and_keeps_heartbeat_and_scratch_root_while_a_left_server_cannot_be_stopped(
    killed_run: Path, surviving_server: list[float], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    # A process stuck in the GPU driver outlives SIGKILL until it leaves the kernel; another user's refuses it.
    heartbeat = _write_heartbeat(killed_run, server=surviving_server, server_launcher=surviving_server)

    def kill_process(process: psutil.Process) -> None:
        if failure == "kill refused":
            raise psutil.AccessDenied(process.pid)

    def killpg(pgid: int, sig: int) -> None:
        if failure == "kill refused":
            raise PermissionError(errno.EPERM, "Operation not permitted")

    with monkeypatch.context() as m:
        m.setattr(runner, "KILL_WAIT_S", 0.05)
        m.setattr(psutil.Process, "kill", kill_process)
        if sys.platform != "win32":
            m.setattr(os, "killpg", killpg)
        with pytest.raises(RunnerError, match=rf"cannot be stopped: pid {int(surviving_server[0])} \("):
            runner.run(killed_run, gpu_factory=_fake_gpu)

    assert (killed_run / HEARTBEAT_FILE).read_bytes() == heartbeat
    assert _scratch_file(killed_run).read_text() == "left by the killed runner"
    assert runner.is_alive(surviving_server)


def _gone(process: psutil.Process) -> bool:
    try:
        return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@contextmanager
def _unreaped_launcher(tmp_path: Path, orphan: bool) -> Iterator[tuple[list[float], psutil.Process | None]]:
    """A server launcher that exited but is not reaped: its parent lives on without waiting for it, as a container's
    PID 1 that reaps nothing does. With `orphan`, a worker it started runs on in its group. Yields the launcher's
    identity and the worker."""

    def publish(path: Path, value: str) -> str:
        # Written whole and renamed into place, so whatever the name holds once it exists is complete.
        tmp = str(path) + ".tmp"
        return f"with open({tmp!r}, 'w') as f: f.write({value})\nos.replace({tmp!r}, {str(path)!r})\n"

    launcher_file, worker_file = tmp_path / "launcher.txt", tmp_path / "worker.txt"
    launcher_code = "import os, subprocess, sys\n"
    if orphan:
        launcher_code += "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        launcher_code += publish(worker_file, "str(worker.pid)")
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, subprocess, sys, time\n"
            f"launcher = subprocess.Popen([sys.executable, '-c', {launcher_code!r}], start_new_session=True)\n"
            + publish(launcher_file, "str(launcher.pid)")
            + "time.sleep(60)\n",
        ]
    )
    worker: psutil.Process | None = None
    try:
        deadline = time.monotonic() + CHILD_TIMEOUT_S
        while not launcher_file.exists():
            assert parent.poll() is None and time.monotonic() < deadline
            time.sleep(CHILD_POLL_S)
        launcher = psutil.Process(int(launcher_file.read_text()))
        while launcher.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(CHILD_POLL_S)
        if orphan:
            worker = psutil.Process(int(worker_file.read_text()))
            assert worker.status() != psutil.STATUS_ZOMBIE
        yield [launcher.pid, launcher.create_time()], worker
    finally:
        if worker is not None and not _gone(worker):
            worker.kill()
        parent.kill()  # its init reaps the launcher then
        parent.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX zombies; Windows keeps no exited process for its parent")
@pytest.mark.parametrize("orphan", [False, True], ids=["alone", "orphaned-worker"])
@pytest.mark.parametrize("command", ["resume", "stop"])
def test_an_unreaped_server_counts_as_gone_but_a_worker_left_in_its_group_is_killed(
    killed_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    orphan: bool,
) -> None:
    # Bounds what a regression would wait for the zombie to exit.
    monkeypatch.setattr(runner, "KILL_WAIT_S", 1.0)
    with _unreaped_launcher(tmp_path, orphan) as (identity, worker):
        assert not runner.is_alive(identity)
        _write_heartbeat(killed_run, server=identity, server_launcher=identity)
        if command == "resume":
            worker_running: list[bool] = []

            def first_write(path: Path, data: bytes) -> None:
                worker_running.append(worker is not None and not _gone(worker))
                raise _FirstHeartbeat

            monkeypatch.setattr(runner, "write_bytes_atomic", first_write)
            with pytest.raises(_FirstHeartbeat):  # not refused
                runner.run(killed_run, gpu_factory=_fake_gpu)
            assert worker_running == [False]
        else:
            capsys.readouterr()
            assert main(["stop", str(killed_run)]) == 0
            # A zombie alone has nothing to kill; a worker in its group is killed through it.
            assert ("killed server_launcher" in capsys.readouterr().out) == orphan
        assert worker is None or _gone(worker)
