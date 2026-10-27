import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest
import yaml

import scripts.family_e2e.__main__ as cli
from scripts.family_e2e import runner, server
from scripts.family_e2e.__main__ import UsageError, _resolve_run, main
from scripts.family_e2e.gpu_memory import Adapter
from scripts.family_e2e.records import RUN_INFO_FILE, RunInfo, load_run_info, write_run_info
from scripts.family_e2e.runner import HEARTBEAT_FILE, now_iso, process_identity

REPO_ROOT = Path(__file__).parents[2]
CHILD_TIMEOUT_S = 60.0
# The runner `start` launches, on a fake GPU, waiting for a line on stdin at its pause: before it begins
# ("unregistered"), or where it opens the GPU, right after its registration ("registered").
STAND_IN_RUNNER = """
import sys
from pathlib import Path

from scripts.family_e2e import runner
from scripts.family_e2e.gpu_memory import Adapter

run_dir, runner.LOCK_DIR, pause = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def wait(at):
    if pause == at:
        print("paused", flush=True)
        sys.stdin.readline()


def open_gpu(description):
    wait("registered")
    return runner.Gpu(adapter=Adapter(luid=1, description=description), process_usage=dict, sampler=lambda pid: None)


wait("unregistered")
sys.exit(runner.run(run_dir, gpu_factory=open_gpu))
"""


def test_runs_resolve_by_directory_id_or_latest_of_the_profile(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "profile": "box",
                "data_dir": "data",
                "gpu": "RTX",
                "server": {"template_root": "t", "python": "p"},
                "defaults": {"prompt": "a fox"},
                "families": [{"family": "f", "cases": [{"id": "a", "model": "m", "steps": 1, "sizes": [[8, 8]]}]}],
            }
        )
    )
    runs = tmp_path / "data" / "runs"
    for name in ("20261016-090000-box", "20261017-080000-box", "20261018-070000-other-box", "20261019-000000-box"):
        (runs / name).mkdir(parents=True)
        if name != "20261019-000000-box":  # a run whose start failed before run.json was written
            (runs / name / "run.json").write_text("{}")

    assert _resolve_run(None, manifest) == runs / "20261017-080000-box"
    assert _resolve_run("20261016-090000-box", manifest) == runs / "20261016-090000-box"
    assert _resolve_run(str(runs / "20261016-090000-box"), None) == (runs / "20261016-090000-box").resolve()
    with pytest.raises(UsageError, match="no run 20261001-000000-box"):
        _resolve_run("20261001-000000-box", manifest)
    with pytest.raises(UsageError, match="--manifest"):
        _resolve_run("20261016-090000-box", None)


def _start_tree(tmp_path: Path) -> tuple[subprocess.Popen[bytes], list[psutil.Process]]:
    """A runner stand-in: a process that starts a sleeping child (its server), plain Popen, no job."""
    pid_file = tmp_path / "child.txt"
    script = tmp_path / "runner.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file) + '.tmp'!r}, 'w').write(str(child.pid))\n"
        f"os.replace({str(pid_file) + '.tmp'!r}, {str(pid_file)!r})\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, str(script)])
    deadline = time.monotonic() + 30
    while not pid_file.exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(0.01)
    return process, [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]


@pytest.mark.parametrize("first_read", ["readable", "refused"])
def test_stop_kills_the_runner_tree_but_no_process_whose_pid_was_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_read: str
) -> None:
    run_id = "20261017-000000-box"
    run_dir = tmp_path / "data" / "runs" / run_id
    run_dir.mkdir(parents=True)
    write_run_info(
        run_dir,
        RunInfo(
            run_id=run_id,
            profile="box",
            created_at="2026-10-17T00:00:00+00:00",
            finished_at=None,
            status="running",
            manifest_path="m.yaml",
            manifest_sha256="0",
            tool_commit=None,
            environments={},
            plan=[],
            case_thresholds={},
            graph_hashes={},
        ),
    )
    scratch = tmp_path / "data" / "scratch-roots" / f"{run_id}-default"
    scratch.mkdir(parents=True)
    (scratch / "invokeai.yaml").write_text("{}")
    # A run of profile "box-x" started in the same second; its scratch root is not this run's.
    other = tmp_path / "data" / "scratch-roots" / f"{run_id}-x-default"
    other.mkdir()

    runner_process, runner_tree = _start_tree(tmp_path)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        heartbeat = {
            "runner": process_identity(runner_process.pid),
            # The server's recorded pid now belongs to another process: its start time gives it away.
            "server": [unrelated.pid, psutil.Process(unrelated.pid).create_time() - 100.0],
            "server_launcher": None,
            "step": "measure",
            "updated_at": now_iso(),
        }
        (run_dir / HEARTBEAT_FILE).write_text(json.dumps(heartbeat))
        if first_read == "refused":
            # Windows refuses to open the file for a moment while the runner replaces it.
            refusals = [{}]
            monkeypatch.setattr(
                cli, "read_heartbeat", lambda d: refusals.pop() if refusals else runner.read_heartbeat(d)
            )

        assert main(["stop", str(run_dir)]) == 0
        _, alive = psutil.wait_procs(runner_tree, timeout=10)
        assert alive == []
        assert unrelated.poll() is None
        assert load_run_info(run_dir).status == "stopped"
        assert not scratch.exists()
        assert other.exists()
    finally:
        unrelated.kill()
        unrelated.wait()
        runner_process.wait()


@pytest.mark.platform_independent
def test_the_tool_imports_neither_torch_nor_invokeai() -> None:
    # A CUDA context in the client would land on the GPU it measures. In-process checks prove nothing here: the
    # test session has imported both already.
    code = (
        "import sys, scripts.family_e2e.__main__, scripts.family_e2e.runner; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'torch', 'invokeai'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=True
    )
    assert result.stdout.strip() == "[]"


@pytest.fixture
def startable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A manifest that passes `start`'s checks on a fake GPU; its server script is empty and would exit at once."""
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(server, "PORT_PROBE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        cli,
        "open_gpu",
        lambda description: runner.Gpu(
            adapter=Adapter(luid=1, description=description), process_usage=dict, sampler=lambda pid: None
        ),
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    (tmp_path / "template").mkdir()
    (tmp_path / "template" / "invokeai.yaml").write_text(yaml.safe_dump({"schema_version": "4.0.3", "port": free}))
    (tmp_path / "code" / "scripts").mkdir(parents=True)
    (tmp_path / "code" / "scripts" / "invokeai-web.py").write_text("")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "profile": "box",
                "data_dir": "data",
                "gpu": "Fake",
                "server": {"template_root": "template", "python": sys.executable, "code_dir": "code"},
                "defaults": {"prompt": "a fox"},
                "families": [{"family": "f", "cases": [{"id": "a", "model": "m", "steps": 1, "sizes": [[8, 8]]}]}],
            }
        )
    )
    return manifest


@pytest.fixture
def stand_ins() -> Iterator[list[subprocess.Popen[bytes]]]:
    """The STAND_IN_RUNNER processes a test starts with `_launch`; what is left of them is killed afterwards."""
    processes: list[subprocess.Popen[bytes]] = []
    yield processes
    for process in processes:
        process.kill()
        process.communicate()


def _launch(processes: list[subprocess.Popen[bytes]], run_dir: Path, pause: str) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [sys.executable, "-c", STAND_IN_RUNNER, str(run_dir), str(runner.LOCK_DIR), pause],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(process)
    return process


def test_start_returns_once_its_runner_registered_so_resume_refuses_and_stop_kills_it(
    startable: Path,
    stand_ins: list[subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "spawn_runner", lambda run_dir: _launch(stand_ins, run_dir, "registered").pid)

    assert main(["start", "--manifest", str(startable)]) == 0
    run_dir = _resolve_run(None, startable)
    [started] = stand_ins
    # The interpreter itself, also where the spawned python is a venv launcher that starts it as its child.
    registered = int(runner.read_heartbeat(run_dir)["runner"][0])
    assert registered in {started.pid, *(child.pid for child in psutil.Process(started.pid).children())}
    assert f"runner  pid {registered}," in capsys.readouterr().out

    # The runner has registered but not begun: it waits where it opens the GPU.
    assert main(["resume", str(run_dir)]) == 2
    assert "still running" in capsys.readouterr().err
    assert stand_ins == [started]

    assert main(["stop", str(run_dir)]) == 0
    started.wait(timeout=CHILD_TIMEOUT_S)
    assert load_run_info(run_dir).status == "stopped"
    # Once the runner is gone, stop still ends what is left of the run.
    assert main(["stop", str(run_dir)]) == 0
    assert load_run_info(run_dir).status == "stopped"


def test_a_runner_that_comes_up_after_stop_leaves_the_stopped_run_untouched(
    startable: Path, stand_ins: list[subprocess.Popen[bytes]]
) -> None:
    run_dir = runner.new_run(startable, [], [])
    late = _launch(stand_ins, run_dir, "unregistered")
    assert late.stdout is not None
    assert late.stdout.readline().strip() == b"paused"

    assert main(["stop", str(run_dir)]) == 0
    stopped = (run_dir / RUN_INFO_FILE).read_bytes()
    assert load_run_info(run_dir).status == "stopped"

    assert late.stdin is not None
    late.stdin.write(b"go\n")
    late.stdin.flush()
    with pytest.raises(runner.RunnerError, match="was stopped before its runner registered"):
        runner.wait_for_runner(run_dir, late.pid, None)
    late.communicate(timeout=CHILD_TIMEOUT_S)
    assert (run_dir / RUN_INFO_FILE).read_bytes() == stopped
    assert not (run_dir / HEARTBEAT_FILE).exists()
    assert sorted(path.name for path in run_dir.iterdir() if path.name.startswith("server-")) == []
    assert not (run_dir.parent.parent / "scratch-roots").exists()
