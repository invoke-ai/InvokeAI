import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
import yaml

from scripts.family_e2e.__main__ import UsageError, _resolve_run, main
from scripts.family_e2e.records import RunInfo, load_run_info, write_run_info
from scripts.family_e2e.runner import HEARTBEAT_FILE, now_iso, process_identity
from tests.family_e2e.test_records import make_run_info

REPO_ROOT = Path(__file__).parents[2]


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


def test_stop_kills_the_runner_tree_but_no_process_whose_pid_was_reused(tmp_path: Path) -> None:
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


def test_stop_saves_the_calibration_store_its_killed_server_left(tmp_path: Path) -> None:
    run_id = "20261017-000000-box"
    run_dir = tmp_path / "data" / "runs" / run_id
    run_dir.mkdir(parents=True)
    write_run_info(
        run_dir, make_run_info(run_id, profile="box", status="running").model_copy(update={"calibrate": True})
    )
    store_dir = tmp_path / "data" / "scratch-roots" / f"{run_id}-default" / "databases" / "working_memory"
    store_dir.mkdir(parents=True)
    store = b'{"schema_version": 1, "points": ["window"]}'
    (store_dir / "0123456789abcdef.json").write_bytes(store)
    (store_dir / "0123456789abcdef.json.lock").write_bytes(b"")

    assert main(["stop", str(run_dir)]) == 0

    info = load_run_info(run_dir)
    assert [(e.lifetime, e.source, e.device_class_id, e.status) for e in info.calibration_exports] == [
        (1, "store_file", "0123456789abcdef", None)
    ]
    assert (run_dir / info.calibration_exports[0].export).read_bytes() == store
    assert not store_dir.exists()


def test_a_stop_that_cannot_save_the_store_marks_the_run_stopped_and_keeps_the_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = "20261017-000000-box"
    run_dir = tmp_path / "data" / "runs" / run_id
    run_dir.mkdir(parents=True)
    write_run_info(
        run_dir,
        make_run_info(run_id, profile="box", status="running", arms=("default", "patched")).model_copy(
            update={"calibrate": True}
        ),
    )
    roots = tmp_path / "data" / "scratch-roots"
    kept_root = roots / f"{run_id}-default"
    store_file = kept_root / "databases" / "working_memory" / "0123456789abcdef.json"
    store_file.parent.mkdir(parents=True)
    store = b'{"schema_version": 1, "points": ["window"]}'
    store_file.write_bytes(store)
    # The other arm's server had recorded no store yet; with nothing to save, its root goes.
    (roots / f"{run_id}-patched" / "databases").mkdir(parents=True)
    # A file where the run's calibration/ directory goes: the copy raises, as on a full disk.
    (run_dir / "calibration").write_text("")

    assert main(["stop", str(run_dir)]) == 2

    info = load_run_info(run_dir)
    assert (info.status, info.calibration_exports) == ("stopped", [])
    assert info.finished_at is not None
    captured = capsys.readouterr()
    assert f"{run_id} stopped; continue with" in captured.out
    assert "could not save the calibration store" in captured.err
    assert f"kept the scratch root {kept_root} with it" in captured.err
    # The only copy of the lifetime's windows, for `resume` to save.
    assert store_file.read_bytes() == store
    assert not (roots / f"{run_id}-patched").exists()


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
