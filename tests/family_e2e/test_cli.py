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
