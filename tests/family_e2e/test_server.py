import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import typing
from contextlib import closing
from pathlib import Path

import psutil
import pytest
import yaml

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from scripts.family_e2e import server
from scripts.family_e2e.manifest import ServerSpec
from scripts.family_e2e.server import (
    DROPPED_SETTINGS,
    SCRATCH_PATHS,
    TEMPLATE_PATHS,
    ServerError,
    ServerProcess,
    build_scratch_root,
    server_env,
    spawn_in_job,
    template_server_running,
    verify_interpreter,
)

# Waiting for a helper process to report that it is ready.
CHILD_POLL_S = 0.01
CHILD_TIMEOUT_S = 30.0


def _rows(database: str | Path, uri: bool = False) -> list[str]:
    with closing(sqlite3.connect(database, uri=uri)) as conn:
        return [row[0] for row in conn.execute("SELECT name FROM models ORDER BY name")]


def _template(tmp_path: Path, settings: dict) -> Path:
    root = tmp_path / "template"
    (root / "databases").mkdir(parents=True)
    conn = sqlite3.connect(root / "databases" / "invokeai.db")
    conn.execute("CREATE TABLE models (name TEXT)")
    conn.execute("INSERT INTO models VALUES ('flux')")
    conn.commit()
    conn.close()
    (root / "databases" / "invokeai_backup_1.db").write_bytes(b"old")
    (root / "invokeai.yaml").write_text(yaml.safe_dump({"schema_version": "4.0.3", **settings}))
    return root


def test_scratch_root_points_back_at_the_template_and_keeps_run_output_local(tmp_path: Path) -> None:
    elsewhere = tmp_path / "nodes-elsewhere"
    template = _template(
        tmp_path,
        {
            "models_dir": "my-models",
            "custom_nodes_dir": str(elsewhere),
            "outputs_dir": "out",
            "fonts_storage_dir": "uploaded",
            "port": 9090,
            "host": "0.0.0.0",
            "log_level": "warning",
            "ssl_certfile": "cert.pem",
            "device_working_mem_gb": 6,
            "enable_partial_loading": False,
        },
    )
    scratch = tmp_path / "scratch"
    log_file = tmp_path / "run" / "server-default.log"
    build_scratch_root(template, scratch, log_file, port=9321)

    config = yaml.safe_load((scratch / "invokeai.yaml").read_text())
    assert config["models_dir"] == str((template / "my-models").resolve())
    assert config["download_cache_dir"] == str((template / "models/.download_cache").resolve())
    assert config["custom_nodes_dir"] == str(elsewhere.resolve())
    # Stores the database owns are reconciled with the scratch database, so they must not be the template's.
    for name, relative in (
        ("outputs_dir", "outputs"),
        ("db_dir", "databases"),
        ("fonts_storage_dir", "fonts-uploaded"),
        ("style_presets_dir", "style_presets"),
        ("workflow_thumbnails_dir", "workflow_thumbnails"),
    ):
        assert config[name] == str(scratch / relative), name
    assert (config["host"], config["port"]) == ("127.0.0.1", 9321)
    assert config["log_handlers"] == ["console", f"file={log_file}"]
    assert (config["log_format"], config["log_level"]) == ("plain", "info")
    assert (config["image_index_enabled"], config["clear_queue_on_startup"]) == (False, True)
    assert "ssl_certfile" not in config
    # Memory settings decide residency and peaks; they must reach the scratch server unchanged.
    assert (config["device_working_mem_gb"], config["enable_partial_loading"]) == (6, False)
    assert [p.name for p in (scratch / "databases").iterdir()] == ["invokeai.db"]
    assert _rows(scratch / "databases" / "invokeai.db") == ["flux"]
    # The template's database was read immutable: SQLite left no -wal/-shm beside it.
    assert sorted(p.name for p in (template / "databases").iterdir()) == ["invokeai.db", "invokeai_backup_1.db"]


def test_snapshot_includes_the_wal_of_an_unclean_stop_without_changing_it(tmp_path: Path) -> None:
    template = _template(tmp_path, {})
    db_dir = template / "databases"
    live = tmp_path / "live"
    live.mkdir()
    shutil.copy(db_dir / "invokeai.db", live / "invokeai.db")
    conn = sqlite3.connect(live / "invokeai.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("INSERT INTO models VALUES ('sdxl')")
    conn.commit()
    # A server killed at this point leaves the committed row in the WAL only.
    for name in ("invokeai.db", "invokeai.db-wal", "invokeai.db-shm"):
        shutil.copy(live / name, db_dir / name)
    conn.close()
    before = {name: (db_dir / name).read_bytes() for name in ("invokeai.db", "invokeai.db-wal")}
    assert _rows(f"{(db_dir / 'invokeai.db').as_uri()}?immutable=1", uri=True) == ["flux"]  # the main file alone

    build_scratch_root(template, tmp_path / "scratch", tmp_path / "a.log", port=1)
    assert _rows(tmp_path / "scratch" / "databases" / "invokeai.db") == ["flux", "sdxl"]
    assert {name: (db_dir / name).read_bytes() for name in before} == before


@pytest.mark.skipif(sys.platform != "win32", reason="open-file detection is Windows-only")
def test_database_open_in_another_process_is_refused(tmp_path: Path) -> None:
    template = _template(tmp_path, {})
    conn = sqlite3.connect(template / "databases" / "invokeai.db")
    try:
        conn.execute("SELECT 1").fetchall()
        with pytest.raises(ServerError, match="open in another process"):
            build_scratch_root(template, tmp_path / "scratch", tmp_path / "a.log", port=1)
    finally:
        conn.close()
    build_scratch_root(template, tmp_path / "scratch", tmp_path / "a.log", port=1)


def test_scratch_root_is_rebuilt_from_scratch(tmp_path: Path) -> None:
    template = _template(tmp_path, {})
    scratch = tmp_path / "scratch"
    build_scratch_root(template, scratch, tmp_path / "a.log", port=1)
    (scratch / "outputs").mkdir()
    (scratch / "outputs" / "stale.png").write_bytes(b"x")
    build_scratch_root(template, scratch, tmp_path / "a.log", port=2)
    assert not (scratch / "outputs").exists()
    assert yaml.safe_load((scratch / "invokeai.yaml").read_text())["models_dir"] == str((template / "models").resolve())


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"db_url": "mariadb+pymysql://u:p@db/invokeai"}, "only a SQLite database"),
        ({"schema_version": "3.0.0"}, "start its server once"),
    ],
)
def test_templates_the_tool_cannot_copy_are_refused(tmp_path: Path, settings: dict, message: str) -> None:
    template = _template(tmp_path, settings)
    with pytest.raises(ServerError, match=message):
        build_scratch_root(template, tmp_path / "scratch", tmp_path / "a.log", port=1)
    assert not (tmp_path / "scratch").exists()


def test_template_without_database_is_refused(tmp_path: Path) -> None:
    template = _template(tmp_path, {})
    (template / "databases" / "invokeai.db").unlink()
    with pytest.raises(ServerError, match="no database"):
        build_scratch_root(template, tmp_path / "scratch", tmp_path / "a.log", port=1)


def test_path_settings_cover_every_path_field_of_the_app_config() -> None:
    # A new path setting must be classified: left relative, it would resolve against the scratch root.
    path_fields = {
        name: field.default
        for name, field in InvokeAIAppConfig.model_fields.items()
        if Path in (field.annotation, *typing.get_args(field.annotation))
    }
    assert set(path_fields) == set(TEMPLATE_PATHS) | set(SCRATCH_PATHS) | {"ssl_certfile", "ssl_keyfile"}
    for name, default in TEMPLATE_PATHS.items():
        assert path_fields[name] == (None if default is None else Path(default)), name
    for name, relative in SCRATCH_PATHS.items():
        assert path_fields[name] == Path(relative), name
    assert set(DROPPED_SETTINGS) <= set(InvokeAIAppConfig.model_fields)


def test_template_port_in_use_is_detected() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert template_server_running({"port": port, "host": "0.0.0.0"})


def test_server_env_drops_inherited_invokeai_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVOKEAI_PORT", "9090")
    monkeypatch.setenv("FAMILY_E2E_TEST_KEEP", "1")
    spec = ServerSpec(template_root=tmp_path, python=Path(sys.executable), code_dir=tmp_path, env={"INVOKEAI_X": "y"})
    env = server_env(spec)
    assert "INVOKEAI_PORT" not in env
    assert (env["FAMILY_E2E_TEST_KEEP"], env["INVOKEAI_X"], env["PYTHONPATH"]) == ("1", "y", str(tmp_path))


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"device": "cuda:1"}, {"device": "cuda:0"}),
        ({"generation_devices": ["cuda:0", "cuda:1"]}, {"generation_devices": "auto"}),
        ({"device": "cuda:1", "generation_devices": ["cuda:1"]}, {"device": "cuda:0", "generation_devices": "auto"}),
        ({"device": "cpu", "generation_devices": ["cpu"]}, {"device": "cpu", "generation_devices": ["cpu"]}),
    ],
)
def test_a_server_pinned_to_one_cuda_device_gets_no_other_cuda_index(
    tmp_path: Path, settings: dict, expected: dict
) -> None:
    template = _template(tmp_path, settings)

    build_scratch_root(template, tmp_path / "pinned", tmp_path / "a.log", port=1, single_cuda_device=True)
    build_scratch_root(template, tmp_path / "unpinned", tmp_path / "a.log", port=1)

    pinned = yaml.safe_load((tmp_path / "pinned" / "invokeai.yaml").read_text())
    assert {key: pinned[key] for key in expected} == expected
    unpinned = yaml.safe_load((tmp_path / "unpinned" / "invokeai.yaml").read_text())
    assert {key: unpinned[key] for key in settings} == settings


def test_the_device_pin_overrides_the_inherited_cuda_visible_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    spec = ServerSpec(template_root=tmp_path, python=Path(sys.executable), code_dir=tmp_path)
    env = server_env(spec, {"CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa"})
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-aaaaaaaa"


def test_interpreter_must_import_invokeai_from_the_code_dir(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    spec = ServerSpec(template_root=tmp_path, python=Path(sys.executable), code_dir=code_dir)
    # Nothing to import in code_dir: the interpreter finds the installed package elsewhere, or none.
    with pytest.raises(ServerError):
        verify_interpreter(spec, server.server_env(spec))
    (code_dir / "invokeai").mkdir()
    (code_dir / "invokeai" / "__init__.py").write_text("")
    assert Path(verify_interpreter(spec, server.server_env(spec))).is_relative_to(code_dir)


def _spawn_tree(tmp_path: Path) -> tuple[subprocess.Popen[bytes], server.KillOnCloseJob | None, list[psutil.Process]]:
    """A process that starts a sleeping child, like a venv launcher and the server it runs; returns the whole tree."""
    pid_file = tmp_path / "pids.txt"
    script = tmp_path / "tree.py"
    # Written whole and closed before the rename, so whatever the name holds once readable is complete.
    script.write_text(
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"tmp = {str(pid_file) + '.tmp'!r}\n"
        "with open(tmp, 'w') as f:\n"
        "    f.write(f'{os.getpid()} {child.pid}')\n"
        f"os.replace(tmp, {str(pid_file)!r})\n"
        "time.sleep(60)\n"
    )
    process, job = spawn_in_job([sys.executable, str(script)])
    deadline = time.monotonic() + CHILD_TIMEOUT_S
    while True:
        assert process.poll() is None and time.monotonic() < deadline
        try:
            # Windows can refuse to open a file right after it was renamed into place, e.g. while a scanner holds it.
            text = pid_file.read_text()
            break
        except OSError:
            time.sleep(CHILD_POLL_S)
    interpreter, grandchild = (int(pid) for pid in text.split())
    tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
    assert {interpreter, grandchild} <= {p.pid for p in tree}
    return process, job, tree


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows-only")
def test_closing_the_job_kills_the_whole_server_tree(tmp_path: Path) -> None:
    # What the runner's death does to its server: the job handle closes.
    process, job, tree = _spawn_tree(tmp_path)
    assert job is not None
    job.close()
    _, alive = psutil.wait_procs(tree, timeout=10)
    assert alive == []
    process.wait(timeout=10)


def test_stop_kills_the_server_tree_and_never_a_process_by_its_recorded_pid(tmp_path: Path) -> None:
    process, job, tree = _spawn_tree(tmp_path)
    # The serving pid recorded at launch now belongs to an unrelated process (pid reuse).
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        console = (tmp_path / "console.log").open("ab")
        stopped = ServerProcess(
            process=process,
            console=console,
            log_path=tmp_path / "server.log",
            base_url="http://127.0.0.1:1",
            serving=psutil.Process(unrelated.pid),
            cmd=[],
            invokeai_file="",
            job=job,
        )
        stopped.stop()
        _, alive = psutil.wait_procs(tree, timeout=10)
        assert alive == []
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


def _gone(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups; on Windows the job object holds the tree")
@pytest.mark.parametrize("reaped", [False, True], ids=["leader-exited", "leader-reaped"])
def test_terminate_kills_the_group_even_after_its_leader_exited(tmp_path: Path, reaped: bool) -> None:
    # A launcher whose child outlives it: the child is in no tree any more, but still in the launcher's group.
    pid_file = tmp_path / "child.txt"
    script = tmp_path / "launcher.py"
    script.write_text(
        "import os, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"tmp = {str(pid_file) + '.tmp'!r}\n"
        "with open(tmp, 'w') as f:\n"
        "    f.write(str(child.pid))\n"
        f"os.replace(tmp, {str(pid_file)!r})\n"
    )
    process, job = spawn_in_job([sys.executable, str(script)])
    leader = psutil.Process(process.pid)
    deadline = time.monotonic() + CHILD_TIMEOUT_S
    # Exited but not reaped, as the runner first meets a crashed launcher.
    while leader.status() != psutil.STATUS_ZOMBIE:
        assert time.monotonic() < deadline
        time.sleep(CHILD_POLL_S)
    child = psutil.Process(int(pid_file.read_text()))
    if reaped:  # as ServerProcess.alive() or Popen.communicate() may have done before
        assert process.poll() is not None
    try:
        server.terminate(process, job)
        _, alive = psutil.wait_procs([child], timeout=CHILD_TIMEOUT_S)
        assert alive == []
    finally:
        if not _gone(child):
            child.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX zombies; Windows does not reap")
@pytest.mark.parametrize("waitid", [True, False], ids=["waitid", "no-waitid"])
def test_a_dead_server_reads_as_dead_but_stays_unreaped_until_terminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waitid: bool
) -> None:
    # Its pid is its group's id: reaped before terminate() killed the group, the id could pass to another group.
    if waitid and not hasattr(os, "waitid"):
        pytest.skip("this platform has no os.waitid (macOS)")
    if not waitid:
        monkeypatch.delattr(os, "waitid", raising=False)  # as on macOS, also where it exists
    process, job = spawn_in_job([sys.executable, "-c", "import sys; sys.exit(3)"])
    leader = psutil.Process(process.pid)
    deadline = time.monotonic() + CHILD_TIMEOUT_S
    while leader.status() != psutil.STATUS_ZOMBIE:
        assert time.monotonic() < deadline
        time.sleep(CHILD_POLL_S)
    dead = ServerProcess(
        process=process,
        console=(tmp_path / "console.log").open("ab"),
        log_path=tmp_path / "server.log",
        base_url="http://127.0.0.1:1",
        serving=leader,
        cmd=[],
        invokeai_file="",
        job=job,
    )

    assert not dead.alive()
    # Without waitid the code of an unreaped process is not to be had.
    assert dead.end_status() == ("exit code 3" if waitid else "exit code unknown")
    assert leader.status() == psutil.STATUS_ZOMBIE

    dead.stop()
    assert not leader.is_running()


def test_terminate_waits_until_the_tree_released_the_scratch_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The server and a process it started hold the scratch database open, as SQLite does. Windows refuses to delete
    # an open file, so a single removal attempt right after terminate() succeeds only if it waited for the tree.
    monkeypatch.setattr(server, "REMOVE_ATTEMPTS", 1)
    root = tmp_path / "scratch"
    (root / "databases").mkdir(parents=True)
    ready = tmp_path / "ready"
    wal = root / "databases" / "invokeai.db-wal"
    child = tmp_path / "child.py"
    child.write_text(f"import time\nwal = open({str(wal)!r}, 'w')\ntime.sleep(60)\n")
    script = tmp_path / "holder.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        f"db = open({str(root / 'databases' / 'invokeai.db')!r}, 'w')\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
        f"while not os.path.exists({str(wal)!r}): time.sleep(0.01)\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(60)\n"
    )
    process, job = spawn_in_job([sys.executable, str(script)])
    deadline = time.monotonic() + CHILD_TIMEOUT_S
    while not ready.exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(CHILD_POLL_S)

    tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
    server.terminate(process, job)
    assert [p.pid for p in tree if p.is_running()] == []
    assert server.remove_tree(root) == []
    assert not root.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to delete an open file")
def test_files_that_cannot_be_removed_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "REMOVE_ATTEMPTS", 2)
    monkeypatch.setattr(server, "REMOVE_RETRY_S", 0.0)
    root = tmp_path / "scratch"
    (root / "databases").mkdir(parents=True)
    held = root / "databases" / "invokeai.db"
    with held.open("w"):
        assert server.remove_tree(root) == [str(held)]
    assert server.remove_tree(root) == []
