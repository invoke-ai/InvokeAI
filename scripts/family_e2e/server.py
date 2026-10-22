"""The server under test: a scratch root built from the template root, and the InvokeAI process serving it.

The tool never attaches to a running server. It copies the template's `invokeai.yaml` and a snapshot of its SQLite
database into a scratch root, points the shared paths back at the template root, and launches the server's own
interpreter on it.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import psutil
import requests
import yaml

from scripts.family_e2e.manifest import ServerSpec

logger = logging.getLogger("family_e2e")

CONFIG_FILE = "invokeai.yaml"
DB_FILE = "invokeai.db"
# The path settings of InvokeAIAppConfig (invokeai/app/services/config/config_default.py) and their defaults, which
# are relative to the root. Stores whose files the database owns go into the scratch root: the server reconciles them
# with its own (scratch) database, e.g. FontService deletes uploaded fonts the database does not list. The rest
# resolves against the template root, as for the template's own server.
SCRATCH_PATHS = {
    "outputs_dir": "outputs",
    "db_dir": "databases",
    "profiles_dir": "profiles",
    "fonts_storage_dir": "fonts-uploaded",
    "style_presets_dir": "style_presets",
    "workflow_thumbnails_dir": "workflow_thumbnails",
}
TEMPLATE_PATHS: dict[str, str | None] = {
    "models_dir": "models",
    "convert_cache_dir": "models/.convert_cache",
    "download_cache_dir": "models/.download_cache",
    "legacy_conf_dir": "configs",
    "fonts_dir": "fonts",
    "custom_nodes_dir": "nodes",
    "legacy_models_yaml_path": None,
}
# The client speaks plain HTTP to the root path.
DROPPED_SETTINGS = ("ssl_certfile", "ssl_keyfile", "base_url")
# ModelInstallService's TMPDIR_PREFIX (invokeai/app/services/model_install/model_install_default.py).
INSTALL_TMPDIR_PREFIX = "tmpinstall_"

RUNNING_LINE = re.compile(r"Invoke running on https?://\S+?:(\d+)")
# Localhost answers at once when something listens; Windows retries a refused connect for about two seconds.
PORT_PROBE_TIMEOUT_S = 0.25
STARTUP_TIMEOUT_S = 900.0
STARTUP_POLL_S = 1.0
STOP_TIMEOUT_S = 60.0
# Windows can refuse a delete for a moment after the last handle of a killed process closed.
REMOVE_ATTEMPTS = 10
REMOVE_RETRY_S = 0.2
GIT_TIMEOUT_S = 30.0
INTERPRETER_CHECK_TIMEOUT_S = 120.0


class ServerError(RuntimeError):
    pass


def read_template_settings(template_root: Path) -> dict[str, Any]:
    """The template's `invokeai.yaml` settings (empty when the root has none), refusing what the tool cannot copy."""
    path = template_root / CONFIG_FILE
    settings: dict[str, Any] = {}
    if path.is_file():
        settings = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not str(settings.get("schema_version", "")).startswith("4."):
            raise ServerError(f"{path} has schema {settings.get('schema_version')!r}; start its server once to migrate")
    if settings.get("db_url"):
        raise ServerError(f"{path} uses db_url; only a SQLite database can be copied into a scratch root")
    return settings


def pending_installs(template_root: Path, settings: dict[str, Any]) -> list[Path]:
    """Unfinished model installs in the template's models directory.

    The scratch server's install service resumes them at startup: the files would land in the template's models
    directory while the model is registered in the scratch database only.
    """
    models_dir = (template_root / settings.get("models_dir", TEMPLATE_PATHS["models_dir"])).resolve()
    return sorted(models_dir.glob(f"{INSTALL_TMPDIR_PREFIX}*"))


def template_server_running(settings: dict[str, Any]) -> bool:
    """Whether something listens on the template's configured port, i.e. its dev server may still be running."""
    host = settings.get("host", "127.0.0.1")
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(PORT_PROBE_TIMEOUT_S)
        return s.connect_ex((host, int(settings.get("port", 9090)))) == 0


_GENERIC_READ = 0x80000000
_OPEN_EXISTING = 3
_ERROR_SHARING_VIOLATION = 32


def database_in_use(path: Path) -> bool:
    """Whether a process has the database file open, e.g. a server of the template root on another port.

    Windows: an exclusive open fails with a sharing violation while any handle is open. Elsewhere this returns False:
    POSIX opens have no exclusive mode, so on Linux only the port check (`template_server_running`) guards the
    template, and a template server on another port goes unnoticed.
    """
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.CreateFileW(str(path), _GENERIC_READ, 0, None, _OPEN_EXISTING, 0, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error == _ERROR_SHARING_VIOLATION:
            return True
        raise ctypes.WinError(error)
    kernel32.CloseHandle(handle)
    return False


def snapshot_database(source: Path, target: Path) -> None:
    """Writes a consistent snapshot of the database, its WAL included, to `target`; the template stays untouched.

    Opening a database sets its journal mode and checkpoints the WAL on close, which would rewrite the template's
    file. So the files (main file, -wal, -shm) are copied first, which is consistent because no process holds them
    (`database_in_use`), and the copy is opened through the database layer and backed up into `target`.
    """
    # Imported here: the client otherwise imports nothing of invokeai, and this pulls in no torch.
    from invokeai.app.services.shared.database.database import Database

    staging = target.parent / ".snapshot"
    staging.mkdir(parents=True)
    try:
        for suffix in ("", "-wal", "-shm"):
            if source.with_name(source.name + suffix).is_file():
                shutil.copy2(source.with_name(source.name + suffix), staging / (source.name + suffix))
        database = Database.open_sqlite(staging / source.name, logger)
        try:
            database.backup(target)
        finally:
            database.dispose()
    finally:
        shutil.rmtree(staging)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_scratch_root(
    template_root: Path, scratch_root: Path, log_file: Path, port: int, single_cuda_device: bool = False
) -> None:
    """(Re)creates the scratch root: the template's settings with absolute paths and a snapshot of its database.

    `single_cuda_device`: the server will see one CUDA device only, the runner's pin (`CUDA_VISIBLE_DEVICES`). A
    template that names a CUDA device by index would then refuse to start, so its `device` becomes `cuda:0` and a
    `generation_devices` list with a CUDA device becomes `auto`.
    """
    settings = read_template_settings(template_root)
    template_db = (template_root / settings.get("db_dir", "databases")).resolve() / DB_FILE
    if not template_db.is_file():
        raise ServerError(f"template root has no database at {template_db}")
    if database_in_use(template_db):
        raise ServerError(f"{template_db} is open in another process; stop the template root's server first")

    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    db_dir = scratch_root / SCRATCH_PATHS["db_dir"]
    db_dir.mkdir(parents=True)
    snapshot_database(template_db, db_dir / DB_FILE)

    config = {k: v for k, v in settings.items() if k not in DROPPED_SETTINGS}
    for name, default in TEMPLATE_PATHS.items():
        value = settings.get(name, default)
        if value is not None:
            config[name] = str((template_root / value).resolve())
    for name, relative in SCRATCH_PATHS.items():
        config[name] = str(scratch_root / relative)
    config.update(
        host="127.0.0.1",
        port=port,
        log_handlers=["console", f"file={log_file}"],
        log_format="plain",
        # The residency check reads INFO lines of the model cache.
        log_level="debug" if settings.get("log_level") == "debug" else "info",
        # The index worker embeds every new image on the GPU between runs; a copied queue may hold pending items.
        image_index_enabled=False,
        clear_queue_on_startup=True,
    )
    if single_cuda_device:
        if str(config.get("device", "")).startswith("cuda:"):
            config["device"] = "cuda:0"
        devices = config.get("generation_devices")
        if isinstance(devices, list) and any(str(device).startswith("cuda") for device in devices):
            config["generation_devices"] = "auto"
    (scratch_root / CONFIG_FILE).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def server_env(spec: ServerSpec, device_env: dict[str, str] | None = None) -> dict[str, str]:
    """The caller's environment without INVOKEAI_*, then the manifest's env, then `device_env`, which pins the GPU."""
    # INVOKEAI_* variables would override the scratch root's settings; the manifest's env is the place for them.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("INVOKEAI_")}
    env["PYTHONPATH"] = str(spec.code_dir)
    env["PYTHONUTF8"] = "1"
    env.update(spec.env)
    env.update(device_env or {})
    return env


def verify_interpreter(spec: ServerSpec, env: dict[str, str]) -> str:
    """`invokeai.__file__` as the server's interpreter resolves it; must lie in the code directory."""
    result = subprocess.run(
        [str(spec.python), "-c", "import invokeai; print(invokeai.__file__)"],
        cwd=spec.code_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=INTERPRETER_CHECK_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        raise ServerError(f"{spec.python} cannot import invokeai:\n{result.stderr[-2000:]}")
    invokeai_file = result.stdout.strip().splitlines()[-1]
    if not Path(invokeai_file).resolve().is_relative_to(spec.code_dir.resolve()):
        raise ServerError(f"{spec.python} imports invokeai from {invokeai_file}, not from {spec.code_dir}")
    return invokeai_file


def git_state(path: Path) -> tuple[str | None, str | None, bool | None]:
    """(commit, branch, dirty) of the checkout at `path`; dirty counts tracked files only."""

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD"), None if status is None else bool(status)


def _serving_process(launched: int, port: int) -> psutil.Process:
    """The process of the launched tree that listens on the port: the interpreter that allocates GPU memory."""
    root = psutil.Process(launched)
    # Per process rather than system-wide: macOS lists other processes' sockets to root only.
    for candidate in [root, *root.children(recursive=True)]:
        try:
            connections = candidate.net_connections(kind="tcp")
        except psutil.Error:
            continue
        if any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port for c in connections):
            return candidate
    raise ServerError(f"no process the tool started listens on port {port}")


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_CREATE_SUSPENDED = 0x00000004


class KillOnCloseJob:
    """A Windows job object that kills its processes when its handle closes.

    The handle is not inheritable, so it closes at the latest when the process that created the job dies, however it
    dies. Processes started inside the job are in it too, including those in jobs of their own (nested jobs).
    """

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel32 = kernel32
        self._handle: int | None = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            self._handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, pid: int) -> None:
        process = self._kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, 0, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self._kernel32.AssignProcessToJobObject(self._handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._kernel32.CloseHandle(process)

    def close(self) -> None:
        """Kills every process still in the job."""
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def spawn_in_job(cmd: list[str], **popen_kwargs: Any) -> tuple[subprocess.Popen[bytes], KillOnCloseJob | None]:
    """Starts `cmd` in its own process group; on Windows also in a job that dies with this process.

    The process starts suspended and is resumed only once it is in the job: a venv's python.exe starts the real
    interpreter as a child right away, and a child created before the assignment would escape the job.
    """
    if sys.platform != "win32":
        return subprocess.Popen(cmd, start_new_session=True, **popen_kwargs), None
    job = KillOnCloseJob()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | _CREATE_SUSPENDED
    try:
        process = subprocess.Popen(cmd, creationflags=flags, **popen_kwargs)
    except BaseException:
        job.close()
        raise
    try:
        job.assign(process.pid)
        psutil.Process(process.pid).resume()
    except BaseException:
        process.kill()
        job.close()
        raise
    return process, job


def terminate(process: subprocess.Popen[Any], job: KillOnCloseJob | None) -> None:
    """Kills a tree started by `spawn_in_job` and waits until every process of it has exited.

    Closing a job kills its processes asynchronously; only an exited process has released its files (the scratch
    database). Neither kill goes by a bare pid: the job holds its processes, and the group id stays reserved while
    the unreaped leader holds it. The tree is captured before the kill, from the root Popen still holds.
    """
    try:
        root = psutil.Process(process.pid)
        tree = [root, *root.children(recursive=True)]
    except psutil.Error:
        tree = []
    if job is not None:
        job.close()
    elif process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(tree, timeout=STOP_TIMEOUT_S)
    if alive:
        logger.warning("processes still running after the kill: %s", ", ".join(str(p.pid) for p in alive))
    try:
        process.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()


def remove_tree(path: Path) -> list[str]:
    """Deletes a directory tree; returns what could still not be deleted after a few attempts.

    Files are named rather than the directories that merely stay non-empty because of them.
    """
    failed: list[str] = []
    for attempt in range(REMOVE_ATTEMPTS):
        failed.clear()
        if not path.exists():
            return []
        shutil.rmtree(path, onexc=lambda _function, failed_path, _error: failed.append(failed_path))
        if not path.exists():
            return []
        if attempt < REMOVE_ATTEMPTS - 1:
            time.sleep(REMOVE_RETRY_S)
    return [p for p in failed if not os.path.isdir(p)] or failed


@dataclass
class ServerProcess:
    process: subprocess.Popen[bytes]
    console: IO[bytes]
    log_path: Path
    base_url: str
    serving: psutil.Process
    """The interpreter serving the API. On Windows a venv's python.exe is a launcher that runs it as a child."""
    cmd: list[str]
    invokeai_file: str
    job: KillOnCloseJob | None
    """Kills the server tree if the runner dies without stopping it (Windows)."""

    @property
    def pid(self) -> int:
        return self.serving.pid

    def alive(self) -> bool:
        return self.process.poll() is None and self.serving.is_running()

    def log_size(self) -> int:
        return self.log_path.stat().st_size

    def read_log(self, offset: int) -> str:
        with self.log_path.open("rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace")

    def stop(self) -> None:
        terminate(self.process, self.job)
        self.console.close()


def launch_server(
    spec: ServerSpec,
    scratch_root: Path,
    log_path: Path,
    console_path: Path,
    device_env: dict[str, str] | None = None,
) -> ServerProcess:
    """Starts the server on a prepared scratch root and waits until its API answers."""
    env = server_env(spec, device_env)
    invokeai_file = verify_interpreter(spec, env)
    cmd = [str(spec.python), str(spec.code_dir / "scripts" / "invokeai-web.py"), "--root", str(scratch_root)]
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    console = console_path.open("ab")
    try:
        process, job = spawn_in_job(
            cmd, cwd=spec.code_dir, env=env, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT
        )
    except BaseException:
        console.close()
        raise
    try:
        port = _wait_for_port(process, log_path, log_offset, console_path)
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_api(process, base_url, console_path)
        serving = _serving_process(process.pid, port)
    except BaseException:
        terminate(process, job)
        console.close()
        raise
    return ServerProcess(process, console, log_path, base_url, serving, cmd, invokeai_file, job)


def _check_alive(process: subprocess.Popen[bytes], console_path: Path) -> None:
    if process.poll() is not None:
        raise ServerError(f"server exited with code {process.returncode}:\n{_tail(console_path)}")


def _wait_for_port(process: subprocess.Popen[bytes], log_path: Path, offset: int, console_path: Path) -> int:
    # The server moves to the next free port when the configured one is taken, so the port comes from its log.
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        _check_alive(process, console_path)
        if log_path.exists():
            with log_path.open("rb") as f:
                f.seek(offset)
                match = RUNNING_LINE.search(f.read().decode("utf-8", errors="replace"))
            if match:
                return int(match.group(1))
        time.sleep(STARTUP_POLL_S)
    raise ServerError(f"server did not log 'Invoke running on' within {STARTUP_TIMEOUT_S:.0f}s:\n{_tail(console_path)}")


def _wait_for_api(process: subprocess.Popen[bytes], base_url: str, console_path: Path) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        _check_alive(process, console_path)
        try:
            if requests.get(f"{base_url}/api/v1/app/version", timeout=10).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(STARTUP_POLL_S)
    raise ServerError(f"server API at {base_url} did not answer within {STARTUP_TIMEOUT_S:.0f}s")
