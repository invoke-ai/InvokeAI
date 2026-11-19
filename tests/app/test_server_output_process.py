"""Server Output through the real entry point: a spawned `run_app` (or a bare `api_app` import under uvicorn) with an
isolated root, a spare port and network logging at `info`, so every request writes an access line."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform == "win32", reason="signals the server with SIGUSR1 and SIGINT"),
]

STARTUP_SECONDS = 120
READ = "/api/v1/app/server_output"

# A node pack whose import prints, leaves a partial line, prints a traceback, and prints on SIGUSR1.
NODE_PACK = """
import signal
import sys
import traceback

print("custom node says hello")
sys.stdout.write("partial line without newline")
try:
    raise RuntimeError("custom node startup failure")
except RuntimeError:
    traceback.print_exc()
signal.signal(signal.SIGUSR1, lambda *_: print("unrelated print"))
"""

# The entry point, recording at exit which streams the process ended with, and which stream InvokeAI's console
# handler (bound during startup) writes to: the tee when capture was installed.
RUN_APP = [
    sys.executable,
    "-c",
    "import atexit, logging, os, sys\n"
    "def record():\n"
    "    handlers = logging.getLogger('InvokeAI').handlers\n"
    "    console = type(handlers[0].stream).__name__ if handlers else '-'\n"
    "    with open(os.environ['EXIT_STREAMS'], 'w') as f:\n"
    "        f.write(' '.join([type(sys.stdout).__name__, type(sys.stderr).__name__, console]))\n"
    "atexit.register(record)\n"
    "from invokeai.app.run_app import run_app\n"
    "run_app()",
]
RESTORED = "TextIOWrapper TextIOWrapper _Tee"
NEVER_INSTALLED = "TextIOWrapper TextIOWrapper TextIOWrapper"


class Server:
    def __init__(self, process: subprocess.Popen[bytes], port: int, terminal: Path, exit_streams: Path) -> None:
        self.process = process
        self.port = port
        self.terminal = terminal
        self.exit_streams = exit_streams

    def get(self, path: str) -> tuple[int, dict[str, str], dict]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as response:
                return response.status, dict(response.headers), json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), json.load(error)

    def read(self, prefix: str = "", cursor: str | None = None) -> dict:
        status, headers, body = self.get(prefix + READ + (f"?cursor={cursor}" if cursor else ""))
        assert status == 200 and headers["cache-control"] == "no-store"
        return body

    def settle(self, prefix: str, cursor: str) -> dict:
        """Read until a pending write from another thread has landed (the reply to an earlier request is sent before
        uvicorn logs it)."""
        deadline = time.monotonic() + 10
        while True:
            body = self.read(prefix, cursor)
            if body["rows"] or time.monotonic() > deadline:
                return body
            time.sleep(0.05)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def serve(tmp_path: Path, command: list[str], env: dict[str, str]) -> Iterator[Server]:
    root = tmp_path / "root"
    exit_streams = tmp_path / "exit-streams.txt"
    (root / "nodes" / "printer").mkdir(parents=True)
    (root / "nodes" / "printer" / "__init__.py").write_text(NODE_PACK)
    port = _free_port()
    terminal = tmp_path / "terminal.log"
    environment = {
        **os.environ,
        "INVOKEAI_ROOT": str(root),
        "INVOKEAI_PORT": str(port),
        "INVOKEAI_LOG_LEVEL_NETWORK": "info",
        "INVOKEAI_IMAGE_INDEX_ENABLED": "false",
        "EXIT_STREAMS": str(exit_streams),
        **env,
    }
    command = [part.replace("{port}", str(port)) for part in command]
    with open(terminal, "wb") as output:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
    server = Server(process, port, terminal, exit_streams)
    try:
        deadline = time.monotonic() + STARTUP_SECONDS
        while True:
            assert process.poll() is None, terminal.read_text()
            try:
                server.get("/api/v1/app/version")
                break
            except OSError:
                assert time.monotonic() < deadline, "server did not start"
                time.sleep(0.2)
        yield server
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("base_url", [None, "/invoke"])
def test_entry_point_mirrors_the_console_but_not_its_own_reads(tmp_path: Path, base_url: str | None) -> None:
    env = {"INVOKEAI_BASE_URL": base_url} if base_url else {}
    # Behind a preserving proxy requests carry the base path; behind a stripping one they arrive without it.
    prefixes = ["", base_url] if base_url else [""]
    with serve(tmp_path, RUN_APP, env) as server:
        status, headers, config = server.get("/api/v1/app/frontend_config")
        assert headers["cache-control"] == "private, no-store"
        assert config["can_view_server_output"] is True

        snapshot = server.read()
        texts = [row["text"] for row in snapshot["rows"]]
        assert "custom node says hello" in texts
        assert "partial line without newlineTraceback (most recent call last):" in texts
        assert "RuntimeError: custom node startup failure" in texts
        assert any("Started server process" in text for text in texts)

        cursor = snapshot["cursor"]
        for prefix in prefixes:
            for _ in range(5):
                idle = server.read(prefix, cursor)
                assert idle["rows"] == [] and idle["cursor"] == cursor

        # Another route's access line reaches the mirror.
        server.get("/api/v1/app/version")
        version = server.settle("", cursor)
        assert len(version["rows"]) == 1
        assert '"GET /api/v1/app/version HTTP/1.1" 200' in version["rows"][0]["text"]
        cursor = version["cursor"]

        server.process.send_signal(signal.SIGUSR1)
        printed = server.settle("", cursor)
        assert [row["text"] for row in printed["rows"]] == ["unrelated print"]

        server.process.send_signal(signal.SIGINT)
        assert server.process.wait(timeout=60) == 0

    terminal = server.terminal.read_text()
    assert terminal.count("GET " + (prefixes[-1] or "") + READ) >= 5
    assert "custom node says hello" in terminal and "unrelated print" in terminal
    assert server.exit_streams.read_text() == RESTORED


def test_opting_out_leaves_the_streams_alone_and_the_reader_unavailable(tmp_path: Path) -> None:
    with serve(tmp_path, RUN_APP, {"INVOKEAI_ENABLE_SERVER_OUTPUT": "false"}) as server:
        _, _, config = server.get("/api/v1/app/frontend_config")
        assert config["can_view_server_output"] is False
        status, headers, _ = server.get(READ)
        assert status == 404 and headers["cache-control"] == "no-store"
    assert server.exit_streams.read_text() == NEVER_INSTALLED


def test_a_bare_api_app_import_serves_without_capture(tmp_path: Path) -> None:
    command = [sys.executable, "-m", "uvicorn", "invokeai.app.api_app:app", "--port", "{port}"]
    with serve(tmp_path, command, {}) as server:
        _, _, config = server.get("/api/v1/app/frontend_config")
        assert config["can_view_server_output"] is False
        assert server.get(READ)[0] == 404


def test_a_startup_failure_restores_the_streams(tmp_path: Path) -> None:
    exit_streams = tmp_path / "exit-streams.txt"
    # An invalid value the entry point validates after capture is installed, before serving. Should it stop failing,
    # the spare port and the short ceiling keep a server from hanging the run.
    env = {
        **os.environ,
        "INVOKEAI_ROOT": str(tmp_path / "root"),
        "INVOKEAI_PORT": str(_free_port()),
        "EXIT_STREAMS": str(exit_streams),
        "INVOKE_KREA2_SDPA_BACKEND": "bogus",
    }
    result = subprocess.run(RUN_APP, env=env, capture_output=True, timeout=60)

    assert result.returncode != 0
    assert b"INVOKE_KREA2_SDPA_BACKEND='bogus' is not a valid value" in result.stderr
    # Installed before the failure (startup's console handler writes to the tee), then restored.
    assert exit_streams.read_text() == RESTORED
