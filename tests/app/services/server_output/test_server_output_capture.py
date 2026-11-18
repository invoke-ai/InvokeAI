import io
import logging
import os
import sys
import threading

import pytest
import tqdm
from uvicorn.protocols.utils import get_path_with_query_string

from invokeai.app.services.server_output.server_output_capture import ServerOutputCapture
from invokeai.app.services.server_output.server_output_terminal import TerminalDecoder

SAMPLE = "plain \x1b[31;1mred\x1b[0m caf\u00e9 \U0001f600\r\x1b[Kredrawn\n\tpartial"


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch):
    """A capture restored at teardown. Tests replace the streams themselves (`replace_streams`): pytest resets
    `sys.stdout` between the setup and call phases."""
    capture = ServerOutputCapture()
    yield capture
    capture.restore()


def replace_streams(monkeypatch: pytest.MonkeyPatch) -> tuple[io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def texts(capture: ServerOutputCapture) -> list[str]:
    return [row.text for row in capture.store.read().rows]


def test_tee_delivers_text_unchanged_and_restore_returns_the_originals(
    capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, err = replace_streams(monkeypatch)
    capture.install()
    assert sys.stdout is not out and sys.stderr is not err

    print(SAMPLE, end="")
    sys.stderr.writelines(["e1\n", "e2\n"])
    assert sys.stdout.write("!") == 1
    capture.restore()

    assert sys.stdout is out and sys.stderr is err
    assert out.getvalue() == SAMPLE + "!"
    assert err.getvalue() == "e1\ne2\n"
    # Both streams share one screen, as on a terminal: stderr continues the row stdout left open.
    assert texts(capture) == ["redrawn", "        partiale1", "e2", "!"]


def test_pipe_backed_stream_keeps_bytes_and_attributes(
    capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_streams(monkeypatch)
    read_fd, write_fd = os.pipe()
    with io.open(read_fd, "rb") as reader:
        original = io.TextIOWrapper(io.FileIO(write_fd, "w"), encoding="utf-8", errors="backslashreplace")
        sys.stdout = original
        capture.install()
        tee = sys.stdout

        for name in ("encoding", "errors", "line_buffering", "write_through", "buffer", "closed", "name", "newlines"):
            assert getattr(tee, name) == getattr(original, name), name
        assert tee.fileno() == write_fd
        assert tee.isatty() is False
        tee.write(SAMPLE + "\udcff\n")
        tee.flush()
        original.close()

        assert reader.read() == (SAMPLE + "\\udcff\n").encode()
    assert texts(capture)[-1] == "        partial\ufffd"


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pseudo-terminal")
def test_interactive_terminal_sees_identical_output(
    capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pty
    import select

    replace_streams(monkeypatch)

    def read_available(fd: int) -> bytes:
        data = b""
        while select.select([fd], [], [], 5)[0]:
            data += os.read(fd, 65536)
            if data.endswith(b"<end>"):
                return data
        raise AssertionError("terminal output did not arrive")

    master, slave = pty.openpty()
    try:
        original = io.TextIOWrapper(io.FileIO(slave, "w", closefd=False), encoding="utf-8", line_buffering=True)
        sys.stderr = original
        original.write(SAMPLE + "<end>")
        original.flush()
        direct = read_available(master)

        capture.install()
        tee = sys.stderr
        assert tee.isatty() is True and tee.fileno() == slave
        tee.write(SAMPLE + "<end>")
        tee.flush()

        assert read_available(master) == direct
        assert texts(capture) == ["redrawn", "        partial<end>"]
    finally:
        capture.restore()
        original.close()
        os.close(master)
        os.close(slave)


def test_mirror_failure_still_delivers_and_stops_mirroring(
    capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, err = replace_streams(monkeypatch)
    capture.install()
    print("kept")
    assert not capture.store.stopped

    def broken(self: TerminalDecoder, text: str) -> None:
        raise RuntimeError("parser bug")

    monkeypatch.setattr(TerminalDecoder, "feed", broken)
    print("first")
    print("second")
    monkeypatch.undo()

    assert out.getvalue() == "kept\nfirst\nsecond\n"
    assert err.getvalue() == "Server output capture stopped after an internal error (RuntimeError).\n"
    assert texts(capture) == ["kept"]
    assert capture.store.stopped  # readers now report the transcript unavailable


def test_suspended_writes_reach_only_the_terminal_from_their_own_thread(
    capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = replace_streams(monkeypatch)
    capture.install()
    entered = threading.Event()
    other_done = threading.Event()

    def other() -> None:
        entered.wait()
        print("other thread")
        other_done.set()

    thread = threading.Thread(target=other)
    thread.start()
    with capture.suspended():
        print("GET /api/v1/app/server_output 200")
        entered.set()
        other_done.wait()
    thread.join()
    print("after")

    assert out.getvalue() == "GET /api/v1/app/server_output 200\nother thread\nafter\n"
    assert texts(capture) == ["other thread", "after"]


def test_concurrent_writers_keep_whole_writes_in_program_order(
    capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_streams(monkeypatch)
    capture.install()
    start = threading.Barrier(4)

    def writer(name: str) -> None:
        start.wait()
        for n in range(500):
            sys.stdout.write(f"{name} {n:03}\n")

    threads = [threading.Thread(target=writer, args=(name,)) for name in "abcd"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = texts(capture)
    assert len(lines) == 2000
    for name in "abcd":
        assert [line for line in lines if line.startswith(name)] == [f"{name} {n:03}" for n in range(500)]


def test_real_tqdm_position_bars_converge(capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch) -> None:
    _, err = replace_streams(monkeypatch)
    capture.install()
    print("Traceback (most recent call last):\nValueError: earlier failure")
    with tqdm.tqdm(total=30, position=0, desc="outer", file=sys.stderr, ncols=60, mininterval=0) as outer:
        for _ in range(30):
            with tqdm.tqdm(total=20, position=1, desc="inner", file=sys.stderr, ncols=60, mininterval=0) as inner:
                for _ in range(20):
                    inner.update()
            outer.update()

    lines = texts(capture)
    assert lines[:2] == ["Traceback (most recent call last):", "ValueError: earlier failure"]
    # As on a terminal: each inner bar closing with leave=True stays on the outer bar's row and pushes it down.
    assert [line[:11] for line in lines[2:]] == ["inner: 100%"] * 30 + ["outer: 100%"]
    assert "30/30" in lines[-1]
    assert "\x1b[A" in err.getvalue()


def test_successful_transcript_reads_in_the_access_log_reach_only_the_terminal(
    capture: ServerOutputCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _ = replace_streams(monkeypatch)
    capture.install()
    access = logging.getLogger("test.server_output.access")
    access.propagate = False
    handler = logging.StreamHandler(sys.stdout)  # what uvicorn's dictConfig builds, bound to the tee
    handler.setFormatter(logging.Formatter("%(message)s"))
    access.handlers = [handler]
    access.setLevel(logging.INFO)
    route = "/api/v1/app/server_output"
    base = "/my app/\u00fcber"
    capture.keep_reads_off_the_transcript(access, {route, base + route})

    def log(method: str, path: str, status: int, query: bytes = b"") -> None:
        # The path as uvicorn writes it: percent-encoded, with the raw query string.
        logged = get_path_with_query_string({"path": path, "query_string": query})  # type: ignore[typeddict-item]
        access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:50000", method, logged, "1.1", status)

    log("GET", route, 200)
    log("GET", base + route, 200, b"cursor=abc.1.1.1")
    log("GET", route + "/", 304)
    terminal_only = 3
    log("GET", route, 403)
    log("GET", route, 422, b"cursor=bad")
    log("GET", route, 500)
    log("POST", route, 200)
    log("GET", route + "_other", 200)
    log("GET", "/api/v1/app/version", 200)
    access.info("not an access record")

    terminal = out.getvalue().splitlines()
    assert len(terminal) == 10
    assert texts(capture) == terminal[terminal_only:]
    assert "/my%20app/%C3%BCber/api" in terminal[1]
