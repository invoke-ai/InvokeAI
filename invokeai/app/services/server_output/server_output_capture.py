"""Mirrors this process's `sys.stdout` and `sys.stderr` text into a `ServerOutputStore`.

`ServerOutputCapture.install()` replaces both streams with tees and `restore()` puts the originals back; importing
this module changes nothing. A tee hands every write to the original stream first, unchanged, and returns its result;
every other attribute (`encoding`, `errors`, `isatty`, `fileno`, `buffer`, `flush`, `line_buffering`, ...) is the
original's. Only `str` writes through `write`/`writelines` are mirrored; bytes written to `.buffer` or to the file
descriptor are not.

Ordering. The original write happens outside the store lock, then the text is mirrored under it. A stalled terminal
or pipe therefore never blocks readers or the other stream, and each write is mirrored whole, in one lock
acquisition, so the transcript is one interleaving of whole write calls in which every thread's writes keep their
program order. Concurrent writes from different threads may land on the terminal and in the transcript in different
orders; holding the lock across terminal I/O to match them would let one blocked stream stall every writer and
reader.

Failures. A mirror that raises stops mirroring for the rest of the run, after one notice written straight to the
original stderr; the original write has already happened. The store is then marked stopped, so readers report the
transcript unavailable instead of serving it frozen. A write made while this thread is already mirroring (a
signal handler, or anything the store calls printing) or inside `suspended()` goes to the original only.
"""

import logging
import sys
import threading
import urllib.parse
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from invokeai.app.services.server_output.server_output_store import ServerOutputChannel, ServerOutputStore


class _MirrorDepth(threading.local):
    """Per thread: inside `suspended()` blocks or a mirror call when above zero."""

    depth = 0


class _Tee:
    __slots__ = ("_original", "_channel", "_capture")

    def __init__(self, original: TextIO, channel: ServerOutputChannel, capture: "ServerOutputCapture") -> None:
        self._original = original
        self._channel = channel
        self._capture: ServerOutputCapture | None = capture

    def write(self, text: str) -> int:
        written = self._original.write(text)
        capture = self._capture
        if capture is not None and type(text) is str and text:
            capture._mirror(self._channel, text)
        return written

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        if name in _Tee.__slots__:  # unset during copy or unpickling; avoid recursing
            raise AttributeError(name)
        return getattr(self._original, name)

    def __repr__(self) -> str:
        return f"<server output tee of {self._original!r}>"


class ServerOutputCapture:
    """Owns the stream tees for one server run and the store they write to."""

    def __init__(self) -> None:
        self.store = ServerOutputStore()
        self._local = _MirrorDepth()
        self._installed: list[tuple[str, TextIO, _Tee]] | None = None
        self._failed = False

    def install(self) -> None:
        """Replace `sys.stdout` and `sys.stderr` (those that exist) with tees."""
        if self._installed is not None:
            raise RuntimeError("Server output capture is already installed")
        installed = []
        for name in ("stdout", "stderr"):
            original = getattr(sys, name)
            if original is not None:
                tee = _Tee(original, self.store.open_channel(), self)
                installed.append((name, original, tee))
                setattr(sys, name, tee)
        self._installed = installed

    def restore(self) -> None:
        """Put back the original streams. Tees still referenced elsewhere (a logging handler bound earlier) keep
        forwarding to the originals but stop mirroring, and the store is marked stopped. A stream someone replaced
        after `install` is left alone."""
        installed, self._installed = self._installed, None
        if installed is not None:
            self.store.stop()
        for name, original, tee in installed or ():
            tee._capture = None
            if getattr(sys, name) is tee:
                setattr(sys, name, original)

    def keep_reads_off_the_transcript(self, access_logger: logging.Logger, paths: Iterable[str]) -> None:
        """Send `access_logger`'s records of successful `GET` requests for `paths` to the terminal only, so a reader
        polling the transcript does not keep adding its own requests to it. Refused or failed reads (401, 403, 422,
        5xx) stay in the transcript, where an administrator can see them. Replaces the logger's stream handlers
        (uvicorn's access handler) with equivalents that make that distinction; every other record is written as
        before. `paths` are the decoded request paths; uvicorn logs them percent-encoded, so they are matched in that
        form."""
        terminal_only = frozenset(urllib.parse.quote(path) for path in paths)
        access_logger.handlers = [
            _AccessLogHandler(handler, self, terminal_only) if isinstance(handler, logging.StreamHandler) else handler
            for handler in access_logger.handlers
        ]

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Writes from this thread inside the block reach the original streams only."""
        local = self._local
        local.depth += 1
        try:
            yield
        finally:
            local.depth -= 1

    def _mirror(self, channel: ServerOutputChannel, text: str) -> None:
        local = self._local
        if self._failed or local.depth:
            return
        local.depth = 1
        try:
            channel.write(text)
        except Exception as error:
            self._fail(error)
        finally:
            local.depth = 0

    def _fail(self, error: Exception) -> None:
        if self._failed:
            return
        self._failed = True
        # The transcript stops following the console; readers report it unavailable rather than show it frozen.
        self.store.stop()
        stderr = next((original for name, original, _ in self._installed or () if name == "stderr"), None)
        if stderr is None:
            return
        try:
            stderr.write(f"Server output capture stopped after an internal error ({type(error).__name__}).\n")
            stderr.flush()
        except Exception:
            pass


class _AccessLogHandler(logging.StreamHandler[TextIO]):
    """A copy of an access-log stream handler that writes transcript reads past the mirror."""

    def __init__(
        self, handler: logging.StreamHandler[TextIO], capture: ServerOutputCapture, paths: frozenset[str]
    ) -> None:
        super().__init__(handler.stream)
        self.setLevel(handler.level)
        self.setFormatter(handler.formatter)
        for record_filter in handler.filters:
            self.addFilter(record_filter)
        self._capture = capture
        self._paths = paths

    def emit(self, record: logging.LogRecord) -> None:
        if _is_get_of(record, self._paths):
            with self._capture.suspended():
                super().emit(record)
        else:
            super().emit(record)


def _is_get_of(record: logging.LogRecord, paths: frozenset[str]) -> bool:
    """Whether a uvicorn access record, `(client, method, path with query, http version, status)`, is a successful
    GET (2xx or 304) of one of `paths`, ignoring the query string and a trailing slash."""
    args = record.args
    if not (isinstance(args, tuple) and len(args) == 5 and args[1] == "GET" and isinstance(args[2], str)):
        return False
    status = args[4]
    if not (isinstance(status, int) and (200 <= status < 300 or status == 304)):
        return False
    return args[2].split("?", 1)[0].rstrip("/") in paths
