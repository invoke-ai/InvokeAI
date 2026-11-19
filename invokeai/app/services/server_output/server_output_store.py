"""The server console transcript: a bounded screen of rows, its change revisions, and the read contract.

Rows. A row is created when the cursor first writes to it or a newline passes it, at the bottom, so row IDs increase
in display order. The newest `EDITABLE_ROWS` rows are editable: cursor-up reaches at most that many rows above the
row after the last one, and movement past the top clips to the oldest editable row (`ESC M` never scrolls). Older
rows are finalized and never change again. Finalized rows are retained while their wire size (text plus style
metadata, as serialized) totals at most `RETAINED_BYTES`; the oldest are evicted first. A row holds at most
`MAX_ROW_CELLS` cells: text past that is dropped and the row is marked truncated until an erase removes the end of
the row. A row keeps at most `MAX_STYLE_RUNS` style runs; text beyond them is shown unstyled.

Revisions. Every change to a row takes the next revision of one store-wide counter and stores it on the row, so the
retained rows ordered by revision are exactly the order of their latest changes.

Reads. A cursor is `<run id>.<revision>.<floor>.<end>`: the revision the client has applied, the lowest row ID it
holds, and one past the highest row ID it has received. It is opaque to clients and at most `MAX_CURSOR_LENGTH`
characters. A read:

- without a cursor answers a snapshot with reset "initial": the newest rows in display order that fit in
  `RESPONSE_BYTES`, with `snapshot_truncated` when earlier rows of this run are left out;
- with a cursor from another run answers a snapshot with reset "restarted";
- with a cursor whose undelivered rows were evicted (the retention floor passed its end), or whose pending changes
  exceed `CATCH_UP_BYTES`, answers a snapshot with reset "expired";
- otherwise answers the retained rows changed after the cursor's revision, each once at its latest value, skipping
  rows below the cursor's floor (rows a truncated snapshot left out). Changes are taken in revision order and cut at a
  row boundary so the response fits in `RESPONSE_BYTES`; `has_more` marks a cut, and the returned cursor then covers
  only the revisions returned, so the next read resumes after them. A row changed again meanwhile has a newer
  revision and comes again with its latest value.

A cursor naming a revision or row this run has not reached raises `InvalidServerOutputCursorError`.

Writes reach the rows only through channels, each holding the store's lock while it decodes. `stop()` marks a
transcript that no longer follows the console (capture restored or failed); reads still work, and the HTTP reader
treats a stopped store as unavailable.

All budgets are module constants read when a store is constructed; they will be revisited after measurement.
"""

import json
import re
import secrets
import threading
from collections import OrderedDict, deque
from collections.abc import Iterator
from typing import Any, NamedTuple

from invokeai.app.services.server_output.server_output_common import (
    InvalidServerOutputCursorError,
    ResetReason,
    ServerOutputRead,
    ServerOutputRow,
    ServerOutputStyleRun,
    Style,
)
from invokeai.app.services.server_output.server_output_terminal import TerminalDecoder

RETAINED_BYTES = 2 * 1024 * 1024
EDITABLE_ROWS = 64
MAX_ROW_CELLS = 16 * 1024
MAX_STYLE_RUNS = 64
RESPONSE_BYTES = 256 * 1024
CATCH_UP_BYTES = 1024 * 1024
MAX_CURSOR_LENGTH = 128

TAB_WIDTH = 8
# Tiny appends to one row are joined once this many chunks pile up.
_MAX_ROW_PARTS = 64
_MAX_INT_DIGITS = 19
_CURSOR = re.compile(r"([0-9a-f]{32})\.(0|[1-9][0-9]{0,18})\.(0|[1-9][0-9]{0,18})\.(0|[1-9][0-9]{0,18})")


class _FrozenRow(NamedTuple):
    id: int
    rev: int
    text: str
    runs: tuple[tuple[int, int, Style], ...]
    truncated: bool
    wire: int


class _Row:
    """An editable row. Text may sit in several chunks until it is read or overwritten."""

    __slots__ = ("id", "rev", "_parts", "length", "runs", "truncated", "frozen")

    def __init__(self, row_id: int) -> None:
        self.id = row_id
        self.rev = 0
        self._parts: list[str] = []
        self.length = 0
        self.runs: list[list[Any]] = []  # [start, end, style], sorted, non-overlapping, style never None
        self.truncated = False
        self.frozen: _FrozenRow | None = None

    def text(self) -> str:
        parts = self._parts
        if len(parts) != 1:
            if not parts:
                return ""
            self._parts = parts = ["".join(parts)]
        return parts[0]

    def write(self, col: int, chunk: str, style: Style | None, max_runs: int) -> None:
        end = col + len(chunk)
        if col >= self.length:
            if col > self.length:
                self._parts.append(" " * (col - self.length))
            self._parts.append(chunk)
            if len(self._parts) > _MAX_ROW_PARTS:
                self.text()
            self.length = end
        else:
            text = self.text()
            self._parts = [text[:col] + chunk + text[end:]]
            self.length = max(self.length, end)
        self._paint(col, end, style, max_runs)

    def erase(self, mode: int, col: int, max_runs: int) -> bool:
        """Apply erase-in-line; return whether the row changed."""
        if mode == 1:
            end = min(col + 1, self.length)
            if end == 0:
                return False
            text = self.text()
            self._parts = [" " * end + text[end:]]
            self._paint(0, end, None, max_runs)
            return True
        if mode == 0 and col > 0:
            changed = self.truncated or col < self.length
            if col < self.length:
                self._parts = [self.text()[:col]]
                self.length = col
                self.runs = [[s, min(e, col), st] for s, e, st in self.runs if s < col]
            self.truncated = False
            return changed
        if not (self.length or self.truncated):
            return False
        self._parts = []
        self.length = 0
        self.runs = []
        self.truncated = False
        return True

    def _paint(self, start: int, end: int, style: Style | None, max_runs: int) -> None:
        runs = self.runs
        if not runs or runs[-1][1] <= start:
            if style is None:
                return
            if runs and runs[-1][1] == start and runs[-1][2] == style:
                runs[-1][1] = end
            elif len(runs) < max_runs:
                runs.append([start, end, style])
            return
        out: list[list[Any]] = []
        for s, e, st in runs:
            if e <= start or s >= end:
                out.append([s, e, st])
                continue
            if s < start:
                out.append([s, start, st])
            if e > end:
                out.append([end, e, st])
        if style is not None:
            at = 0
            while at < len(out) and out[at][0] < start:
                at += 1
            out.insert(at, [start, end, style])
        merged: list[list[Any]] = []
        for run in out:
            if merged and merged[-1][1] == run[0] and merged[-1][2] == run[2]:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        del merged[max_runs:]
        self.runs = merged


def _text_wire_bytes(text: str) -> int:
    """JSON string body size: UTF-8, with `"` and `\\` escaped. Stored text holds no control characters."""
    return len(text.encode("utf-8")) + text.count('"') + text.count("\\")


def _utf16_offsets(text: str, points: list[int]) -> list[int]:
    """Map ascending code point offsets to UTF-16 code unit offsets."""
    if text.isascii():
        return points
    out = []
    units = 0
    last = 0
    for p in points:
        units += len(text[last:p].encode("utf-16-le")) // 2
        last = p
        out.append(units)
    return out


def _row_dto(row: _FrozenRow) -> ServerOutputRow:
    styles: list[ServerOutputStyleRun] = []
    if row.runs:
        points = [p for s, e, _ in row.runs for p in (s, e)]
        units = _utf16_offsets(row.text, points)
        for k, (_, _, (fg, bg, bold, dim, italic, underline, inverse)) in enumerate(row.runs):
            start, end = units[2 * k], units[2 * k + 1]
            styles.append(
                ServerOutputStyleRun(
                    offset=start,
                    length=end - start,
                    fg=fg,
                    bg=bg,
                    bold=bold,
                    dim=dim,
                    italic=italic,
                    underline=underline,
                    inverse=inverse,
                )
            )
    return ServerOutputRow(id=row.id, text=row.text, styles=styles, truncated=row.truncated)


def _wire_size(model: ServerOutputRead | ServerOutputRow | ServerOutputStyleRun) -> int:
    return len(json.dumps(model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")).encode())


def encode_cursor(run_id: str, revision: int, floor: int, end: int) -> str:
    return f"{run_id}.{revision}.{floor}.{end}"


def _decode_cursor(cursor: str) -> tuple[str, int, int, int]:
    match = _CURSOR.fullmatch(cursor) if len(cursor) <= MAX_CURSOR_LENGTH else None
    if match is None:
        raise InvalidServerOutputCursorError()
    return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))


class _Snapshot(NamedTuple):
    """What a read copies under the lock; the response is built from it outside."""

    rows: list[_FrozenRow]
    reset: ResetReason | None
    revision: int
    floor: int
    end: int
    truncated: bool


class _Limits(NamedTuple):
    editable_rows: int
    max_cells: int
    max_runs: int
    retained: int
    response: int
    catch_up: int
    # Worst-case serialized sizes, so every size bound holds whatever the field values.
    run_wire: int
    row_wire: int
    envelope_wire: int


def _read_limits() -> _Limits:
    if EDITABLE_ROWS < 1:
        raise ValueError("EDITABLE_ROWS must be at least 1")
    widest = 10**_MAX_INT_DIGITS - 1
    units = 2 * MAX_ROW_CELLS
    run_id = "f" * 32
    limits = _Limits(
        editable_rows=EDITABLE_ROWS,
        max_cells=MAX_ROW_CELLS,
        max_runs=MAX_STYLE_RUNS,
        retained=RETAINED_BYTES,
        response=RESPONSE_BYTES,
        catch_up=CATCH_UP_BYTES,
        run_wire=1 + _wire_size(ServerOutputStyleRun(offset=units, length=units, fg="#ffffff", bg="#ffffff")),
        row_wire=1 + _wire_size(ServerOutputRow(id=widest, text="", styles=[], truncated=False)),
        envelope_wire=_wire_size(
            ServerOutputRead(
                run_id=run_id,
                cursor=encode_cursor(run_id, widest, widest, widest),
                reset="restarted",
                rows=[],
                floor_id=widest,
                has_more=False,
                snapshot_truncated=False,
            )
        ),
    )
    # The largest row: 4 UTF-8 bytes per code point bounds both astral characters and escaped quotes.
    max_row = limits.row_wire + 4 * limits.max_cells + limits.max_runs * limits.run_wire
    if limits.envelope_wire + max_row > limits.response:
        raise ValueError("RESPONSE_BYTES must hold the largest row, so every read makes progress")
    return limits


class ServerOutputStore:
    """Thread-safe transcript of console output. Writes come through channels (`open_channel`); reads through
    `read`. One lock covers parsing and storing a write and copying a read; serialization happens outside it."""

    def __init__(self) -> None:
        self.run_id = secrets.token_hex(16)
        self._limits = _read_limits()
        self._lock = threading.Lock()
        self._screen = _Screen(self._limits)
        self._stopped = False

    def open_channel(self) -> "ServerOutputChannel":
        """A writer with its own decoding state (escape sequence in progress, colors) onto this shared screen."""
        return ServerOutputChannel(self._lock, TerminalDecoder(self._screen))

    @property
    def stopped(self) -> bool:
        """The transcript no longer follows the console (capture was restored, or failed)."""
        return self._stopped

    def stop(self) -> None:
        self._stopped = True

    def read(self, cursor: str | None = None) -> ServerOutputRead:
        position = None if cursor is None else _decode_cursor(cursor)
        with self._lock:
            if position is None:
                snapshot = self._screen.copy_snapshot("initial")
            elif position[0] != self.run_id:
                snapshot = self._screen.copy_snapshot("restarted")
            else:
                snapshot = self._screen.copy_changes(*position[1:])
        return self._respond(snapshot)

    def _respond(self, snapshot: _Snapshot) -> ServerOutputRead:
        limits = self._limits
        rows = snapshot.rows
        revision = snapshot.revision
        has_more = False
        if snapshot.reset is None:
            budget = limits.response - limits.envelope_wire
            for count, row in enumerate(rows):
                budget -= row.wire
                if budget < 0:
                    rows = rows[:count]
                    revision = rows[-1].rev
                    has_more = True
                    break
            rows = sorted(rows, key=lambda r: r.id)
        end = max(snapshot.end, rows[-1].id + 1) if rows else snapshot.end
        return ServerOutputRead(
            run_id=self.run_id,
            cursor=encode_cursor(self.run_id, revision, snapshot.floor, end),
            reset=snapshot.reset,
            rows=[_row_dto(row) for row in rows],
            floor_id=snapshot.floor,
            has_more=has_more,
            snapshot_truncated=snapshot.truncated,
        )


class _Screen:
    """The rows, the cursor and the retention behind a store. Not thread-safe: only the store's lock holders (its
    channels' decoders and its reads) touch it."""

    def __init__(self, limits: _Limits) -> None:
        self._editable_rows = limits.editable_rows
        self._max_cells = limits.max_cells
        self._max_runs = limits.max_runs
        self._retained_limit = limits.retained
        self._response_limit = limits.response
        self._catch_up_limit = limits.catch_up
        self._run_wire = limits.run_wire
        self._row_wire = limits.row_wire
        self._envelope_wire = limits.envelope_wire
        self._revision = 0
        self._next_id = 1
        self._editable: list[_Row] = []
        self._cursor = 0  # index into _editable; len(_editable) is the row after the last one
        self._col = 0
        self._finalized: deque[_FrozenRow] = deque()
        self._retained = 0
        # Every retained row, ordered by its latest revision.
        self._by_revision: OrderedDict[int, _Row | _FrozenRow] = OrderedDict()

    def copy_changes(self, revision: int, floor: int, end: int) -> _Snapshot:
        """The rows changed after `revision` for a client holding rows from `floor` up to `end`, or a snapshot
        when that history is gone or too long."""
        if revision > self._revision or not 0 < floor <= end <= self._next_id:
            raise InvalidServerOutputCursorError()
        store_floor = self._floor()
        if store_floor > end:
            return self.copy_snapshot("expired")
        floor = max(floor, store_floor)
        changed: list[_FrozenRow] = []
        size = 0
        for row in reversed(self._by_revision.values()):
            if row.rev <= revision:
                break
            if row.id < floor:
                continue
            frozen = self._freeze(row)
            size += frozen.wire
            if size > self._catch_up_limit:
                return self.copy_snapshot("expired")
            changed.append(frozen)
        changed.reverse()
        return _Snapshot(changed, None, self._revision, floor, end, False)

    def copy_snapshot(self, reset: ResetReason) -> _Snapshot:
        budget = self._response_limit - self._envelope_wire
        rows: list[_FrozenRow] = []
        for row in self._newest_first():
            frozen = self._freeze(row)
            budget -= frozen.wire
            if budget < 0:
                break
            rows.append(frozen)
        rows.reverse()
        floor = rows[0].id if rows else self._next_id
        return _Snapshot(rows, reset, self._revision, floor, self._next_id, floor > 1)

    def _newest_first(self) -> Iterator[_Row | _FrozenRow]:
        yield from reversed(self._editable)
        yield from reversed(self._finalized)

    def _floor(self) -> int:
        if self._finalized:
            return self._finalized[0].id
        if self._editable:
            return self._editable[0].id
        return self._next_id

    # `TerminalScreen` operations.

    def put_text(self, text: str, start: int, end: int, style: Style | None) -> None:
        row = self._current_row()
        col = self._col
        room = self._max_cells - col
        count = end - start
        if room <= 0:
            if not row.truncated:
                row.truncated = True
                self._touch(row)
            return
        if count > room:
            row.truncated = True
            end = start + room
            count = room
        row.write(col, text[start:end], style, self._max_runs)
        self._col = col + count
        self._touch(row)

    def line_feed(self) -> None:
        if self._cursor == len(self._editable):
            self._current_row()
        self._cursor += 1
        self._col = 0

    def carriage_return(self) -> None:
        self._col = 0

    def cursor_up(self, count: int) -> None:
        self._cursor = max(0, self._cursor - count)

    def cursor_down(self, count: int) -> None:
        self._cursor = min(len(self._editable), self._cursor + count)

    def cursor_forward(self, count: int) -> None:
        self._col = min(self._max_cells, self._col + count)

    def cursor_back(self, count: int) -> None:
        self._col = max(0, self._col - count)

    def cursor_column(self, column: int) -> None:
        self._col = min(self._max_cells, column)

    def tab(self) -> None:
        self._col = min(self._max_cells, (self._col // TAB_WIDTH + 1) * TAB_WIDTH)

    def erase_line(self, mode: int) -> None:
        if self._cursor < len(self._editable):
            row = self._editable[self._cursor]
            if row.erase(mode, self._col, self._max_runs):
                self._touch(row)

    def _current_row(self) -> _Row:
        editable = self._editable
        if self._cursor < len(editable):
            return editable[self._cursor]
        row = _Row(self._next_id)
        self._next_id += 1
        editable.append(row)
        self._by_revision[row.id] = row
        self._touch(row)
        if len(editable) > self._editable_rows:
            self._finalize(editable.pop(0))
        self._cursor = len(editable) - 1
        return row

    def _freeze(self, row: _Row | _FrozenRow) -> _FrozenRow:
        if isinstance(row, _FrozenRow):
            return row
        if row.frozen is None:
            text = row.text()
            runs = tuple((s, e, st) for s, e, st in row.runs)
            wire = self._row_wire + _text_wire_bytes(text) + len(runs) * self._run_wire
            row.frozen = _FrozenRow(row.id, row.rev, text, runs, row.truncated, wire)
        return row.frozen

    def _touch(self, row: _Row) -> None:
        self._revision += 1
        row.rev = self._revision
        row.frozen = None
        self._by_revision.move_to_end(row.id)

    def _finalize(self, row: _Row) -> None:
        frozen = self._freeze(row)
        self._by_revision[row.id] = frozen  # replacing a key keeps its revision position
        self._finalized.append(frozen)
        self._retained += frozen.wire
        while self._retained > self._retained_limit:
            evicted = self._finalized.popleft()
            self._retained -= evicted.wire
            del self._by_revision[evicted.id]


class ServerOutputChannel:
    """One output stream's writer onto a store."""

    __slots__ = ("_lock", "_decoder")

    def __init__(self, lock: threading.Lock, decoder: TerminalDecoder) -> None:
        self._lock = lock
        self._decoder = decoder

    def write(self, text: str) -> None:
        with self._lock:
            self._decoder.feed(text)
