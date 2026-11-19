import json
import logging
import traceback

import pytest

from invokeai.app.services.server_output import server_output_store
from invokeai.app.services.server_output.server_output_common import (
    InvalidServerOutputCursorError,
    ServerOutputRead,
)
from invokeai.app.services.server_output.server_output_store import ServerOutputStore, encode_cursor
from invokeai.backend.util.logging import InvokeAIColorLogFormatter


@pytest.fixture
def small_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_output_store, "EDITABLE_ROWS", 4)
    monkeypatch.setattr(server_output_store, "MAX_ROW_CELLS", 64)
    monkeypatch.setattr(server_output_store, "MAX_STYLE_RUNS", 4)
    monkeypatch.setattr(server_output_store, "RETAINED_BYTES", 2_000)
    monkeypatch.setattr(server_output_store, "RESPONSE_BYTES", 1_500)
    monkeypatch.setattr(server_output_store, "CATCH_UP_BYTES", 4_000)


class Client:
    """Applies reads the way the contract tells a client to."""

    def __init__(self, store: ServerOutputStore) -> None:
        self.store = store
        self.rows: dict[int, str] = {}
        self.cursor: str | None = None
        self.resets: list[str] = []

    def apply(self, read: ServerOutputRead) -> ServerOutputRead:
        if read.reset is not None:
            self.resets.append(read.reset)
            self.rows = {}
        self.rows = {i: t for i, t in self.rows.items() if i >= read.floor_id}
        self.rows.update({row.id: row.text for row in read.rows})
        self.cursor = read.cursor
        return read

    def poll(self) -> ServerOutputRead:
        return self.apply(self.store.read(self.cursor))

    def drain(self) -> int:
        pages = 1
        while self.poll().has_more:
            pages += 1
        return pages

    def texts(self) -> list[str]:
        return [self.rows[i] for i in sorted(self.rows)]


def texts(store: ServerOutputStore) -> list[str]:
    return [row.text for row in store.read().rows]


def retained_texts(store: ServerOutputStore) -> list[str]:
    """Every retained row, even past one snapshot: a client catching up from before the first change."""
    run_id, _, _, end = store.read().cursor.split(".")
    reference = Client(store)
    reference.cursor = encode_cursor(run_id, 0, 1, int(end))
    reference.drain()
    assert reference.resets == []
    return reference.texts()


def write(store: ServerOutputStore, *chunks: str) -> None:
    channel = store.open_channel()
    for chunk in chunks:
        channel.write(chunk)


def wire_bytes(read: ServerOutputRead) -> int:
    return len(json.dumps(read.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")).encode())


def test_invokeai_log_colors_survive_escape_sequences_split_anywhere() -> None:
    formatter = InvokeAIColorLogFormatter()
    lines = [
        formatter.format(logging.makeLogRecord({"name": "InvokeAI", "levelno": level, "levelname": name, "msg": name}))
        + "\n"
        for level, name in [(logging.INFO, "INFO"), (logging.WARNING, "WARNING"), (logging.CRITICAL, "CRITICAL")]
    ]
    text = "".join(lines) + "\x1b[1;4;38;5;208;48;2;1;2;3mx\x1b[22;24;39;49mplain\n"
    whole = ServerOutputStore()
    write(whole, text)
    split = ServerOutputStore()
    write(split, *text)  # one character per write

    rows = whole.read().rows
    assert [(r.text, r.styles) for r in rows] == [(r.text, r.styles) for r in split.read().rows]
    assert rows[0].text.endswith("::INFO --> INFO") and rows[0].styles == []  # `38;20` renders as the default
    assert [(s.fg, s.bold) for s in rows[1].styles] == [("3", False)]
    assert [(s.fg, s.bold, s.offset, s.length) for s in rows[2].styles] == [("1", True, 0, len(rows[2].text))]
    x = rows[3].styles
    assert rows[3].text == "xplain"
    assert [(s.offset, s.length, s.fg, s.bg, s.bold, s.underline) for s in x] == [(0, 1, "208", "#010203", True, True)]


def test_carriage_return_overwrites_cells_of_the_same_row() -> None:
    store = ServerOutputStore()
    write(store, "before\n", "\rloading 10%", "\rloading 100% done", "\rshort", "\n", "after")

    assert texts(store) == ["before", "shortng 100% done", "after"]


def test_erase_line_variants() -> None:
    store = ServerOutputStore()
    write(store, "abcdef\x1b[3D\x1b[K\n", "abcdef\x1b[3D\x1b[1K\n", "abcdef\x1b[2Kxy\n", "\tz\x08Z\n")

    assert texts(store) == ["abc", "    ef", "      xy", "        Z"]


def test_parallel_bars_converge_below_an_older_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        trace = traceback.format_exc()
    store = ServerOutputStore()
    client = Client(store)
    write(store, "start\n", trace)
    client.drain()

    channel = store.open_channel()
    channel.write("\ra:   0%\n\rb:   0%\x1b[A")
    for step in range(1, 3001):
        # tqdm's update of position 0, then position 1 via newline and cursor-up.
        channel.write(f"\ra: {step}")
        channel.write("\n")
        channel.write(f"\rb: {step * 2}\x1b[A")
        if step % 700 == 0:
            client.drain()
    channel.write("\ra: done\n\rb: done\n")

    expected = ["start", *trace.rstrip("\n").split("\n"), "a: done", "b: done"]
    assert texts(store) == expected
    client.drain()
    assert client.texts() == expected
    assert client.resets == ["initial"]


def test_huge_unterminated_write_keeps_a_truncated_row(small_limits: None) -> None:
    store = ServerOutputStore()
    write(store, "y" * 5_000_000, "z" * 1_000, "\nnext")

    rows = store.read().rows
    assert [(r.text, r.truncated) for r in rows] == [("y" * 64, True), ("next", False)]

    write(store, "\x1b[A\r\x1b[K")  # erasing the end clears the flag
    assert [(r.text, r.truncated) for r in store.read().rows] == [("", False), ("next", False)]


def test_many_tiny_writes_build_rows_and_retention_stays_in_budget(small_limits: None) -> None:
    store = ServerOutputStore()
    channel = store.open_channel()
    for line in range(500):
        for c in f"line {line:03}":
            channel.write(c)
        channel.write("\n")

    snapshot = store.read()
    assert snapshot.rows[-1].text == "line 499"
    assert snapshot.snapshot_truncated
    # Retained rows end with the newest and have consecutive IDs from the floor.
    assert [r.id for r in snapshot.rows] == list(range(snapshot.floor_id, snapshot.rows[-1].id + 1))
    assert all(r.text == f"line {r.id - 1:03}" for r in snapshot.rows)


def test_eviction_drops_oldest_finalized_rows_first(small_limits: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_output_store, "RESPONSE_BYTES", 20_000)  # a snapshot shows every retained row
    store = ServerOutputStore()
    write(store, *(f"{n:04}\n" for n in range(200)))

    snapshot = store.read()
    finalized = snapshot.rows[:-4]
    retained = sum(len(row.model_dump_json()) for row in finalized)
    # The budget counts worst-case serialized sizes, so actual rows use somewhat less of it.
    assert 1_000 < retained <= 2_000
    assert snapshot.rows[-1].text == "0199"
    assert snapshot.floor_id == snapshot.rows[0].id > 1


def test_snapshot_fits_the_response_budget_and_reports_truncation(small_limits: None) -> None:
    store = ServerOutputStore()
    write(store, *(f"row {n}\n" for n in range(40)))

    snapshot = store.read()
    assert snapshot.reset == "initial"
    assert snapshot.snapshot_truncated
    assert wire_bytes(snapshot) <= 1_500
    assert snapshot.rows[-1].text == "row 39"
    assert snapshot.floor_id == snapshot.rows[0].id


def test_current_cursor_reads_nothing_and_a_retried_cursor_reads_the_same() -> None:
    store = ServerOutputStore()
    write(store, "one\n")
    first = store.read()
    assert store.read(first.cursor) == first.model_copy(update={"reset": None, "rows": [], "snapshot_truncated": False})

    write(store, "two\n", "\rbar 1")
    retried = [store.read(first.cursor) for _ in range(2)]
    assert retried[0] == retried[1]
    assert [r.text for r in retried[0].rows] == ["two", "bar 1"]
    assert store.read(retried[0].cursor).rows == []


def test_catch_up_sends_a_rewritten_old_row_once_beside_appended_rows() -> None:
    store = ServerOutputStore()
    client = Client(store)
    channel = store.open_channel()
    channel.write("\rbar 0\nsteady\n")
    client.poll()

    for step in range(1, 6):
        channel.write(f"log {step}\n\x1b[{step + 2}A\rbar {step}\x1b[{step + 2}B\r")
    read = client.poll()

    assert [(r.id, r.text) for r in read.rows] == [(1, "bar 5"), *((n + 2, f"log {n}") for n in range(1, 6))]
    assert client.texts() == texts(store)


def test_rows_a_truncated_snapshot_left_out_stay_out_of_catch_up(small_limits: None) -> None:
    store = ServerOutputStore()
    client = Client(store)
    channel = store.open_channel()
    # Rows of the largest size: four style runs and 64 four-byte characters. A snapshot holds one of them.
    big = "".join(f"\x1b[3{k}m" + "\U0001f600" * 16 for k in range(4)) + "\x1b[0m"
    channel.write(f"{big}\n{big}\n{big}\n")
    snapshot = client.poll()
    assert [r.id for r in snapshot.rows] == [3] and snapshot.snapshot_truncated

    channel.write("\x1b[3A\rleft out\x1b[3B\rnew\n")  # rewrite editable row 1, below the client's floor
    read = client.poll()
    assert [(r.id, r.text) for r in read.rows] == [(4, "new")]
    assert read.floor_id == 3


def test_cursor_up_clips_to_the_editable_window(small_limits: None) -> None:
    store = ServerOutputStore()
    write(store, *(f"r{n}\n" for n in range(6)), "\x1b[99AX\x1bMY")

    # Four editable rows above the cursor: r2..r5; r0 and r1 are final. `ESC M` at the top does not scroll.
    assert texts(store) == ["r0", "r1", "XY", "r3", "r4", "r5"]


def test_paging_continues_without_skipping_rows_rewritten_meanwhile(
    small_limits: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_output_store, "RETAINED_BYTES", 10_000)
    monkeypatch.setattr(server_output_store, "CATCH_UP_BYTES", 10_000)
    store = ServerOutputStore()
    client = Client(store)
    channel = store.open_channel()
    client.poll()

    channel.write("".join(f"line {n:02}\n" for n in range(60)) + "\rbar a")
    pages = 0
    while True:
        read = client.poll()
        pages += 1
        assert wire_bytes(read) <= 1_500
        if not read.has_more:
            break
        # Rewrite the bar (already delivered or not) and add rows between pages.
        channel.write(f"\rbar {pages}\nextra {pages}\n\x1b[2A")
    assert pages > 3
    assert client.resets == ["initial"]
    assert client.texts() == retained_texts(store)


def test_eviction_during_catch_up_advances_the_floor(small_limits: None) -> None:
    store = ServerOutputStore()
    client = Client(store)
    channel = store.open_channel()
    channel.write("".join(f"old {n:02}\n" for n in range(20)))
    client.poll()
    channel.write("".join(f"new {n:02}\n" for n in range(20)))
    first = client.poll()
    assert first.has_more

    # Evicts rows the client already holds, not rows it has yet to receive.
    channel.write("".join(f"more {n:02}\n" for n in range(6)))
    rest = client.poll()
    assert rest.reset is None and rest.floor_id > first.floor_id
    client.drain()
    assert client.texts() == retained_texts(store)

    # Rows the client never received are evicted: history expired.
    channel.write("".join(f"flood {n:03}\n" for n in range(200)))
    expired = client.poll()
    assert expired.reset == "expired"
    assert client.texts() == texts(store)


def test_catch_up_larger_than_the_limit_expires(small_limits: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_output_store, "RETAINED_BYTES", 50_000)
    monkeypatch.setattr(server_output_store, "CATCH_UP_BYTES", 3_000)
    store = ServerOutputStore()
    client = Client(store)
    client.poll()
    write(store, *(f"line {n}\n" for n in range(30)))
    assert client.poll().has_more  # within the limit: paged

    write(store, *(f"line {n}\n" for n in range(60)))
    assert client.poll().reset == "expired"


def test_cursor_from_another_run_resets_with_restarted() -> None:
    before = ServerOutputStore()
    after = ServerOutputStore()
    for store in (before, after):
        write(store, "same\n", "output\n")
    cursor = before.read().cursor
    write(after, "newer\n")  # revisions overlap and pass the old cursor's

    read = after.read(cursor)
    assert read.reset == "restarted"
    assert read.run_id == after.run_id != before.run_id
    assert [r.text for r in read.rows] == ["same", "output", "newer"]


@pytest.mark.parametrize(
    "make_cursor",
    [
        lambda run, cursor: "",
        lambda run, cursor: "not a cursor",
        lambda run, cursor: cursor + "0" * 200,
        lambda run, cursor: cursor.upper(),
        lambda run, cursor: f"{run}.-1.1.1",
        lambda run, cursor: f"{run}.01.1.1",
        lambda run, cursor: f"{run}.1.1.1.1",
        lambda run, cursor: f"{run}.{10**25}.1.1",
        lambda run, cursor: f"{run}.999.1.1",  # a revision this run has not reached
        lambda run, cursor: f"{run}.0.0.1",
        lambda run, cursor: f"{run}.0.3.2",
        lambda run, cursor: f"{run}.0.1.99",  # a row this run has not created
    ],
)
def test_malformed_cursors_raise_without_echoing(make_cursor) -> None:
    store = ServerOutputStore()
    write(store, "secret text\n")
    bad = make_cursor(store.run_id, store.read().cursor)

    with pytest.raises(InvalidServerOutputCursorError) as raised:
        store.read(bad)
    assert str(raised.value) == "Invalid server output cursor"


def test_unsupported_control_sequences_are_consumed() -> None:
    sequences = [
        "\x1b]0;window title\x07",
        "\x1b]52;c;" + "QUJD" * 1000 + "\x1b\\",  # clipboard write
        "\x1b]8;;https://example.invalid\x1b\\link\x1b]8;;\x1b\\",
        "\x1bPq#0;2;0;0;0\x1b\\",
        "\x1b[?25l\x1b[?2004h\x1b[>4;1m\x1b(B\x1b7\x1b8\x1b=\x07\x00\x9b",
        "\x1b[3J\x1b[2J\x1b[H\x1b[5;5r",
    ]
    whole = ServerOutputStore()
    write(whole, "a" + "".join(sequences) + "b\n", "\x1b]0;unterminated\nshown\n")
    split = ServerOutputStore()
    write(split, *("a" + "".join(sequences) + "b\n"), "\x1b]0;unterminated\nshown\n")

    assert texts(whole) == texts(split) == ["alinkb", "", "shown"]


def test_eight_bit_control_introducers_act_like_their_escape_forms() -> None:
    text = (
        "a\x9d52;c;SGVsbG8=\x9cb"  # OSC 52 clipboard write
        "\x900;1|data\x9c"  # DCS
        "\x98sos\x9c\x9epm\x9c\x9fapc\x9c"
        "\x9b31mc\x9b0m"  # CSI SGR
        "\x9cd\n"  # a stray ST is ignored
    )
    whole = ServerOutputStore()
    write(whole, text)
    split = ServerOutputStore()
    write(split, *text)

    rows = whole.read().rows
    assert [(r.text, r.styles) for r in rows] == [(r.text, r.styles) for r in split.read().rows]
    assert rows[0].text == "abcd"
    assert [(s.offset, s.length, s.fg) for s in rows[0].styles] == [(2, 1, "1")]


def test_lone_surrogates_and_style_runs_stay_bounded(small_limits: None) -> None:
    store = ServerOutputStore()
    write(store, "\ud800ok" + "".join(f"\x1b[3{n % 8}m{n % 10}" for n in range(20)) + "\x1b[0m\n")

    row = store.read().rows[0]
    assert row.text == "�ok" + "".join(str(n % 10) for n in range(20))
    assert len(row.styles) == 4
    row.model_dump_json().encode("utf-8")  # encodable


def test_wire_bytes_stay_in_budget_for_worst_case_text(small_limits: None) -> None:
    store = ServerOutputStore()
    hostile = '"\\\U0001f600'
    for _ in range(30):
        styled = "".join(f"\x1b[{31 + k % 2};4{k % 2}m{hostile[k % 3]}" for k in range(10))
        write(store, styled + hostile * 20 + "\x1b[0m\n")

    snapshot = store.read()
    assert wire_bytes(snapshot) <= 1_500
    assert len(snapshot.model_dump_json().encode()) <= 1_500
    emoji_row = snapshot.rows[-1]
    # Offsets count UTF-16 code units, as JavaScript strings do: the emoji is two units wide.
    assert [(s.offset, s.length) for s in emoji_row.styles[:3]] == [(0, 1), (1, 1), (2, 2)]
