"""Write-workload benchmark: SQLite vs MariaDB.

Three scenarios, both engines, same workload definitions:

1. **single_thread_insert** — single connection inserts N rows. Pure raw
   write throughput; isolates fsync/commit cost.
2. **concurrent_insert** — K worker threads, each with its own connection,
   inserting N/K rows simultaneously. Shows SQLite's serialization vs
   MariaDB's row-level locking.
3. **mixed_insert_read** — one writer thread inserts rows continuously while
   K reader threads execute `get_many` and `get_image_names` in a loop.
   Measures read latency under write pressure.

Run via project venv:

    .venv/bin/python scripts/benchmark_write_workload.py \
        --mariadb-host 127.0.0.1 --mariadb-port 3307 \
        --rows-per-test 10000 --workers 1 4 16

Requires pymysql in the venv.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pymysql
import pymysql.cursors


# ---------------------------------------------------------------------------
# Schema setup — same columns/indexes as production migration_1
# ---------------------------------------------------------------------------

SQLITE_CREATE_IMAGES = """
CREATE TABLE images (
    image_name      TEXT NOT NULL PRIMARY KEY,
    image_origin    TEXT NOT NULL,
    image_category  TEXT NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    session_id      TEXT,
    node_id         TEXT,
    metadata        TEXT,
    is_intermediate INTEGER NOT NULL DEFAULT 0,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at      DATETIME,
    starred         INTEGER NOT NULL DEFAULT 0,
    has_workflow    INTEGER NOT NULL DEFAULT 0,
    user_id         TEXT NOT NULL DEFAULT 'system'
)
"""
SQLITE_INDEXES = [
    "CREATE INDEX idx_images_image_origin   ON images(image_origin)",
    "CREATE INDEX idx_images_image_category ON images(image_category)",
    "CREATE INDEX idx_images_created_at     ON images(created_at)",
    "CREATE INDEX idx_images_starred        ON images(starred)",
    "CREATE INDEX idx_images_user_id        ON images(user_id)",
]
SQLITE_CREATE_BOARD_IMAGES = """
CREATE TABLE board_images (
    image_name TEXT NOT NULL PRIMARY KEY,
    board_id   TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at DATETIME
)
"""

MARIADB_CREATE_IMAGES = """
CREATE TABLE images (
    image_name      VARCHAR(255) NOT NULL PRIMARY KEY,
    image_origin    VARCHAR(64)  NOT NULL,
    image_category  VARCHAR(64)  NOT NULL,
    width           INT          NOT NULL,
    height          INT          NOT NULL,
    session_id      VARCHAR(255),
    node_id         VARCHAR(255),
    `metadata`      LONGTEXT,
    is_intermediate TINYINT(1)   NOT NULL DEFAULT 0,
    created_at      DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at      DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    deleted_at      DATETIME(6),
    starred         TINYINT(1)   NOT NULL DEFAULT 0,
    has_workflow    TINYINT(1)   NOT NULL DEFAULT 0,
    user_id         VARCHAR(255) NOT NULL DEFAULT 'system'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""
MARIADB_INDEXES = [
    "CREATE INDEX idx_images_image_origin   ON images(image_origin)",
    "CREATE INDEX idx_images_image_category ON images(image_category)",
    "CREATE INDEX idx_images_created_at     ON images(created_at)",
    "CREATE INDEX idx_images_starred        ON images(starred)",
    "CREATE INDEX idx_images_user_id        ON images(user_id)",
]
MARIADB_CREATE_BOARD_IMAGES = """
CREATE TABLE board_images (
    image_name VARCHAR(255) NOT NULL PRIMARY KEY,
    board_id   VARCHAR(255) NOT NULL,
    created_at DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    deleted_at DATETIME(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


# ---------------------------------------------------------------------------
# Engine adapters — each provides:
#   - connect(): a callable returning a fresh DB-API connection
#   - reset_schema(): drop+recreate tables on a long-lived control connection
#   - insert_sql / query_sql / count_sql: paramstyle-correct SQL strings
# ---------------------------------------------------------------------------


@dataclass
class EngineAdapter:
    name: str
    connect: Callable[[], object]      # returns DB-API conn
    reset_schema: Callable[[], None]
    insert_sql: str                     # single-row INSERT
    get_many_sql: str
    get_names_sql: str
    placeholder: str                    # "?" or "%s"


def make_sqlite_adapter(db_path: Path) -> EngineAdapter:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect() -> sqlite3.Connection:
        # check_same_thread=False so each worker thread can use its own
        # connection. WAL + busy_timeout match production sqlite_database.py.
        conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA synchronous = NORMAL")  # production default for WAL
        return conn

    def reset_schema() -> None:
        # Wipe the file fully so journal/WAL state is fresh.
        for p in [db_path, db_path.with_suffix(db_path.suffix + "-wal"),
                  db_path.with_suffix(db_path.suffix + "-shm")]:
            if p.exists():
                p.unlink()
        conn = connect()
        conn.execute(SQLITE_CREATE_IMAGES)
        for idx in SQLITE_INDEXES:
            conn.execute(idx)
        conn.execute(SQLITE_CREATE_BOARD_IMAGES)
        conn.commit()
        conn.close()

    return EngineAdapter(
        name="sqlite",
        connect=connect,
        reset_schema=reset_schema,
        insert_sql=(
            "INSERT INTO images "
            "(image_name, image_origin, image_category, width, height, "
            " is_intermediate, starred, has_workflow, user_id) "
            "VALUES (?, 'internal', 'general', 512, 512, ?, ?, 0, 'user1')"
        ),
        get_many_sql=(
            "SELECT images.* FROM images "
            "LEFT JOIN board_images ON board_images.image_name = images.image_name "
            "WHERE images.image_category IN ('general') "
            "ORDER BY images.starred DESC, images.created_at DESC "
            "LIMIT ? OFFSET ?"
        ),
        get_names_sql=(
            "SELECT images.image_name FROM images "
            "LEFT JOIN board_images ON board_images.image_name = images.image_name "
            "WHERE images.image_category IN ('general') "
            "ORDER BY images.starred DESC, images.created_at DESC"
        ),
        placeholder="?",
    )


def make_mariadb_adapter(host: str, port: int, user: str, password: str, database: str) -> EngineAdapter:
    def connect() -> pymysql.connections.Connection:
        return pymysql.connect(
            host=host, port=port, user=user, password=password, database=database,
            autocommit=False, cursorclass=pymysql.cursors.Cursor,
        )

    def reset_schema() -> None:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS board_images")
                cur.execute("DROP TABLE IF EXISTS images")
                cur.execute(MARIADB_CREATE_IMAGES)
                for idx in MARIADB_INDEXES:
                    cur.execute(idx)
                cur.execute(MARIADB_CREATE_BOARD_IMAGES)
            conn.commit()
        finally:
            conn.close()

    return EngineAdapter(
        name="mariadb",
        connect=connect,
        reset_schema=reset_schema,
        insert_sql=(
            "INSERT INTO images "
            "(image_name, image_origin, image_category, width, height, "
            " is_intermediate, starred, has_workflow, user_id) "
            "VALUES (%s, 'internal', 'general', 512, 512, %s, %s, 0, 'user1')"
        ),
        get_many_sql=(
            "SELECT images.* FROM images "
            "LEFT JOIN board_images ON board_images.image_name = images.image_name "
            "WHERE images.image_category IN ('general') "
            "ORDER BY images.starred DESC, images.created_at DESC "
            "LIMIT %s OFFSET %s"
        ),
        get_names_sql=(
            "SELECT images.image_name FROM images "
            "LEFT JOIN board_images ON board_images.image_name = images.image_name "
            "WHERE images.image_category IN ('general') "
            "ORDER BY images.starred DESC, images.created_at DESC"
        ),
        placeholder="%s",
    )


# ---------------------------------------------------------------------------
# Workload primitives
# ---------------------------------------------------------------------------


def _row_tuple(i: int) -> tuple:
    return (f"img_{i:08d}.png", 1 if i % 5 == 0 else 0, 1 if i % 10 == 0 else 0)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _insert_range(adapter: EngineAdapter, start: int, count: int,
                  per_row_latencies_ms: list[float] | None = None) -> None:
    """Insert `count` rows starting at `start`. One row per INSERT, committed
    per row — matches the SqlModel/SQLite `save()` API which commits each save.
    """
    conn = adapter.connect()
    try:
        cur = conn.cursor()
        for i in range(start, start + count):
            t0 = time.perf_counter()
            cur.execute(adapter.insert_sql, _row_tuple(i))
            conn.commit()
            if per_row_latencies_ms is not None:
                per_row_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        cur.close()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scenario 1: single-thread INSERT throughput
# ---------------------------------------------------------------------------


@dataclass
class SingleThreadResult:
    engine: str
    rows: int
    total_s: float
    rows_per_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def run_single_thread_insert(adapter: EngineAdapter, rows: int) -> SingleThreadResult:
    adapter.reset_schema()
    latencies: list[float] = []
    t0 = time.perf_counter()
    _insert_range(adapter, 0, rows, latencies)
    total_s = time.perf_counter() - t0
    return SingleThreadResult(
        engine=adapter.name,
        rows=rows,
        total_s=total_s,
        rows_per_s=rows / total_s if total_s > 0 else 0.0,
        p50_ms=statistics.median(latencies),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ---------------------------------------------------------------------------
# Scenario 2: concurrent INSERT (K workers, own connection each)
# ---------------------------------------------------------------------------


@dataclass
class ConcurrentResult:
    engine: str
    workers: int
    rows_per_worker: int
    total_rows: int
    wall_s: float
    rows_per_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def run_concurrent_insert(adapter: EngineAdapter, workers: int, rows_per_worker: int) -> ConcurrentResult:
    adapter.reset_schema()
    all_latencies: list[float] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        local: list[float] = []
        # Each worker writes a disjoint name range so PK collisions never happen.
        start = wid * rows_per_worker
        _insert_range(adapter, start, rows_per_worker, local)
        with lock:
            all_latencies.extend(local)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, i) for i in range(workers)]
        for f in as_completed(futures):
            f.result()
    wall_s = time.perf_counter() - t0

    total = workers * rows_per_worker
    return ConcurrentResult(
        engine=adapter.name,
        workers=workers,
        rows_per_worker=rows_per_worker,
        total_rows=total,
        wall_s=wall_s,
        rows_per_s=total / wall_s if wall_s > 0 else 0.0,
        p50_ms=statistics.median(all_latencies),
        p95_ms=_percentile(all_latencies, 95),
        p99_ms=_percentile(all_latencies, 99),
    )


# ---------------------------------------------------------------------------
# Scenario 3: mixed INSERT + READ
# One writer thread inserts continuously. K readers run get_many in a loop.
# We measure read latency *under write pressure* over a fixed duration.
# ---------------------------------------------------------------------------


@dataclass
class MixedResult:
    engine: str
    readers: int
    duration_s: float
    writer_rows: int
    writer_rows_per_s: float
    reader_calls: int
    read_p50_ms: float
    read_p95_ms: float
    read_p99_ms: float
    seed_rows: int


def run_mixed_workload(adapter: EngineAdapter, readers: int, duration_s: float, seed_rows: int) -> MixedResult:
    adapter.reset_schema()
    # Seed the table so reads have something to chew on (matches a populated install).
    _insert_range(adapter, 0, seed_rows)

    stop_flag = threading.Event()
    writer_count = [0]  # boxed counter
    read_latencies: list[float] = []
    read_lock = threading.Lock()

    def writer() -> None:
        conn = adapter.connect()
        cur = conn.cursor()
        i = seed_rows
        try:
            while not stop_flag.is_set():
                cur.execute(adapter.insert_sql, _row_tuple(i))
                conn.commit()
                writer_count[0] += 1
                i += 1
        finally:
            cur.close()
            conn.close()

    def reader(rid: int) -> None:
        conn = adapter.connect()
        cur = conn.cursor()
        local: list[float] = []
        page = 20
        c = 0
        try:
            while not stop_flag.is_set():
                # Vary offset like in read benchmark.
                # We don't know exact n_rows during run, but seed_rows is the floor.
                max_offset = max(0, seed_rows - page)
                offset = (c * page * 7) % (max_offset + 1)
                t0 = time.perf_counter()
                if adapter.placeholder == "?":
                    cur.execute(adapter.get_many_sql, (page, offset))
                else:
                    cur.execute(adapter.get_many_sql, (page, offset))
                cur.fetchall()
                local.append((time.perf_counter() - t0) * 1000.0)
                c += 1
        finally:
            cur.close()
            conn.close()
        with read_lock:
            read_latencies.extend(local)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=writer, name="writer", daemon=True)]
    for i in range(readers):
        threads.append(threading.Thread(target=reader, args=(i,), name=f"reader-{i}", daemon=True))
    for t in threads:
        t.start()

    time.sleep(duration_s)
    stop_flag.set()
    for t in threads:
        t.join(timeout=30)

    wall_s = time.perf_counter() - t0
    return MixedResult(
        engine=adapter.name,
        readers=readers,
        duration_s=wall_s,
        writer_rows=writer_count[0],
        writer_rows_per_s=writer_count[0] / wall_s if wall_s > 0 else 0.0,
        reader_calls=len(read_latencies),
        read_p50_ms=statistics.median(read_latencies) if read_latencies else 0.0,
        read_p95_ms=_percentile(read_latencies, 95),
        read_p99_ms=_percentile(read_latencies, 99),
        seed_rows=seed_rows,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csvs(out_dir: Path,
               single: list[SingleThreadResult],
               concurrent: list[ConcurrentResult],
               mixed: list[MixedResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "write_single_thread.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "rows", "total_s", "rows_per_s", "p50_ms", "p95_ms", "p99_ms"])
        for r in single:
            w.writerow([r.engine, r.rows, f"{r.total_s:.3f}", f"{r.rows_per_s:.1f}",
                        f"{r.p50_ms:.3f}", f"{r.p95_ms:.3f}", f"{r.p99_ms:.3f}"])

    with (out_dir / "write_concurrent.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "workers", "rows_per_worker", "total_rows", "wall_s",
                    "rows_per_s", "p50_ms", "p95_ms", "p99_ms"])
        for r in concurrent:
            w.writerow([r.engine, r.workers, r.rows_per_worker, r.total_rows, f"{r.wall_s:.3f}",
                        f"{r.rows_per_s:.1f}", f"{r.p50_ms:.3f}", f"{r.p95_ms:.3f}", f"{r.p99_ms:.3f}"])

    with (out_dir / "write_mixed.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "readers", "duration_s", "seed_rows", "writer_rows", "writer_rows_per_s",
                    "reader_calls", "read_p50_ms", "read_p95_ms", "read_p99_ms"])
        for r in mixed:
            w.writerow([r.engine, r.readers, f"{r.duration_s:.2f}", r.seed_rows,
                        r.writer_rows, f"{r.writer_rows_per_s:.1f}",
                        r.reader_calls, f"{r.read_p50_ms:.3f}", f"{r.read_p95_ms:.3f}", f"{r.read_p99_ms:.3f}"])


def write_markdown(out_dir: Path,
                   single: list[SingleThreadResult],
                   concurrent: list[ConcurrentResult],
                   mixed: list[MixedResult],
                   rows_per_test: int,
                   workers: list[int],
                   mixed_duration_s: float,
                   mixed_seed_rows: int) -> None:
    lines = ["# Write-workload benchmark — SQLite vs MariaDB", ""]
    lines.append("Both engines on disk (SQLite WAL+NORMAL synchronous; MariaDB InnoDB ")
    lines.append("buffer pool 1G, default flush-log-at-trx-commit=1). One row per INSERT, ")
    lines.append("commit-per-row — same semantics as the production `save()` calls.")
    lines.append("")

    # 1) single-thread
    lines.append("## 1) Single-thread INSERT throughput")
    lines.append("")
    lines.append(f"Insert {rows_per_test} rows from a single connection, commit per row.")
    lines.append("")
    lines.append("| engine | total | rows/s | p50 ms | p95 ms | p99 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in single:
        lines.append(f"| {r.engine} | {r.total_s:.2f} s | {r.rows_per_s:,.0f} "
                     f"| {r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} |")
    lines.append("")

    # 2) concurrent
    lines.append("## 2) Concurrent INSERT")
    lines.append("")
    lines.append(f"K workers, each with its own connection, each inserting "
                 f"{rows_per_test // max(workers)} rows. Total = K × rows_per_worker.")
    lines.append("")
    lines.append("| engine | workers | total rows | wall s | rows/s | p50 ms | p95 ms | p99 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in concurrent:
        lines.append(f"| {r.engine} | {r.workers} | {r.total_rows} | {r.wall_s:.2f} "
                     f"| {r.rows_per_s:,.0f} | {r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} |")
    lines.append("")
    lines.append("`p50/p95/p99` are **per-INSERT** latencies. If they shoot up with more workers, "
                 "the engine is serializing writes.")
    lines.append("")

    # 3) mixed
    lines.append("## 3) Mixed — writer running, K readers under pressure")
    lines.append("")
    lines.append(f"Table seeded with {mixed_seed_rows} rows. One continuous writer thread + "
                 f"K reader threads run `get_many(LIMIT 20)` in a loop for {mixed_duration_s:.0f}s.")
    lines.append("")
    lines.append("| engine | readers | writer rows/s | reads total | read p50 ms | read p95 ms | read p99 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in mixed:
        lines.append(f"| {r.engine} | {r.readers} | {r.writer_rows_per_s:,.0f} | {r.reader_calls} "
                     f"| {r.read_p50_ms:.2f} | {r.read_p95_ms:.2f} | {r.read_p99_ms:.2f} |")
    lines.append("")

    (out_dir / "write_benchmark.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mariadb-host", default="127.0.0.1")
    ap.add_argument("--mariadb-port", type=int, default=3307)
    ap.add_argument("--mariadb-user", default="root")
    ap.add_argument("--mariadb-password", default="bench")
    ap.add_argument("--mariadb-database", default="invokeai_bench")
    ap.add_argument("--rows-per-test", type=int, default=10_000,
                    help="Total rows for single-thread; also divided among concurrent workers.")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--mixed-duration-s", type=float, default=10.0)
    ap.add_argument("--mixed-seed-rows", type=int, default=20_000)
    ap.add_argument("--sqlite-path", type=Path, default=None,
                    help="SQLite file path (default: tmp file).")
    ap.add_argument("--report-dir", type=Path, default=Path("/tmp/invokeai-bench-mariadb"))
    args = ap.parse_args()

    sqlite_path = args.sqlite_path or Path(tempfile.gettempdir()) / "invokeai_bench_sqlite_writes.db"
    print(f"[write-bench] sqlite: {sqlite_path}")
    print(f"[write-bench] mariadb: {args.mariadb_host}:{args.mariadb_port}/{args.mariadb_database}")
    print(f"[write-bench] rows_per_test={args.rows_per_test}  workers={args.workers}  "
          f"mixed={args.mixed_duration_s}s seed={args.mixed_seed_rows}")

    sqlite_adapter = make_sqlite_adapter(sqlite_path)
    mariadb_adapter = make_mariadb_adapter(
        args.mariadb_host, args.mariadb_port, args.mariadb_user,
        args.mariadb_password, args.mariadb_database,
    )
    adapters = [sqlite_adapter, mariadb_adapter]

    # 1) single-thread
    print("\n=== 1) single-thread INSERT ===")
    single_results: list[SingleThreadResult] = []
    for ad in adapters:
        print(f"  [{ad.name}] inserting {args.rows_per_test} rows...")
        r = run_single_thread_insert(ad, args.rows_per_test)
        single_results.append(r)
        print(f"    total={r.total_s:.2f}s  {r.rows_per_s:,.0f} rows/s  "
              f"p50={r.p50_ms:.2f} p95={r.p95_ms:.2f} p99={r.p99_ms:.2f}")

    # 2) concurrent
    print("\n=== 2) concurrent INSERT ===")
    concurrent_results: list[ConcurrentResult] = []
    for w in args.workers:
        rows_per_worker = max(1, args.rows_per_test // w)
        for ad in adapters:
            print(f"  [{ad.name}] {w} workers × {rows_per_worker} rows...")
            r = run_concurrent_insert(ad, w, rows_per_worker)
            concurrent_results.append(r)
            print(f"    wall={r.wall_s:.2f}s  {r.rows_per_s:,.0f} rows/s  "
                  f"p50={r.p50_ms:.2f} p95={r.p95_ms:.2f} p99={r.p99_ms:.2f}")

    # 3) mixed
    print("\n=== 3) mixed INSERT + READ ===")
    mixed_results: list[MixedResult] = []
    for w in args.workers:
        for ad in adapters:
            print(f"  [{ad.name}] readers={w}  duration={args.mixed_duration_s}s  "
                  f"seed={args.mixed_seed_rows}")
            r = run_mixed_workload(ad, w, args.mixed_duration_s, args.mixed_seed_rows)
            mixed_results.append(r)
            print(f"    writer={r.writer_rows_per_s:,.0f} rows/s  "
                  f"reads={r.reader_calls}  p50={r.read_p50_ms:.2f} "
                  f"p95={r.read_p95_ms:.2f} p99={r.read_p99_ms:.2f}")

    write_csvs(args.report_dir, single_results, concurrent_results, mixed_results)
    write_markdown(args.report_dir, single_results, concurrent_results, mixed_results,
                   args.rows_per_test, args.workers, args.mixed_duration_s, args.mixed_seed_rows)
    print(f"\n[write-bench] reports in: {args.report_dir}")


if __name__ == "__main__":
    main()
