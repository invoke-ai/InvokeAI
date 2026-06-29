"""Standalone scaling benchmark against a MariaDB/MySQL server.

Mirrors the workload of
tests/app/services/test_sqlmodel_services/test_benchmark_image_scaling.py
(same row counts per step, same 4 queries, same iteration counts) so the
output CSV can be compared row-for-row against the SQLite numbers.

Run via the project venv:

    .venv/bin/python scripts/benchmark_mariadb_images.py \
        --host 127.0.0.1 --port 3307 --user root --password bench \
        --database invokeai_bench --max-rows 100000

Requires `pymysql` in the venv (`uv pip install pymysql`).

The script:
- Drops and recreates the `images` and `board_images` tables to match the
  schema from migration_1 (images columns + the same indexes).
- Walks row count from 1k -> 10k in 1k steps, then 20k -> --max-rows in 10k steps.
- For each step: inserts the new batch, then times each of the 4 queries
  --iters times and records p50 / p95.
- Writes `mariadb_scaling_benchmark.csv` next to the SQLite csv format.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import pymysql
import pymysql.cursors

# ---------------------------------------------------------------------------
# Schema — mirrors invokeai/app/services/shared/sqlite_migrator/migrations/migration_1.py
# ---------------------------------------------------------------------------

# `images` table: same columns as SQLite. `metadata` is the actual column name
# (SQLModel maps the python attr `metadata_` onto it). DATETIME(6) for
# microsecond precision so ORDER BY created_at is unambiguous like in SQLite.
CREATE_IMAGES_SQL = """
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

# Same indexes as migration_1 + migration_27 (idx_images_user_id).
CREATE_IMAGE_INDEXES_SQL = [
    "CREATE INDEX idx_images_image_origin   ON images(image_origin)",
    "CREATE INDEX idx_images_image_category ON images(image_category)",
    "CREATE INDEX idx_images_created_at     ON images(created_at)",
    "CREATE INDEX idx_images_starred        ON images(starred)",
    "CREATE INDEX idx_images_user_id        ON images(user_id)",
]

# Junction table for board joins (get_many does a LEFT JOIN even when no board
# is filtered, so we recreate it to get realistic plans).
CREATE_BOARD_IMAGES_SQL = """
CREATE TABLE board_images (
    image_name VARCHAR(255) NOT NULL PRIMARY KEY,
    board_id   VARCHAR(255) NOT NULL,
    created_at DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    deleted_at DATETIME(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

CREATE_BOARD_IMAGE_INDEXES_SQL = [
    "CREATE INDEX idx_board_images_board_id ON board_images(board_id)",
    "CREATE INDEX idx_board_images_board_id_created_at ON board_images(board_id, created_at)",
]


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------


def _build_scale_steps(max_rows: int) -> list[int]:
    steps: list[int] = []
    cur = 1000
    while cur <= min(max_rows, 10_000):
        steps.append(cur)
        cur += 1000
    cur = 20_000
    while cur <= max_rows:
        steps.append(cur)
        cur += 10_000
    if not steps or steps[-1] != max_rows:
        steps.append(max_rows)
    return sorted(set(steps))


@dataclass
class StepResult:
    n_rows: int
    op: str
    median_ms: float
    p95_ms: float
    iters: int


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# Workload — matches the SQLite/SQLModel benchmark exactly:
#   - same row generation (prefix `img_NNNNNNNN.png`, is_intermediate = i%5==0, starred = i%10==0, GENERAL/INTERNAL)
#   - same 4 queries
# ---------------------------------------------------------------------------


INSERT_SQL = (
    "INSERT INTO images "
    "(image_name, image_origin, image_category, width, height, "
    " is_intermediate, starred, has_workflow, user_id) "
    "VALUES (%s, 'internal', 'general', 512, 512, %s, %s, 0, 'user1')"
)


def insert_batch(conn: pymysql.connections.Connection, start: int, count: int) -> None:
    rows = [(f"img_{i:08d}.png", 1 if i % 5 == 0 else 0, 1 if i % 10 == 0 else 0) for i in range(start, start + count)]
    with conn.cursor() as cur:
        # executemany is much faster than per-row INSERTs in pymysql.
        cur.executemany(INSERT_SQL, rows)
    conn.commit()


# ---- Query 1: get_by_pk ----
GET_BY_PK_SQL = "SELECT * FROM images WHERE image_name = %s"


# ---- Query 2: get_many — same shape as SqlModelImageRecordStorage.get_many ----
GET_MANY_SQL = (
    "SELECT images.* "
    "FROM images "
    "LEFT JOIN board_images ON board_images.image_name = images.image_name "
    "WHERE images.image_category IN ('general') "
    "ORDER BY images.starred DESC, images.created_at DESC "
    "LIMIT %s OFFSET %s"
)


# ---- Query 3: get_image_names — full names list, no limit ----
GET_IMAGE_NAMES_SQL = (
    "SELECT images.image_name "
    "FROM images "
    "LEFT JOIN board_images ON board_images.image_name = images.image_name "
    "WHERE images.image_category IN ('general') "
    "ORDER BY images.starred DESC, images.created_at DESC"
)


# ---- Query 4: get_intermediates_count ----
INTERMEDIATES_COUNT_SQL = "SELECT COUNT(*) FROM images WHERE is_intermediate = 1"


def run_get_by_pk(conn: pymysql.connections.Connection, n: int, c: int) -> None:
    target = f"img_{(c * 997) % n:08d}.png"
    with conn.cursor() as cur:
        cur.execute(GET_BY_PK_SQL, (target,))
        cur.fetchone()


def run_get_many(conn: pymysql.connections.Connection, n: int, c: int, page_limit: int) -> None:
    max_offset = max(0, n - page_limit)
    offset = (c * page_limit * 7) % (max_offset + 1)
    with conn.cursor() as cur:
        cur.execute(GET_MANY_SQL, (page_limit, offset))
        cur.fetchall()


def run_get_image_names(conn: pymysql.connections.Connection, n: int, c: int) -> None:
    with conn.cursor() as cur:
        cur.execute(GET_IMAGE_NAMES_SQL)
        cur.fetchall()


def run_intermediates_count(conn: pymysql.connections.Connection, n: int, c: int) -> None:
    with conn.cursor() as cur:
        cur.execute(INTERMEDIATES_COUNT_SQL)
        cur.fetchone()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def setup_schema(conn: pymysql.connections.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS board_images")
        cur.execute("DROP TABLE IF EXISTS images")
        cur.execute(CREATE_IMAGES_SQL)
        for idx_sql in CREATE_IMAGE_INDEXES_SQL:
            cur.execute(idx_sql)
        cur.execute(CREATE_BOARD_IMAGES_SQL)
        for idx_sql in CREATE_BOARD_IMAGE_INDEXES_SQL:
            cur.execute(idx_sql)
    conn.commit()


def write_csv(path: Path, results: list[StepResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "n_rows", "op", "median_ms", "p95_ms", "iterations"])
        for r in results:
            w.writerow(["mariadb", r.n_rows, r.op, f"{r.median_ms:.4f}", f"{r.p95_ms:.4f}", r.iters])


def write_markdown(path: Path, results: list[StepResult], max_rows: int, iters: int, page: int) -> None:
    rows = sorted({r.n_rows for r in results})
    ops_order = ["get_by_pk", "get_many", "get_image_names", "intermediates_cnt"]
    lines = ["# Image table scaling benchmark — MariaDB", ""]
    lines.append(f"- query iterations per step: {iters}")
    lines.append(f"- get_many page size: {page}")
    lines.append(f"- max rows: {max_rows}")
    lines.append("")
    for op in ops_order:
        lines.append(f"## {op}")
        lines.append("")
        lines.append("| rows | mariadb p50 (ms) | mariadb p95 (ms) |")
        lines.append("|---:|---:|---:|")
        for n in rows:
            r = next((x for x in results if x.n_rows == n and x.op == op), None)
            if r:
                lines.append(f"| {n} | {r.median_ms:.2f} | {r.p95_ms:.2f} |")
            else:
                lines.append(f"| {n} | n/a | n/a |")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3307)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="bench")
    ap.add_argument("--database", default="invokeai_bench")
    ap.add_argument("--max-rows", type=int, default=100_000)
    ap.add_argument("--iters", type=int, default=25, help="query iterations per step")
    ap.add_argument("--page", type=int, default=20, help="get_many page size")
    ap.add_argument("--report-dir", type=Path, default=Path("/tmp/invokeai-bench-mariadb"))
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        autocommit=False,
        cursorclass=pymysql.cursors.Cursor,
    )

    print(f"[mariadb-bench] connected to {args.host}:{args.port}/{args.database}")
    setup_schema(conn)
    print("[mariadb-bench] schema reset")

    steps = _build_scale_steps(args.max_rows)
    print(f"[mariadb-bench] steps={steps}  iters={args.iters}  page={args.page}")

    ops = {
        "get_by_pk": lambda n, c: run_get_by_pk(conn, n, c),
        "get_many": lambda n, c: run_get_many(conn, n, c, args.page),
        "get_image_names": lambda n, c: run_get_image_names(conn, n, c),
        "intermediates_cnt": lambda n, c: run_intermediates_count(conn, n, c),
    }

    results: list[StepResult] = []
    inserted = 0

    for target in steps:
        to_add = target - inserted
        t0 = time.perf_counter()
        # Chunk inserts to avoid one giant statement -> matches SQLite per-row save() behavior.
        # 1000 rows per executemany is a sane batch size for MariaDB.
        for chunk_start in range(inserted, inserted + to_add, 1000):
            chunk_count = min(1000, inserted + to_add - chunk_start)
            insert_batch(conn, chunk_start, chunk_count)
        t_ins = time.perf_counter() - t0
        inserted = target
        print(f"\n[mariadb-bench] rows={inserted}  inserted_batch={to_add}  insert={t_ins:.1f}s")

        for op_name, op_fn in ops.items():
            samples_ms: list[float] = []
            for c in range(args.iters):
                t_start = time.perf_counter()
                op_fn(inserted, c)
                samples_ms.append((time.perf_counter() - t_start) * 1000.0)
            median_ms = statistics.median(samples_ms)
            p95_ms = _percentile(samples_ms, 95)
            results.append(
                StepResult(n_rows=inserted, op=op_name, median_ms=median_ms, p95_ms=p95_ms, iters=args.iters)
            )
            print(f"  mariadb  {op_name:>18}  p50={median_ms:7.2f} ms  p95={p95_ms:7.2f} ms")

    csv_path = args.report_dir / "mariadb_scaling_benchmark.csv"
    md_path = args.report_dir / "mariadb_scaling_benchmark.md"
    write_csv(csv_path, results)
    write_markdown(md_path, results, args.max_rows, args.iters, args.page)
    print(f"\n[mariadb-bench] csv: {csv_path}")
    print(f"[mariadb-bench] md:  {md_path}")

    conn.close()


if __name__ == "__main__":
    main()
