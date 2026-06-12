"""A/B benchmark: MariaDB over Unix socket vs localhost TCP.

Same connection, same DB, only the transport differs. Measures the
roundtrip-tax for the hot paths where roundtrip cost dominates:

- get_by_pk (single row by PK, 1 roundtrip)
- get_many (paginated LIMIT 20, 1 roundtrip, small payload)
- get_image_names (full names list, 1 roundtrip, large payload)
- intermediates_cnt (COUNT, 1 roundtrip)
- INSERT commit-per-row (2 roundtrips per row: exec + commit)

Run after the table is already populated from the main MariaDB bench
(otherwise pass --seed-rows N to seed first).

Usage:

    .venv/bin/python scripts/benchmark_mariadb_socket_vs_tcp.py \
        --socket /tmp/invokeai-bench-mariadb-socket/mysqld.sock \
        --host 127.0.0.1 --port 3307 \
        --rows-target 100000
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import pymysql
import pymysql.cursors

# ---------------------------------------------------------------------------
# Workload (subset of the main MariaDB bench, identical SQL)
# ---------------------------------------------------------------------------


GET_BY_PK_SQL = "SELECT * FROM images WHERE image_name = %s"
GET_MANY_SQL = (
    "SELECT images.* FROM images "
    "LEFT JOIN board_images ON board_images.image_name = images.image_name "
    "WHERE images.image_category IN ('general') "
    "ORDER BY images.starred DESC, images.created_at DESC "
    "LIMIT %s OFFSET %s"
)
GET_NAMES_SQL = (
    "SELECT images.image_name FROM images "
    "LEFT JOIN board_images ON board_images.image_name = images.image_name "
    "WHERE images.image_category IN ('general') "
    "ORDER BY images.starred DESC, images.created_at DESC"
)
INTERMEDIATES_COUNT_SQL = "SELECT COUNT(*) FROM images WHERE is_intermediate = 1"
INSERT_SQL = (
    "INSERT INTO images "
    "(image_name, image_origin, image_category, width, height, "
    " is_intermediate, starred, has_workflow, user_id) "
    "VALUES (%s, 'internal', 'general', 512, 512, %s, %s, 0, 'user1')"
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


@dataclass
class OpResult:
    transport: str
    op: str
    iters: int
    p50_ms: float
    p95_ms: float
    p99_ms: float


def _time_op(name: str, fn: Callable[[int], None], iters: int) -> tuple[float, float, float]:
    samples: list[float] = []
    for c in range(iters):
        t0 = time.perf_counter()
        fn(c)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), _percentile(samples, 95), _percentile(samples, 99)


def run_read_ops(conn: pymysql.connections.Connection, n_rows: int, iters: int, transport: str) -> list[OpResult]:
    cur = conn.cursor()
    results: list[OpResult] = []
    page = 20

    def get_by_pk(c: int) -> None:
        target = f"img_{(c * 997) % n_rows:08d}.png"
        cur.execute(GET_BY_PK_SQL, (target,))
        cur.fetchone()

    def get_many(c: int) -> None:
        max_offset = max(0, n_rows - page)
        offset = (c * page * 7) % (max_offset + 1)
        cur.execute(GET_MANY_SQL, (page, offset))
        cur.fetchall()

    def get_names(c: int) -> None:
        cur.execute(GET_NAMES_SQL)
        cur.fetchall()

    def intermediates(c: int) -> None:
        cur.execute(INTERMEDIATES_COUNT_SQL)
        cur.fetchone()

    for op_name, fn in [
        ("get_by_pk", get_by_pk),
        ("get_many", get_many),
        ("get_image_names", get_names),
        ("intermediates_cnt", intermediates),
    ]:
        # Warm up once
        fn(0)
        p50, p95, p99 = _time_op(op_name, fn, iters)
        results.append(OpResult(transport=transport, op=op_name, iters=iters, p50_ms=p50, p95_ms=p95, p99_ms=p99))
        print(f"  {transport:>6}  {op_name:>18}  p50={p50:7.3f} ms  p95={p95:7.3f} ms  p99={p99:7.3f} ms")

    cur.close()
    return results


def run_insert_op(conn: pymysql.connections.Connection, start_offset: int, count: int, transport: str) -> OpResult:
    """Insert `count` rows commit-per-row. Names use `bench_socket_{i}` prefix to avoid PK collisions with the main read dataset."""
    cur = conn.cursor()
    samples: list[float] = []
    for i in range(count):
        name = f"bench_{transport}_{start_offset + i:08d}.png"
        t0 = time.perf_counter()
        cur.execute(INSERT_SQL, (name, 1 if i % 5 == 0 else 0, 1 if i % 10 == 0 else 0))
        conn.commit()
        samples.append((time.perf_counter() - t0) * 1000.0)
    cur.close()
    p50 = statistics.median(samples)
    p95 = _percentile(samples, 95)
    p99 = _percentile(samples, 99)
    print(f"  {transport:>6}  {'insert':>18}  p50={p50:7.3f} ms  p95={p95:7.3f} ms  p99={p99:7.3f} ms  ({count} rows)")
    return OpResult(transport=transport, op="insert", iters=count, p50_ms=p50, p95_ms=p95, p99_ms=p99)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket", default="/tmp/invokeai-bench-mariadb-socket/mysqld.sock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3307)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="bench")
    ap.add_argument("--database", default="invokeai_bench")
    ap.add_argument(
        "--rows-target",
        type=int,
        default=100_000,
        help="Expected row count in the images table (for offset calculation)",
    )
    ap.add_argument("--read-iters", type=int, default=50, help="Iterations per read op")
    ap.add_argument("--insert-rows", type=int, default=500, help="Rows for the INSERT benchmark")
    args = ap.parse_args()

    conn_tcp = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        autocommit=False,
        cursorclass=pymysql.cursors.Cursor,
    )
    conn_sock = pymysql.connect(
        unix_socket=args.socket,
        user=args.user,
        password=args.password,
        database=args.database,
        autocommit=False,
        cursorclass=pymysql.cursors.Cursor,
    )

    # Verify row count so the random PK lookups hit existing rows.
    with conn_tcp.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM images")
        actual_rows = cur.fetchone()[0]
    print(f"[socket-vs-tcp] table has {actual_rows} rows; using rows_target={args.rows_target}")
    n_rows = min(actual_rows, args.rows_target) if actual_rows else args.rows_target
    if n_rows < 1000:
        print(f"[socket-vs-tcp] WARN: only {n_rows} rows — results will be noisy. Run main benchmark first.")

    print(f"\n=== READ ops (iters={args.read_iters} each) ===")
    print("--- TCP localhost ---")
    tcp_reads = run_read_ops(conn_tcp, n_rows, args.read_iters, "tcp")
    print("--- Unix socket ---")
    sock_reads = run_read_ops(conn_sock, n_rows, args.read_iters, "socket")

    print(f"\n=== INSERT op (rows={args.insert_rows}, commit-per-row) ===")
    print("--- TCP localhost ---")
    # Clean up any prior bench inserts so we don't hit duplicate-PK errors on rerun.
    with conn_tcp.cursor() as cur:
        cur.execute("DELETE FROM images WHERE image_name LIKE 'bench_tcp_%' OR image_name LIKE 'bench_socket_%'")
    conn_tcp.commit()
    tcp_ins = run_insert_op(conn_tcp, 0, args.insert_rows, "tcp")
    print("--- Unix socket ---")
    sock_ins = run_insert_op(conn_sock, 0, args.insert_rows, "socket")

    conn_tcp.close()
    conn_sock.close()

    # ---- Comparison table ----
    print("\n=== Socket vs TCP — speedup (TCP_p50 / socket_p50) ===")
    print(f"{'op':>20}  {'tcp p50 ms':>12}  {'sock p50 ms':>12}  {'speedup':>10}")
    print("-" * 60)
    pairs = list(zip(tcp_reads, sock_reads, strict=False)) + [(tcp_ins, sock_ins)]
    for t, s in pairs:
        speedup = t.p50_ms / s.p50_ms if s.p50_ms > 0 else float("inf")
        print(f"{t.op:>20}  {t.p50_ms:>12.3f}  {s.p50_ms:>12.3f}  {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
