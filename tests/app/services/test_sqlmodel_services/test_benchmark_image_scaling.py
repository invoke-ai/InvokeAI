"""Scaling benchmark: how do common image queries degrade as the images table grows?

For each scale step (1k, 2k, ..., 10k, 20k, ..., 100k+), inserts a batch of
images, then runs each query a number of times and reports median latency.
Both the legacy raw-SQLite implementation and the new SQLModel implementation
are measured side-by-side, on an on-disk database (file in tmp_path) so WAL
and fsync behaviour is realistic.

Marked `slow` because runtime grows with the largest step. Run explicitly:

    pytest tests/app/services/test_sqlmodel_services/test_benchmark_image_scaling.py \
        -v -s -m slow

Steering via env vars:

    INVOKE_BENCH_MAX_IMAGES   default 100000  (cap, inclusive)
    INVOKE_BENCH_QUERY_ITERS  default 25      (per query, per step)
    INVOKE_BENCH_PAGE_LIMIT   default 20      (get_many page size)
    INVOKE_BENCH_REPORT_DIR   default tests output  (where to write CSV/MD)

Output:
    - human-readable table to stdout (use -s to see it live)
    - CSV at `<report_dir>/image_scaling_benchmark.csv`
    - Markdown summary at `<report_dir>/image_scaling_benchmark.md`
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Callable

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.image_records.image_records_sqlmodel import (
    SqlModelImageRecordStorage as _SqlModelImageRecordStorageBase,
)
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


class SqlModelImageRecordStorage(_SqlModelImageRecordStorageBase):
    """Concrete subclass for benchmarking.

    The base SQLModel implementation is still abstract because two date-based
    methods (`get_image_dates`, `get_image_names_by_date`) haven't been ported
    yet on this branch. They aren't in scope for this benchmark, so we stub
    them with NotImplementedError to satisfy the ABC.
    """

    def get_image_dates(self, user_id=None, is_admin=False) -> list[VirtualSubBoardDTO]:
        raise NotImplementedError("Not benchmarked")

    def get_image_names_by_date(
        self, date, starred_first=True, order_dir=None, categories=None,
        search_term=None, user_id=None, is_admin=False,
    ):
        raise NotImplementedError("Not benchmarked")


# ---------------------------------------------------------------------------
# Scaling configuration
# ---------------------------------------------------------------------------


def _build_scale_steps(max_images: int) -> list[int]:
    """Steps of 1000 up to 10k, then 10000 up to max_images (inclusive)."""
    steps: list[int] = []
    cur = 1000
    while cur <= min(max_images, 10_000):
        steps.append(cur)
        cur += 1000
    cur = 20_000
    while cur <= max_images:
        steps.append(cur)
        cur += 10_000
    if not steps or steps[-1] != max_images:
        steps.append(max_images)
    # dedupe & sort
    return sorted(set(steps))


MAX_IMAGES = int(os.environ.get("INVOKE_BENCH_MAX_IMAGES", "100000"))
QUERY_ITERS = int(os.environ.get("INVOKE_BENCH_QUERY_ITERS", "25"))
PAGE_LIMIT = int(os.environ.get("INVOKE_BENCH_PAGE_LIMIT", "20"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    impl: str            # "sqlite" or "sqlmodel"
    n_rows: int          # total rows in images table at this step
    op: str              # which query
    median_ms: float
    p95_ms: float
    iters: int


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _time_calls(fn: Callable[[], None], iterations: int) -> tuple[float, float]:
    """Run fn `iterations` times, return (median_ms, p95_ms)."""
    samples_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples_ms), _percentile(samples_ms, 95)


def _make_disk_db(tmp_path: Path, logger: Logger, name: str) -> SqliteDatabase:
    """Create a real on-disk SqliteDatabase with all migrations applied."""
    db_dir = tmp_path / name
    db_dir.mkdir(parents=True, exist_ok=True)
    config = InvokeAIAppConfig(use_memory_db=False, db_dir=Path("databases"))
    # _root is a PrivateAttr, not a Field — set it directly so config.db_path
    # resolves under tmp_path instead of polluting the user's invokeai root.
    config._root = db_dir
    return create_mock_sqlite_database(config=config, logger=logger)


def _insert_batch(storage, prefix: str, start: int, count: int) -> None:
    """Insert `count` images with predictable distribution (matches benchmark above)."""
    for i in range(start, start + count):
        storage.save(
            image_name=f"{prefix}_{i:08d}.png",
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=512,
            height=512,
            has_workflow=False,
            is_intermediate=(i % 5 == 0),
            starred=(i % 10 == 0),
            user_id="user1",
        )


# ---------------------------------------------------------------------------
# Query operations under test
#
# Each op takes a storage, an int `n` (current row count), and an int call index `c`,
# and returns the callable that will be timed. The call index lets us vary the
# accessed key/offset so we don't measure pure cache hits.
# ---------------------------------------------------------------------------


def _op_get_by_pk(storage, n: int, c: int) -> Callable[[], None]:
    # Walk a range of names so we sample different rows. Names use 8-digit zero-pad,
    # matching the prefix used by _insert_batch.
    target = f"img_{(c * 997) % n:08d}.png"
    return lambda: storage.get(target)


def _op_get_many(storage, n: int, c: int) -> Callable[[], None]:
    # Page through the table by varying offset. Caps at table size minus limit.
    max_offset = max(0, n - PAGE_LIMIT)
    offset = (c * PAGE_LIMIT * 7) % (max_offset + 1)
    return lambda: storage.get_many(
        offset=offset,
        limit=PAGE_LIMIT,
        starred_first=True,
        order_dir=SQLiteDirection.Descending,
        categories=[ImageCategory.GENERAL],
    )


def _op_get_image_names(storage, n: int, c: int) -> Callable[[], None]:
    # Full names list, mimics the optimistic-update path used by the frontend.
    # No offset/limit on this API.
    return lambda: storage.get_image_names(
        starred_first=True,
        order_dir=SQLiteDirection.Descending,
        categories=[ImageCategory.GENERAL],
    )


def _op_intermediates_count(storage, n: int, c: int) -> Callable[[], None]:
    return lambda: storage.get_intermediates_count()


OPS: dict[str, Callable] = {
    "get_by_pk":          _op_get_by_pk,
    "get_many":           _op_get_many,
    "get_image_names":    _op_get_image_names,
    "intermediates_cnt":  _op_intermediates_count,
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_table(results: list[StepResult]) -> str:
    """Pretty table grouped by n_rows, showing both impls per op."""
    rows = sorted({r.n_rows for r in results})
    ops = list(OPS.keys())
    lines = []
    header = f"{'rows':>8}  " + "  ".join(f"{op:>22}" for op in ops)
    lines.append(header)
    lines.append("-" * len(header))
    for n in rows:
        cells = [f"{n:>8}"]
        for op in ops:
            sqlite_r = next(
                (r for r in results if r.n_rows == n and r.op == op and r.impl == "sqlite"), None
            )
            sqlmodel_r = next(
                (r for r in results if r.n_rows == n and r.op == op and r.impl == "sqlmodel"), None
            )
            s = f"{sqlite_r.median_ms:>7.2f}" if sqlite_r else "    n/a"
            m = f"{sqlmodel_r.median_ms:>7.2f}" if sqlmodel_r else "    n/a"
            cells.append(f"  {s} / {m}")  # ms median, sqlite/sqlmodel
        lines.append("  ".join(cells))
    lines.append("")
    lines.append("Cells show MEDIAN milliseconds: sqlite / sqlmodel.")
    return "\n".join(lines)


def _write_csv(path: Path, results: list[StepResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "n_rows", "op", "median_ms", "p95_ms", "iterations"])
        for r in results:
            w.writerow([r.impl, r.n_rows, r.op, f"{r.median_ms:.4f}", f"{r.p95_ms:.4f}", r.iters])


def _write_markdown(path: Path, results: list[StepResult]) -> None:
    rows = sorted({r.n_rows for r in results})
    ops = list(OPS.keys())
    lines = ["# Image table scaling benchmark", ""]
    lines.append(f"- query iterations per step: {QUERY_ITERS}")
    lines.append(f"- get_many page size: {PAGE_LIMIT}")
    lines.append(f"- max rows: {max(rows) if rows else 0}")
    lines.append("")
    for op in ops:
        lines.append(f"## {op}")
        lines.append("")
        lines.append("| rows | sqlite p50 (ms) | sqlite p95 (ms) | sqlmodel p50 (ms) | sqlmodel p95 (ms) |")
        lines.append("|---:|---:|---:|---:|---:|")
        for n in rows:
            s = next((r for r in results if r.n_rows == n and r.op == op and r.impl == "sqlite"), None)
            m = next((r for r in results if r.n_rows == n and r.op == op and r.impl == "sqlmodel"), None)
            lines.append(
                f"| {n} "
                f"| {s.median_ms:.2f} | {s.p95_ms:.2f} "
                f"| {m.median_ms:.2f} | {m.p95_ms:.2f} |"
                if s and m
                else f"| {n} | n/a | n/a | n/a | n/a |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_image_query_scaling(tmp_path: Path) -> None:
    """Walk row count from 1k to MAX_IMAGES, measure all four queries on both impls."""
    logger = InvokeAILogger.get_logger()
    steps = _build_scale_steps(MAX_IMAGES)
    print(f"\n[scaling-bench] steps={steps}  iters={QUERY_ITERS}  page={PAGE_LIMIT}")

    # Two separate on-disk databases; prefix the saved images differently so
    # row counts match exactly across both stores at each step.
    sqlite_db = _make_disk_db(tmp_path, logger, "sqlite_bench")
    sqlmodel_db = _make_disk_db(tmp_path, logger, "sqlmodel_bench")
    sqlite_storage = SqliteImageRecordStorage(db=sqlite_db)
    sqlmodel_storage = SqlModelImageRecordStorage(db=sqlmodel_db)

    # Same logical prefix across both stores so the random get_by_pk lookup
    # constructs a name that exists. (Each store has its own file.)
    prefix = "img"

    results: list[StepResult] = []
    inserted = 0

    for target in steps:
        to_add = target - inserted
        t_ins_sqlite_start = time.perf_counter()
        _insert_batch(sqlite_storage, prefix, inserted, to_add)
        t_ins_sqlite = time.perf_counter() - t_ins_sqlite_start

        t_ins_sqlmodel_start = time.perf_counter()
        _insert_batch(sqlmodel_storage, prefix, inserted, to_add)
        t_ins_sqlmodel = time.perf_counter() - t_ins_sqlmodel_start

        inserted = target
        print(
            f"\n[scaling-bench] rows={inserted}  "
            f"inserted_batch={to_add}  "
            f"insert_sqlite={t_ins_sqlite:.1f}s  insert_sqlmodel={t_ins_sqlmodel:.1f}s"
        )

        # Run each op on each impl.
        for op_name, op_factory in OPS.items():
            for impl_name, storage in (("sqlite", sqlite_storage), ("sqlmodel", sqlmodel_storage)):
                # Build a fresh closure per iteration to vary the accessed key/offset.
                # Loop variables are bound via defaults so each closure sees its own values.
                def run_one(c: int = 0, _op=op_factory, _storage=storage, _inserted=inserted) -> None:
                    _op(_storage, _inserted, c)()

                samples_ms: list[float] = []
                for c in range(QUERY_ITERS):
                    t0 = time.perf_counter()
                    run_one(c)
                    samples_ms.append((time.perf_counter() - t0) * 1000.0)

                median_ms = statistics.median(samples_ms)
                p95_ms = _percentile(samples_ms, 95)
                results.append(
                    StepResult(
                        impl=impl_name,
                        n_rows=inserted,
                        op=op_name,
                        median_ms=median_ms,
                        p95_ms=p95_ms,
                        iters=QUERY_ITERS,
                    )
                )
                print(
                    f"  {impl_name:>8}  {op_name:>18}  "
                    f"p50={median_ms:7.2f} ms  p95={p95_ms:7.2f} ms"
                )

    # Report
    print("\n" + _format_table(results))

    report_dir = Path(os.environ.get("INVOKE_BENCH_REPORT_DIR") or tmp_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "image_scaling_benchmark.csv"
    md_path = report_dir / "image_scaling_benchmark.md"
    _write_csv(csv_path, results)
    _write_markdown(md_path, results)
    print(f"\n[scaling-bench] csv:  {csv_path}")
    print(f"[scaling-bench] md:   {md_path}")
