#!/usr/bin/env python
"""Times representative database operations through the services, to compare two versions of the code.

Run it once per version and compare the results, e.g. a branch against its base:

    PYTHONPATH=<checkout of the base>   python scripts/benchmark_database.py --json base.json
    PYTHONPATH=<checkout of the branch> python scripts/benchmark_database.py --json head.json
    python scripts/benchmark_database.py --compare base.json head.json

The database is a SQLite file in a temporary directory, built by the real migrations and seeded through
the services, so what is timed is the production path including its transaction handling. Each operation
runs once to warm up and then repeatedly; the median and p95 time per call are reported, and the number
of SQL statements per call, counted in a separate pass with statement logging on so that the logging
does not distort the timings. Timings vary between runs on a busy machine: compare runs made back to
back, and repeat a comparison before trusting a small difference.
"""

import argparse
import json
import logging
import random
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

# A median may grow by this fraction or by this many milliseconds, whichever is larger. The floor is the fixed
# cost of a call through the query layer: SQLAlchemy Core and an explicit transaction make a point read on
# SQLite about 11 microseconds slower than a raw cursor did (measured). Calls that do real work stay bound by
# the fraction.
BUDGET_FRACTION = 0.10
BUDGET_FLOOR_MS = 0.05


class _StatementCounter(logging.Handler):
    """Counts the statements SQLite reports through the trace callback a verbose database installs."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.count = 0

    def handle(self, record: logging.LogRecord) -> bool:
        self.count += 1
        return True


class Services:
    def __init__(self, db: Any) -> None:
        from invokeai.app.services.board_image_records.board_image_records_sqlite import (
            SqliteBoardImageRecordStorage,
        )
        from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
        from invokeai.app.services.gallery.gallery_default import SqliteGalleryService
        from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage

        self.image_records = SqliteImageRecordStorage(db=db)
        self.board_records = SqliteBoardRecordStorage(db=db)
        self.board_image_records = SqliteBoardImageRecordStorage(db=db)
        self.gallery = SqliteGalleryService(db=db)
        # Listing gallery items builds their URLs through the invoker's URL service.
        from invokeai.app.services.urls.urls_default import LocalUrlService

        invoker = mock.Mock()
        invoker.services.urls = LocalUrlService()
        self.gallery.start(invoker)


def _metadata(rng: random.Random) -> str:
    words = ["portrait", "landscape", "cinematic", "volumetric", "light", "detailed", "film", "grain", "neon"]
    prompt = " ".join(rng.choice(words) for _ in range(120))
    return json.dumps(
        {
            "generation_mode": "txt2img",
            "positive_prompt": prompt,
            "negative_prompt": prompt[:400],
            "model": {"key": str(uuid.UUID(int=rng.getrandbits(128))), "name": "Some Model", "base": "flux"},
            "seed": rng.getrandbits(32),
            "steps": 30,
            "cfg_scale": 3.5,
            "width": 1024,
            "height": 1024,
            "scheduler": "euler",
            "loras": [{"key": str(uuid.UUID(int=rng.getrandbits(128))), "weight": 0.75} for _ in range(3)],
        }
    )


def _seed(services: Services, images: int, boards: int, rng: random.Random) -> list[str]:
    from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

    board_ids = [services.board_records.save(board_name=f"Board {i}", user_id="system").board_id for i in range(boards)]
    names: list[str] = []
    for i in range(images):
        name = f"{uuid.UUID(int=rng.getrandbits(128))}.png"
        services.image_records.save(
            image_name=name,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=1024,
            height=1024,
            has_workflow=False,
            is_intermediate=i % 20 == 0,
            starred=i % 50 == 0,
            metadata=_metadata(rng),
            user_id="system",
        )
        names.append(name)
        if i % 2 == 0:
            services.board_image_records.add_image_to_board(board_id=board_ids[i % boards], image_name=name)
    return names


def _operations(
    services: Services, names: list[str], rng: random.Random
) -> dict[str, tuple[Callable[[], object], int]]:
    """Operation name -> (one call, how many calls to time)."""
    from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy
    from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
    from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection

    sample = iter(rng.choices(names, k=100_000))
    general = [ImageCategory.GENERAL]

    def save_image() -> object:
        return services.image_records.save(
            image_name=f"{uuid.uuid4()}.png",
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=1024,
            height=1024,
            has_workflow=False,
            metadata=_metadata(rng),
            user_id="system",
        )

    return {
        "image_records.get": (lambda: services.image_records.get(next(sample)), 1000),
        "image_records.get_many(page of 100)": (
            lambda: services.image_records.get_many(
                limit=100, categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            50,
        ),
        "image_records.get_image_names(all)": (
            lambda: services.image_records.get_image_names(
                categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            10,
        ),
        "gallery.get_item_names(all)": (
            lambda: services.gallery.get_item_names(
                categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            10,
        ),
        "gallery.list_items(page of 100)": (
            lambda: services.gallery.list_items(
                limit=100, categories=general, is_intermediate=False, user_id="system", is_admin=True
            ),
            50,
        ),
        "board_records.get_all": (
            lambda: services.board_records.get_all(
                user_id="system",
                is_admin=True,
                order_by=BoardRecordOrderBy.CreatedAt,
                direction=SQLiteDirection.Descending,
            ),
            50,
        ),
        "image_records.save": (save_image, 200),
    }


def _time(call: Callable[[], object], repeats: int) -> tuple[float, float]:
    call()
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        call()
        durations.append((time.perf_counter_ns() - started) / 1e6)
    durations.sort()
    return statistics.median(durations), durations[min(len(durations) - 1, int(len(durations) * 0.95))]


def run(images: int, boards: int, seed: int) -> dict[str, Any]:
    import invokeai.app
    from invokeai.app.services.config.config_default import InvokeAIAppConfig
    from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
    from invokeai.app.services.shared.sqlite.sqlite_util import init_db

    quiet = logging.getLogger("benchmark_database.quiet")
    quiet.setLevel(logging.WARNING)
    counting = logging.getLogger("benchmark_database.statements")
    counting.setLevel(logging.DEBUG)
    counting.propagate = False
    counter = _StatementCounter()
    counting.addHandler(counter)

    # Code from before the database layer cannot close its connection, and Windows refuses to delete an
    # open database file: such a run leaves its temporary directory behind.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # Seeding commits tens of thousands of times; `normal` keeps that short. The timed connection
        # below uses the default `full`, so commits are timed as an install makes them.
        config = InvokeAIAppConfig(db_dir=Path(tmp), db_synchronous="normal")
        # The migrations clean up files under the root (legacy caches and models): never a real install's.
        config._root = Path(tmp)
        seeding_db = init_db(config=config, logger=quiet, image_files=mock.Mock(spec=ImageFileStorageBase))
        rng = random.Random(seed)
        started = time.perf_counter()
        names = _seed(Services(seeding_db), images, boards, rng)
        seeded_in = time.perf_counter() - started

        timed_db = SqliteDatabase(config.db_path, quiet)
        counted_db = SqliteDatabase(config.db_path, counting, verbose=True)
        timed = _operations(Services(timed_db), names, random.Random(seed + 1))
        counted = _operations(Services(counted_db), names, random.Random(seed + 1))

        results: dict[str, dict[str, float]] = {}
        for name, (call, repeats) in timed.items():
            median_ms, p95_ms = _time(call, repeats)
            count_call, _ = counted[name]
            count_call()  # warm up the same way the timed pass did
            counter.count = 0
            count_call()
            results[name] = {"median_ms": median_ms, "p95_ms": p95_ms, "statements": counter.count, "calls": repeats}

        for db in (seeding_db, timed_db, counted_db):
            database = getattr(db, "database", None)
            if database is not None:
                database.dispose()

    return {
        # `invokeai.app`, because an editable install leaves the top-level package without a `__file__`.
        "code": str(Path(invokeai.app.__file__).parent.parent),
        "python": sys.version.split()[0],
        "images": images,
        "boards": boards,
        "seeded_in_s": round(seeded_in, 1),
        "operations": results,
    }


def compare(base: dict[str, Any], head: dict[str, Any]) -> int:
    print(f"base: {base['code']}\nhead: {head['code']}\n")
    print(f"{'operation':40} {'base ms':>9} {'head ms':>9} {'change':>8} {'stmts':>9}  budget")
    over_budget = 0
    for name, after in head["operations"].items():
        before = base["operations"].get(name)
        if before is None:
            print(f"{name:40} {'-':>9} {after['median_ms']:9.3f}")
            continue
        change = (after["median_ms"] - before["median_ms"]) / before["median_ms"]
        allowed = max(before["median_ms"] * BUDGET_FRACTION, BUDGET_FLOOR_MS)
        within = after["median_ms"] - before["median_ms"] <= allowed
        over_budget += not within
        statements = f"{before['statements']}->{after['statements']}"
        verdict = "ok" if within else "OVER"
        print(
            f"{name:40} {before['median_ms']:9.3f} {after['median_ms']:9.3f} {change:+8.1%} {statements:>9}  {verdict}"
        )
    return 1 if over_budget else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=20_000)
    parser.add_argument("--boards", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write the results to this file")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "HEAD"), help="compare two result files")
    args = parser.parse_args()

    if args.compare:
        base, head = (json.loads(path.read_text()) for path in args.compare)
        return compare(base, head)

    results = run(args.images, args.boards, args.seed)
    for name, result in results["operations"].items():
        print(
            f"{name:40} median {result['median_ms']:8.3f} ms   p95 {result['p95_ms']:8.3f} ms   "
            f"{result['statements']} statements"
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
