"""`invoke-db-copy`: copies an install's SQLite database into a new MySQL or MariaDB database."""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

from invokeai.app.services.config.config_default import InvokeAIAppConfig, load_config_from_root
from invokeai.app.services.shared.database.copy import (
    Problem,
    checksums,
    copy_database,
    count_normalized,
    delete_orphans,
    find_orphans,
    find_oversized,
    merged_tables,
    normalize_for_a_server,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.startup import (
    open_copy_target,
    open_migrated_database,
    redacted_database_url,
)
from invokeai.backend.util.logging import InvokeAILogger


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="invoke-db-copy",
        description="Copy an InvokeAI install's SQLite database into a new, empty MySQL or MariaDB database.",
        epilog=(
            "Stop InvokeAI first: what it writes during the copy is not copied. The SQLite database is not changed; "
            "set `db_url` in invokeai.yaml to the target afterwards to use it."
        ),
    )
    parser.add_argument("--target", required=True, help="URL of the target, e.g. mariadb+pymysql://user:pw@host/db")
    parser.add_argument("--root", type=Path, help="InvokeAI root directory (default: as InvokeAI finds it)")
    parser.add_argument(
        "--check", action="store_true", help="Check the source and the target and report, without copying"
    )
    parser.add_argument(
        "--orphans",
        choices=("fail", "skip"),
        default="fail",
        help="Rows whose foreign keys name no row, which a server refuses: stop (default), or leave them out",
    )
    args = parser.parse_args()
    sys.exit(copy(args.target, root=args.root, check_only=args.check, skip_orphans=args.orphans == "skip"))


def copy(target_url: str, *, root: Optional[Path], check_only: bool, skip_orphans: bool) -> int:
    """Runs the copy and reports on it; the process's exit code."""
    config = _source_config(root)
    logger = InvokeAILogger.get_logger("invoke-db-copy")
    print(f"Source: {config.db_path}")
    print(f"Target: {redacted_database_url(target_url)}")
    try:
        source = open_migrated_database(config, logger)
        try:
            target, max_allowed_packet = open_copy_target(target_url, logger)
        except BaseException:
            source.dispose()
            raise
    except Exception as e:
        # A database not set up for it, a URL of another kind, a server out of reach, the `mysql` extra missing.
        print(f"\nCannot copy: {e}")
        return 1

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # A consistent snapshot to work from: orphans can be left out of it, and the source stays as it is.
        snapshot_path = Path(tmp) / "snapshot.db"
        try:
            source.backup(snapshot_path)
        finally:
            source.dispose()
        snapshot = Database.open_sqlite(snapshot_path, logger)
        try:
            return _copy(snapshot, target, max_allowed_packet, check_only=check_only, skip_orphans=skip_orphans)
        finally:
            snapshot.dispose()
            target.dispose()


def _copy(
    snapshot: Database, target: Database, max_allowed_packet: int, *, check_only: bool, skip_orphans: bool
) -> int:
    blocking: list[Problem] = []
    orphans = find_orphans(snapshot)
    _report("Rows whose foreign keys name no row", orphans)
    if orphans and skip_orphans:
        print(f"  Left out: {delete_orphans(snapshot)} rows, and what the database deletes with them.")
    elif orphans:
        blocking.extend(orphans)
        print("  Copy with --orphans skip to leave them out.")
    oversized = find_oversized(snapshot, max_allowed_packet)
    _report("Values the target cannot store", oversized)
    blocking.extend(oversized)
    _report("Values the copy changes", count_normalized(snapshot))

    if blocking:
        print("\nNothing was copied.")
        return 1
    if check_only:
        print("\nThe check found nothing that stops a copy. Nothing was copied.")
        return 0

    print("\nCopying...")
    copied = copy_database(snapshot, target)
    print(f"Copied {sum(copied.values())} rows of {len(copied)} tables. Verifying...")
    expected = checksums(snapshot, normalize_for_a_server)
    actual = checksums(target)
    mismatches: list[str] = []
    for table, sums in expected.items():
        got = actual.get(table)
        if table in merged_tables():
            if got is None or got.rows > sums.rows:
                mismatches.append(f"{table}: {sums.rows} rows, the target holds {got.rows if got else 0}")
            elif got.rows < sums.rows:
                print(f"  {table}: {sums.rows - got.rows} rows the target treats as equal to others were merged.")
        elif got != sums:
            mismatches.append(f"{table}: rows or contents differ ({sums.rows} copied, {got.rows if got else 0} held)")
    if mismatches:
        print("\nThe copy does not match its source; do not use the target:")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        return 1
    print("\nDone: every table matches its source. Set db_url in invokeai.yaml to the target to use it.")
    return 0


def _source_config(root: Optional[Path]) -> InvokeAIAppConfig:
    if root is None:
        from invokeai.app.services.config import get_config

        root = get_config().root_path
    # The source is the install's SQLite database, whatever db_url says.
    return load_config_from_root(root).model_copy(update={"db_url": None, "use_memory_db": False})


def _report(heading: str, problems: list[Problem]) -> None:
    if not problems:
        return
    print(f"\n{heading}:")
    for problem in problems:
        print(f"  {problem.table}: {problem.count} rows {problem.what}")


if __name__ == "__main__":
    main()
