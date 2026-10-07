"""`invoke-db-copy`: copies an install's SQLite database into a new MySQL or MariaDB database."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from invokeai.app.services.config.config_default import InvokeAIAppConfig, load_config_from_root
from invokeai.app.services.shared.database.copy import (
    Problem,
    copy_database,
    copy_records,
    count_normalized,
    delete_orphans,
    find_orphans,
    find_oversized,
    merged_rows,
    verify_copy,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.startup import (
    open_copy_target,
    open_migrated_database,
    redacted_database_url,
)
from invokeai.backend.util.logging import InvokeAILogger

_START_AGAIN = (
    "The target holds part of the copy, which InvokeAI refuses. Drop the target database, create it again empty, and "
    "run invoke-db-copy again."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="invoke-db-copy",
        description="Copy an InvokeAI install's SQLite database into a new, empty MySQL or MariaDB database.",
        epilog=(
            "Stop InvokeAI first: what it writes during the copy is not copied. The SQLite database is not changed. "
            "The target is the database `db_url` names in invokeai.yaml or INVOKEAI_DB_URL, unless --target names "
            "another; InvokeAI uses the copy once db_url names it."
        ),
    )
    parser.add_argument(
        "--target",
        help="URL of the target, e.g. mariadb+pymysql://user:pw@host/db (default: db_url). A URL here is visible to "
        "other users of this computer and kept in the shell's history; prefer db_url",
    )
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


def copy(target_url: Optional[str], *, root: Optional[Path], check_only: bool, skip_orphans: bool) -> int:
    """Runs the copy and reports on it; the process's exit code.

    :param target_url: The target; by default, the database the install's config names.
    """
    config = _install_config(root)
    target_url = target_url or config.db_url
    if not target_url:
        print("Name the target: set db_url in invokeai.yaml or INVOKEAI_DB_URL, or pass --target.")
        return 1
    # The source is the install's SQLite database, whatever db_url says.
    source_config = config.model_copy(update={"db_url": None, "use_memory_db": False})
    logger = InvokeAILogger.get_logger("invoke-db-copy")
    print(f"Source: {source_config.db_path}")
    print(f"Target: {redacted_database_url(target_url)}")
    try:
        source = open_migrated_database(source_config, logger)
        try:
            target, max_allowed_packet = open_copy_target(target_url, logger)
        except BaseException:
            source.dispose()
            raise
    except Exception as e:
        # A database not set up for it, a URL of another kind, a server out of reach, the `mysql` extra missing.
        print(f"\nCannot copy: {e}")
        return 1

    try:
        try:
            # A consistent snapshot to work from, beside the database, which it is as large as: orphans can be left
            # out of it, and the source stays as it is.
            snapshot_dir = Path(tempfile.mkdtemp(prefix="invoke-db-copy-", dir=source_config.db_path.parent))
        except BaseException:
            source.dispose()
            raise
        try:
            snapshot_path = snapshot_dir / "snapshot.db"
            try:
                source.backup(snapshot_path)
            finally:
                source.dispose()
            snapshot = Database.open_sqlite(snapshot_path, logger)
            try:
                return _copy(snapshot, target, max_allowed_packet, check_only=check_only, skip_orphans=skip_orphans)
            finally:
                snapshot.dispose()
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            if snapshot_dir.exists():
                print(
                    f"\nCould not delete the snapshot at {snapshot_dir}, which holds a copy of the database: delete it."
                )
    except Exception as e:
        # Before anything is copied (the copy reports its own failures): the snapshot, or a check of it.
        print(f"\nCannot copy: {e}\nNothing was copied.")
        return 1
    finally:
        target.dispose()


def _copy(
    snapshot: Database, target: Database, max_allowed_packet: int, *, check_only: bool, skip_orphans: bool
) -> int:
    blocking: list[Problem] = []
    orphans = find_orphans(snapshot)
    _report("Rows whose foreign keys name a missing row", orphans)
    if orphans and skip_orphans:
        _report("Left out of the copy", delete_orphans(snapshot))
    elif orphans:
        blocking.extend(orphans)
        print("  Copy with --orphans skip to leave them out.")
    oversized = find_oversized(snapshot, max_allowed_packet)
    _report("Rows the target cannot store", oversized)
    blocking.extend(oversized)
    _report("Values the copy changes", count_normalized(snapshot))

    if blocking:
        print("\nNothing was copied.")
        return 1
    if check_only:
        print("\nThe check found nothing that stops a copy. Nothing was copied.")
        return 0

    print("\nCopying...")
    try:
        copied = copy_database(snapshot, target, progress=lambda table: print(f"  {table}", flush=True))
        print(f"Copied {_rows(sum(copied.values()))} of {len(copied)} tables. Verifying...")
        mismatches = verify_copy(snapshot, target)
        if not mismatches:
            copy_records(snapshot, target)
            mismatches = verify_copy(snapshot, target, records=True)
        merged = merged_rows(snapshot, target)
    except Exception as e:
        print(f"\nThe copy failed: {e}\n{_START_AGAIN}")
        return 1
    if mismatches:
        print("\nThe copy does not match its source:")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        print(_START_AGAIN)
        return 1
    for table, fewer in merged.items():
        print(f"  {table}: {_rows(fewer)} the target treats as equal to others were merged.")
    print("\nDone: every table matches its source. InvokeAI uses the target once db_url names it.")
    return 0


def _install_config(root: Optional[Path]) -> InvokeAIAppConfig:
    if root is None:
        from invokeai.app.services.config import get_config

        root = get_config().root_path
    return load_config_from_root(root)


def _report(heading: str, problems: list[Problem]) -> None:
    if not problems:
        return
    print(f"\n{heading}:")
    for problem in problems:
        print(f"  {problem.table}: {_rows(problem.count)} {problem.what}")


def _rows(count: int) -> str:
    return f"{count} row" if count == 1 else f"{count} rows"


if __name__ == "__main__":
    main()
