"""Transactional coordination for deleting batches of media records and files.

Images and videos have different stores but the same invariant: files are staged first, records
are deleted only for successful stages, and every stage rolls back if the record batch fails. The
adapter is the seam between that policy and the two storage implementations.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class StagedMediaDeleteAdapter:
    kind: str
    stage: Callable[[str], object]
    delete_records: Callable[[list[str]], None]
    rollback: Callable[[object], None]
    commit: Callable[[object], None]
    notify_deleted: Callable[[str], None]
    log_error: Callable[[str], None]


def delete_media_by_names(names: list[str], adapter: StagedMediaDeleteAdapter) -> tuple[list[str], list[str]]:
    """Delete exactly ``names``, preserving records whenever their files cannot be staged."""
    deleted: list[str] = []
    failed: list[str] = []
    staged: list[tuple[str, object]] = []

    for name in names:
        try:
            token = adapter.stage(name)
            staged.append((name, token))
            deleted.append(name)
        except Exception as error:
            failed.append(name)
            adapter.log_error(f"Failed to delete {adapter.kind} file {name}; keeping record: {error}")

    try:
        adapter.delete_records(deleted)
    except Exception:
        for name, token in staged:
            try:
                adapter.rollback(token)
            except Exception as rollback_error:
                adapter.log_error(f"Failed to restore staged {adapter.kind} files for {name}: {rollback_error}")
        raise

    for _, token in staged:
        try:
            adapter.commit(token)
        except Exception as cleanup_error:
            adapter.log_error(f"Failed to purge staged {adapter.kind} files: {cleanup_error}")

    for name in deleted:
        adapter.notify_deleted(name)

    return deleted, failed
