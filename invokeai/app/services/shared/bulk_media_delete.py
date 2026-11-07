"""Transactional coordination for deleting batches of media records and files.

Images and videos have different stores but the same invariant: files are staged first, records
are deleted only for successful stages, and every stage rolls back if the record batch fails. When
starred media is protected, the record delete itself refuses starred rows, so a star that lands
after the check still keeps its record, and the files staged for it are restored. The adapter is
the seam between that policy and the two storage implementations.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class StagedMediaDeleteAdapter(Generic[RecordT]):
    kind: str
    load: Callable[[str], RecordT]
    is_starred: Callable[[RecordT], bool]
    stage: Callable[[str, RecordT], object]
    delete_records: Callable[[list[str]], None]
    delete_unstarred_records: Callable[[list[str]], list[str]]
    rollback: Callable[[object], None]
    commit: Callable[[object], None]
    notify_deleted: Callable[[str], None]
    log_error: Callable[[str], None]


def delete_media_by_names(
    names: list[str], adapter: StagedMediaDeleteAdapter[RecordT], delete_starred: bool = True
) -> tuple[list[str], list[str], list[str]]:
    """Delete exactly ``names``, preserving records whenever their files cannot be staged.

    Returns ``(deleted, failed, starred_skipped)``. With ``delete_starred=False`` starred items are
    left untouched and reported in ``starred_skipped``, including ones starred after they were read.
    """
    deleted: list[str] = []
    failed: list[str] = []
    starred_skipped: list[str] = []
    staged: list[tuple[str, object]] = []

    for name in names:
        try:
            record = adapter.load(name)
            if not delete_starred and adapter.is_starred(record):
                starred_skipped.append(name)
                continue
            token = adapter.stage(name, record)
            staged.append((name, token))
            deleted.append(name)
        except Exception as error:
            failed.append(name)
            adapter.log_error(f"Failed to delete {adapter.kind} file {name}; keeping record: {error}")

    try:
        if delete_starred:
            adapter.delete_records(deleted)
        else:
            # The starred check and the delete share one transaction, so only a row nobody starred meanwhile goes.
            kept_names = set(adapter.delete_unstarred_records(deleted))
            for name, token in [entry for entry in staged if entry[0] in kept_names]:
                try:
                    adapter.rollback(token)
                except Exception as rollback_error:
                    adapter.log_error(f"Failed to restore starred {adapter.kind} files for {name}: {rollback_error}")
            staged = [entry for entry in staged if entry[0] not in kept_names]
            starred_skipped.extend(name for name in deleted if name in kept_names)
            deleted = [name for name in deleted if name not in kept_names]
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

    return deleted, failed, starred_skipped
