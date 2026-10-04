"""Conditional intermediate deletion shared by the image and video services.

A guard runs on the deleting transaction's cursor, immediately before the `DELETE`, and returns
the subset of the candidate names that may still go. Because the database serializes writers
through one connection and lock, nothing — a project save, an enqueue, a promotion — can make a
name protected between the guard's answer and the record's removal.

Records go first and files second, with a durable journal spanning the two. Staging files before
a conditional record delete would need a restore for every survivor, and a concurrent delete of a
survivor while its files sit staged leaves permanent orphans (PR #9361). Here only the files of
rows this call removed are touched; if the process dies between the commit and the purge, startup
recovery finishes the purge for every journalled name whose record is gone.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence


class IntermediateDeleteGuard(Protocol):
    def __call__(self, cursor: sqlite3.Cursor, names: Sequence[str]) -> list[str]: ...


@dataclass
class IntermediateDeleteResult:
    """What a batch delete removed.

    ``deleted_names`` are records that are committed as gone. ``purge_deferred`` names the subset
    whose files could not be removed yet: the record deletion still stands and startup recovery
    finishes the purge from the journal, so their bytes are not reclaimed yet.
    """

    deleted_names: list[str] = field(default_factory=list)
    purge_deferred: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JournaledDeleteAdapter:
    kind: str
    begin: Callable[[list[tuple[str, str]]], object]
    delete_records: Callable[[list[str], Optional[IntermediateDeleteGuard]], list[str]]
    abandon: Callable[[object], None]
    commit: Callable[[object, list[str]], None]
    notify_deleted: Callable[[str], None]
    log_error: Callable[[str], None]


def delete_journaled_intermediates(
    subfolders: dict[str, str], guard: Optional[IntermediateDeleteGuard], adapter: JournaledDeleteAdapter
) -> IntermediateDeleteResult:
    """Deletes the intermediates in ``subfolders`` (name → subfolder) that the guard still allows.

    The caller holds its media kind's deletion lock across the call so a subfolder move or a
    staged single delete cannot relocate files between the journal and the purge.
    """
    token = adapter.begin(list(subfolders.items()))
    try:
        deleted = adapter.delete_records(list(subfolders.keys()), guard)
    except Exception:
        try:
            adapter.abandon(token)
        except Exception as cleanup_error:
            adapter.log_error(f"Failed to discard the intermediate {adapter.kind} delete journal: {cleanup_error}")
        raise
    result = IntermediateDeleteResult(deleted_names=deleted)
    try:
        # The journal still lists survivors, which is harmless: recovery re-checks every entry
        # against the record store.
        adapter.commit(token, deleted)
    except Exception as cleanup_error:
        # The records are committed as gone; a failed purge is retried at startup and must neither
        # fail the operation nor undo the deletions.
        adapter.log_error(f"Failed to purge intermediate {adapter.kind} files: {cleanup_error}")
        result.purge_deferred = list(deleted)
    for name in deleted:
        adapter.notify_deleted(name)
    return result
