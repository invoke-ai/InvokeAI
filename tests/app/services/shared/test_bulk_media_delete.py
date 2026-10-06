from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.shared.bulk_media_delete import StagedMediaDeleteAdapter, delete_media_by_names


@dataclass
class _Record:
    starred: bool = False


def _adapter(
    records: dict[str, _Record],
    *,
    stage_failures: frozenset[str] = frozenset(),
    delete_records: MagicMock | None = None,
) -> tuple[StagedMediaDeleteAdapter[_Record], MagicMock, MagicMock, MagicMock]:
    rollback = MagicMock()
    commit = MagicMock()
    notify = MagicMock()

    def stage(name: str, record: _Record) -> object:
        if name in stage_failures:
            raise OSError("device busy")
        return f"token:{name}"

    adapter = StagedMediaDeleteAdapter(
        kind="image",
        load=records.__getitem__,
        is_starred=lambda record: record.starred,
        stage=stage,
        delete_records=delete_records or MagicMock(),
        rollback=rollback,
        commit=commit,
        notify_deleted=notify,
        log_error=MagicMock(),
    )
    return adapter, rollback, commit, notify


RECORDS = {"plain.png": _Record(), "star.png": _Record(starred=True), "other.png": _Record()}


def test_starred_names_are_deleted_unless_protection_is_requested() -> None:
    adapter, _, _, _ = _adapter(dict(RECORDS))

    assert delete_media_by_names(["plain.png", "star.png"], adapter) == (["plain.png", "star.png"], [], [])


def test_protection_skips_starred_names_without_staging_or_deleting_them() -> None:
    delete_records = MagicMock()
    adapter, rollback, commit, notify = _adapter(dict(RECORDS), delete_records=delete_records)

    result = delete_media_by_names(["plain.png", "star.png"], adapter, delete_starred=False)

    assert result == (["plain.png"], [], ["star.png"])
    delete_records.assert_called_once_with(["plain.png"])
    commit.assert_called_once_with("token:plain.png")
    notify.assert_called_once_with("plain.png")
    rollback.assert_not_called()


def test_protection_with_only_starred_names_deletes_no_records() -> None:
    delete_records = MagicMock()
    adapter, _, commit, notify = _adapter(dict(RECORDS), delete_records=delete_records)

    assert delete_media_by_names(["star.png"], adapter, delete_starred=False) == ([], [], ["star.png"])

    delete_records.assert_called_once_with([])
    commit.assert_not_called()
    notify.assert_not_called()


def test_a_failed_stage_is_reported_failed_not_starred_and_protection_still_applies_to_the_rest() -> None:
    adapter, _, _, _ = _adapter(dict(RECORDS), stage_failures=frozenset({"other.png"}))

    result = delete_media_by_names(["plain.png", "other.png", "star.png"], adapter, delete_starred=False)

    assert result == (["plain.png"], ["other.png"], ["star.png"])


def test_a_record_delete_failure_restores_only_the_staged_items() -> None:
    delete_records = MagicMock(side_effect=RuntimeError("database is locked"))
    adapter, rollback, commit, notify = _adapter(dict(RECORDS), delete_records=delete_records)

    with pytest.raises(RuntimeError):
        delete_media_by_names(["plain.png", "star.png"], adapter, delete_starred=False)

    rollback.assert_called_once_with("token:plain.png")
    commit.assert_not_called()
    notify.assert_not_called()
