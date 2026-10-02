"""The intermediates cleanup engine against a real database, real files and the real queue tables.

Every test drives the public service interface; the eligibility policy is exercised through the
same SQL that production runs, never through a stub that decides what is safe.
"""

import time
import uuid
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordChanges,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.intermediates import intermediates_default
from invokeai.app.services.intermediates.intermediates_base import IntermediatesCaller
from invokeai.app.services.intermediates.intermediates_common import (
    IntermediatesOperation,
    IntermediatesOperationNotFoundError,
    IntermediatesOperationRequest,
    IntermediatesPreview,
    IntermediatesPreviewNotFoundError,
    IntermediatesPreviewRequest,
    IntermediatesScope,
    IntermediatesScopeForbiddenError,
    IntermediatesScopeInvalidError,
    IntermediatesScopeTarget,
    IntermediatesUnavailableError,
)
from invokeai.app.services.intermediates.intermediates_default import IntermediatesService
from invokeai.app.services.intermediates.intermediates_records_sqlite import IntermediatesRecordsSqlite
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.video_files.video_files_disk import DiskVideoFileStorage
from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException
from invokeai.app.services.videos.videos_default import VideoService

ADMIN = IntermediatesCaller(user_id="admin", is_admin=True)
ALICE = IntermediatesCaller(user_id="alice", is_admin=False)
BOB = IntermediatesCaller(user_id="bob", is_admin=False)


@pytest.fixture
def invoker(mock_invoker: Invoker, tmp_path: Path) -> Invoker:
    """The shared mock invoker with real image/video file storage and a real queue table."""
    services = mock_invoker.services
    services.image_files = DiskImageFileStorage(tmp_path / "outputs")
    services.image_files.start(mock_invoker)
    services.video_files = DiskVideoFileStorage(str(tmp_path / "outputs" / "videos"))
    services.videos = VideoService()
    services.videos.start(mock_invoker)
    services.session_queue = SqliteSessionQueue(db=services.image_records._db)
    services.image_moves = None
    # These are multi-account scenarios; the single-user case is covered explicitly below.
    services.configuration.multiuser = True
    with services.image_records._db.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO users (user_id, email, display_name, password_hash, is_admin, is_active)"
            " VALUES (?, ?, ?, 'x', ?, 1);",
            [
                ("admin", "admin@example.com", "Admin", 1),
                ("alice", "alice@example.com", "Alice", 0),
                ("bob", "bob@example.com", "Bob", 0),
            ],
        )
    return mock_invoker


@pytest.fixture
def service(invoker: Invoker) -> IntermediatesService:
    svc = IntermediatesService(records=IntermediatesRecordsSqlite(db=invoker.services.image_records._db))
    invoker.services.intermediates = svc
    svc.start(invoker)
    yield svc
    svc.stop()


def _seed_image(
    invoker: Invoker,
    name: str,
    *,
    user_id: str = "alice",
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    is_intermediate: bool = True,
    created_at: Optional[str] = "2020-01-01 00:00:00.000",
    with_file: bool = True,
    size: Optional[int] = None,
) -> None:
    records = invoker.services.image_records
    records.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        has_workflow=False,
        is_intermediate=is_intermediate,
        session_id=session_id,
        user_id=user_id,
        project_id=project_id,
    )
    if created_at is not None:
        with records._db.transaction() as cursor:
            cursor.execute("UPDATE images SET created_at = ? WHERE image_name = ?;", (created_at, name))
    if with_file:
        invoker.services.image_files.save(image=Image.new("RGB", (8, 8)), image_name=name)
        measured = invoker.services.image_files.get_file_size_bytes(name)
        records.set_file_size_bytes(name, size if size is not None else measured)


def _seed_video(invoker: Invoker, name: str, *, user_id: str = "alice", is_intermediate: bool = True) -> None:
    records = invoker.services.video_records
    records.save(
        video_name=name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        is_intermediate=is_intermediate,
        user_id=user_id,
    )
    with records._db.transaction() as cursor:
        cursor.execute("UPDATE videos SET created_at = '2020-01-01 00:00:00.000' WHERE video_name = ?;", (name,))
    files = invoker.services.video_files
    path = files.get_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    records.set_file_size_bytes(name, 64)


def _enqueue_row(
    invoker: Invoker,
    *,
    session_id: str,
    status: str,
    user_id: str = "alice",
    session_json: str = "{}",
    root_item_id: Optional[int] = None,
    parent_item_id: Optional[int] = None,
) -> int:
    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO session_queue "
            "(queue_id, session, session_id, batch_id, priority, user_id, status, root_item_id, parent_item_id)"
            " VALUES ('default', ?, ?, ?, 0, ?, ?, ?, ?);",
            (session_json, session_id, uuid.uuid4().hex, user_id, status, root_item_id, parent_item_id),
        )
        return int(cursor.lastrowid or 0)


def _project(invoker: Invoker, user_id: str, name: str, data: dict) -> str:
    return invoker.services.project_records.create(user_id, name, data).project_id


def _end_cached_media_grace(service: IntermediatesService, invoker: Invoker) -> None:
    """Lets the recency grace of cached media whose consuming sessions have ended run out."""
    service._records.summarize(None)
    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute(
            "UPDATE temp.intermediates_session_media SET released_at = '2000-01-01 00:00:00.000' "
            "WHERE released_at IS NOT NULL;"
        )


def _wait(
    service: IntermediatesService, operation_id: str, caller: IntermediatesCaller = ALICE
) -> IntermediatesOperation:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        operation = service.get_operation(operation_id, caller)
        if operation.status in ("completed", "failed"):
            return operation
        time.sleep(0.02)
    raise AssertionError("operation did not finish")


def _start(
    service: IntermediatesService, preview: IntermediatesPreview, caller: IntermediatesCaller = ALICE
) -> IntermediatesOperation:
    return service.start_operation(IntermediatesOperationRequest(preview_id=preview.preview_id), caller)


def _run(service: IntermediatesService, caller: IntermediatesCaller, scope: IntermediatesScope, mode: str = "safe"):
    preview = service.create_preview(IntermediatesPreviewRequest(mode=mode, scope=scope), caller)
    return preview, _wait(service, _start(service, preview, caller).operation_id, caller)


def _before_first_delete(
    invoker: Invoker, monkeypatch: pytest.MonkeyPatch, mutate: Callable[[], None], *, kind: str = "image"
) -> None:
    """Runs ``mutate`` between the first batch's page read and its final check.

    Batches are paged live, so a change made before the operation starts is simply never paged;
    this window is the one only the guard on the deleting transaction covers.
    """
    owner = invoker.services.images if kind == "image" else invoker.services.videos
    real = owner.delete_intermediates_by_names
    done = False

    def wrapped(names, guard=None):
        nonlocal done
        if not done:
            done = True
            mutate()
        return real(names, guard)

    monkeypatch.setattr(owner, "delete_intermediates_by_names", wrapped)


def _owner(user_id: str) -> IntermediatesScope:
    return IntermediatesScope(kind="owner", user_id=user_id)


def _exists(invoker: Invoker, name: str) -> bool:
    try:
        invoker.services.image_records.get(name)
        return True
    except ImageRecordNotFoundException:
        return False


# ── summary ──


def test_summary_groups_rows_and_classifies_under_the_policy(invoker: Invoker, service: IntermediatesService) -> None:
    project = _project(
        invoker, "alice", "Portraits", {"canvas": {"stagingArea": {"pendingImages": [{"imageName": "staged.png"}]}}}
    )
    _seed_image(invoker, "safe.png", project_id=project, size=100)
    _seed_image(invoker, "staged.png", project_id=project, size=50)
    _seed_image(invoker, "active-output.png", project_id=project, session_id="s-active")
    _seed_image(invoker, "active-input.png", size=30)
    _seed_image(invoker, "fresh.png", created_at=None)
    _seed_image(invoker, "unmeasured.png", project_id=project, size=None)
    invoker.services.image_records.set_file_size_bytes("unmeasured.png", None)
    _seed_image(invoker, "deleted-project.png", project_id="gone", size=7)
    _seed_image(invoker, "durable.png", project_id=project, is_intermediate=False)
    invoker.services.board_image_records.add_image_to_board(
        invoker.services.project_records.get_board_id("alice", project), "durable.png"
    )
    _seed_image(invoker, "bobs.png", user_id="bob", size=9)
    _enqueue_row(invoker, session_id="s-active", status="in_progress")
    _enqueue_row(
        invoker,
        session_id="s-pending",
        status="pending",
        session_json='{"graph": {"nodes": {"a": {"image": {"image_name": "active-input.png"}}}}}',
    )
    _enqueue_row(invoker, session_id="s-done", status="completed", session_json='{"image_name": "safe.png"}')

    summary = service.get_summary(
        ALICE, owner_id=None, search=None, sort="reclaimable_bytes", descending=True, offset=0, limit=50
    )

    rows = {(row.project_id, row.project_name): row for row in summary.items}
    assert set(rows) == {(project, "Portraits"), (None, None)}
    portraits = rows[(project, "Portraits")]
    assert (portraits.images.safe, portraits.images.referenced, portraits.images.active, portraits.images.recent) == (
        2,
        1,
        1,
        0,
    )
    # The board's newest durable image is the row's thumbnail; intermediates never are.
    assert portraits.cover_image_name == "durable.png"
    assert portraits.reclaimable_bytes == 100
    assert portraits.referenced_bytes == 50
    assert portraits.unknown_size_count == 1
    unassigned = rows[(None, None)]
    assert (unassigned.images.safe, unassigned.images.active, unassigned.images.recent) == (1, 1, 1)
    assert unassigned.reclaimable_bytes == 7
    assert summary.totals.safe_images == 3
    assert summary.totals.in_use_images == 4
    assert summary.totals.reclaimable_bytes == 107
    assert summary.can_manage_everyone is False
    assert all(row.user_id == "alice" for row in summary.items)


def test_summary_scopes_owners_search_sort_and_pages(invoker: Invoker, service: IntermediatesService) -> None:
    alpha = _project(invoker, "alice", "Alpha", {})
    beta = _project(invoker, "alice", "Beta", {})
    _seed_image(invoker, "a.png", project_id=alpha, size=10)
    _seed_image(invoker, "b.png", project_id=beta, size=20)
    _seed_image(invoker, "bob.png", user_id="bob", size=30)

    everyone = service.get_summary(
        ADMIN, owner_id=None, search=None, sort="reclaimable_bytes", descending=True, offset=0, limit=50
    )
    assert [(row.user_id, row.project_name) for row in everyone.items] == [
        ("bob", None),
        ("alice", "Beta"),
        ("alice", "Alpha"),
    ]
    assert everyone.items[0].user_display_name == "Bob"
    assert everyone.can_manage_everyone is True

    by_name = service.get_summary(
        ADMIN, owner_id="alice", search=None, sort="project_name", descending=False, offset=1, limit=1
    )
    assert [row.project_name for row in by_name.items] == ["Beta"]
    assert by_name.total == 2 and by_name.totals.reclaimable_bytes == 30

    focused = service.get_summary(
        ALICE,
        owner_id=None,
        project_id=beta,
        search=None,
        sort="project_name",
        descending=False,
        offset=0,
        limit=1,
    )
    assert [(row.project_id, row.project_name) for row in focused.items] == [(beta, "Beta")]
    assert focused.total == 1

    searched = service.get_summary(
        ADMIN, owner_id=None, search="bo", sort="project_name", descending=False, offset=0, limit=50
    )
    assert [row.user_id for row in searched.items] == ["bob"]
    # Owner names are not searchable for a non-admin, whose rows are all their own anyway.
    assert (
        service.get_summary(
            ALICE, owner_id=None, search="alice", sort="project_name", descending=False, offset=0, limit=50
        ).items
        == []
    )

    with pytest.raises(IntermediatesScopeForbiddenError):
        service.get_summary(
            ALICE, owner_id="bob", search=None, sort="project_name", descending=False, offset=0, limit=50
        )


def test_summary_triggers_measurement_of_unmeasured_sizes(invoker: Invoker, service: IntermediatesService) -> None:
    _seed_image(invoker, "later.png", size=None)
    invoker.services.image_records.set_file_size_bytes("later.png", None)
    _seed_image(invoker, "missing.png", with_file=False)

    first = service.get_summary(
        ALICE, owner_id=None, search=None, sort="project_name", descending=False, offset=0, limit=50
    )
    assert first.measuring is True

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not service._records.has_unmeasured_intermediates():
            break
        time.sleep(0.02)
    expected = invoker.services.image_files.get_file_size_bytes("later.png")
    assert invoker.services.image_records.get("later.png").file_size_bytes == expected
    # A missing file measures as nothing on disk, not as unknown.
    assert invoker.services.image_records.get("missing.png").file_size_bytes == 0


# ── previews and policy ──


def test_safe_cleanup_deletes_only_safe_items_and_reports_what_it_kept(
    invoker: Invoker, service: IntermediatesService
) -> None:
    project = _project(invoker, "alice", "P", {"layers": [{"imageName": "referenced.png"}]})
    _seed_image(invoker, "safe.png", project_id=project, size=100)
    _seed_image(invoker, "referenced.png", project_id=project)
    _seed_image(invoker, "active.png", project_id=project, session_id="s1")
    _seed_image(invoker, "fresh.png", project_id=project, created_at=None)
    _seed_video(invoker, "safe.mp4")
    _enqueue_row(invoker, session_id="s1", status="pending")
    deleted_callbacks: list[str] = []
    invoker.services.images.on_deleted(deleted_callbacks.append)

    preview, operation = _run(service, ALICE, _owner("alice"))

    assert (preview.impact.delete_images, preview.impact.delete_videos) == (1, 1)
    assert (
        preview.impact.keep_referenced_images,
        preview.impact.keep_active_images,
        preview.impact.keep_recent_images,
    ) == (1, 1, 1)
    assert preview.impact.reclaimable_bytes == 164
    assert preview.affected_documents == []
    assert operation.status == "completed"
    assert (operation.target_images, operation.target_videos) == (1, 1)
    assert (operation.progress.deleted_images, operation.progress.deleted_videos) == (1, 1)
    assert operation.progress.reclaimed_bytes == 164
    assert not _exists(invoker, "safe.png")
    assert not invoker.services.image_files.get_path("safe.png").exists()
    assert not invoker.services.video_files.get_path("safe.mp4").exists()
    with pytest.raises(VideoRecordNotFoundException):
        invoker.services.video_records.get("safe.mp4")
    assert all(_exists(invoker, name) for name in ("referenced.png", "active.png", "fresh.png"))
    assert deleted_callbacks == ["safe.png"]
    assert list(invoker.services.image_files.image_root.glob(".delete_*")) == []


def test_force_cleanup_deletes_referenced_items_but_never_active_or_recent(
    invoker: Invoker, service: IntermediatesService
) -> None:
    project = _project(
        invoker,
        "alice",
        "Drafts",
        {"stagingArea": {"pendingImages": [{"imageName": "staged.png"}, {"imageName": "shared.png"}]}},
    )
    workflow = invoker.services.workflow_records.create(_workflow_naming("shared.png"), user_id="bob")
    _seed_image(invoker, "staged.png", project_id=project, size=40)
    _seed_image(invoker, "shared.png", project_id=project, size=40)
    _seed_image(invoker, "active.png", project_id=project, session_id="s1")
    _seed_image(invoker, "fresh.png", project_id=project, created_at=None)
    _enqueue_row(invoker, session_id="s1", status="waiting")

    preview = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("alice")), ALICE)
    # Bob's workflow is not Alice's to break: the media it names is kept, not merely undescribed.
    assert (preview.impact.delete_images, preview.impact.keep_referenced_images) == (1, 1)
    assert (preview.impact.keep_active_images, preview.impact.keep_recent_images) == (1, 1)
    assert preview.impact.reclaimable_bytes == 40
    assert [(doc.kind, doc.name, doc.references) for doc in preview.affected_documents] == [("project", "Drafts", 1)]

    admin_preview = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("alice")), ADMIN)
    assert admin_preview.impact.delete_images == 2
    assert {
        (doc.kind, doc.owner_id, doc.references, doc.user_display_name) for doc in admin_preview.affected_documents
    } == {
        ("project", project, 2, "Alice"),
        ("workflow", workflow.workflow_id, 1, "Bob"),
    }

    operation = _wait(service, _start(service, preview).operation_id)
    assert operation.status == "completed"
    assert not _exists(invoker, "staged.png")
    assert all(_exists(invoker, name) for name in ("shared.png", "active.png", "fresh.png"))


def test_force_cleanup_keeps_media_another_account_references_after_the_preview(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(invoker, "alice", "Drafts", {"layers": [{"imageName": "staged.png"}]})
    _seed_image(invoker, "staged.png", project_id=project)
    preview = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("alice")), ALICE)
    assert preview.impact.delete_images == 1
    _before_first_delete(
        invoker,
        monkeypatch,
        lambda: invoker.services.workflow_records.create(_workflow_naming("staged.png"), user_id="bob"),
    )

    operation = _wait(service, _start(service, preview).operation_id)

    assert operation.progress.retained_images == 1
    assert _exists(invoker, "staged.png")


def _workflow_naming(image_name: str):
    from invokeai.app.services.workflow_records.workflow_records_common import (
        WorkflowCategory,
        WorkflowMeta,
        WorkflowWithoutID,
    )

    return WorkflowWithoutID(
        name="W",
        author="",
        description="",
        version="1.0.0",
        contact="",
        tags="",
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.User),
        nodes=[{"id": "n", "data": {"image": {"image_name": image_name}}}],
        edges=[],
    )


def test_selection_scope_targets_exactly_the_chosen_rows(invoker: Invoker, service: IntermediatesService) -> None:
    keep = _project(invoker, "alice", "Keep", {})
    clear = _project(invoker, "alice", "Clear", {})
    _seed_image(invoker, "keep.png", project_id=keep)
    _seed_image(invoker, "clear.png", project_id=clear)
    _seed_image(invoker, "unassigned.png")
    _seed_image(invoker, "orphaned-project.png", project_id="gone")

    scope = IntermediatesScope(
        kind="selection",
        targets=[
            IntermediatesScopeTarget(user_id="alice", project_id=clear),
            IntermediatesScopeTarget(user_id="alice", project_id=None),
            IntermediatesScopeTarget(user_id="alice", project_id=None),
        ],
    )
    preview, operation = _run(service, ALICE, scope)

    assert preview.target_rows == 2
    assert preview.impact.delete_images == 3
    assert operation.progress.deleted_images == 3
    assert _exists(invoker, "keep.png")
    assert not any(_exists(invoker, n) for n in ("clear.png", "unassigned.png", "orphaned-project.png"))


def test_a_row_that_was_recent_at_preview_time_stays_recent_for_the_operation(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta, timezone

    from invokeai.app.services.intermediates import intermediates_records_sqlite
    from invokeai.app.services.intermediates.intermediates_common import RECENT_GRACE_SECONDS

    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    readings = iter([start] + [start + timedelta(hours=1)] * 100)
    monkeypatch.setattr(intermediates_records_sqlite, "_utc_now", lambda: next(readings))
    crossing = start - timedelta(seconds=RECENT_GRACE_SECONDS - 1)
    _seed_image(invoker, "crossing.png", created_at=crossing.strftime("%Y-%m-%d %H:%M:%S.000"))
    _seed_image(invoker, "old.png")

    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert (preview.impact.keep_recent_images, preview.impact.delete_images) == (1, 1)

    # An hour later the row would be safe under a live clock; the operation judges recency as of its preview.
    operation = _wait(service, _start(service, preview).operation_id)
    assert operation.progress.deleted_images == 1
    assert _exists(invoker, "crossing.png") and not _exists(invoker, "old.png")


def test_the_final_check_keeps_targets_protected_after_the_page_read(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(invoker, "alice", "P", {})
    _seed_image(invoker, "promoted.png", project_id=project)
    _seed_image(invoker, "now-referenced.png", project_id=project)
    _seed_image(invoker, "now-active.png", project_id=project, session_id="s-late")
    _seed_image(invoker, "still-safe.png", project_id=project)
    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert preview.impact.delete_images == 4

    def protect_three() -> None:
        invoker.services.image_records.update("promoted.png", ImageRecordChanges(is_intermediate=False))
        invoker.services.project_records.update(
            "alice", project, expected_revision=1, name="P", data={"imageName": "now-referenced.png"}
        )
        _enqueue_row(invoker, session_id="s-late", status="pending")

    _before_first_delete(invoker, monkeypatch, protect_three)
    operation = _wait(service, _start(service, preview).operation_id)

    assert operation.progress.deleted_images == 1
    assert operation.progress.retained_images == 3
    assert all(_exists(invoker, n) for n in ("promoted.png", "now-referenced.png", "now-active.png"))
    assert not _exists(invoker, "still-safe.png")
    assert invoker.services.image_files.get_path("promoted.png").exists()


# ── authorization ──


def test_non_admins_are_confined_to_their_own_account(invoker: Invoker, service: IntermediatesService) -> None:
    with pytest.raises(IntermediatesScopeForbiddenError):
        service.create_preview(
            IntermediatesPreviewRequest(mode="safe", scope=IntermediatesScope(kind="everyone")), ALICE
        )
    with pytest.raises(IntermediatesScopeForbiddenError):
        service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("bob")), ALICE)
    with pytest.raises(IntermediatesScopeForbiddenError):
        service.create_preview(
            IntermediatesPreviewRequest(
                mode="safe",
                scope=IntermediatesScope(
                    kind="selection",
                    targets=[IntermediatesScopeTarget(user_id="alice"), IntermediatesScopeTarget(user_id="bob")],
                ),
            ),
            ALICE,
        )
    with pytest.raises(IntermediatesScopeForbiddenError):
        service.create_preview(
            IntermediatesPreviewRequest(mode="safe", scope=IntermediatesScope(kind="matching", user_id="bob")), ALICE
        )


def test_admins_can_clear_everyone_and_a_chosen_account(invoker: Invoker, service: IntermediatesService) -> None:
    _seed_image(invoker, "alice.png", user_id="alice")
    _seed_image(invoker, "bob.png", user_id="bob")
    _seed_image(invoker, "admin.png", user_id="admin")

    _, first = _run(service, ADMIN, _owner("bob"))
    assert first.progress.deleted_images == 1 and _exists(invoker, "alice.png")

    _, second = _run(service, ADMIN, IntermediatesScope(kind="everyone"))
    assert second.progress.deleted_images == 2
    assert not any(_exists(invoker, n) for n in ("alice.png", "bob.png", "admin.png"))


def test_a_demoted_admin_stops_at_the_next_batch(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intermediates_default, "DELETE_BATCH_SIZE", 1)
    for index in range(3):
        _seed_image(invoker, f"bob-{index}.png", user_id="bob")
    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("bob")), ADMIN)
    real_get = invoker.services.users.get
    calls = {"n": 0}

    def demote_after_first_batch(user_id: str):
        user = real_get(user_id)
        calls["n"] += 1
        if user is not None and user_id == "admin" and calls["n"] > 1:
            return user.model_copy(update={"is_admin": False})
        return user

    monkeypatch.setattr(invoker.services.users, "get", demote_after_first_batch)

    operation = _wait(service, _start(service, preview, ADMIN).operation_id, ADMIN)

    assert operation.status == "failed"
    assert "no longer administers" in (operation.error or "")
    assert operation.progress.deleted_images == 1
    assert sum(_exists(invoker, f"bob-{i}.png") for i in range(3)) == 2


def test_a_demoted_admin_keeps_media_other_accounts_documents_name_in_its_own_scope(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_image(invoker, "admin-own.png", user_id="admin")
    invoker.services.workflow_records.create(_workflow_naming("admin-own.png"), user_id="bob")
    preview = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("admin")), ADMIN)
    assert preview.impact.delete_images == 1
    assert [doc.user_id for doc in preview.affected_documents] == ["bob"]
    real_get = invoker.services.users.get

    def demoted(user_id: str):
        user = real_get(user_id)
        return user.model_copy(update={"is_admin": False}) if user is not None and user_id == "admin" else user

    monkeypatch.setattr(invoker.services.users, "get", demoted)

    operation = _wait(service, _start(service, preview, ADMIN).operation_id, ADMIN)

    assert operation.status == "completed"
    assert operation.progress.deleted_images == 0
    assert _exists(invoker, "admin-own.png")


# ── operations ──


def test_operations_are_single_use_and_private(invoker: Invoker, service: IntermediatesService) -> None:
    _seed_image(invoker, "a.png")
    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)

    with pytest.raises(IntermediatesPreviewNotFoundError):
        _start(service, preview, BOB)

    first = _start(service, preview)
    _wait(service, first.operation_id)
    with pytest.raises(IntermediatesPreviewNotFoundError):
        _start(service, preview)

    with pytest.raises(IntermediatesOperationNotFoundError):
        service.get_operation(first.operation_id, BOB)
    assert service.get_operation(first.operation_id, ADMIN).status == "completed"


def test_operations_are_listed_newest_first_and_only_to_their_caller(
    invoker: Invoker, service: IntermediatesService
) -> None:
    _seed_image(invoker, "a.png")
    _seed_image(invoker, "b.png", user_id="bob")
    _, first = _run(service, ALICE, _owner("alice"))
    _, second = _run(service, ALICE, _owner("alice"))
    _, bobs = _run(service, BOB, _owner("bob"))

    assert [op.operation_id for op in service.list_operations(ALICE)] == [second.operation_id, first.operation_id]
    assert [op.operation_id for op in service.list_operations(BOB)] == [bobs.operation_id]
    assert service.list_operations(ADMIN) == []


def test_a_failed_batch_is_reported_and_a_later_run_finishes_it(
    invoker: Invoker, service: IntermediatesService, monkeypatch
) -> None:
    monkeypatch.setattr(intermediates_default, "DELETE_BATCH_SIZE", 1)
    _seed_image(invoker, "ok.png")
    _seed_image(invoker, "flaky.png")
    real_delete = invoker.services.images.delete_intermediates_by_names

    def fail_flaky_once(names, guard=None):
        if "flaky.png" in names and not getattr(fail_flaky_once, "failed", False):
            fail_flaky_once.failed = True  # type: ignore[attr-defined]
            raise OSError("disk hiccup")
        return real_delete(names, guard)

    monkeypatch.setattr(invoker.services.images, "delete_intermediates_by_names", fail_flaky_once)

    _, operation = _run(service, ALICE, _owner("alice"))
    assert operation.status == "completed"
    assert (operation.progress.deleted_images, operation.progress.failed_images) == (1, 1)
    assert _exists(invoker, "flaky.png")

    _seed_image(invoker, "new-since.png", created_at=None)
    second, finished = _run(service, ALICE, _owner("alice"))
    assert second.impact.delete_images == 1
    assert finished.progress.deleted_images == 1
    assert not _exists(invoker, "flaky.png")
    assert _exists(invoker, "new-since.png")


def test_a_purge_failure_counts_as_pending_disk_cleanup_not_reclaimed(
    invoker: Invoker, service: IntermediatesService, monkeypatch
) -> None:
    _seed_image(invoker, "stuck.png", size=500)
    monkeypatch.setattr(invoker.services.image_files, "commit_delete", MagicMock(side_effect=OSError("busy")))

    _, operation = _run(service, ALICE, _owner("alice"))

    assert operation.status == "completed"
    assert operation.progress.deleted_images == 1
    assert operation.progress.reclaimed_bytes == 0
    assert operation.progress.pending_disk_cleanup == 1
    assert not _exists(invoker, "stuck.png")
    # The journal survives for the next startup to finish the purge.
    assert len(list(invoker.services.image_files.image_root.glob(".delete_*"))) == 1


def test_failed_video_purge_is_recovered_after_restart(invoker: Invoker, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_video(invoker, "recover.mp4")
    files = invoker.services.video_files
    path = files.get_path("recover.mp4")
    original_unlink = Path.unlink
    failed_once = False

    def fail_once(candidate: Path, *args, **kwargs) -> None:
        nonlocal failed_once
        if candidate == path and not failed_once:
            failed_once = True
            raise OSError("video file temporarily locked")
        original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    result = invoker.services.videos.delete_intermediates_by_names(["recover.mp4"])

    assert result.deleted_names == ["recover.mp4"]
    assert result.purge_deferred == ["recover.mp4"]
    assert path.exists()
    DiskVideoFileStorage(path.parent).start(invoker)
    assert not path.exists()


def test_cleanup_refuses_to_start_or_continue_during_storage_maintenance(
    invoker: Invoker, service: IntermediatesService, monkeypatch
) -> None:
    monkeypatch.setattr(intermediates_default, "DELETE_BATCH_SIZE", 1)
    _seed_image(invoker, "one.png")
    _seed_image(invoker, "two.png")
    moves = MagicMock()
    moves.is_maintenance_active.return_value = True
    invoker.services.image_moves = moves
    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    with pytest.raises(IntermediatesUnavailableError):
        _start(service, preview)

    invoker.services.image_moves = None
    started = _start(service, preview)
    # Halt after the first batch: the second sees maintenance and stops with the rest untouched.
    real_delete = invoker.services.images.delete_intermediates_by_names

    def delete_then_start_maintenance(names, guard=None):
        result = real_delete(names, guard)
        invoker.services.image_moves = moves
        return result

    monkeypatch.setattr(invoker.services.images, "delete_intermediates_by_names", delete_then_start_maintenance)
    operation = _wait(service, started.operation_id)
    assert operation.status == "failed"
    assert operation.progress.deleted_images == 1
    assert operation.progress.failed_images == 0
    invoker.services.image_moves = None
    monkeypatch.setattr(invoker.services.images, "delete_intermediates_by_names", real_delete)
    second, finished = _run(service, ALICE, _owner("alice"))
    assert second.impact.delete_images == 1 and finished.progress.deleted_images == 1


def test_operation_events_reach_the_bus(invoker: Invoker, service: IntermediatesService) -> None:
    from invokeai.app.services.events.events_common import IntermediatesOperationChangedEvent

    _seed_image(invoker, "a.png")
    _, operation = _run(service, ALICE, _owner("alice"))

    statuses = [
        e.operation.status
        for e in invoker.services.events.events  # type: ignore[attr-defined]
        if isinstance(e, IntermediatesOperationChangedEvent) and e.operation.operation_id == operation.operation_id
    ]
    assert statuses[0] == "pending" and statuses[-1] == "completed" and "running" in statuses
    assert all(
        e.user_id == "alice"
        for e in invoker.services.events.events
        if isinstance(e, IntermediatesOperationChangedEvent)
    )  # type: ignore[attr-defined]


def test_a_restart_forgets_operations_and_a_fresh_run_finishes_the_work(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_image(invoker, "restart-flaky.png")
    real_delete = invoker.services.images.delete_intermediates_by_names
    monkeypatch.setattr(
        invoker.services.images, "delete_intermediates_by_names", MagicMock(side_effect=OSError("busy"))
    )
    _, failed = _run(service, ALICE, _owner("alice"))
    assert failed.progress.failed_images == 1
    service.stop()

    restored = IntermediatesService(IntermediatesRecordsSqlite(invoker.services.image_records._db))
    invoker.services.intermediates = restored
    restored.start(invoker)
    monkeypatch.setattr(invoker.services.images, "delete_intermediates_by_names", real_delete)
    try:
        with pytest.raises(IntermediatesOperationNotFoundError):
            restored.get_operation(failed.operation_id, ALICE)
        assert restored.list_operations(ALICE) == []
        _, finished = _run(restored, ALICE, _owner("alice"))
        assert finished.progress.deleted_images == 1
        assert not _exists(invoker, "restart-flaky.png")
    finally:
        restored.stop()


# ── legacy ──


def test_legacy_clear_all_deletes_every_safe_image_and_nothing_else(
    invoker: Invoker, service: IntermediatesService
) -> None:
    _project(invoker, "alice", "P", {"imageName": "referenced.png"})
    _seed_image(invoker, "alice-safe.png", user_id="alice")
    _seed_image(invoker, "bob-safe.png", user_id="bob")
    _seed_image(invoker, "referenced.png")
    _seed_image(invoker, "active.png", session_id="s1")
    _seed_video(invoker, "video.mp4")
    _enqueue_row(invoker, session_id="s1", status="pending")

    with pytest.raises(IntermediatesScopeForbiddenError):
        service.clear_all_images_now(ALICE)
    assert service.clear_all_images_now(ADMIN) == 2

    assert not _exists(invoker, "alice-safe.png") and not _exists(invoker, "bob-safe.png")
    assert _exists(invoker, "referenced.png") and _exists(invoker, "active.png")
    assert invoker.services.video_records.get("video.mp4").is_intermediate is True


# ── review follow-ups ──


def test_single_user_mode_clears_everyone_without_an_admin_row(invoker: Invoker, service: IntermediatesService) -> None:
    """The default install: every request is the local administrator, but the `system` row is not an admin."""
    invoker.services.configuration.multiuser = False
    _seed_image(invoker, "one.png", user_id="system")
    _seed_image(invoker, "two.png", user_id="alice")

    _, operation = _run(
        service, IntermediatesCaller(user_id="system", is_admin=True), IntermediatesScope(kind="everyone")
    )

    assert operation.status == "completed", operation.error
    assert operation.progress.deleted_images == 2


def test_inputs_added_to_an_active_session_after_it_was_scanned_are_protected(
    invoker: Invoker, service: IntermediatesService
) -> None:
    """A workflow-call parent is rewritten with its child's outputs while it stays active."""
    _seed_image(invoker, "child-output.png")
    item_id = _enqueue_row(invoker, session_id="parent", status="waiting", session_json='{"graph": {}}')
    first = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert first.impact.delete_images == 1

    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute(
            "UPDATE session_queue SET session = ?, status = 'pending' WHERE item_id = ?;",
            ('{"results": {"n": {"image": {"image_name": "child-output.png"}}}}', item_id),
        )

    second = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert second.impact.delete_images == 0
    assert second.impact.keep_active_images == 1


def test_a_save_cannot_slip_between_the_final_check_and_the_delete(
    invoker: Invoker, service: IntermediatesService, monkeypatch
) -> None:
    """The guard and the DELETE share one transaction under the database lock, so a project save that
    would protect a target either lands before the check (target kept) or after the commit."""
    import threading

    project = _project(invoker, "alice", "P", {})
    _seed_image(invoker, "raced.png", project_id=project)
    records = service._records
    real_guard_factory = records.make_delete_guard
    guard_entered = threading.Event()
    guard_may_finish = threading.Event()
    delete_committed_at: list[float] = []

    def slow_guard_factory(kind, **kwargs):
        guard = real_guard_factory(kind, **kwargs)

        def paused(cursor, names):
            guard_entered.set()
            assert guard_may_finish.wait(timeout=10)
            return guard(cursor, names)

        return paused

    monkeypatch.setattr(records, "make_delete_guard", slow_guard_factory)
    real_record_delete = invoker.services.image_records.delete_intermediates_by_names

    def timed_record_delete(names, guard=None):
        result = real_record_delete(names, guard=guard)
        delete_committed_at.append(time.monotonic())
        return result

    monkeypatch.setattr(invoker.services.image_records, "delete_intermediates_by_names", timed_record_delete)

    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    started = _start(service, preview)
    assert guard_entered.wait(timeout=10)

    save_finished_at: list[float] = []

    def save_reference() -> None:
        invoker.services.project_records.update(
            "alice", project, expected_revision=1, name="P", data={"imageName": "raced.png"}
        )
        save_finished_at.append(time.monotonic())

    saver = threading.Thread(target=save_reference)
    saver.start()
    saver.join(timeout=0.5)
    assert saver.is_alive(), "the save committed while the delete's final check was in progress"

    guard_may_finish.set()
    saver.join(timeout=10)
    operation = _wait(service, started.operation_id)

    assert operation.progress.deleted_images == 1
    assert save_finished_at[0] >= delete_committed_at[0]


def test_active_inputs_survive_a_rolled_back_batch(
    invoker: Invoker, service: IntermediatesService, monkeypatch
) -> None:
    """A failed batch rolls the temp table back; the next check must still see everything active."""
    monkeypatch.setattr(intermediates_default, "DELETE_BATCH_SIZE", 1)
    _seed_image(invoker, "first.png")
    _seed_image(invoker, "consumed.png")
    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert preview.impact.delete_images == 2
    real_delete = invoker.services.image_records.delete_intermediates_by_names
    calls = {"n": 0}

    def fail_first_batch_after_the_guard(names, guard=None):
        calls["n"] += 1
        if calls["n"] == 1:
            with invoker.services.image_records._db.transaction() as cursor:
                if guard is not None:
                    guard(cursor, names)
                raise RuntimeError("simulated I/O failure after the guard rebuilt the temp table")
        # Enqueued after the second batch was paged: only its final check can protect it.
        _enqueue_row(
            invoker, session_id="s-late", status="pending", session_json='{"a": {"image_name": "consumed.png"}}'
        )
        return real_delete(names, guard=guard)

    monkeypatch.setattr(
        invoker.services.image_records, "delete_intermediates_by_names", fail_first_batch_after_the_guard
    )

    operation = _wait(service, _start(service, preview).operation_id)

    assert operation.progress.failed_images == 1
    assert operation.progress.retained_images == 1
    assert _exists(invoker, "consumed.png"), "an input of pending work was deleted after a rolled-back batch"


def test_the_acknowledged_documents_bound_what_a_force_clear_may_break(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirming acknowledges documents, not (document, item) pairs: an acknowledged document that names
    another target during the run does not protect it, while a document saved after the preview does."""
    acknowledged = _project(invoker, "alice", "Acknowledged", {"imageName": "t1.png"})
    for name in ("t1.png", "t2.png", "t3.png"):
        _seed_image(invoker, name)
    preview = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("alice")), ALICE)
    assert [doc.name for doc in preview.affected_documents] == ["Acknowledged"]
    assert preview.impact.delete_images == 3

    def reference_more() -> None:
        invoker.services.project_records.update(
            "alice",
            acknowledged,
            expected_revision=1,
            name="Acknowledged",
            data={"images": [{"imageName": "t1.png"}, {"imageName": "t2.png"}]},
        )
        _project(invoker, "alice", "New", {"imageName": "t3.png"})

    _before_first_delete(invoker, monkeypatch, reference_more)
    finished = _wait(service, _start(service, preview).operation_id)

    assert (finished.progress.deleted_images, finished.progress.retained_images) == (2, 1)
    assert not _exists(invoker, "t1.png") and not _exists(invoker, "t2.png")
    assert _exists(invoker, "t3.png")


def test_an_account_holds_media_for_a_bounded_number_of_leases(invoker: Invoker, service: IntermediatesService) -> None:
    from invokeai.app.services.intermediates.intermediates_common import (
        MAX_BROWSER_HOLD_LEASES_PER_USER,
        IntermediatesBrowserHoldRequest,
    )

    _seed_image(invoker, "bobs-held.png", user_id="bob")
    service.replace_browser_hold(BOB, "bob-tab", IntermediatesBrowserHoldRequest(images=["bobs-held.png"]))
    for tab in range(MAX_BROWSER_HOLD_LEASES_PER_USER + 1):
        _seed_image(invoker, f"held-{tab}.png")
        service.replace_browser_hold(ALICE, f"tab-{tab}", IntermediatesBrowserHoldRequest(images=[f"held-{tab}.png"]))
        # Refresh order decides which lease lapses; stagger it so the first tab is the stalest.
        with invoker.services.image_records._db.transaction() as cursor:
            cursor.execute(
                "UPDATE intermediates_browser_holds SET expires_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', ?)"
                " WHERE user_id = 'alice' AND lease_id = ?;",
                (f"+{60 + tab} seconds", f"tab-{tab}"),
            )

    alice, operation = _run(service, ALICE, _owner("alice"))
    assert (alice.impact.keep_active_images, alice.impact.delete_images) == (MAX_BROWSER_HOLD_LEASES_PER_USER, 1)
    assert operation.progress.deleted_images == 1
    assert not _exists(invoker, "held-0.png")
    assert all(_exists(invoker, f"held-{tab}.png") for tab in range(1, MAX_BROWSER_HOLD_LEASES_PER_USER + 1))
    bob = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("bob")), BOB)
    assert (bob.impact.keep_active_images, bob.impact.delete_images) == (1, 0)


def test_browser_hold_protects_unsaved_and_undo_references(invoker: Invoker, service: IntermediatesService) -> None:
    from invokeai.app.services.intermediates.intermediates_common import IntermediatesBrowserHoldRequest

    _seed_image(invoker, "browser-only.png")
    service.replace_browser_hold(ALICE, "tab-1", IntermediatesBrowserHoldRequest(images=["browser-only.png"]))
    preview = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("alice")), ALICE)
    assert preview.impact.keep_active_images == 1
    assert preview.impact.delete_images == 0

    service.release_browser_hold(BOB, "tab-1")
    still_held = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert still_held.impact.keep_active_images == 1
    service.release_browser_hold(ALICE, "tab-1")
    service.replace_browser_hold(BOB, "tab-2", IntermediatesBrowserHoldRequest(images=["browser-only.png"]))
    available = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert available.impact.delete_images == 1


def test_one_account_cannot_fill_the_cleanup_queue(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    monkeypatch.setattr(intermediates_default, "MAX_ACTIVE_OPERATIONS", 2)
    _seed_image(invoker, "alice.png")
    _seed_image(invoker, "bob.png", user_id="bob")
    _seed_image(invoker, "admin.png", user_id="admin")
    release = threading.Event()
    real_delete = invoker.services.images.delete_intermediates_by_names

    def blocked_delete(names, guard=None):
        release.wait(10)
        return real_delete(names, guard)

    monkeypatch.setattr(invoker.services.images, "delete_intermediates_by_names", blocked_delete)

    def start(caller: IntermediatesCaller) -> IntermediatesOperation:
        preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner(caller.user_id)), caller)
        return _start(service, preview, caller)

    try:
        confirmed = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
        started = [_start(service, confirmed)]
        # A lost-response replay learns the preview is consumed rather than waiting for capacity.
        with pytest.raises(IntermediatesPreviewNotFoundError):
            _start(service, confirmed)
        waiting = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
        with pytest.raises(IntermediatesUnavailableError, match="your running deletion"):
            _start(service, waiting)
        started.append(start(BOB))
        with pytest.raises(IntermediatesUnavailableError, match="Too many"):
            start(ADMIN)
    finally:
        release.set()
    for operation in started:
        assert _wait(service, operation.operation_id, ADMIN).status == "completed"
    assert not _exists(invoker, "alice.png") and not _exists(invoker, "bob.png")
    assert _exists(invoker, "admin.png")
    # A capacity refusal keeps the preview for once the running deletion finishes.
    assert _wait(service, _start(service, waiting).operation_id).status == "completed"


def test_the_final_check_judges_cached_media_recency_as_of_the_preview(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard shares the connection's session-hold table: judged live, it would sweep a hold the preview
    counted recent, and the next page would then collect the row the preview promised to keep."""
    from datetime import datetime, timedelta, timezone

    from invokeai.app.services.intermediates import intermediates_records_sqlite
    from invokeai.app.services.shared.media_references import MediaReferences

    monkeypatch.setattr(intermediates_default, "DELETE_BATCH_SIZE", 1)
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    release_time = start - timedelta(minutes=29)
    readings = iter([release_time, release_time, start] + [start + timedelta(hours=1)] * 100)
    monkeypatch.setattr(intermediates_records_sqlite, "_utc_now", lambda: next(readings))
    _seed_image(invoker, "old.png")
    _seed_image(invoker, "cached.png")
    consumer = _enqueue_row(invoker, session_id="consumer", status="in_progress")
    assert service.hold_cached_media("consumer", MediaReferences(images={"cached.png"}))
    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'completed' WHERE item_id = ?;", (consumer,))
    service._records.summarize(None)

    preview = service.create_preview(IntermediatesPreviewRequest(mode="safe", scope=_owner("alice")), ALICE)
    assert (preview.impact.keep_recent_images, preview.impact.delete_images) == (1, 1)

    operation = _wait(service, _start(service, preview).operation_id)
    assert operation.progress.deleted_images == 1
    assert _exists(invoker, "cached.png") and not _exists(invoker, "old.png")


def test_matching_scope_resolves_filtered_rows_minus_exclusions(
    invoker: Invoker, service: IntermediatesService
) -> None:
    alpha = _project(invoker, "alice", "Alpha", {})
    beta = _project(invoker, "alice", "Beta", {})
    _seed_image(invoker, "alpha.png", project_id=alpha)
    _seed_image(invoker, "beta.png", project_id=beta)
    _seed_image(invoker, "unassigned.png")
    _seed_image(invoker, "bob.png", user_id="bob")

    everyone = service.create_preview(
        IntermediatesPreviewRequest(mode="safe", scope=IntermediatesScope(kind="matching")), ADMIN
    )
    assert everyone.target_rows == 4
    searched = service.create_preview(
        IntermediatesPreviewRequest(mode="safe", scope=IntermediatesScope(kind="matching", search="alp")), ADMIN
    )
    assert (searched.target_rows, searched.impact.delete_images) == (1, 1)
    with pytest.raises(IntermediatesScopeInvalidError):
        service.create_preview(
            IntermediatesPreviewRequest(mode="safe", scope=IntermediatesScope(kind="matching", search="nothing")),
            ADMIN,
        )

    scope = IntermediatesScope(kind="matching", excluded=[IntermediatesScopeTarget(user_id="alice", project_id=beta)])
    preview, operation = _run(service, ALICE, scope)

    assert preview.scope == scope and operation.scope == scope
    assert (preview.target_rows, preview.impact.delete_images) == (2, 2)
    assert operation.progress.deleted_images == 2
    assert not _exists(invoker, "alpha.png") and not _exists(invoker, "unassigned.png")
    assert _exists(invoker, "beta.png") and _exists(invoker, "bob.png")


def test_selection_paging_seeks_each_row_without_sorting_the_scope(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window of an explicit-row scope is an index seek in rowid order. One OR over many rows would make
    SQLite sort the whole remaining scope on every window, holding the database lock for the run."""
    records = service._records
    targets = [("alice", _project(invoker, "alice", f"P{index}", {})) for index in range(3)] + [("alice", None)]
    for index in range(len(targets)):
        _seed_image(invoker, f"row-{index}.png", project_id=targets[index][1])
    real_scope_clauses = records._scope_clauses
    paging_clauses: list[list[tuple[str, list[object]]]] = []

    def recording_scope_clauses(**kwargs):
        clauses = real_scope_clauses(**kwargs)
        if kwargs.get("per_statement") == 1:
            paging_clauses.append(clauses)
        return clauses

    monkeypatch.setattr(records, "_scope_clauses", recording_scope_clauses)
    batches = list(
        records.iter_deletable_batches(
            "image",
            user_id=None,
            targets=targets,
            mode="safe",
            is_admin=False,
            caller_user_id="alice",
            recent_cutoff=None,
            limit=10,
        )
    )
    assert sorted(name for batch in batches for name, _ in batch) == [f"row-{index}.png" for index in range(4)]
    assert len(paging_clauses) == 1 and len(paging_clauses[0]) == len(targets)

    deletable_sql, deletable_params = records._deletable_sql("image", "safe", is_admin=False, caller_user_id="alice")
    with invoker.services.image_records._db.transaction() as cursor:
        clock = records._prepare(cursor)
        for clause, params in paging_clauses[0]:
            cursor.execute(
                "EXPLAIN QUERY PLAN " + records._window_sql("image", clause, deletable_sql),
                [*deletable_params, *clock, *params, 0, 10],
            )
            plan = " ".join(str(row[-1]) for row in cursor.fetchall())
            assert "B-TREE FOR ORDER BY" not in plan, plan


@pytest.mark.parametrize("cleanup_wins_race", [False, True])
def test_running_graph_keeps_cached_media_or_recomputes_if_cleanup_won(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch, cleanup_wins_race: bool
) -> None:
    from threading import Event

    from invokeai.app.invocations.image import BlankImageInvocation, ImageCropInvocation
    from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
    from invokeai.app.services.names.names_default import SimpleNameService
    from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
    from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
    from invokeai.app.services.shared.graph import Graph, GraphExecutionState
    from invokeai.app.services.urls.urls_default import LocalUrlService
    from tests.app.services.workflow_call_test_utils import _DummyStats
    from tests.test_nodes import create_edge

    services = invoker.services
    services.session_queue.start(invoker)
    services.configuration.node_cache_size = 512
    services.names = SimpleNameService()
    services.urls = LocalUrlService()
    services.performance_statistics = _DummyStats()
    services.progress_previews = MemoryProgressPreviews()
    services.tensors = MagicMock()
    services.conditioning = MagicMock()
    cache = services.invocation_cache = MemoryInvocationCache(max_cache_size=512)
    cache.start(invoker)
    runner = DefaultSessionRunner()
    runner.start(services, Event())
    graph = Graph()
    graph.add_node(BlankImageInvocation(id="blank", width=8, height=8, is_intermediate=True))
    graph.add_node(ImageCropInvocation(id="crop", x=0, y=0, width=4, height=4, use_cache=False))
    graph.add_edge(create_edge("blank", "image", "crop", "image"))

    def queued_session():
        session = GraphExecutionState(graph=graph.model_copy(deep=True))
        item_id = _enqueue_row(
            invoker, session_id=session.id, status="in_progress", session_json=session.model_dump_json()
        )
        return services.session_queue.get_queue_item(item_id)

    first = queued_session()
    first_blank = first.session.next()
    runner.run_node(first_blank, first)
    assert not first.session.has_error()
    image_name = first.session.results[first_blank.id].image.image_name
    runner.run_node(first.session.next(), first)
    assert first.session.is_complete() and not first.session.has_error()
    services.session_queue.save_queue_item_session(first.item_id, first.session)
    with services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'completed' WHERE item_id = ?", (first.item_id,))
        cursor.execute("UPDATE images SET created_at = '2020-01-01 00:00:00.000' WHERE image_name = ?", (image_name,))

    if cleanup_wins_race:
        real_get = cache.get

        def get_then_cleanup(key):
            output = real_get(key)
            if output is not None:
                _, completed = _run(service, ALICE, _owner("alice"))
                assert completed.progress.deleted_images == 1
            return output

        monkeypatch.setattr(cache, "get", get_then_cleanup)

    second = queued_session()
    second_blank = second.session.next()
    runner.run_node(second_blank, second)
    assert not second.session.has_error()
    assert cache.get_status().hits == 1
    second_image = second.session.results[second_blank.id].image.image_name
    if cleanup_wins_race:
        assert second_image != image_name
    else:
        assert second_image == image_name
        record = services.image_records.get(image_name)
        assert record.session_id == first.session_id
        stored_second = services.session_queue.get_queue_item(second.item_id)
        assert image_name not in stored_second.session.model_dump_json()
        for mode in ("safe", "force"):
            preview, completed = _run(service, ALICE, _owner("alice"), mode=mode)
            assert preview.impact.keep_active_images == 1
            assert completed.progress.deleted_images == 0
        assert services.image_files.get_path(image_name, image_subfolder=record.image_subfolder).exists()

    runner.run_node(second.session.next(), second)
    assert second.session.is_complete() and not second.session.has_error()
    services.session_queue.save_queue_item_session(second.item_id, second.session)
    with services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'completed' WHERE item_id = ?", (second.item_id,))
    if not cleanup_wins_race:
        _end_cached_media_grace(service, invoker)
        _, completed = _run(service, ALICE, _owner("alice"))
        assert completed.progress.deleted_images == 1


@pytest.mark.parametrize("mode", ["safe", "force"])
def test_cached_media_holds_keep_rows_out_of_a_running_cleanup_until_all_consuming_sessions_end(
    invoker: Invoker, service: IntermediatesService, mode: str
) -> None:
    from invokeai.app.services.shared.media_references import MediaReferences

    _seed_image(invoker, "cached.png")
    _seed_video(invoker, "cached.mp4")
    preview = service.create_preview(IntermediatesPreviewRequest(mode=mode, scope=_owner("alice")), ALICE)
    first = _enqueue_row(invoker, session_id="first-consumer", status="in_progress")
    second = _enqueue_row(invoker, session_id="second-consumer", status="waiting")
    references = MediaReferences(images={"cached.png"}, videos={"cached.mp4"})
    assert service.hold_cached_media("first-consumer", references)
    assert service.hold_cached_media("second-consumer", references)
    protected = _wait(service, _start(service, preview).operation_id)
    assert protected.progress.deleted_images == protected.progress.deleted_videos == 0
    assert _exists(invoker, "cached.png")

    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'completed' WHERE item_id = ?", (first,))
    preview, _ = _run(service, ALICE, _owner("alice"), mode=mode)
    assert preview.impact.keep_active_images == preview.impact.keep_active_videos == 1
    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'canceled' WHERE item_id = ?", (second,))
    preview, _ = _run(service, ALICE, _owner("alice"), mode=mode)
    # A cache hit reused old rows; the grace a fresh output gets now runs from the session's end.
    assert preview.impact.keep_recent_images == preview.impact.keep_recent_videos == 1
    _end_cached_media_grace(service, invoker)
    _, completed = _run(service, ALICE, _owner("alice"), mode=mode)
    assert completed.progress.deleted_images == completed.progress.deleted_videos == 1


def test_missing_cached_media_does_not_leave_a_partial_hold(invoker: Invoker, service: IntermediatesService) -> None:
    from invokeai.app.services.shared.media_references import MediaReferences

    _seed_image(invoker, "survivor.png")
    _enqueue_row(invoker, session_id="consumer", status="in_progress")
    assert not service.hold_cached_media(
        "consumer", MediaReferences(images={"survivor.png"}, videos={"already-deleted.mp4"})
    )
    _, completed = _run(service, ALICE, _owner("alice"))
    assert completed.progress.deleted_images == 1


@pytest.mark.parametrize("mode", ["safe", "force"])
@pytest.mark.parametrize("reload_records", [False, True])
def test_completed_child_media_survives_until_its_root_finishes(
    invoker: Invoker, service: IntermediatesService, mode: str, reload_records: bool
) -> None:
    from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
    from invokeai.app.invocations.fields import ImageField, VideoField
    from invokeai.app.invocations.workflow_return import WorkflowReturnOutput
    from invokeai.app.services.shared.graph import Graph, GraphExecutionState
    from invokeai.app.services.shared.media_references import MediaReferences

    queue = invoker.services.session_queue
    queue.start(invoker)
    _seed_image(invoker, "cached.png")
    _seed_video(invoker, "cached.mp4")
    early = service.create_preview(IntermediatesPreviewRequest(mode=mode, scope=_owner("alice")), ALICE)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="workflow"))
    root = GraphExecutionState(graph=graph)
    invocation = root.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)
    frame = root.build_workflow_call_frame(invocation.id, "workflow")
    root.begin_waiting_on_workflow_call(frame)
    children = [root.create_child_workflow_execution_state(Graph(), frame) for _ in range(2)]
    root.attach_waiting_workflow_call_child_sessions(children)
    root_id = _enqueue_row(invoker, session_id=root.id, status="in_progress", session_json=root.model_dump_json())
    queue.enqueue_workflow_call_children(queue.get_queue_item(root_id), [(session, None) for session in children])
    child = queue.dequeue()
    assert child is not None
    assert service.hold_cached_media(child.session_id, MediaReferences(images={"cached.png"}, videos={"cached.mp4"}))
    output = WorkflowReturnOutput(
        values={"image": ImageField(image_name="cached.png"), "video": VideoField(video_name="cached.mp4")}
    )
    child.session.results["return"] = output
    queue.save_queue_item_session(child.item_id, child.session)
    _seed_image(invoker, "produced.png", session_id=child.session_id)
    queue.complete_queue_item(child.item_id)
    # This is the real gap: the child is terminal, its sibling is pending, and the waiting
    # parent's persisted session does not contain the returned media yet.
    assert "cached.png" not in queue.get_queue_item(root_id).session.model_dump_json()

    # Clearing completed queue history must not discard an unconsumed child result.
    assert queue.prune("default").deleted == 0

    if reload_records:
        service.stop()
        with invoker.services.image_records._db.transaction() as cursor:
            cursor.execute("DROP TABLE IF EXISTS temp.intermediates_session_media;")
        service = IntermediatesService(IntermediatesRecordsSqlite(invoker.services.image_records._db))
        service.start(invoker)
    try:
        if not reload_records:
            completed = _wait(service, _start(service, early).operation_id)
            assert completed.progress.deleted_images == completed.progress.deleted_videos == 0
        preview, completed = _run(service, ALICE, _owner("alice"), mode=mode)
        assert preview.impact.keep_active_images == 2
        assert preview.impact.keep_active_videos == 1
        assert completed.progress.deleted_images == completed.progress.deleted_videos == 0
        assert _exists(invoker, "cached.png") and _exists(invoker, "produced.png")

        queue.record_workflow_call_child_completion(root_id, child.item_id, output.values)
        queue.cancel_queue_item(root_id)
        _end_cached_media_grace(service, invoker)
        _, completed = _run(service, ALICE, _owner("alice"), mode=mode)
        assert completed.progress.deleted_images == 2
        assert completed.progress.deleted_videos == 1
    finally:
        if reload_records:
            service.stop()


def test_completed_nested_child_keeps_media_while_its_root_is_active(
    invoker: Invoker, service: IntermediatesService
) -> None:
    from invokeai.app.services.shared.media_references import MediaReferences

    root = _enqueue_row(invoker, session_id="root", status="waiting")
    parent = _enqueue_row(invoker, session_id="parent", status="completed", root_item_id=root, parent_item_id=root)
    child = _enqueue_row(invoker, session_id="leaf", status="in_progress", root_item_id=root, parent_item_id=parent)
    _seed_image(invoker, "produced.png", session_id="leaf")
    _seed_video(invoker, "cached.mp4")
    assert service.hold_cached_media("leaf", MediaReferences(videos={"cached.mp4"}))
    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'completed' WHERE item_id = ?", (child,))

    for mode in ("safe", "force"):
        preview, completed = _run(service, ALICE, _owner("alice"), mode=mode)
        assert preview.impact.keep_active_images == preview.impact.keep_active_videos == 1
        assert completed.progress.deleted_images == completed.progress.deleted_videos == 0

    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'completed' WHERE item_id = ?", (root,))
    _end_cached_media_grace(service, invoker)
    _, completed = _run(service, ALICE, _owner("alice"))
    assert completed.progress.deleted_images == completed.progress.deleted_videos == 1


@pytest.mark.parametrize("bulk", [False, True])
def test_video_cleanup_cannot_purge_files_during_staged_deletion(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch, bulk: bool
) -> None:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    _seed_video(invoker, "racy.mp4")
    files = invoker.services.video_files
    video_path = files.get_path("racy.mp4")
    thumbnail_path = files.get_path("racy.mp4", thumbnail=True)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path.write_bytes(b"thumbnail")
    staging_paused = Event()
    resume_staging = Event()
    cleanup_started = Event()
    real_replace = Path.replace

    def pause_thumbnail_move(path, target):
        if path == thumbnail_path:
            staging_paused.set()
            assert resume_staging.wait(10)
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", pause_thumbnail_move)

    def delete_video():
        if bulk:
            assert invoker.services.videos.delete_videos_by_names(["racy.mp4"]) == (["racy.mp4"], [])
        else:
            invoker.services.videos.delete("racy.mp4")

    def cleanup():
        cleanup_started.set()
        return _run(service, ALICE, _owner("alice"))[1]

    with ThreadPoolExecutor(max_workers=2) as executor:
        deletion = executor.submit(delete_video)
        try:
            assert staging_paused.wait(10)
            clearing = executor.submit(cleanup)
            assert cleanup_started.wait(10)
            # Without serialization, cleanup removes the thumbnail after stage_delete's
            # exists() check, making its replace() fail and roll the MP4 back as an orphan.
            with pytest.raises(TimeoutError):
                clearing.result(timeout=0.5)
        finally:
            resume_staging.set()
        deletion.result(timeout=10)
        completed = clearing.result(timeout=10)
    assert completed.progress.deleted_videos == 0
    with pytest.raises(VideoRecordNotFoundException):
        invoker.services.video_records.get("racy.mp4")
    assert not video_path.exists()
    assert not thumbnail_path.exists()
    assert not list(video_path.parent.glob(".delete_*"))


# ── documents outside projects and workflows ──


def test_legacy_client_state_protects_its_canvas_media_until_it_changes(
    invoker: Invoker, service: IntermediatesService
) -> None:
    client_state = invoker.services.client_state_persistence
    client_state.set_by_key("alice", "canvas", '{"layers": [{"image": {"image_name": "layer.png"}}]}')
    _seed_image(invoker, "layer.png")

    preview, operation = _run(service, ALICE, _owner("alice"))
    assert (preview.impact.delete_images, preview.impact.keep_referenced_images) == (0, 1)
    force = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("alice")), ALICE)
    assert [(doc.kind, doc.owner_id, doc.name) for doc in force.affected_documents] == [
        ("client_state", "canvas", None)
    ]

    client_state.set_by_key("alice", "canvas", '{"layers": []}')
    _, operation = _run(service, ALICE, _owner("alice"))
    assert operation.progress.deleted_images == 1


def test_quarantined_projects_are_named_when_a_force_clear_would_break_them(
    invoker: Invoker, service: IntermediatesService
) -> None:
    from invokeai.app.services.shared.media_references import MediaReferences, replace_media_references

    _seed_image(invoker, "recovered.png")
    with invoker.services.image_records._db.transaction() as cursor:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS orphaned_projects_2026_08_06 (project_id TEXT, user_id TEXT, name TEXT,"
            " data TEXT, PRIMARY KEY (user_id, project_id));"
        )
        cursor.execute("INSERT INTO orphaned_projects_2026_08_06 VALUES ('p1', 'alice', 'Before boards', '{}');")
        replace_media_references(
            cursor,
            owner_kind="quarantined_project",
            user_id="alice",
            owner_id="p1",
            references=MediaReferences(images={"recovered.png"}),
        )

    safe, _ = _run(service, ALICE, _owner("alice"))
    assert safe.impact.keep_referenced_images == 1
    force = service.create_preview(IntermediatesPreviewRequest(mode="force", scope=_owner("alice")), ADMIN)
    assert [(doc.kind, doc.name) for doc in force.affected_documents] == [("quarantined_project", "Before boards")]


# ── bounds ──


def test_legacy_clear_deletes_every_safe_image_in_bounded_batches(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intermediates_default, "DELETE_BATCH_SIZE", 2)
    # A whole page of kept rows in front must neither stop the clear nor be re-read per batch.
    _project(invoker, "alice", "P", {"images": [{"imageName": "kept.png"}, {"imageName": "also-kept.png"}]})
    _seed_image(invoker, "kept.png")
    _seed_image(invoker, "also-kept.png")
    for index in range(5):
        _seed_image(invoker, f"safe-{index}.png")

    batch_sizes: list[int] = []
    delete = invoker.services.images.delete_intermediates_by_names
    monkeypatch.setattr(
        invoker.services.images,
        "delete_intermediates_by_names",
        lambda names, guard: batch_sizes.append(len(names)) or delete(names, guard),
    )

    assert service.count_safe_images(ADMIN) == 5
    assert service.clear_all_images_now(ADMIN) == 5
    assert batch_sizes == [2, 2, 1]
    assert service.count_safe_images(ADMIN) == 0
    assert _exists(invoker, "kept.png") and _exists(invoker, "also-kept.png")


def test_background_measurement_never_overwrites_the_writers_size(invoker: Invoker) -> None:
    _seed_image(invoker, "written.png", size=10)

    invoker.services.image_records.set_file_sizes_bytes({"written.png": 0})

    assert invoker.services.image_records.get("written.png").file_size_bytes == 10


def test_a_non_admin_force_preview_is_not_starved_by_foreign_referenced_rows(
    invoker: Invoker, service: IntermediatesService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Items another account's document names are kept before any window or batch is bounded, so they never crowd
    out the caller's own eligible items: the old code capped candidates first and could offer nothing."""
    monkeypatch.setattr(intermediates_default, "DELETE_BATCH_SIZE", 1)
    foreign = [f"shared-{index}.png" for index in range(2)]
    for name in foreign:
        _seed_image(invoker, name)
        invoker.services.workflow_records.create(_workflow_naming(name), user_id="bob")
    _seed_image(invoker, "own-referenced.png")
    _project(invoker, "alice", "Drafts", {"imageName": "own-referenced.png"})
    batches: list[list[str]] = []
    real_delete = invoker.services.images.delete_intermediates_by_names
    monkeypatch.setattr(
        invoker.services.images,
        "delete_intermediates_by_names",
        lambda names, guard=None: batches.append(list(names)) or real_delete(names, guard),
    )

    preview, operation = _run(service, ALICE, _owner("alice"), mode="force")

    assert preview.impact.delete_images == 1
    assert preview.impact.keep_referenced_images == len(foreign)
    assert [(doc.kind, doc.name) for doc in preview.affected_documents] == [("project", "Drafts")]
    # The kept rows are never offered to a batch, so the final check has nothing to retain.
    assert batches == [["own-referenced.png"]]
    assert (operation.progress.deleted_images, operation.progress.retained_images) == (1, 0)
    assert not _exists(invoker, "own-referenced.png")
    assert all(_exists(invoker, name) for name in foreign)
