"""The intermediates storage on every database backend: what protects an intermediate from a cleanup, how a
cleanup pages and checks, and the state the cleanup keeps in tables.

The service tests (`test_intermediates_service.py`) drive the same storage through the queue and the file
storages, on SQLite only until the queue is ported.
"""

import itertools
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest
from sqlalchemy import insert, select, update

from invokeai.app.services.client_state_persistence.client_state_persistence_default import (
    ClientStatePersistence,
)
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.intermediates import intermediates_records_default
from invokeai.app.services.intermediates.intermediates_common import BROWSER_HOLD_TTL_SECONDS
from invokeai.app.services.intermediates.intermediates_records_default import (
    IntermediatesRecords,
    ReferenceOwner,
)
from invokeai.app.services.project_records.project_records_default import ProjectRecordsStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.intermediates import (
    intermediates_browser_holds,
    intermediates_session_holds,
    intermediates_unmeasurable,
)
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.media_references import MediaReferences
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowCategory,
    WorkflowMeta,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage
from tests.fixtures.races import while_in_flight

OLD = "2020-01-01 00:00:00.000"
item_ids = itertools.count(1000)


@pytest.fixture
def records(database: Database) -> IntermediatesRecords:
    return IntermediatesRecords(database)


@pytest.fixture
def accounts(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(users),
            [{"user_id": user, "email": f"{user}@example.com", "password_hash": "-"} for user in ("alice", "bob")],
        )


def _image(
    database: Database,
    name: str,
    *,
    user_id: str = "alice",
    created_at: str = OLD,
    session_id: Optional[str] = None,
    file_size_bytes: Optional[int] = 10,
    is_intermediate: bool = True,
) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images).values(
                image_name=name,
                image_origin="internal",
                image_category="general",
                width=8,
                height=8,
                is_intermediate=is_intermediate,
                user_id=user_id,
                created_at=created_at,
                session_id=session_id,
                file_size_bytes=file_size_bytes,
            )
        )


def _queue_item(
    database: Database,
    session_id: str,
    *,
    status: str = "pending",
    session: str = "{}",
    root_item_id: Optional[int] = None,
) -> int:
    item_id = next(item_ids)
    with database.begin(write=True) as conn:
        conn.execute(
            insert(session_queue).values(
                item_id=item_id,
                batch_id="batch",
                queue_id="default",
                session_id=session_id,
                session=session,
                status=status,
                root_item_id=root_item_id,
            )
        )
    return item_id


def _reference(database: Database, name: str, *, user_id: str = "alice", owner_id: str = "p1") -> None:
    database.queries.media_references.replace(
        owner_kind="project", user_id=user_id, owner_id=owner_id, references=MediaReferences(images={name})
    )


def _deletable(
    records: IntermediatesRecords, *, mode: str = "safe", caller: str = "admin", admin: bool = True
) -> set[str]:
    batches = records.iter_deletable_batches(
        "image",
        user_id=None,
        targets=None,
        mode=mode,  # type: ignore[arg-type]
        is_admin=admin,
        caller_user_id=caller,
        recent_cutoff=None,
        limit=50,
    )
    return {name for batch in batches for name, _ in batch}


def _guarded(database: Database, records: IntermediatesRecords, names: list[str], **guard: Any) -> list[str]:
    settings = {"mode": "safe", "allowed_user_ids": None, "caller_user_id": "admin", "is_admin": True, **guard}
    return ImageRecordStorage(database).delete_intermediates_by_names(
        names, records.make_delete_guard("image", **settings)
    )


def _exists(database: Database, name: str) -> bool:
    with database.begin(write=False) as conn:
        return conn.execute(select(images.c.image_name).where(images.c.image_name == name)).first() is not None


def _protect_by_queue_input(database: Database, records: IntermediatesRecords) -> None:
    _queue_item(database, "s-input", session='{"node": {"image": {"image_name": "protected.png"}}}')


def _protect_by_producing_session(database: Database, records: IntermediatesRecords) -> None:
    with database.begin(write=True) as conn:
        conn.execute(update(images).where(images.c.image_name == "protected.png").values(session_id="s-producer"))
    _queue_item(database, "s-producer", status="in_progress")


def _protect_by_completed_child_of_an_active_root(database: Database, records: IntermediatesRecords) -> None:
    root = _queue_item(database, "s-root", status="waiting")
    _queue_item(
        database,
        "s-child",
        status="completed",
        root_item_id=root,
        session='{"out": {"image_name": "protected.png"}}',
    )


def _protect_as_output_of_a_completed_child_of_an_active_root(
    database: Database, records: IntermediatesRecords
) -> None:
    root = _queue_item(database, "s-root", status="waiting")
    _queue_item(database, "s-child", status="completed", root_item_id=root)
    with database.begin(write=True) as conn:
        conn.execute(update(images).where(images.c.image_name == "protected.png").values(session_id="s-child"))


def _protect_by_browser_lease(database: Database, records: IntermediatesRecords) -> None:
    records.replace_browser_hold("alice", "tab-1", ["protected.png"], [])


def _protect_by_cached_output(database: Database, records: IntermediatesRecords) -> None:
    _queue_item(database, "s-reuser", status="in_progress")
    assert records.hold_cached_media("s-reuser", MediaReferences(images={"protected.png"}))


def _protect_by_recency(database: Database, records: IntermediatesRecords) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000")
    with database.begin(write=True) as conn:
        conn.execute(update(images).where(images.c.image_name == "protected.png").values(created_at=now))


def _protect_by_reference(database: Database, records: IntermediatesRecords) -> None:
    _reference(database, "protected.png")


@pytest.mark.parametrize(
    "protect",
    [
        _protect_by_queue_input,
        _protect_by_producing_session,
        _protect_by_completed_child_of_an_active_root,
        _protect_as_output_of_a_completed_child_of_an_active_root,
        _protect_by_browser_lease,
        _protect_by_cached_output,
        _protect_by_recency,
        _protect_by_reference,
    ],
)
def test_a_protected_intermediate_is_neither_paged_nor_deleted_by_a_safe_cleanup(
    database: Database, records: IntermediatesRecords, protect: Callable[[Database, IntermediatesRecords], None]
) -> None:
    _image(database, "protected.png")
    _image(database, "safe.png")
    _image(database, "durable.png", is_intermediate=False)
    protect(database, records)

    assert _deletable(records) == {"safe.png"}
    assert _guarded(database, records, ["protected.png", "safe.png", "durable.png"]) == ["safe.png"]
    assert _exists(database, "protected.png") and _exists(database, "durable.png")


def test_the_summary_counts_each_classification_with_its_bytes(
    database: Database, records: IntermediatesRecords
) -> None:
    _image(database, "safe.png", file_size_bytes=5)
    _image(database, "unmeasured.png", file_size_bytes=None)
    _image(database, "referenced.png", file_size_bytes=7)
    _reference(database, "referenced.png")
    _image(database, "active.png")
    records.replace_browser_hold("alice", "tab-1", ["active.png"], [])
    _image(database, "bob.png", user_id="bob")

    rows = records.summarize(None)
    alice = rows[("alice", None)]["image"]
    only_bob = records.summarize(["bob"])

    assert (alice.counts.safe, alice.counts.referenced, alice.counts.active, alice.counts.recent) == (2, 1, 1, 0)
    assert (alice.safe_bytes, alice.referenced_bytes, alice.unknown_size_count) == (5, 7, 1)
    assert list(only_bob) == [("bob", None)] and only_bob[("bob", None)]["image"].counts.safe == 1
    assert records.summarize([]) == {}


def test_windows_page_in_creation_order_with_names_breaking_ties(
    database: Database, records: IntermediatesRecords
) -> None:
    for name, created_at in (
        ("c.png", "2020-01-01 00:00:00.000"),
        ("a.png", "2020-01-02 00:00:00.000"),
        ("b.png", "2020-01-02 00:00:00.000"),
        ("d.png", "2020-01-03 00:00:00.000"),
        ("e.png", "2020-01-03 00:00:00.000"),
    ):
        _image(database, name, created_at=created_at)

    batches = list(
        records.iter_deletable_batches(
            "image",
            user_id="alice",
            targets=None,
            mode="safe",
            is_admin=False,
            caller_user_id="alice",
            recent_cutoff=None,
            limit=2,
        )
    )

    assert [[name for name, _ in batch] for batch in batches] == [["c.png", "a.png"], ["b.png", "d.png"], ["e.png"]]


def test_a_force_cleanup_breaks_only_documents_its_caller_may_break(
    database: Database, records: IntermediatesRecords
) -> None:
    _image(database, "own-reference.png")
    _reference(database, "own-reference.png", user_id="alice", owner_id="mine")
    _image(database, "bobs-reference.png")
    _reference(database, "bobs-reference.png", user_id="bob", owner_id="his")

    _image(database, "shared-reference.png")
    _reference(database, "shared-reference.png", user_id="alice", owner_id="mine-too")
    _reference(database, "shared-reference.png", user_id="bob", owner_id="his-too")

    caller = {"caller": "alice", "admin": False}
    assert _deletable(records, mode="force", **caller) == {"own-reference.png"}
    assert _deletable(records, mode="force") == {"own-reference.png", "bobs-reference.png", "shared-reference.png"}

    names = ["own-reference.png", "bobs-reference.png"]
    mine = frozenset({ReferenceOwner("project", "alice", "mine")})
    # A referenced item goes only while every document naming it is acknowledged.
    assert _guarded(database, records, names, mode="force", is_admin=True, acknowledged=frozenset()) == []
    assert _guarded(database, records, names, mode="force", is_admin=True, acknowledged=mine) == ["own-reference.png"]


def test_the_guard_re_reads_ownership(database: Database, records: IntermediatesRecords) -> None:
    _image(database, "alices.png")
    _image(database, "bobs.png", user_id="bob")

    kept = _guarded(database, records, ["alices.png", "bobs.png"], allowed_user_ids=frozenset({"alice"}))

    assert kept == ["alices.png"] and _exists(database, "bobs.png")


def test_a_cached_output_is_held_while_its_session_runs_then_for_the_grace(
    database: Database, records: IntermediatesRecords
) -> None:
    _image(database, "cached.png")
    item = _queue_item(database, "s-running", status="in_progress")

    assert not records.hold_cached_media("s-ended", MediaReferences(images={"cached.png"}))
    assert not records.hold_cached_media("s-running", MediaReferences(images={"cached.png", "gone.png"}))
    assert records.hold_cached_media("s-running", MediaReferences(images={"cached.png"}))
    summary = records.summarize(None)[("alice", None)]["image"].counts
    assert summary.active == 1

    with database.begin(write=True) as conn:
        conn.execute(update(session_queue).where(session_queue.c.item_id == item).values(status="completed"))
    assert records.summarize(None)[("alice", None)]["image"].counts.recent == 1

    with database.begin(write=True) as conn:
        conn.execute(update(intermediates_session_holds).values(released_at="2000-01-01 00:00:00.000"))
    assert records.summarize(None)[("alice", None)]["image"].counts.safe == 1
    with database.begin(write=False) as conn:
        assert conn.execute(select(intermediates_session_holds)).all() == []


def test_a_new_process_starts_without_the_last_ones_holds_and_marks(
    database: Database, records: IntermediatesRecords
) -> None:
    _image(database, "cached.png")
    _image(database, "leased.png")
    _queue_item(database, "s-running", status="in_progress")
    assert records.hold_cached_media("s-running", MediaReferences(images={"cached.png"}))
    records.mark_unmeasurable("image", ["cached.png"])
    records.replace_browser_hold("alice", "tab-1", ["leased.png"], [])

    IntermediatesRecords(database).start()

    with database.begin(write=False) as conn:
        assert conn.execute(select(intermediates_session_holds)).all() == []
        assert conn.execute(select(intermediates_unmeasurable)).all() == []
    # A browser lease is the tab's, not the process's: an open editor keeps its media across a restart.
    assert "leased.png" not in _deletable(records)


def test_past_the_lease_cap_the_leases_refreshed_longest_ago_lapse(
    database: Database, records: IntermediatesRecords, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(intermediates_records_default, "MAX_BROWSER_HOLD_LEASES_PER_USER", 2)
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    instants: Iterator[datetime] = (start + timedelta(seconds=i) for i in itertools.count())
    monkeypatch.setattr(intermediates_records_default, "_utc_now", lambda: next(instants))
    for name in ("one.png", "two.png", "three.png"):
        _image(database, name)

    records.replace_browser_hold("alice", "tab-1", ["one.png"], [])
    records.replace_browser_hold("alice", "tab-2", ["two.png"], [])
    records.replace_browser_hold("alice", "tab-3", ["three.png"], [])
    # Refreshing tab-2 keeps it, so tab-1 lapses rather than tab-2.
    records.replace_browser_hold("alice", "tab-2", ["two.png"], [])
    records.replace_browser_hold("alice", "tab-4", ["three.png"], [])

    with database.begin(write=False) as conn:
        leases = {row[0] for row in conn.execute(select(intermediates_browser_holds.c.lease_id))}
    assert leases == {"tab-2", "tab-4"}


def test_a_lease_holds_only_the_accounts_own_intermediates(database: Database, records: IntermediatesRecords) -> None:
    _image(database, "alices.png")
    _image(database, "bobs.png", user_id="bob")
    _image(database, "durable.png", is_intermediate=False)

    records.replace_browser_hold("alice", "tab-1", ["alices.png", "bobs.png", "durable.png", "missing.png"], [])

    assert _deletable(records) == {"bobs.png"}


class TestMeasurementQueue:
    def test_the_unmeasured_come_oldest_first_and_young_or_unmeasurable_rows_wait(
        self, database: Database, records: IntermediatesRecords
    ) -> None:
        young = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S.000")
        _image(database, "measured.png", file_size_bytes=100)
        _image(database, "durable.png", file_size_bytes=None, is_intermediate=False)
        _image(database, "unreadable.png", file_size_bytes=None)
        _image(database, "young.png", file_size_bytes=None, created_at=young)
        _image(database, "z-first.png", file_size_bytes=None, created_at="2020-01-01 00:00:00.000")
        _image(database, "a-second.png", file_size_bytes=None, created_at="2020-01-02 00:00:00.000")
        records.mark_unmeasurable("image", ["unreadable.png"])
        records.mark_unmeasurable("image", ["unreadable.png"])

        assert records.next_unmeasured("image", 200, min_age_seconds=10) == [("z-first.png", ""), ("a-second.png", "")]
        assert records.next_unmeasured("image", 1, min_age_seconds=10) == [("z-first.png", "")]
        assert records.has_unmeasured_intermediates()

        records.mark_unmeasurable("image", ["z-first.png", "a-second.png", "young.png"])
        assert not records.has_unmeasured_intermediates()

    @pytest.mark.sqlite_only  # Counts SQLite's virtual machine steps.
    def test_late_batches_do_not_rescan_the_completed_work(
        self, database: Database, records: IntermediatesRecords
    ) -> None:
        with database.begin(write=True) as conn:
            conn.execute(
                insert(images),
                [
                    {
                        "image_name": f"asset-{index:06d}",
                        "image_origin": "internal",
                        "image_category": "general",
                        "width": 1,
                        "height": 1,
                        "is_intermediate": True,
                        "created_at": OLD,
                    }
                    for index in range(100_000)
                ],
            )

        def next_batch() -> tuple[list[tuple[str, str]], int]:
            steps = 0

            def count() -> int:
                nonlocal steps
                steps += 100
                return 0

            with database.begin(write=False) as conn:
                driver = conn.connection.driver_connection
            assert driver is not None
            driver.set_progress_handler(count, 100)
            try:
                return records.next_unmeasured("image", 200, min_age_seconds=10), steps
            finally:
                driver.set_progress_handler(None, 0)

        first, first_work = next_batch()
        with database.begin(write=True) as conn:
            conn.execute(update(images).where(images.c.image_name < "asset-099800").values(file_size_bytes=100))
        late, late_work = next_batch()

        assert first == [(f"asset-{index:06d}", "") for index in range(200)]
        assert late == [(f"asset-{index:06d}", "") for index in range(99800, 100_000)]
        # Deterministic enough for a generous bound, and independent of machine speed: a scan of the measured rows
        # would exceed it by two orders of magnitude on the late batch.
        assert 0 < first_work < 10_000
        assert late_work < 2 * first_work


@pytest.mark.parametrize("kind", ["image", "video"])
def test_a_cleanup_waits_for_a_reference_in_flight_and_then_keeps_the_media(
    database: Database, records: IntermediatesRecords, kind: str
) -> None:
    """A reference written but not committed is invisible to a server's other transactions: without the lock, the
    cleanup's check would find the media unreferenced and delete it under the save."""
    if kind == "image":
        _image(database, "media")
        storage: Any = ImageRecordStorage(database)
    else:
        with database.begin(write=True) as conn:
            conn.execute(
                insert(videos).values(
                    video_name="media",
                    video_origin="internal",
                    video_category="general",
                    width=8,
                    height=8,
                    is_intermediate=True,
                    created_at=OLD,
                )
            )
        storage = VideoRecordStorage(database)
    references = MediaReferences(images={"media"}) if kind == "image" else MediaReferences(videos={"media"})

    def reference(q: Queries) -> None:
        q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
        q.media_references.replace(owner_kind="project", user_id="alice", owner_id="p1", references=references)

    guard = records.make_delete_guard(kind, mode="safe", allowed_user_ids=None, caller_user_id="admin", is_admin=True)  # type: ignore[arg-type]
    deleted: list[str] = []
    errors = while_in_flight(
        database, reference, lambda: deleted.extend(storage.delete_intermediates_by_names(["media"], guard))
    )

    assert errors == [] and deleted == []


def _workflow(name: str) -> WorkflowWithoutID:
    return WorkflowWithoutID(
        name=name,
        author="",
        description="",
        version="1.0.0",
        contact="",
        tags="",
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.User),
        nodes=[{"id": "n", "data": {"image": {"image_name": "img.png"}}}],
        edges=[],
    )


@pytest.mark.usefixtures("accounts")
@pytest.mark.parametrize(
    "write",
    [
        "project create",
        "project update",
        "workflow create",
        "workflow update",
        "client state",
        "browser lease",
        "cached output",
    ],
)
def test_every_write_that_protects_media_waits_for_a_cleanup_in_flight(
    database: Database, records: IntermediatesRecords, write: str
) -> None:
    """Shared with the cleanup's exclusive lock: a cleanup's check and delete are never interleaved with a write
    that makes the media it checked protected."""
    _image(database, "img.png")
    projects = ProjectRecordsStorage(database)
    workflows = WorkflowRecordsStorage(database)
    project = projects.create("alice", "P", {})
    workflow = workflows.create(_workflow("W"), user_id="alice")
    _queue_item(database, "s-reuser", status="in_progress")
    writes: dict[str, Callable[[], object]] = {
        "project create": lambda: projects.create("alice", "Q", {"imageName": "img.png"}),
        "project update": lambda: projects.update(
            "alice", project.project_id, project.revision, "P", {"imageName": "img.png"}
        ),
        "workflow create": lambda: workflows.create(_workflow("V"), user_id="alice"),
        "workflow update": lambda: workflows.update(Workflow(**workflow.workflow.model_dump()), user_id="alice"),
        "client state": lambda: ClientStatePersistence(database).set_by_key(
            "alice", "canvas", '{"imageName": "img.png"}'
        ),
        "browser lease": lambda: records.replace_browser_hold("alice", "tab", ["img.png"], []),
        "cached output": lambda: records.hold_cached_media("s-reuser", MediaReferences(images={"img.png"})),
    }

    errors = while_in_flight(database, lambda q: q.locks.acquire(DatabaseLock.MEDIA_PROTECTION), writes[write])

    assert errors == []


@pytest.mark.parametrize("selected", [3, 33])
def test_a_preview_of_selected_rows_counts_exactly_those_rows(
    database: Database, records: IntermediatesRecords, selected: int
) -> None:
    # Statements of up to 32 rows, each padded to a power of two: 3 rows fill 4 slots, 33 rows need two statements.
    for owner in range(40):
        _image(database, f"owner-{owner}.png", user_id=f"owner-{owner}")

    preview = records.preview_scope(
        user_id=None,
        targets=[(f"owner-{owner}", None) for owner in range(selected)],
        mode="safe",
        is_admin=True,
        caller_user_id="admin",
        max_acknowledged=10,
    )

    assert preview.deletable["image"].count == selected
    assert preview.rows == {(f"owner-{owner}", None) for owner in range(selected)}


def test_windows_page_ties_within_a_second_one_row_at_a_time(database: Database, records: IntermediatesRecords) -> None:
    for name, created_at in (
        ("c.png", "2020-01-02 00:00:00.123"),
        ("a.png", "2020-01-02 00:00:00.500"),
        ("b.png", "2020-01-02 00:00:00.500"),
        ("d.png", "2020-01-02 00:00:00.999"),
        ("e.png", "2020-01-02 00:00:01.000"),
    ):
        _image(database, name, created_at=created_at)

    batches = records.iter_deletable_batches(
        "image",
        user_id=None,
        targets=None,
        mode="safe",
        is_admin=True,
        caller_user_id="admin",
        recent_cutoff=None,
        limit=1,
    )

    assert [[name for name, _ in batch] for batch in batches] == [["c.png"], ["a.png"], ["b.png"], ["d.png"], ["e.png"]]


@pytest.mark.usefixtures("accounts")
def test_a_project_row_holds_the_media_of_its_project_and_the_unassigned_row_the_rest(
    database: Database, records: IntermediatesRecords
) -> None:
    project = ProjectRecordsStorage(database).create("alice", "P", {})
    _image(database, "in-project.png")
    _image(database, "dangling.png")
    _image(database, "loose.png")
    with database.begin(write=True) as conn:
        for name, project_id in (("in-project.png", project.project_id), ("dangling.png", "deleted-project")):
            conn.execute(update(images).where(images.c.image_name == name).values(project_id=project_id))

    def deletable(targets: list[tuple[str, Optional[str]]]) -> set[str]:
        batches = records.iter_deletable_batches(
            "image",
            user_id=None,
            targets=targets,
            mode="safe",
            is_admin=True,
            caller_user_id="admin",
            recent_cutoff=None,
            limit=10,
        )
        return {name for batch in batches for name, _ in batch}

    assert deletable([("alice", project.project_id)]) == {"in-project.png"}
    # Media of a project that no longer exists belong to the owner's unassigned row.
    assert deletable([("alice", None)]) == {"dangling.png", "loose.png"}
    preview = records.preview_scope(
        user_id=None,
        targets=[("alice", project.project_id)],
        mode="safe",
        is_admin=True,
        caller_user_id="admin",
        max_acknowledged=10,
    )
    assert preview.rows == {("alice", project.project_id)} and preview.deletable["image"].count == 1
    summary = records.summarize(["alice"])
    assert set(summary) == {("alice", project.project_id), ("alice", None)}
    assert summary[("alice", None)]["image"].counts.safe == 2


def test_an_expired_lease_stops_protecting_and_is_swept(
    database: Database, records: IntermediatesRecords, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [datetime(2030, 1, 1, tzinfo=timezone.utc)]
    monkeypatch.setattr(intermediates_records_default, "_utc_now", lambda: now[0])
    _image(database, "held.png")
    _image(database, "other.png")
    records.replace_browser_hold("alice", "tab-1", ["held.png"], [])
    assert records.summarize(None)[("alice", None)]["image"].counts.active == 1

    now[0] += timedelta(seconds=BROWSER_HOLD_TTL_SECONDS + 1)

    assert records.summarize(None)[("alice", None)]["image"].counts.active == 0
    # The next lease of any tab sweeps the expired one away.
    records.replace_browser_hold("alice", "tab-2", ["other.png"], [])
    with database.begin(write=False) as conn:
        assert {row[0] for row in conn.execute(select(intermediates_browser_holds.c.lease_id))} == {"tab-2"}


def test_a_lease_refreshed_twice_in_one_transaction_keeps_one_row_per_medium(
    database: Database, records: IntermediatesRecords
) -> None:
    """As when two refreshes of one lease meet on a server: both cleared the lease, and both insert its rows."""
    _image(database, "held.png")

    with database.queries.transaction() as q:
        for expires_at in ("2030-01-01 00:00:00.000", "2030-01-01 00:01:00.000"):
            q.intermediates.hold_for_lease(
                "image", user_id="alice", lease_id="tab", names={"held.png"}, expires_at=expires_at
            )

    with database.begin(write=False) as conn:
        assert conn.execute(select(intermediates_browser_holds.c.expires_at)).scalars().all() == [
            "2030-01-01 00:01:00.000"
        ]


def test_documents_found_by_several_statements_are_counted_once_up_to_the_cap(
    database: Database, records: IntermediatesRecords
) -> None:
    for owner in range(40):
        _image(database, f"o{owner}.png", user_id=f"owner-{owner}")
    # One document naming media of two rows that fall into different statements (32 rows each).
    database.queries.media_references.replace(
        owner_kind="project",
        user_id="owner-0",
        owner_id="doc",
        references=MediaReferences(images={"o0.png", "o32.png"}),
    )

    def preview(targets: Optional[list[tuple[str, Optional[str]]]]) -> Any:
        return records.preview_scope(
            user_id=None, targets=targets, mode="force", is_admin=True, caller_user_id="admin", max_acknowledged=1
        )

    merged = preview([(f"owner-{owner}", None) for owner in range(33)])
    assert not merged.acknowledged_overflow
    assert merged.acknowledged == {ReferenceOwner("project", "owner-0", "doc"): 2}

    database.queries.media_references.replace(
        owner_kind="project", user_id="owner-1", owner_id="doc-2", references=MediaReferences(images={"o1.png"})
    )
    assert preview(None).acknowledged_overflow


def test_a_hold_stays_active_while_its_finished_session_belongs_to_an_active_root(
    database: Database, records: IntermediatesRecords
) -> None:
    """A workflow-call child completes before its outputs reach its waiting parent; the media it reused stay held."""
    _image(database, "cached.png")
    root = _queue_item(database, "s-root", status="waiting")
    child = _queue_item(database, "s-child", status="in_progress", root_item_id=root)
    assert records.hold_cached_media("s-child", MediaReferences(images={"cached.png"}))

    with database.begin(write=True) as conn:
        conn.execute(update(session_queue).where(session_queue.c.item_id == child).values(status="completed"))

    assert records.summarize(None)[("alice", None)]["image"].counts.active == 1
