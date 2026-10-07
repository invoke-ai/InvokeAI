"""The session queue's reads on every backend: what each returns, in which order, and under which filters.

The queue's writes still run on the SQLite cursor, so these tests write their rows directly, and read through the
service with only its database: its reads need nothing else.
"""

import json
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
from pydantic_core import to_jsonable_python
from sqlalchemy import insert

from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.session_queue import session_queue, session_queue_enqueue_receipts
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.execution_state_migration import dump_execution_state
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

QUEUE = "default"


@pytest.fixture
def accounts(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(users),
            [
                {"user_id": "alice", "email": "alice@example.com", "display_name": "Alice", "password_hash": "-"},
                {"user_id": "bob", "email": "bob@example.com", "display_name": "Bob", "password_hash": "-"},
            ],
        )


@pytest.fixture
def service(database: Database) -> SqliteSessionQueue:
    return SqliteSessionQueue(cast(SqliteDatabase, SimpleNamespace(database=database)))


def _item(
    database: Database,
    *,
    status: str = "pending",
    priority: int = 0,
    user_id: str = "alice",
    created_at: str = "2026-01-01 00:00:00.000",
    batch_id: str = "batch",
    queue_id: str = QUEUE,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    parent_item_id: Optional[int] = None,
    workflow: Optional[str] = None,
) -> int:
    session = GraphExecutionState(graph=Graph())
    values: dict[str, Any] = {
        "queue_id": queue_id,
        "batch_id": batch_id,
        "session_id": session.id,
        "session": json.dumps(dump_execution_state(session), default=to_jsonable_python),
        "status": status,
        "priority": priority,
        "user_id": user_id,
        "created_at": created_at,
        "origin": origin,
        "destination": destination,
        "parent_item_id": parent_item_id,
        "workflow": workflow,
        "field_values": json.dumps([{"node_path": "n", "field_name": "seed", "value": 7}]),
    }
    with database.begin(write=True) as conn:
        return int(conn.execute(insert(session_queue).values(**values)).inserted_primary_key[0])


@pytest.mark.usefixtures("accounts")
class TestItems:
    def test_an_item_carries_every_column_and_its_owners_name(self, database: Database) -> None:
        item_id = _item(database, workflow='{"name": "w"}', origin="canvas", destination="gallery")

        item = database.queries.session_queue.item(item_id)

        assert item is not None
        assert {column.name for column in session_queue.columns} <= item.keys()
        assert (item["item_id"], item["user_display_name"], item["user_email"]) == (
            item_id,
            "Alice",
            "alice@example.com",
        )
        assert (item["origin"], item["destination"], item["workflow"]) == ("canvas", "gallery", '{"name": "w"}')
        assert database.queries.session_queue.workflow(item_id) == '{"name": "w"}'

    def test_a_missing_item_reads_as_none(self, database: Database) -> None:
        assert database.queries.session_queue.item(404) is None
        assert database.queries.session_queue.workflow(404) is None

    def test_an_item_of_an_account_that_is_gone_keeps_its_row(self, database: Database) -> None:
        item_id = _item(database, user_id="nobody")

        item = database.queries.session_queue.item(item_id)

        assert item is not None and item["user_display_name"] is None and item["user_email"] is None

    def test_the_next_pending_item_is_the_highest_priority_then_the_oldest(self, database: Database) -> None:
        _item(database, priority=0, created_at="2026-01-01 00:00:00.000")
        _item(database, priority=1, created_at="2026-01-04 00:00:00.000")
        oldest = _item(database, priority=1, created_at="2026-01-03 00:00:00.000")
        _item(database, priority=1, created_at="2026-01-03 00:00:00.000")
        _item(database, priority=5, status="completed")

        item = database.queries.session_queue.item_by_status(QUEUE, "pending", None)

        assert item is not None and item["item_id"] == oldest

    def test_the_current_item_is_the_earliest_enqueued_of_several(self, database: Database) -> None:
        first = _item(database, status="in_progress")
        _item(database, status="in_progress")

        item = database.queries.session_queue.item_by_status(QUEUE, "in_progress", None)

        assert item is not None and item["item_id"] == first

    def test_an_origin_prefix_ignores_case_and_takes_wildcards_as_written(self, database: Database) -> None:
        _item(database, origin="canvasXworkflow", priority=2)
        wanted = _item(database, origin="Canvas_workflow:1", priority=1)

        item = database.queries.session_queue.item_by_status(QUEUE, "pending", "canvas_")

        assert item is not None and item["item_id"] == wanted

    def test_other_queues_are_left_out(self, database: Database) -> None:
        _item(database, queue_id="other")

        assert database.queries.session_queue.item_by_status(QUEUE, "pending", None) is None
        assert database.queries.session_queue.count(QUEUE) == 0
        assert database.queries.session_queue.count("other") == 1


@pytest.mark.usefixtures("accounts")
class TestListings:
    def test_all_items_come_highest_priority_first_then_in_enqueue_order(self, database: Database) -> None:
        low = _item(database, priority=0)
        high = _item(database, priority=3, destination="canvas")
        middle = _item(database, priority=1)
        _item(database, queue_id="other")

        listed = [item["item_id"] for item in database.queries.session_queue.all_items(QUEUE, None)]
        on_canvas = [item["item_id"] for item in database.queries.session_queue.all_items(QUEUE, "canvas")]

        assert listed == [high, middle, low]
        assert on_canvas == [high]

    def test_a_later_page_keeps_every_filter(self, database: Database) -> None:
        # Before the port, the keyset's OR escaped the filters from the second page on: the items after the cursor
        # at its own priority were listed whatever their queue, status or destination.
        kept = [_item(database, priority=1, status="pending", destination="canvas") for _ in range(3)]
        _item(database, priority=1, status="completed", destination="canvas")
        _item(database, priority=1, status="pending", destination="gallery")
        _item(database, priority=1, status="pending", destination="canvas", queue_id="other")
        kept.append(_item(database, priority=0, status="pending", destination="canvas"))

        def page(after: Optional[tuple[int, int]]) -> list[int]:
            rows = database.queries.session_queue.page(
                QUEUE, limit=2, after=after, status="pending", destination="canvas"
            )
            return [row["item_id"] for row in rows]

        first = page(None)
        second = page((1, first[-1]))

        assert first + second == kept

    def test_item_ids_follow_creation_and_keep_a_batch_in_enqueue_order(self, database: Database) -> None:
        # A batch's items share their creation time; both directions list them in enqueue order, as SQLite always
        # returned them, so that the queue's views window a large batch the same way.
        old = _item(database, created_at="2026-01-01 00:00:00.000")
        batch = [_item(database, created_at="2026-01-02 00:00:00.000") for _ in range(3)]
        _item(database, user_id="bob", created_at="2026-01-03 00:00:00.000")
        _item(database, queue_id="other", created_at="2026-01-04 00:00:00.000")

        newest_first = database.queries.session_queue.item_ids(
            QUEUE, descending=True, user_id="alice", origin_prefix=None
        )
        oldest_first = database.queries.session_queue.item_ids(
            QUEUE, descending=False, user_id="alice", origin_prefix=None
        )

        assert newest_first == [*batch, old]
        assert oldest_first == [old, *batch]

    def test_item_ids_by_origin(self, database: Database) -> None:
        wanted = _item(database, origin="workflows:1")
        _item(database, origin="canvas:1")

        assert database.queries.session_queue.item_ids(
            QUEUE, descending=True, user_id=None, origin_prefix="WORKFLOWS"
        ) == [wanted]

    def test_summaries_are_of_the_named_items_of_the_queue(self, database: Database) -> None:
        item_ids = [_item(database, destination=f"d{i}") for i in range(5)]
        foreign = _item(database, queue_id="other")

        summaries = database.queries.session_queue.summaries(QUEUE, [*item_ids[1:], foreign, 404])

        assert sorted(summary["item_id"] for summary in summaries) == item_ids[1:]
        assert {summary["user_display_name"] for summary in summaries} == {"Alice"}


@pytest.mark.usefixtures("accounts")
class TestCounts:
    def test_status_counts_are_global_with_the_accounts_own_and_the_earliest_current_item(
        self, database: Database
    ) -> None:
        for status, user_id in [
            ("pending", "alice"),
            ("pending", "bob"),
            ("pending", "bob"),
            ("in_progress", "bob"),
            ("in_progress", "alice"),
            ("completed", "alice"),
        ]:
            _item(database, status=status, user_id=user_id)
        _item(database, status="pending", queue_id="other")

        counts, own, current = database.queries.session_queue.status_counts(QUEUE, "alice", None)

        assert counts == {"pending": 3, "in_progress": 2, "completed": 1}
        assert own == {"pending": 1, "in_progress": 1, "completed": 1}
        assert current is not None and current.user_id == "bob"

    def test_status_counts_by_origin_also_scope_the_accounts_own(self, database: Database) -> None:
        # A project's badge: the account's items of other origins are not its own count in this one.
        _item(database, status="in_progress", origin="canvas")
        _item(database, status="pending", origin="project:1:q:")
        _item(database, status="pending", origin="project:1:q:", user_id="bob")
        _item(database, status="pending", origin="project:2:q:")

        counts, own, current = database.queries.session_queue.status_counts(QUEUE, "alice", "project:1:q:")
        nobodys = database.queries.session_queue.status_counts(QUEUE, None, "project:1:q:")

        assert (counts, own, current) == ({"pending": 2}, {"pending": 1}, None)
        assert nobodys.own_counts == {}

    def test_batch_counts_and_the_batchs_origin(self, database: Database) -> None:
        _item(database, batch_id="b", status="pending", origin="canvas", destination="gallery")
        _item(database, batch_id="b", status="failed", origin="canvas", destination="gallery", user_id="bob")
        _item(database, batch_id="b", status="failed", queue_id="other")
        _item(database, batch_id="other", status="pending")

        everyone = database.queries.session_queue.batch_counts(QUEUE, "b", None)
        bobs = database.queries.session_queue.batch_counts(QUEUE, "b", "bob")
        none = database.queries.session_queue.batch_counts(QUEUE, "missing", None)

        assert everyone == ({"pending": 1, "failed": 1}, "canvas", "gallery")
        assert bobs.counts == {"failed": 1}
        assert none == ({}, None, None)

    def test_destination_counts(self, database: Database) -> None:
        _item(database, destination="canvas", status="pending")
        _item(database, destination="canvas", status="pending", user_id="bob")
        _item(database, destination="canvas", status="pending", queue_id="other")
        _item(database, destination="gallery", status="pending")

        assert database.queries.session_queue.destination_counts(QUEUE, "canvas", None) == {"pending": 2}
        assert database.queries.session_queue.destination_counts(QUEUE, "canvas", "bob") == {"pending": 1}


@pytest.mark.usefixtures("accounts")
class TestWorkflowCallChains:
    def test_a_chain_is_the_ancestors_the_item_and_the_roots_descendants_breadth_first(
        self, database: Database
    ) -> None:
        # Deep enough that breadth first and depth first differ: depth first would list a1 before b.
        root = _item(database, status="waiting")
        a = _item(database, parent_item_id=root)
        b = _item(database, parent_item_id=root)
        a1 = _item(database, parent_item_id=a)
        b1 = _item(database, parent_item_id=b)

        assert database.queries.session_queue.chain_item_ids(a1) == [a, root, a1, b, b1]
        assert database.queries.session_queue.descendant_ids(root) == [a, b, a1, b1]

    def test_a_chain_with_a_missing_item_or_ancestor_reads_as_none(self, database: Database) -> None:
        orphan = _item(database, parent_item_id=404)

        assert database.queries.session_queue.chain_item_ids(orphan) is None
        assert database.queries.session_queue.chain_item_ids(405) is None

    def test_a_cycle_ends_the_walk(self, database: Database) -> None:
        # The app links no cycle; a corrupt one must not hold the transaction forever.
        first = _item(database)
        second = _item(database, parent_item_id=first)
        with database.begin(write=True) as conn:
            conn.execute(session_queue.update().where(session_queue.c.item_id == first).values(parent_item_id=second))

        assert set(database.queries.session_queue.chain_item_ids(second) or []) == {first, second}
        assert database.queries.session_queue.descendant_ids(first) == [second]

    def test_the_current_chains_are_those_of_every_item_in_progress(self, database: Database) -> None:
        first_root = _item(database, status="waiting")
        first = _item(database, status="in_progress", parent_item_id=first_root)
        second = _item(database, status="in_progress")
        _item(database, status="pending")
        _item(database, status="in_progress", queue_id="other")

        assert database.queries.session_queue.current_chain_item_ids(QUEUE) == {first_root, first, second}

    def test_a_running_item_whose_parent_row_is_gone_still_protects_its_chain(self, database: Database) -> None:
        # "Cancel all except current" must neither fail nor cancel a running item's children because an ancestor
        # row vanished: what of the chain is left stays protected.
        grandparent = _item(database, status="waiting")
        parent = _item(database, status="waiting", parent_item_id=grandparent)
        running = _item(database, status="in_progress", parent_item_id=parent)
        child = _item(database, status="pending", parent_item_id=running)
        with database.begin(write=True) as conn:
            conn.execute(session_queue.delete().where(session_queue.c.item_id == grandparent))
        other = _item(database, status="in_progress")

        assert database.queries.session_queue.current_chain_item_ids(QUEUE) == {parent, running, child, other}

    def test_without_one_in_progress_the_chain_of_the_next_waiting_child_is_current(self, database: Database) -> None:
        # The gap between two children of a suspended workflow call: the child that runs next keeps its chain, not
        # another suspended chain, and not a pending child of a parent that no longer waits.
        waiting = _item(database, status="waiting")
        next_child = _item(database, status="pending", parent_item_id=waiting, priority=2)
        sibling = _item(database, status="pending", parent_item_id=waiting, priority=1)
        other_waiting = _item(database, status="waiting")
        _item(database, status="pending", parent_item_id=other_waiting, priority=0)
        other_queue_waiting = _item(database, status="waiting", queue_id="other")
        _item(database, status="pending", parent_item_id=other_queue_waiting, priority=9, queue_id="other")
        finished = _item(database, status="completed")
        _item(database, status="pending", parent_item_id=finished, priority=3)
        _item(database, status="pending", priority=4)

        assert database.queries.session_queue.current_chain_item_ids(QUEUE) == {waiting, next_child, sibling}

    def test_nothing_running_means_no_current_chain(self, database: Database) -> None:
        _item(database, status="pending")

        assert database.queries.session_queue.current_chain_item_ids(QUEUE) == set()


@pytest.mark.usefixtures("accounts")
class TestServiceReads:
    """The DTOs the service builds from these rows, on every backend."""

    def test_an_item_reads_whole_and_for_the_api(self, database: Database, service: SqliteSessionQueue) -> None:
        item_id = _item(database, origin="canvas", destination="gallery")

        item = service.get_queue_item(item_id)
        for_api = service.get_queue_item_for_api(item_id)

        assert item._snapshot_readable and for_api._snapshot_readable
        assert (item.item_id, item.user_display_name, item.field_values[0].value) == (item_id, "Alice", 7)  # type: ignore[index]
        assert (for_api.status, for_api.origin, for_api.destination) == ("pending", "canvas", "gallery")

    def test_summaries_follow_the_callers_order(self, database: Database, service: SqliteSessionQueue) -> None:
        item_ids = [_item(database) for _ in range(3)]
        child = _item(database, parent_item_id=item_ids[0])

        summaries = service.get_queue_item_summaries_by_ids(QUEUE, [child, *reversed(item_ids)])

        assert [summary.item_id for summary in summaries] == [child, *reversed(item_ids)]
        assert (summaries[0].parent_item_id, summaries[0].user_email) == (item_ids[0], "alice@example.com")

    def test_batch_status_and_destination_counts(self, database: Database, service: SqliteSessionQueue) -> None:
        _item(database, batch_id="b", status="pending", origin="canvas", destination="gallery")
        _item(database, batch_id="b", status="completed", origin="canvas", destination="gallery")

        batch = service.get_batch_status(QUEUE, "b")
        destination = service.get_counts_by_destination(QUEUE, "gallery")

        assert (batch.origin, batch.destination, batch.pending, batch.completed, batch.total) == (
            "canvas",
            "gallery",
            1,
            1,
            2,
        )
        assert (destination.pending, destination.completed, destination.total) == (1, 1, 2)

    def test_queue_status_hides_another_accounts_current_item(
        self, database: Database, service: SqliteSessionQueue
    ) -> None:
        bobs = _item(database, status="in_progress", user_id="bob")
        _item(database, status="pending")

        as_alice = service.get_queue_status(QUEUE, user_id="alice")
        as_bob = service.get_queue_status(QUEUE, user_id="bob")

        assert (as_alice.item_id, as_alice.user_pending, as_alice.in_progress) == (None, 1, 1)
        assert (as_bob.item_id, as_bob.user_in_progress, as_bob.total) == (bobs, 1, 2)

    def test_current_next_and_ids(self, database: Database, service: SqliteSessionQueue) -> None:
        running = _item(database, status="in_progress", created_at="2026-01-01 00:00:00.000")
        waiting = _item(database, status="pending", created_at="2026-01-02 00:00:00.000")

        current, next_item = service.get_current(QUEUE), service.get_next_for_api(QUEUE)
        ids = service.get_queue_item_ids(QUEUE, SQLiteDirection.Descending)

        assert (current and current.item_id, next_item and next_item.item_id) == (running, waiting)
        assert (ids.item_ids, ids.total_count) == ([waiting, running], 2)

    def test_list_queue_items_pages_by_its_cursor(self, database: Database, service: SqliteSessionQueue) -> None:
        item_ids = [_item(database, priority=1) for _ in range(3)]

        first = service.list_queue_items(QUEUE, limit=2, priority=0)
        rest = service.list_queue_items(QUEUE, limit=2, priority=1, cursor=item_ids[1])

        assert ([item.item_id for item in first.items], first.has_more) == (item_ids[:2], True)
        assert ([item.item_id for item in rest.items], rest.has_more) == (item_ids[2:], False)


@pytest.mark.usefixtures("accounts")
def test_an_enqueue_receipt_is_found_by_queue_account_and_key(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(session_queue_enqueue_receipts).values(
                queue_id=QUEUE,
                user_id="alice",
                idempotency_key="key",
                payload_hash="hash",
                batch_id="b",
                requested=3,
                enqueued=2,
                priority=0,
                item_ids="[1,2]",
                byte_size=10,
            )
        )

    receipt = database.queries.session_queue.enqueue_receipt(QUEUE, "alice", "key")

    assert receipt == ("b", 3, 2, "[1,2]")
    assert database.queries.session_queue.enqueue_receipt(QUEUE, "bob", "key") is None
