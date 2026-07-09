"""Regression tests for multiuser queue status / list scoping.

The queue badge in multiuser mode shows "X/Y" where X is the requesting user's own
pending+in_progress jobs and Y is the global total across all users. For this to work,
get_queue_status must report GLOBAL aggregate counts and ADDITIONALLY return the requesting
user's own counts in user_pending/user_in_progress.

Separately, the virtualized queue list fetches ids via get_queue_item_ids and hydrates them
via get_queue_items_by_item_ids (which redacts other users' items). So get_queue_item_ids must
return every user's ids — otherwise a non-admin never sees the (redacted) entries belonging to
other users.

A regression (#9018) scoped both of these to the calling user, which (a) collapsed the badge to
a single own-count and (b) hid other users' redacted entries entirely. These tests guard the
restored behavior.
"""

import uuid

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import PromptTestInvocation


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def _insert_queue_item(session_queue: SqliteSessionQueue, user_id: str) -> int:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    session = GraphExecutionState(graph=graph)
    session_json = session.model_dump_json(warnings=False, exclude_none=True)
    batch_id = str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, field_values,
                priority, workflow, origin, destination, retried_from_item_id, user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("default", session_json, session.id, batch_id, None, 0, None, None, None, None, user_id),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def test_status_aggregate_counts_are_global_with_user_subcounts(session_queue: SqliteSessionQueue) -> None:
    """A non-admin caller (user_id set) sees global aggregate counts plus their own subcounts."""
    user_a = "user-a"
    user_b = "user-b"
    _insert_queue_item(session_queue, user_id=user_a)
    _insert_queue_item(session_queue, user_id=user_a)
    _insert_queue_item(session_queue, user_id=user_b)

    status = session_queue.get_queue_status(queue_id="default", user_id=user_a)

    # Global counts span every user's pending items.
    assert status.pending == 3
    assert status.total == 3
    # Per-user subcounts reflect only user A's items → badge renders "2/3".
    assert status.user_pending == 2
    assert status.user_in_progress == 0


def test_status_admin_global_call_omits_user_subcounts(session_queue: SqliteSessionQueue) -> None:
    """An admin/global caller (user_id=None) gets global counts and no per-user subcounts."""
    _insert_queue_item(session_queue, user_id="user-a")
    _insert_queue_item(session_queue, user_id="user-b")

    status = session_queue.get_queue_status(queue_id="default")

    assert status.pending == 2
    assert status.total == 2
    assert status.user_pending is None
    assert status.user_in_progress is None


def test_status_current_item_redacted_for_non_owner_but_counts_global(session_queue: SqliteSessionQueue) -> None:
    """When the in-progress item belongs to another user, its identifiers are hidden from a
    non-owner, but the aggregate counts (and the user's own subcounts) remain populated."""
    user_a = "user-a"
    user_b = "user-b"
    b_item_id = _insert_queue_item(session_queue, user_id=user_b)
    _insert_queue_item(session_queue, user_id=user_a)

    in_progress = session_queue.dequeue()
    assert in_progress is not None and in_progress.item_id == b_item_id

    status = session_queue.get_queue_status(queue_id="default", user_id=user_a)

    # B's in-progress item identifiers are hidden from A.
    assert status.item_id is None
    assert status.session_id is None
    assert status.batch_id is None
    # But counts are still global, and A's own subcounts are present.
    assert status.in_progress == 1  # B's item, counted globally
    assert status.user_in_progress == 0  # A owns none in progress
    assert status.user_pending == 1  # A's single pending item


def test_get_queue_item_ids_returns_all_users_ids(session_queue: SqliteSessionQueue) -> None:
    """get_queue_item_ids returns ids for every user so the virtualized list can show the
    (redacted) entries belonging to other users. Redaction happens at hydration time."""
    a_item_id = _insert_queue_item(session_queue, user_id="user-a")
    b_item_id = _insert_queue_item(session_queue, user_id="user-b")

    result = session_queue.get_queue_item_ids(queue_id="default")

    assert set(result.item_ids) == {a_item_id, b_item_id}
    assert result.total_count == 2
