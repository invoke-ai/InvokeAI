"""The session queue's writes on every backend: enqueueing, claiming, status changes, bulk cancels and deletes, sessions
and history, including what each does while another transaction is in flight.

On MySQL and MariaDB transactions run side by side, so every check-then-write of the queue is one conditional
statement or runs under a lock; the race tests hold one transaction open and show the other waits for it or loses
cleanly. On SQLite every transaction excludes the others, so those tests pass there trivially.
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
from pydantic_core import to_jsonable_python
from sqlalchemy import insert, select, update

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent,
    QueueItemsCanceledEvent,
    QueueItemStatusChangedEvent,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import Batch, EnqueueProjectNotFoundError
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries import session_queue as queue_queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.database.queries.session_queue import SessionQueueQueries
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.execution_state_migration import dump_execution_state
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.fixtures.races import waiting, when_called, while_in_flight
from tests.test_nodes import PromptTestInvocation, TestEventService

QUEUE = "default"


@pytest.fixture
def accounts(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(users),
            [
                {"user_id": user, "email": f"{user}@example.com", "display_name": user.title(), "password_hash": "-"}
                for user in ("alice", "bob", "carol")
            ],
        )


@pytest.fixture
def events() -> TestEventService:
    return TestEventService()


def _service(database: Database, events: TestEventService, **config: Any) -> SessionQueue:
    configuration = InvokeAIAppConfig(use_memory_db=True, **config)
    services = SimpleNamespace(
        configuration=configuration, events=events, logger=logging.getLogger("queue"), model_manager=None
    )
    queue = SessionQueue(database)
    queue.start(cast(Invoker, SimpleNamespace(services=services)))
    return queue


@pytest.fixture
def queue(database: Database, events: TestEventService, accounts: None) -> SessionQueue:
    return _service(database, events)


def _batch(prompt: str = "test", runs: int = 1, *, key: Optional[str] = None, origin: Optional[str] = None) -> Batch:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt=prompt))
    return Batch(graph=graph, runs=runs, idempotency_key=key, origin=origin)


def _item(
    database: Database,
    *,
    status: str = "pending",
    user_id: str = "alice",
    priority: int = 0,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    parent_item_id: Optional[int] = None,
    root_item_id: Optional[int] = None,
    prompt: str = "",
    created_at: str = "2026-01-01 00:00:00.000",
    started_at: Optional[str] = None,
) -> int:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt=prompt))
    session = GraphExecutionState(graph=graph)
    values = {
        "queue_id": QUEUE,
        "batch_id": "batch",
        "session_id": session.id,
        "session": json.dumps(dump_execution_state(session), default=to_jsonable_python),
        "status": status,
        "priority": priority,
        "user_id": user_id,
        "origin": origin,
        "destination": destination,
        "parent_item_id": parent_item_id,
        "root_item_id": root_item_id,
        "created_at": created_at,
        "started_at": started_at,
    }
    with database.begin(write=True) as conn:
        return int(conn.execute(insert(session_queue).values(**values)).inserted_primary_key[0])


def _row(database: Database, item_id: int) -> Optional[dict[str, Any]]:
    with database.begin(write=False) as conn:
        row = conn.execute(select(session_queue).where(session_queue.c.item_id == item_id)).first()
    return row._asdict() if row is not None else None


def _status_events(events: TestEventService) -> list[tuple[int, str]]:
    return [(event.item_id, event.status) for event in events.events if isinstance(event, QueueItemStatusChangedEvent)]


class TestEnqueue:
    def test_an_enqueue_stamps_its_items_and_a_retry_with_its_key_returns_them(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        batch = _batch(runs=2, key="k")
        first = asyncio.run(queue.enqueue_batch(QUEUE, batch, False, "alice"))
        again = asyncio.run(queue.enqueue_batch(QUEUE, batch, False, "alice"))

        assert (first.enqueued, len(first.item_ids), again.item_ids) == (2, 2, first.item_ids)
        row = _row(database, first.item_ids[0])
        assert row is not None and row["created_at"] == row["updated_at"] and row["status"] == "pending"
        assert queue.get_enqueue_receipt(QUEUE, "k", "alice") is not None
        assert [type(event) for event in events.events].count(BatchEnqueuedEvent) == 1

    def test_an_enqueue_takes_only_what_fits_and_prepends_above_the_pending(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        queue = _service(database, events, max_queue_size=3)
        asyncio.run(queue.enqueue_batch(QUEUE, _batch(runs=2), False, "alice"))
        prepended = asyncio.run(queue.enqueue_batch(QUEUE, _batch(runs=2), True, "bob"))

        assert (prepended.enqueued, prepended.priority) == (1, 1)
        assert queue.get_queue_status(QUEUE).pending == 3

    def test_an_enqueue_for_a_project_of_another_account_is_refused(self, queue: SessionQueue) -> None:
        batch = _batch().model_copy(update={"project_id": "not-alices"})

        with pytest.raises(EnqueueProjectNotFoundError):
            asyncio.run(queue.enqueue_batch(QUEUE, batch, False, "alice"))

    def test_an_enqueue_waits_for_one_admitting(self, database: Database, queue: SessionQueue) -> None:
        # Two enqueues counting the pending items side by side could both fit into the last free place.
        errors = while_in_flight(
            database,
            lambda q: q.locks.acquire(DatabaseLock.SESSION_QUEUE_ADMISSION),
            lambda: asyncio.run(queue.enqueue_batch(QUEUE, _batch(), False, "alice")),
        )

        assert errors == []
        assert queue.get_queue_status(QUEUE).pending == 1

    def test_an_enqueue_and_a_retry_share_the_media_protection(self, database: Database, queue: SessionQueue) -> None:
        # The intermediates cleanup holds it exclusively between its check and its delete.
        errors = while_in_flight(
            database,
            lambda q: q.locks.acquire(DatabaseLock.MEDIA_PROTECTION),
            lambda: asyncio.run(queue.enqueue_batch(QUEUE, _batch(), False, "alice")),
        )

        assert errors == []


class TestDequeue:
    def test_a_claim_records_its_start_and_device_and_signals_once(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        item_id = _item(database)

        claimed = queue.dequeue(device="cuda:1")

        assert claimed is not None and (claimed.item_id, claimed.status, claimed.device) == (
            item_id,
            "in_progress",
            "cuda:1",
        )
        row = _row(database, item_id)
        assert row is not None and row["started_at"] is not None and row["status_sequence"] == 1
        assert _status_events(events) == [(item_id, "in_progress")]
        assert queue.dequeue() is None

    def test_two_workers_never_claim_one_item(
        self, database: Database, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The second worker claims the item the first one chose, before the first claims it: the first then
        # loses its claim and takes the next item.
        first_item, second_item = _item(database), _item(database)
        real = SessionQueue._apply_device_affinity
        claimed_by_other: list[Any] = []

        def other_worker_first(self: SessionQueue, candidate: Any, resident_keys: set[str]) -> Any:
            if not claimed_by_other:
                claimed_by_other.append(None)
                claimed_by_other[0] = queue.dequeue(device="other")
            return real(self, candidate, resident_keys)

        monkeypatch.setattr(SessionQueue, "_apply_device_affinity", other_worker_first)

        mine = queue.dequeue(device="mine")

        other = claimed_by_other[0]
        assert other is not None and mine is not None
        assert {other.item_id, mine.item_id} == {first_item, second_item}
        assert (other.device, mine.device) == ("other", "mine")

    def test_round_robin_serves_the_account_served_longest_ago(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        queue = _service(database, events, multiuser=True, session_queue_mode="round_robin")
        _item(database, user_id="alice", status="completed", started_at="2026-01-02 00:00:00.000")
        _item(database, user_id="bob", status="completed", started_at="2026-01-01 00:00:00.000")
        _item(database, user_id="alice", priority=5)
        _item(database, user_id="bob", priority=0)
        bobs_best = _item(database, user_id="bob", priority=1)

        served = queue.dequeue()

        assert served is not None and served.item_id == bobs_best

    def test_a_worker_prefers_an_item_whose_models_it_has_loaded(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        queue = _service(database, events, multiuser=True, session_queue_mode="round_robin")
        cache = SimpleNamespace(cached_model_keys=lambda: ["model-key-b"])
        queue._SessionQueue__invoker.services.model_manager = SimpleNamespace(  # type: ignore[attr-defined]
            load=SimpleNamespace(ram_caches={"cuda:0": cache})
        )
        _item(database, prompt="model-key-a")
        warm = _item(database, prompt="model-key-b")

        served = queue.dequeue(device="cuda:0")

        assert served is not None and served.item_id == warm


class TestStatusChanges:
    def test_a_finish_stamps_its_time_and_a_finished_item_stays_finished(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        item_id = _item(database, status="in_progress")

        completed = queue.complete_queue_item(item_id)
        canceled = queue.cancel_queue_item(item_id)

        assert (completed.status, canceled.status) == ("completed", "completed")
        row = _row(database, item_id)
        assert row is not None and row["completed_at"] is not None and row["status_sequence"] == 1
        assert _status_events(events) == [(item_id, "completed")]

    def test_a_status_change_racing_another_loses_cleanly(self, database: Database, queue: SessionQueue) -> None:
        # The other transaction completes the item but has not committed: the cancel waits, then finds it finished.
        item_id = _item(database, status="in_progress")
        outcomes: list[tuple[Any, bool]] = []

        def cancel() -> None:
            outcomes.append(queue._transition_queue_item_status(item_id, "canceled"))

        errors = while_in_flight(database, lambda q: q.session_queue.transition(item_id, "completed"), cancel)

        assert errors == []
        ((item, changed),) = outcomes
        assert (item.status, changed) == ("completed", False)

    def test_a_move_to_in_progress_stamps_its_start(self, database: Database) -> None:
        item_id = _item(database, status="waiting")

        exists, changed = database.queries.session_queue.transition(item_id, "in_progress", device="cpu")

        assert exists and changed is not None
        assert (changed["status"], changed["device"], changed["completed_at"]) == ("in_progress", "cpu", None)
        assert changed["started_at"] is not None

    def test_startup_cancels_interrupted_chains(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        root = _item(database, status="waiting")
        child = _item(database, status="in_progress", parent_item_id=root, root_item_id=root)
        sibling = _item(database, status="pending", parent_item_id=root, root_item_id=root)
        unrelated = _item(database, status="pending")

        _service(database, events)

        statuses = {item_id: (_row(database, item_id) or {})["status"] for item_id in (root, child, sibling, unrelated)}
        assert statuses == {root: "canceled", child: "canceled", sibling: "canceled", unrelated: "pending"}


class TestBulk:
    def test_cancel_by_origin_takes_its_wildcards_as_written_and_signals_running_items_one_by_one(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        waiting = _item(database, origin="canvas_1", user_id="alice")
        bobs = _item(database, origin="Canvas_2", user_id="bob")
        running = _item(database, origin="canvas_3", status="in_progress")
        _item(database, origin="canvasX", user_id="alice")
        _item(database, origin="canvas_4", status="completed")

        result = queue.cancel_by_queue_id(QUEUE, origin_prefix="canvas_")

        assert result.canceled == 3
        bulk = [event for event in events.events if isinstance(event, QueueItemsCanceledEvent)]
        assert [event.canceled_item_ids_by_user for event in bulk] == [{"alice": [waiting], "bob": [bobs]}]
        assert _status_events(events) == [(running, "canceled")]

    def test_cancel_by_batch_cancels_only_the_batchs_items(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        first = asyncio.run(queue.enqueue_batch(QUEUE, _batch(runs=2), False, "alice"))
        second = asyncio.run(queue.enqueue_batch(QUEUE, _batch(runs=2), False, "alice"))

        result = queue.cancel_by_batch_ids(QUEUE, [first.batch.batch_id])

        assert result.canceled == 2
        assert [queue.get_queue_item(item_id).status for item_id in first.item_ids] == ["canceled", "canceled"]
        assert [queue.get_queue_item(item_id).status for item_id in second.item_ids] == ["pending", "pending"]

    def test_cancel_all_except_current_spares_the_running_chain(self, database: Database, queue: SessionQueue) -> None:
        root = _item(database, status="waiting")
        _item(database, status="in_progress", parent_item_id=root, root_item_id=root)
        next_child = _item(database, status="pending", parent_item_id=root, root_item_id=root)
        other = _item(database, status="pending")

        result = queue.cancel_all_except_current(QUEUE)

        assert result.canceled == 1
        assert (_row(database, other) or {})["status"] == "canceled"
        assert (_row(database, next_child) or {})["status"] == "pending"

    def test_delete_by_destination_deletes_running_items_after_canceling_them(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        running = _item(database, destination="canvas", status="in_progress")
        waiting = _item(database, destination="canvas")
        kept = _item(database, destination="gallery")

        result = queue.delete_by_destination(QUEUE, "canvas")

        assert result.deleted == 2
        assert (_row(database, running), _row(database, waiting)) == (None, None)
        assert _row(database, kept) is not None
        bulk = [event for event in events.events if isinstance(event, QueueItemsCanceledEvent)]
        assert [event.canceled_item_ids for event in bulk] == [[waiting]]

    def test_clear_cancels_the_accounts_running_items_and_deletes_only_its_own(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        alices = [_item(database, status="in_progress"), _item(database)]
        bobs = _item(database, user_id="bob", status="in_progress")

        result = queue.clear(QUEUE, user_id="alice")

        assert result.deleted == 2
        assert [_row(database, item_id) for item_id in alices] == [None, None]
        assert (_row(database, bobs) or {})["status"] == "in_progress"
        assert _status_events(events) == [(alices[0], "canceled")]

    def test_prune_keeps_history_an_active_workflow_call_still_needs(
        self, database: Database, queue: SessionQueue
    ) -> None:
        root = _item(database, status="waiting")
        needed = _item(database, status="completed", parent_item_id=root, root_item_id=root)
        old = [_item(database, status=status) for status in ("completed", "failed", "canceled")]
        pending = _item(database)

        assert queue.prune(QUEUE).deleted == 3
        assert [_row(database, item_id) for item_id in old] == [None, None, None]
        assert all(_row(database, item_id) is not None for item_id in (root, needed, pending))

    def test_history_is_pruned_to_its_latest_items(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        finished = [_item(database, status="completed") for _ in range(4)]
        with database.begin(write=True) as conn:
            for day, item_id in enumerate(finished):
                conn.execute(
                    update(session_queue)
                    .where(session_queue.c.item_id == item_id)
                    .values(completed_at=f"2026-01-0{day + 1} 00:00:00.000")
                )

        _service(database, events, max_queue_history=2)

        assert [_row(database, item_id) is not None for item_id in finished] == [False, False, True, True]


class TestSessions:
    def test_every_session_write_moves_the_revision(self, database: Database, queue: SessionQueue) -> None:
        item_id = _item(database, status="in_progress")
        session = queue.get_queue_item(item_id).session

        queue.save_queue_item_session(item_id, session)
        queue.save_queue_item_session(item_id, session)

        assert (_row(database, item_id) or {})["session_revision"] == 2
        assert queue._save_queue_item_session_if_active(item_id, session) is True
        queue.complete_queue_item(item_id)
        assert queue._save_queue_item_session_if_active(item_id, session) is False

    def test_a_session_write_shares_the_media_protection(self, database: Database, queue: SessionQueue) -> None:
        item_id = _item(database, status="in_progress")
        session = queue.get_queue_item(item_id).session

        errors = while_in_flight(
            database,
            lambda q: q.locks.acquire(DatabaseLock.MEDIA_PROTECTION),
            lambda: queue.save_queue_item_session(item_id, session),
        )

        assert errors == []

    def test_a_parent_waits_for_its_children_only_with_the_session_it_was_read_with(self, database: Database) -> None:
        item_id = _item(database, status="in_progress")
        stored = (_row(database, item_id) or {})["session"]

        stale = database.queries.session_queue.wait_for_children(item_id, session="{}", expected_session="other")
        waited = database.queries.session_queue.wait_for_children(item_id, session="{}", expected_session=stored)

        row = _row(database, item_id) or {}
        assert (stale, waited) == (False, True)
        assert (row["status"], row["session"], row["session_revision"]) == ("waiting", "{}", 1)

    def test_a_retry_enqueues_failed_roots_once(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        failed = _item(database, status="failed")
        _item(database, status="pending")

        result = queue.retry_items_by_id(QUEUE, [failed, failed, 404])

        assert result.retried_item_ids == [failed]
        assert queue.get_queue_status(QUEUE).pending == 2


def _session_json(session: GraphExecutionState) -> str:
    return json.dumps(dump_execution_state(session), default=to_jsonable_python)


def _waiting_parent(database: Database, children: int = 2) -> tuple[int, list[GraphExecutionState]]:
    """An item in progress whose session waits on a workflow call with `children` child sessions to enqueue."""
    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    parent_session = GraphExecutionState(graph=graph)
    invocation = parent_session.next()
    frame = parent_session.build_workflow_call_frame(invocation.id, invocation.workflow_id)
    parent_session.begin_waiting_on_workflow_call(frame)
    child_sessions = [parent_session.create_child_workflow_execution_state(Graph(), frame) for _ in range(children)]
    parent_session.attach_waiting_workflow_call_child_sessions(child_sessions)
    parent = _item(database, status="in_progress")
    with database.begin(write=True) as conn:
        conn.execute(
            update(session_queue).where(session_queue.c.item_id == parent).values(session=_session_json(parent_session))
        )
    return parent, child_sessions


def _pending_values(parent: Optional[int] = None) -> dict[str, Any]:
    session = GraphExecutionState(graph=Graph())
    return {
        "queue_id": QUEUE,
        "batch_id": "other",
        "session_id": session.id,
        "session": _session_json(session),
        "status": "pending",
        "user_id": "bob",
        "parent_item_id": parent,
        "root_item_id": parent,
    }


class TestRaces:
    """What a write does while another transaction is in flight; on SQLite the other one has to wait anyway."""

    def test_capacity_is_recounted_after_admission_waits(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        # An enqueue that counted the free places before another filled them must not overfill the queue.
        queue = _service(database, events, max_queue_size=2)

        def fill(q: Queries) -> None:
            q.locks.acquire(DatabaseLock.SESSION_QUEUE_ADMISSION)
            for _ in range(2):
                q.session_queue.insert_item(_pending_values())

        errors = while_in_flight(
            database, fill, lambda: asyncio.run(queue.enqueue_batch(QUEUE, _batch(runs=2), False, "alice"))
        )

        assert errors == []
        assert queue.get_queue_status(QUEUE).pending == 2

    @pytest.mark.parametrize("lock", [DatabaseLock.SESSION_QUEUE_ADMISSION, DatabaseLock.MEDIA_PROTECTION])
    def test_a_retry_waits_for_the_admission_and_the_cleanup(
        self, database: Database, queue: SessionQueue, lock: DatabaseLock
    ) -> None:
        failed = _item(database, status="failed")

        errors = while_in_flight(
            database, lambda q: q.locks.acquire(lock), lambda: queue.retry_items_by_id(QUEUE, [failed])
        )

        assert errors == []

    @pytest.mark.parametrize("lock", [DatabaseLock.SESSION_QUEUE_ADMISSION, DatabaseLock.MEDIA_PROTECTION])
    def test_a_child_enqueue_waits_for_the_admission_and_the_cleanup(
        self, database: Database, queue: SessionQueue, lock: DatabaseLock
    ) -> None:
        parent, child_sessions = _waiting_parent(database)

        errors = while_in_flight(
            database,
            lambda q: q.locks.acquire(lock),
            lambda: queue.enqueue_workflow_call_children(
                queue.get_queue_item(parent), [(session, None) for session in child_sessions]
            ),
        )

        assert errors == []

    def test_a_child_completion_shares_the_media_protection(self, database: Database, queue: SessionQueue) -> None:
        parent, child_sessions = _waiting_parent(database)
        children = queue.enqueue_workflow_call_children(
            queue.get_queue_item(parent), [(session, None) for session in child_sessions]
        )
        queue.complete_queue_item(children[0].item_id)

        errors = while_in_flight(
            database,
            lambda q: q.locks.acquire(DatabaseLock.MEDIA_PROTECTION),
            lambda: queue.record_workflow_call_child_completion(parent, children[0].item_id, {"r": 1}),
        )

        assert errors == []

    def test_two_children_completing_at_once_both_record(
        self, database: Database, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The second records into the session the first wrote, not into the one both read.
        parent, child_sessions = _waiting_parent(database)
        children = queue.enqueue_workflow_call_children(
            queue.get_queue_item(parent), [(session, None) for session in child_sessions]
        )
        for child in children:
            queue.complete_queue_item(child.item_id)
        results: list[Any] = []
        ended = when_called(
            monkeypatch,
            SessionQueueQueries,
            "set_session",
            lambda: results.append(queue.record_workflow_call_child_completion(parent, children[1].item_id, {"r": 2})),
        )

        results.append(queue.record_workflow_call_child_completion(parent, children[0].item_id, {"r": 1}))

        assert ended() == []
        execution = queue.get_queue_item(parent).session.waiting_workflow_call_execution
        assert execution is not None
        assert set(execution.completed_child_item_ids) == {children[0].item_id, children[1].item_id}
        assert sum(result is not None and result.should_resume for result in results) == 1

    def test_a_bulk_cancel_does_not_report_an_item_claimed_meanwhile(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        # The claim in flight wins: the bulk cancel must neither name the item in its event nor count it twice.
        claimed = _item(database)
        outcomes: list[Any] = []

        errors = while_in_flight(
            database,
            lambda q: q.session_queue.claim(claimed, "gpu"),
            lambda: outcomes.append(queue.cancel_by_queue_id(QUEUE)),
        )

        assert errors == []
        (result,) = outcomes
        bulk = [event for event in events.events if isinstance(event, QueueItemsCanceledEvent)]
        assert (result.canceled, [item for event in bulk for item in event.canceled_item_ids]) == (1, [])
        assert (_row(database, claimed) or {})["status"] == "canceled"

    def test_a_claim_waits_for_a_cancel_of_everything_but_the_running_chains(
        self, database: Database, queue: SessionQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The cancel locks its rows before it reads which chains run: a worker claiming one of them meanwhile waits,
        # and then finds it canceled, instead of starting a child whose parent the cancel takes away.
        _item(database, status="in_progress", user_id="bob")
        parent = _item(database, status="waiting")
        child = _item(database, parent_item_id=parent, root_item_id=parent)
        real_current_chain = queue_queries._current_chain
        claims: list[Any] = []

        def current_chain_then_claim(conn: Any, queue_id: str) -> set[int]:
            if not claims:
                claims.append(waiting(lambda: database.queries.session_queue.claim(child, "gpu")))
            return real_current_chain(conn, queue_id)

        monkeypatch.setattr(queue_queries, "_current_chain", current_chain_then_claim)

        queue.cancel_all_except_current(QUEUE)

        assert claims[0]() == []
        assert [(_row(database, item_id) or {})["status"] for item_id in (parent, child)] == ["canceled", "canceled"]

    @pytest.mark.parametrize("operation", ["cancel", "delete"])
    def test_the_children_a_running_item_enqueues_meanwhile_are_spared(
        self, database: Database, queue: SessionQueue, operation: str
    ) -> None:
        # The running item enqueues its children and waits for them while the cancel or delete is in flight: the
        # chains are read after the rows are locked, so the children and the parent count as running.
        parent = _item(database, status="in_progress")
        children: list[int] = []

        def enqueue_children(q: Queries) -> None:
            children.extend(q.session_queue.insert_item(_pending_values(parent)) for _ in range(2))
            q.session_queue.transition(parent, "waiting")

        change = queue.cancel_all_except_current if operation == "cancel" else queue.delete_all_except_current

        errors = while_in_flight(database, enqueue_children, lambda: change(QUEUE))

        assert errors == []
        assert [(_row(database, item_id) or {}).get("status") for item_id in children] == ["pending", "pending"]


class TestStamps:
    def test_a_bulk_cancel_and_a_startup_cancel_stamp_completion(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        queue = _service(database, events)
        bulk = _item(database)
        queue.cancel_by_queue_id(QUEUE)
        interrupted = _item(database, status="in_progress")

        _service(database, events)

        for item_id in (bulk, interrupted):
            row = _row(database, item_id) or {}
            assert row["status"] == "canceled" and row["completed_at"] is not None

    def test_a_status_change_moves_updated_at(self, database: Database, queue: SessionQueue) -> None:
        item_id = _item(database, status="in_progress")
        with database.begin(write=True) as conn:
            conn.execute(
                update(session_queue)
                .where(session_queue.c.item_id == item_id)
                .values(updated_at="2020-01-01 00:00:00.000")
            )

        queue.complete_queue_item(item_id)

        assert (_row(database, item_id) or {})["updated_at"] > "2020-01-01 00:00:00.000"


class TestScopes:
    def test_a_prune_of_an_account_keeps_the_others_history(self, database: Database, queue: SessionQueue) -> None:
        alices = _item(database, status="completed")
        bobs = _item(database, status="completed", user_id="bob")

        assert queue.prune(QUEUE, user_id="alice").deleted == 1
        assert (_row(database, alices), _row(database, bobs) is not None) == (None, True)

    def test_delete_all_except_current_keeps_history(self, database: Database, queue: SessionQueue) -> None:
        done = _item(database, status="completed")
        waiting_item = _item(database)

        assert queue.delete_all_except_current(QUEUE).deleted == 1
        assert (_row(database, done) is not None, _row(database, waiting_item)) == (True, None)

    def test_capacity_counts_only_pending_items(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        queue = _service(database, events, max_queue_size=1)
        _item(database, status="completed")

        assert asyncio.run(queue.enqueue_batch(QUEUE, _batch(), False, "alice")).enqueued == 1

    def test_startup_keeps_finished_chain_items_and_cancels_lone_waiting_items_and_orphans(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        root = _item(database, status="waiting")
        done_child = _item(database, status="completed", parent_item_id=root, root_item_id=root)
        next_child = _item(database, status="pending", parent_item_id=root, root_item_id=root)
        orphan_parent = _item(database, status="waiting", parent_item_id=10**9)
        orphan_child = _item(database, status="pending", parent_item_id=orphan_parent)

        _service(database, events)

        statuses = [
            (_row(database, item_id) or {})["status"]
            for item_id in (root, done_child, next_child, orphan_parent, orphan_child)
        ]
        assert statuses == ["canceled", "completed", "canceled", "canceled", "canceled"]

    def test_fifo_serves_the_highest_priority_first(
        self, database: Database, events: TestEventService, accounts: None
    ) -> None:
        queue = _service(database, events, session_queue_mode="FIFO")
        _item(database, priority=0)
        high = _item(database, priority=5)

        served = queue.dequeue()

        assert served is not None and served.item_id == high

    def test_a_status_event_carries_its_batch_counts(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        first, _ = _item(database), _item(database)

        queue.cancel_queue_item(first)

        (event,) = [event for event in events.events if isinstance(event, QueueItemStatusChangedEvent)]
        assert (event.batch_status.batch_id, event.batch_status.canceled, event.batch_status.pending) == ("batch", 1, 1)

    def test_an_enqueue_event_goes_to_its_owner(
        self, database: Database, queue: SessionQueue, events: TestEventService
    ) -> None:
        asyncio.run(queue.enqueue_batch(QUEUE, _batch(), False, "bob"))

        (event,) = [event for event in events.events if isinstance(event, BatchEnqueuedEvent)]
        assert event.user_id == "bob"
