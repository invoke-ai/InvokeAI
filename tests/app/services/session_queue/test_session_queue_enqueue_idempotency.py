import asyncio
import threading
from typing import Any

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue import session_queue_sqlite
from invokeai.app.services.session_queue.session_queue_common import (
    Batch,
    EnqueueIdempotencyConflictError,
    EnqueueProjectNotFoundError,
    EnqueueReceiptLimitError,
)
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph
from tests.test_nodes import PromptTestInvocation


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO users (user_id, email, password_hash) VALUES (?, ?, ?);",
            [("user-1", "user-1@example.com", "test"), ("user-2", "user-2@example.com", "test")],
        )
    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    queue.start(mock_invoker)
    return queue


def _batch(
    *,
    batch_id: str | None = None,
    idempotency_key: str = "webv2:item-1",
    origin: str = "webv2:project-1:item-1",
    prompt: str = "test",
    project_id: str | None = None,
    runs: int = 2,
) -> Batch:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt=prompt))
    return Batch(
        **({"batch_id": batch_id} if batch_id is not None else {}),
        idempotency_key=idempotency_key,
        graph=graph,
        origin=origin,
        project_id=project_id,
        runs=runs,
    )


def test_enqueue_batch_retries_return_the_original_items(session_queue: SqliteSessionQueue) -> None:
    batch = _batch()
    first = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))
    retry = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))

    assert retry.item_ids == first.item_ids
    assert retry.batch.batch_id == first.batch.batch_id
    assert retry.enqueued == first.enqueued == 2
    assert len(session_queue.list_all_queue_items("default")) == 2


def test_distinct_idempotent_enqueues_cannot_share_a_caller_batch_id(session_queue: SqliteSessionQueue) -> None:
    caller_batch_id = "caller-reused-batch"
    first = asyncio.run(
        session_queue.enqueue_batch(
            "default", _batch(batch_id=caller_batch_id, idempotency_key="first"), False, "user-1"
        )
    )
    second = asyncio.run(
        session_queue.enqueue_batch(
            "default", _batch(batch_id=caller_batch_id, idempotency_key="second"), False, "user-1"
        )
    )

    assert first.batch.batch_id == caller_batch_id
    assert second.batch.batch_id != caller_batch_id
    assert second.batch.batch_id != first.batch.batch_id
    assert set(first.item_ids).isdisjoint(second.item_ids)
    assert len(first.item_ids) == first.enqueued == 2
    assert len(second.item_ids) == second.enqueued == 2

    session_queue.cancel_by_batch_ids("default", [second.batch.batch_id], user_id="user-1")

    assert all(session_queue.get_queue_item(item_id).status == "pending" for item_id in first.item_ids)
    assert all(session_queue.get_queue_item(item_id).status == "canceled" for item_id in second.item_ids)


def test_concurrent_enqueue_retries_commit_once(session_queue: SqliteSessionQueue) -> None:
    batch = _batch()

    async def enqueue_twice():
        return await asyncio.gather(
            session_queue.enqueue_batch("default", batch, False, "user-1"),
            session_queue.enqueue_batch("default", batch, False, "user-1"),
        )

    first, second = asyncio.run(enqueue_twice())

    assert second.item_ids == first.item_ids
    assert len(session_queue.list_all_queue_items("default")) == 2


def test_acknowledgement_does_not_erase_identity_for_an_inflight_retry(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _batch()
    first_prepared = threading.Event()
    second_prepared = threading.Event()
    release_second = threading.Event()
    prepare_call_count = 0
    original = session_queue_sqlite.prepare_values_to_insert

    def interleave_prepare(*args: Any, **kwargs: Any):
        nonlocal prepare_call_count
        values = original(*args, **kwargs)
        prepare_call_count += 1
        if prepare_call_count == 1:
            first_prepared.set()
            assert second_prepared.wait(timeout=5)
        else:
            second_prepared.set()
            assert release_second.wait(timeout=5)
        return values

    monkeypatch.setattr(session_queue_sqlite, "prepare_values_to_insert", interleave_prepare)

    async def exercise_interleaving():
        first_task = asyncio.create_task(session_queue.enqueue_batch("default", batch, False, "user-1"))
        assert await asyncio.to_thread(first_prepared.wait, 5)
        second_task = asyncio.create_task(session_queue.enqueue_batch("default", batch, False, "user-1"))
        assert await asyncio.to_thread(second_prepared.wait, 5)
        first = await first_task
        session_queue.acknowledge_enqueue("default", batch.idempotency_key or "", "user-1")
        release_second.set()
        return first, await second_task

    first, delayed_retry = asyncio.run(exercise_interleaving())

    assert delayed_retry.item_ids == first.item_ids
    assert len(session_queue.list_all_queue_items("default")) == 2


def test_enqueue_batch_id_is_scoped_by_user(session_queue: SqliteSessionQueue) -> None:
    first = asyncio.run(session_queue.enqueue_batch("default", _batch(), False, "user-1"))
    second = asyncio.run(session_queue.enqueue_batch("default", _batch(), False, "user-2"))

    assert len(first.item_ids) == len(second.item_ids) == 2
    assert set(first.item_ids).isdisjoint(second.item_ids)
    assert first.batch.batch_id != second.batch.batch_id
    assert len(session_queue.list_all_queue_items("default")) == 4


def test_enqueue_batch_id_is_scoped_by_queue(session_queue: SqliteSessionQueue) -> None:
    first = asyncio.run(session_queue.enqueue_batch("queue-1", _batch(), False, "user-1"))
    second = asyncio.run(session_queue.enqueue_batch("queue-2", _batch(), False, "user-1"))

    assert len(first.item_ids) == len(second.item_ids) == 2
    assert set(first.item_ids).isdisjoint(second.item_ids)
    assert first.batch.batch_id != second.batch.batch_id


def test_enqueue_batch_rejects_identity_reuse_for_changed_payload(session_queue: SqliteSessionQueue) -> None:
    asyncio.run(session_queue.enqueue_batch("default", _batch(), False, "user-1"))

    with pytest.raises(EnqueueIdempotencyConflictError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(prompt="changed"), False, "user-1"))

    with pytest.raises(EnqueueIdempotencyConflictError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(runs=3), False, "user-1"))

    with pytest.raises(EnqueueIdempotencyConflictError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(), True, "user-1"))


def test_enqueue_receipt_survives_queue_item_deletion(session_queue: SqliteSessionQueue) -> None:
    batch = _batch()
    first = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))
    session_queue.delete_queue_items_by_id(first.item_ids)

    retry = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))

    assert retry.item_ids == first.item_ids
    assert retry.enqueued == first.enqueued == 2
    assert session_queue.list_all_queue_items("default") == []


def test_enqueue_receipt_does_not_expire_while_unacknowledged(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    batch = _batch()
    first = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))
    session_queue.delete_queue_items_by_id(first.item_ids)
    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue_enqueue_receipts SET created_at = '2000-01-01';")

    retry = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))

    assert retry.item_ids == first.item_ids
    assert session_queue.list_all_queue_items("default") == []


def test_acknowledged_enqueue_receipt_still_settles_delayed_retries(session_queue: SqliteSessionQueue) -> None:
    batch = _batch()
    first = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))
    session_queue.delete_queue_items_by_id(first.item_ids)

    session_queue.acknowledge_enqueue("default", batch.idempotency_key or "", "user-1")
    second = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))

    assert second.item_ids == first.item_ids
    assert session_queue.list_all_queue_items("default") == []


def test_enqueue_receipt_lookup_is_scoped_by_user_and_queue(session_queue: SqliteSessionQueue) -> None:
    batch = _batch()
    accepted = asyncio.run(session_queue.enqueue_batch("queue-1", batch, False, "user-1"))

    receipt = session_queue.get_enqueue_receipt("queue-1", batch.idempotency_key or "", "user-1")

    assert receipt is not None
    assert receipt.batch_id == accepted.batch.batch_id
    assert receipt.item_ids == accepted.item_ids
    assert receipt.requested == accepted.requested
    assert receipt.enqueued == accepted.enqueued
    assert session_queue.get_enqueue_receipt("queue-2", batch.idempotency_key or "", "user-1") is None
    assert session_queue.get_enqueue_receipt("queue-1", batch.idempotency_key or "", "user-2") is None


def test_acknowledge_enqueue_is_scoped_by_user_and_queue(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_queue.session_queue_sqlite.MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPTS_PER_OWNER", 1
    )
    batch = _batch()
    asyncio.run(session_queue.enqueue_batch("queue-1", batch, False, "user-1"))

    session_queue.acknowledge_enqueue("queue-2", batch.idempotency_key or "", "user-1")
    session_queue.acknowledge_enqueue("queue-1", batch.idempotency_key or "", "user-2")

    with pytest.raises(EnqueueReceiptLimitError):
        asyncio.run(session_queue.enqueue_batch("queue-2", _batch(idempotency_key="second"), False, "user-1"))

    session_queue.acknowledge_enqueue("queue-1", batch.idempotency_key or "", "user-1")
    accepted = asyncio.run(session_queue.enqueue_batch("queue-2", _batch(idempotency_key="second"), False, "user-1"))
    assert accepted.enqueued == 2


def test_expired_acknowledged_receipts_are_collected_on_admission(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    first = _batch(idempotency_key="first")
    asyncio.run(session_queue.enqueue_batch("default", first, False, "user-1"))
    session_queue.acknowledge_enqueue("default", "first", "user-1")
    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.execute(
            "UPDATE session_queue_enqueue_receipts SET acknowledged_at = '2000-01-01' WHERE idempotency_key = 'first';"
        )

    asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="second"), False, "user-1"))

    assert session_queue.get_enqueue_receipt("default", "first", "user-1") is None


def test_acknowledged_receipts_are_retained_for_seven_days(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    first = _batch(idempotency_key="first")
    asyncio.run(session_queue.enqueue_batch("default", first, False, "user-1"))
    session_queue.acknowledge_enqueue("default", "first", "user-1")
    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.execute(
            """--sql
            UPDATE session_queue_enqueue_receipts
            SET acknowledged_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', '-6 days')
            WHERE idempotency_key = 'first';
            """
        )

    asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="second"), False, "user-1"))

    assert session_queue.get_enqueue_receipt("default", "first", "user-1") is not None


def test_unacknowledged_receipts_are_bounded_without_eviction(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_queue.session_queue_sqlite.MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPTS_PER_OWNER", 1
    )
    asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="first"), False, "user-1"))

    with pytest.raises(EnqueueReceiptLimitError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="second"), False, "user-1"))

    session_queue.acknowledge_enqueue("default", "first", "user-1")
    result = asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="second"), False, "user-1"))
    assert result.enqueued == 2


def test_unacknowledged_receipt_count_limit_is_per_user_across_queues(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_queue.session_queue_sqlite.MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPTS_PER_OWNER", 1
    )
    first_batch = _batch(idempotency_key="first")
    asyncio.run(session_queue.enqueue_batch("queue-1", first_batch, False, "user-1"))

    with pytest.raises(EnqueueReceiptLimitError):
        asyncio.run(session_queue.enqueue_batch("queue-2", _batch(idempotency_key="second"), False, "user-1"))

    retry = asyncio.run(session_queue.enqueue_batch("queue-1", first_batch, False, "user-1"))
    assert retry.enqueued == 2
    other_user = asyncio.run(session_queue.enqueue_batch("queue-2", _batch(idempotency_key="second"), False, "user-2"))
    assert other_user.enqueued == 2


def test_unacknowledged_receipt_bytes_are_bounded_atomically(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_queue.session_queue_sqlite.MAX_UNACKNOWLEDGED_ENQUEUE_RECEIPT_BYTES_PER_OWNER",
        1,
    )

    with pytest.raises(EnqueueReceiptLimitError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(), False, "user-1"))

    assert session_queue.list_all_queue_items("default") == []
    assert session_queue.get_enqueue_receipt("default", "webv2:item-1", "user-1") is None


def test_total_receipt_count_includes_acknowledged_tombstones(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("invokeai.app.services.session_queue.session_queue_sqlite.MAX_ENQUEUE_RECEIPTS_PER_OWNER", 1)
    first_batch = _batch(idempotency_key="first")
    asyncio.run(session_queue.enqueue_batch("default", first_batch, False, "user-1"))
    session_queue.acknowledge_enqueue("default", "first", "user-1")
    retry = asyncio.run(session_queue.enqueue_batch("default", first_batch, False, "user-1"))
    assert retry.enqueued == 2

    with pytest.raises(EnqueueReceiptLimitError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="second"), False, "user-1"))


def test_total_receipt_bytes_are_bounded_atomically(
    session_queue: SqliteSessionQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_queue.session_queue_sqlite.MAX_ENQUEUE_RECEIPT_BYTES_PER_OWNER", 1
    )

    with pytest.raises(EnqueueReceiptLimitError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(), False, "user-1"))

    assert session_queue.list_all_queue_items("default") == []


def test_project_scoped_enqueue_requires_an_owned_project(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    mock_invoker.services.project_records.create("system", "Owned", {}, project_id="project-1")

    with pytest.raises(EnqueueProjectNotFoundError):
        asyncio.run(
            session_queue.enqueue_batch(
                "default", _batch(idempotency_key="missing", project_id="missing"), False, "system"
            )
        )
    with pytest.raises(EnqueueProjectNotFoundError):
        asyncio.run(
            session_queue.enqueue_batch(
                "default", _batch(idempotency_key="wrong-owner", project_id="project-1"), False, "user-1"
            )
        )

    assert session_queue.list_all_queue_items("default") == []


def test_project_deletion_wins_before_new_queue_admission(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    mock_invoker.services.project_records.create("system", "Doomed", {}, project_id="project-1")
    mock_invoker.services.project_records.delete("system", "project-1")

    with pytest.raises(EnqueueProjectNotFoundError):
        asyncio.run(
            session_queue.enqueue_batch(
                "default", _batch(idempotency_key="after-delete", project_id="project-1"), False, "system"
            )
        )

    assert session_queue.list_all_queue_items("default") == []


def test_existing_enqueue_receipt_settles_after_project_deletion(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    mock_invoker.services.project_records.create("system", "Doomed", {}, project_id="project-1")
    batch = _batch(project_id="project-1")
    first = asyncio.run(session_queue.enqueue_batch("default", batch, False, "system"))
    mock_invoker.services.project_records.delete("system", "project-1")

    retry = asyncio.run(session_queue.enqueue_batch("default", batch, False, "system"))

    assert retry.item_ids == first.item_ids


def test_full_queue_does_not_consume_the_idempotency_key(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mock_invoker.services.configuration, "max_queue_size", 0)
    batch = _batch()

    rejected = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))
    monkeypatch.setattr(mock_invoker.services.configuration, "max_queue_size", 10)
    accepted = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))

    assert rejected.enqueued == 0
    assert accepted.enqueued == 2


def test_partial_acceptance_is_settled_idempotently_after_capacity_changes(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mock_invoker.services.configuration, "max_queue_size", 1)
    batch = _batch(runs=2)
    first = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))
    session_queue.delete_queue_items_by_id(first.item_ids)
    monkeypatch.setattr(mock_invoker.services.configuration, "max_queue_size", 10)

    retry = asyncio.run(session_queue.enqueue_batch("default", batch, False, "user-1"))

    assert first.enqueued == retry.enqueued == 1
    assert retry.item_ids == first.item_ids
    assert session_queue.list_all_queue_items("default") == []


def test_full_queue_does_not_prepare_session_payloads(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mock_invoker.services.configuration, "max_queue_size", 1)
    asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="first", runs=1), False, "user-1"))
    requested_prepare_limits: list[int] = []
    original = session_queue_sqlite.prepare_values_to_insert

    def observe_prepare_limit(*args: Any, **kwargs: Any):
        requested_prepare_limits.append(kwargs["max_new_queue_items"])
        return original(*args, **kwargs)

    monkeypatch.setattr(session_queue_sqlite, "prepare_values_to_insert", observe_prepare_limit)

    result = asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="second"), False, "user-1"))

    assert result.enqueued == 0
    assert requested_prepare_limits == [0]


def test_enqueue_batch_rejects_identity_reuse_for_another_origin(session_queue: SqliteSessionQueue) -> None:
    asyncio.run(session_queue.enqueue_batch("default", _batch(), False, "user-1"))

    with pytest.raises(EnqueueIdempotencyConflictError):
        asyncio.run(session_queue.enqueue_batch("default", _batch(origin="webv2:project-2:item-2"), False, "user-1"))


def test_enqueued_items_carry_their_project(session_queue: SqliteSessionQueue, mock_invoker: Invoker) -> None:
    mock_invoker.services.project_records.create("system", "Owned", {}, project_id="project-1")

    asyncio.run(
        session_queue.enqueue_batch(
            "default", _batch(idempotency_key="scoped", project_id="project-1"), False, "system"
        )
    )
    asyncio.run(session_queue.enqueue_batch("default", _batch(idempotency_key="unscoped"), False, "system"))

    items = session_queue.list_all_queue_items("default")
    assert sorted(item.project_id for item in items if item.project_id is not None) == ["project-1", "project-1"]
    assert [item.project_id for item in items].count(None) == 2
