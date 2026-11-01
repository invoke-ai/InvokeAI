"""Which browser stages a queue item's results: a claim per item that one claimant holds until it confirms the delivery,
releases the claim or lets it lapse, decided by one conditional statement on every backend."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic_core import to_jsonable_python
from sqlalchemy import insert, select, update

from invokeai.app.services.session_queue.session_queue_common import (
    QueueItemDeliveryResult,
    QueueItemOwner,
    SessionQueueItemNotFoundError,
)
from invokeai.app.services.session_queue.session_queue_default import QUEUE_ITEM_DELIVERY_CLAIM_TTL, SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.session_queue import Delivery, SessionQueueQueries
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.types import timestamp_text
from invokeai.app.services.shared.execution_state_migration import dump_execution_state
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.fixtures.races import while_in_flight

CLAIMED = QueueItemDeliveryResult(status="claimed")
DELIVERED = QueueItemDeliveryResult(status="delivered")
RELEASED = QueueItemDeliveryResult(status="released")


@pytest.fixture
def queue(database: Database) -> SessionQueue:
    return SessionQueue(database)


@pytest.fixture
def item_id(database: Database) -> int:
    session = GraphExecutionState(graph=Graph())
    values = {
        "queue_id": "default",
        "batch_id": "batch",
        "session_id": session.id,
        "session": json.dumps(dump_execution_state(session), default=to_jsonable_python),
        "status": "completed",
        "user_id": "system",
    }
    with database.begin(write=True) as conn:
        return int(conn.execute(insert(session_queue).values(**values)).inserted_primary_key[0])


def _delivery(database: Database, item_id: int) -> dict[str, Any]:
    q = session_queue.c
    columns = (q.delivery_claimant, q.delivery_claimed_at, q.delivered_at, q.updated_at)
    with database.begin(write=False) as conn:
        return conn.execute(select(*columns).where(q.item_id == item_id)).one()._asdict()


def _claimed_ago(database: Database, item_id: int, age: timedelta) -> None:
    claimed_at = timestamp_text(datetime.now(timezone.utc) - age)
    with database.begin(write=True) as conn:
        conn.execute(
            update(session_queue).where(session_queue.c.item_id == item_id).values(delivery_claimed_at=claimed_at)
        )


def test_a_claim_is_held_for_its_claimant_and_renewed_by_it(database: Database, queue: SessionQueue, item_id: int):
    assert queue.claim_queue_item_delivery(item_id, "tab-a") == CLAIMED
    _claimed_ago(database, item_id, timedelta(seconds=100))
    stale = _delivery(database, item_id)

    assert queue.claim_queue_item_delivery(item_id, "tab-a") == CLAIMED

    delivery = _delivery(database, item_id)
    assert delivery["delivery_claimant"] == "tab-a"
    assert delivery["delivery_claimed_at"] > stale["delivery_claimed_at"]
    # Delivery is not a change of the item the queue's views show.
    assert delivery["updated_at"] == stale["updated_at"]


def test_another_claimant_is_held_off_until_the_claim_lapses(database: Database, queue: SessionQueue, item_id: int):
    queue.claim_queue_item_delivery(item_id, "tab-a")
    _claimed_ago(database, item_id, QUEUE_ITEM_DELIVERY_CLAIM_TTL - timedelta(seconds=20))

    held = queue.claim_queue_item_delivery(item_id, "tab-b")

    assert held.status == "held"
    assert held.retry_after_ms is not None and 15_000 < held.retry_after_ms <= 20_000
    assert _delivery(database, item_id)["delivery_claimant"] == "tab-a"

    _claimed_ago(database, item_id, QUEUE_ITEM_DELIVERY_CLAIM_TTL + timedelta(seconds=1))
    assert queue.claim_queue_item_delivery(item_id, "tab-b") == CLAIMED
    assert queue.claim_queue_item_delivery(item_id, "tab-a").status == "held"


def test_a_confirmed_delivery_is_final(database: Database, queue: SessionQueue, item_id: int):
    queue.claim_queue_item_delivery(item_id, "tab-a")

    # Any claimant confirms: the results were delivered, whoever holds the claim.
    assert queue.confirm_queue_item_delivery(item_id) == DELIVERED
    delivered_at = _delivery(database, item_id)["delivered_at"]
    assert delivered_at is not None

    _claimed_ago(database, item_id, QUEUE_ITEM_DELIVERY_CLAIM_TTL * 2)
    assert queue.claim_queue_item_delivery(item_id, "tab-a") == DELIVERED
    assert queue.claim_queue_item_delivery(item_id, "tab-b") == DELIVERED
    assert queue.release_queue_item_delivery(item_id, "tab-a") == RELEASED
    assert queue.confirm_queue_item_delivery(item_id) == DELIVERED
    assert queue.claim_queue_item_delivery(item_id, "tab-b") == DELIVERED
    assert _delivery(database, item_id)["delivered_at"] == delivered_at


def test_a_delivery_is_confirmed_without_a_claim(queue: SessionQueue, item_id: int):
    assert queue.confirm_queue_item_delivery(item_id) == DELIVERED
    assert queue.claim_queue_item_delivery(item_id, "tab-a") == DELIVERED


def test_only_its_claimant_releases_a_claim(database: Database, queue: SessionQueue, item_id: int):
    queue.claim_queue_item_delivery(item_id, "tab-a")

    assert queue.release_queue_item_delivery(item_id, "tab-b") == RELEASED
    assert queue.claim_queue_item_delivery(item_id, "tab-b").status == "held"

    assert queue.release_queue_item_delivery(item_id, "tab-a") == RELEASED
    assert _delivery(database, item_id)["delivery_claimant"] is None
    assert queue.release_queue_item_delivery(item_id, "tab-a") == RELEASED
    assert queue.claim_queue_item_delivery(item_id, "tab-b") == CLAIMED


def test_a_claim_racing_another_waits_and_is_held_off(database: Database, queue: SessionQueue, item_id: int):
    stale_before = timestamp_text(datetime.now(timezone.utc) - QUEUE_ITEM_DELIVERY_CLAIM_TTL)
    results: list[QueueItemDeliveryResult] = []

    errors = while_in_flight(
        database,
        lambda q: q.session_queue.claim_delivery(
            item_id, "tab-a", now=timestamp_text(datetime.now(timezone.utc)), stale_before=stale_before
        ),
        lambda: results.append(queue.claim_queue_item_delivery(item_id, "tab-b")),
    )

    assert errors == []
    assert [result.status for result in results] == ["held"]
    assert _delivery(database, item_id)["delivery_claimant"] == "tab-a"


def test_a_claim_released_before_its_refusal_is_explained_is_retried_at_once(
    monkeypatch: pytest.MonkeyPatch, queue: SessionQueue, item_id: int
):
    queue.claim_queue_item_delivery(item_id, "tab-a")
    # tab-a releases between tab-b's refused claim and the read that explains the refusal.
    monkeypatch.setattr(SessionQueueQueries, "delivery", lambda self, item_id: Delivery(None, None, None))

    assert queue.claim_queue_item_delivery(item_id, "tab-b") == QueueItemDeliveryResult(status="held", retry_after_ms=1)


def test_an_items_owner_is_read_without_its_session(queue: SessionQueue, item_id: int):
    assert queue.get_queue_item_owner(item_id) == QueueItemOwner(queue_id="default", user_id="system")
    with pytest.raises(SessionQueueItemNotFoundError):
        queue.get_queue_item_owner(item_id + 1000)


def test_a_missing_item_has_no_delivery(queue: SessionQueue, item_id: int):
    missing = item_id + 1000
    with pytest.raises(SessionQueueItemNotFoundError):
        queue.claim_queue_item_delivery(missing, "tab-a")
    with pytest.raises(SessionQueueItemNotFoundError):
        queue.confirm_queue_item_delivery(missing)
    with pytest.raises(SessionQueueItemNotFoundError):
        queue.release_queue_item_delivery(missing, "tab-a")
