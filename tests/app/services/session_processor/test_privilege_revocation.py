"""Tests that queued execution is revoked when the owning account is deactivated
or deleted.

Policy (see queue_owner_is_active):
- Pending items are rejected (canceled) at dequeue, before any invocation runs.
- Running items are stopped at the next node boundary; canceling also sets the
  processor's cancel event, which stops step-callback nodes mid-node.
- Single-user mode and the ``system`` user are exempt.
"""

from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
    queue_owner_is_active,
)
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItemNotFoundError


def _services(multiuser: bool = True, users_by_id: dict | None = None) -> SimpleNamespace:
    users_by_id = users_by_id or {}
    return SimpleNamespace(
        configuration=SimpleNamespace(multiuser=multiuser),
        users=SimpleNamespace(get=lambda user_id: users_by_id.get(user_id)),
        session_queue=MagicMock(),
        logger=MagicMock(),
    )


def _queue_item(user_id: str = "user-1", item_id: int = 7) -> SimpleNamespace:
    return SimpleNamespace(user_id=user_id, item_id=item_id)


def _active(user_id: str) -> SimpleNamespace:
    return SimpleNamespace(user_id=user_id, is_active=True, is_admin=False)


def _inactive(user_id: str) -> SimpleNamespace:
    return SimpleNamespace(user_id=user_id, is_active=False, is_admin=False)


class TestQueueOwnerIsActive:
    def test_single_user_mode_is_always_active(self) -> None:
        services = _services(multiuser=False)
        assert queue_owner_is_active(services, _queue_item(user_id="anyone")) is True

    def test_system_user_is_always_active(self) -> None:
        services = _services(multiuser=True)
        assert queue_owner_is_active(services, _queue_item(user_id="system")) is True

    def test_active_user(self) -> None:
        services = _services(users_by_id={"user-1": _active("user-1")})
        assert queue_owner_is_active(services, _queue_item()) is True

    def test_deactivated_user(self) -> None:
        services = _services(users_by_id={"user-1": _inactive("user-1")})
        assert queue_owner_is_active(services, _queue_item()) is False

    def test_deleted_user(self) -> None:
        services = _services(users_by_id={})
        assert queue_owner_is_active(services, _queue_item()) is False


class TestDequeueRejection:
    """Items whose owner was deactivated while pending are canceled at dequeue and
    never executed."""

    def _processor(self, services: SimpleNamespace) -> DefaultSessionProcessor:
        processor = DefaultSessionProcessor.__new__(DefaultSessionProcessor)
        processor._invoker = SimpleNamespace(services=services)
        return processor

    def test_inactive_owner_item_is_canceled(self) -> None:
        services = _services(users_by_id={"user-1": _inactive("user-1")})
        processor = self._processor(services)
        item = _queue_item()

        assert processor._cancel_queue_item_if_owner_inactive(item) is True
        services.session_queue.cancel_queue_item.assert_called_once_with(7)

    def test_deleted_owner_item_is_canceled(self) -> None:
        services = _services(users_by_id={})
        processor = self._processor(services)

        assert processor._cancel_queue_item_if_owner_inactive(_queue_item()) is True
        services.session_queue.cancel_queue_item.assert_called_once()

    def test_active_owner_item_is_executed(self) -> None:
        services = _services(users_by_id={"user-1": _active("user-1")})
        processor = self._processor(services)

        assert processor._cancel_queue_item_if_owner_inactive(_queue_item()) is False
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_system_item_is_executed(self) -> None:
        services = _services(multiuser=True)
        processor = self._processor(services)

        assert processor._cancel_queue_item_if_owner_inactive(_queue_item(user_id="system")) is False
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_missing_queue_item_does_not_raise(self) -> None:
        """The item may be deleted concurrently; rejection still stands."""
        services = _services(users_by_id={})
        services.session_queue.cancel_queue_item.side_effect = SessionQueueItemNotFoundError("gone")
        processor = self._processor(services)

        assert processor._cancel_queue_item_if_owner_inactive(_queue_item()) is True


class TestUserAccessChangedCancelsCurrentItem:
    """Deactivating a user cancels their currently running queue item immediately."""

    def _processor(self, services: SimpleNamespace, queue_item: SimpleNamespace | None) -> DefaultSessionProcessor:
        processor = DefaultSessionProcessor.__new__(DefaultSessionProcessor)
        processor._invoker = SimpleNamespace(services=services)
        processor._queue_item = queue_item
        return processor

    def _event(self, user_id: str, is_active: bool) -> tuple:
        return (
            "user_access_changed",
            SimpleNamespace(user_id=user_id, is_admin=False, is_active=is_active),
        )

    @pytest.mark.anyio
    async def test_deactivation_cancels_owned_running_item(self) -> None:
        services = _services()
        processor = self._processor(services, _queue_item(user_id="user-1", item_id=11))

        await processor._on_user_access_changed(self._event("user-1", is_active=False))

        services.session_queue.cancel_queue_item.assert_called_once_with(11)

    @pytest.mark.anyio
    async def test_deactivation_of_other_user_does_not_cancel(self) -> None:
        services = _services()
        processor = self._processor(services, _queue_item(user_id="user-1"))

        await processor._on_user_access_changed(self._event("user-2", is_active=False))

        services.session_queue.cancel_queue_item.assert_not_called()

    @pytest.mark.anyio
    async def test_role_change_alone_does_not_cancel(self) -> None:
        services = _services()
        processor = self._processor(services, _queue_item(user_id="user-1"))

        await processor._on_user_access_changed(self._event("user-1", is_active=True))

        services.session_queue.cancel_queue_item.assert_not_called()

    @pytest.mark.anyio
    async def test_no_current_item_is_a_noop(self) -> None:
        services = _services()
        processor = self._processor(services, None)

        await processor._on_user_access_changed(self._event("user-1", is_active=False))

        services.session_queue.cancel_queue_item.assert_not_called()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestRunnerStopsBetweenNodes:
    """A session whose owner is deactivated mid-run stops before the next node."""

    def _runner_with_services(self, services: SimpleNamespace) -> DefaultSessionRunner:
        runner = DefaultSessionRunner()
        runner.start(services=services, cancel_event=ThreadEvent(), profiler=None)
        return runner

    def _multi_node_queue_item(self, nodes: list, user_id: str = "user-1") -> SimpleNamespace:
        """A queue item whose session yields `nodes` then None."""
        node_iter = iter([*nodes, None])
        session = SimpleNamespace(
            next=lambda: next(node_iter),
            is_complete=lambda: False,
        )
        return SimpleNamespace(user_id=user_id, item_id=21, status="in_progress", session=session)

    def test_deactivation_after_first_node_stops_later_nodes(self) -> None:
        node1, node2 = SimpleNamespace(id="n1"), SimpleNamespace(id="n2")
        # Owner is active for the first check, deactivated afterwards.
        answers = iter([_active("user-1"), _inactive("user-1"), _inactive("user-1")])
        services = _services()
        services.users = SimpleNamespace(get=lambda user_id: next(answers))
        runner = self._runner_with_services(services)
        executed = []
        runner.run_node = lambda invocation, queue_item: executed.append(invocation.id)  # type: ignore[method-assign]
        queue_item = self._multi_node_queue_item([node1, node2])

        runner._run_session_loop(queue_item)

        assert executed == ["n1"]
        services.session_queue.cancel_queue_item.assert_called_once_with(21)

    def test_active_owner_runs_all_nodes(self) -> None:
        node1, node2 = SimpleNamespace(id="n1"), SimpleNamespace(id="n2")
        services = _services(users_by_id={"user-1": _active("user-1")})
        runner = self._runner_with_services(services)
        executed = []
        runner.run_node = lambda invocation, queue_item: executed.append(invocation.id)  # type: ignore[method-assign]
        queue_item = self._multi_node_queue_item([node1, node2])

        runner._run_session_loop(queue_item)

        assert executed == ["n1", "n2"]
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_single_user_mode_runs_all_nodes(self) -> None:
        node1, node2 = SimpleNamespace(id="n1"), SimpleNamespace(id="n2")
        services = _services(multiuser=False)
        runner = self._runner_with_services(services)
        executed = []
        runner.run_node = lambda invocation, queue_item: executed.append(invocation.id)  # type: ignore[method-assign]
        queue_item = self._multi_node_queue_item([node1, node2], user_id="system")

        runner._run_session_loop(queue_item)

        assert executed == ["n1", "n2"]
        services.session_queue.cancel_queue_item.assert_not_called()
