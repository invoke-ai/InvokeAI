"""Tests that queued execution is revoked when the owning account is deactivated
or deleted.

Policy (see queue_owner_is_active):
- Pending items are rejected (canceled) at dequeue, before any invocation runs.
- Running items are stopped at the next node boundary; canceling also sets the
  processor's cancel event, which stops step-callback nodes mid-node.
- Single-user mode is exempt. The ``system`` user is not special-cased: it has a real,
  active database row and passes on its own merits.
"""

from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.session_processor import session_processor_default
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

    def test_system_user_passes_on_its_own_row(self) -> None:
        """migration_27 creates an active `system` row, so no exemption is needed."""
        services = _services(users_by_id={"system": _active("system")})
        assert queue_owner_is_active(services, _queue_item(user_id="system")) is True

    def test_system_user_without_a_row_is_rejected(self) -> None:
        """Agrees with the `invocation_context` save gates, which have no exemption either.

        Exempting `system` here would let the item consume GPU time and then fail at its
        first save; rejecting it at dequeue is the coherent outcome.
        """
        services = _services(users_by_id={})
        assert queue_owner_is_active(services, _queue_item(user_id="system")) is False

    def test_active_user(self) -> None:
        services = _services(users_by_id={"user-1": _active("user-1")})
        assert queue_owner_is_active(services, _queue_item()) is True

    def test_deactivated_user(self) -> None:
        services = _services(users_by_id={"user-1": _inactive("user-1")})
        assert queue_owner_is_active(services, _queue_item()) is False

    def test_deleted_user(self) -> None:
        services = _services(users_by_id={})
        assert queue_owner_is_active(services, _queue_item()) is False

    def test_an_unreadable_owner_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failing open here would make unknown database state executable.

        The account may have been deactivated a moment ago; nothing else stands between
        that and GPU time spent on its behalf. Failing closed costs a still-valid user a
        cancellation, which is retryable.
        """
        monkeypatch.setattr(session_processor_default, "OWNER_LOOKUP_RETRY_SECONDS", 0)
        services = _services()
        attempts = []

        def explode(user_id: str) -> None:
            attempts.append(user_id)
            raise RuntimeError("database is locked")

        services.users.get = explode

        assert queue_owner_is_active(services, _queue_item()) is False
        assert len(attempts) == session_processor_default.OWNER_LOOKUP_ATTEMPTS

    def test_a_transient_read_failure_is_retried_not_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A busy-timeout under write contention must not cost the user their queued work."""
        monkeypatch.setattr(session_processor_default, "OWNER_LOOKUP_RETRY_SECONDS", 0)
        services = _services()
        answers = iter([RuntimeError("database is locked"), _active("user-1")])

        def flaky(user_id: str):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        services.users.get = flaky

        assert queue_owner_is_active(services, _queue_item()) is True


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
        services = _services(users_by_id={"system": _active("system")})
        processor = self._processor(services)

        assert processor._cancel_queue_item_if_owner_inactive(_queue_item(user_id="system")) is False
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_missing_queue_item_does_not_raise(self) -> None:
        """The item may be deleted concurrently; rejection still stands."""
        services = _services(users_by_id={})
        services.session_queue.cancel_queue_item.side_effect = SessionQueueItemNotFoundError("gone")
        processor = self._processor(services)

        assert processor._cancel_queue_item_if_owner_inactive(_queue_item()) is True

    def test_unreadable_owner_item_is_canceled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An owner the database cannot answer for does not get to run."""
        monkeypatch.setattr(session_processor_default, "OWNER_LOOKUP_RETRY_SECONDS", 0)
        services = _services()

        def explode(user_id: str) -> None:
            raise RuntimeError("database is locked")

        services.users.get = explode
        processor = self._processor(services)

        assert processor._cancel_queue_item_if_owner_inactive(_queue_item()) is True
        services.session_queue.cancel_queue_item.assert_called_once_with(7)


class TestUserAccessChangedCancelsCurrentItem:
    """Deactivating a user cancels their currently running queue item immediately."""

    def _processor(self, services: SimpleNamespace, *queue_items: SimpleNamespace | None) -> DefaultSessionProcessor:
        processor = DefaultSessionProcessor.__new__(DefaultSessionProcessor)
        processor._invoker = SimpleNamespace(services=services)
        # Each worker runs at most one item; an idle worker has `queue_item is None`.
        processor._workers = [SimpleNamespace(queue_item=item) for item in queue_items]
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

    @pytest.mark.anyio
    async def test_deactivation_cancels_items_on_every_worker(self) -> None:
        """One user may occupy several workers at once; all of their items must stop."""
        services = _services()
        processor = self._processor(
            services,
            _queue_item(user_id="user-1", item_id=11),
            _queue_item(user_id="user-2", item_id=12),
            None,
            _queue_item(user_id="user-1", item_id=13),
        )

        await processor._on_user_access_changed(self._event("user-1", is_active=False))

        assert sorted(c.args[0] for c in services.session_queue.cancel_queue_item.call_args_list) == [11, 13]

    @pytest.mark.anyio
    async def test_reactivation_before_the_cancel_lands_spares_the_item(self) -> None:
        """The owner is re-read at the point of decision rather than trusted from the event.

        Each event is dispatched as its own task, so a deactivate immediately followed by a
        reactivate can leave the first handler still parked while the second has come and
        gone (it returns early). Cancelling on the stale snapshot would kill a running item
        of an account the database says is active, and nothing undoes a cancellation.
        """
        services = _services(users_by_id={"user-1": _active("user-1")})
        processor = self._processor(services, _queue_item(user_id="user-1", item_id=11))

        await processor._on_user_access_changed(self._event("user-1", is_active=False))

        services.session_queue.cancel_queue_item.assert_not_called()

    @pytest.mark.anyio
    async def test_single_user_mode_does_not_cancel(self) -> None:
        """Ownership is not enforced anywhere else in single-user mode either."""
        services = _services(multiuser=False)
        processor = self._processor(services, _queue_item(user_id="user-1", item_id=11))

        await processor._on_user_access_changed(self._event("user-1", is_active=False))

        services.session_queue.cancel_queue_item.assert_not_called()

    @pytest.mark.anyio
    async def test_a_failed_re_read_still_cancels(self) -> None:
        """A read that cannot contradict the event must not override it.

        The gates fail closed on an unreadable database too, but this handler has the
        stronger claim: it is the only thing that stops a single-node graph, which is
        checked once before it starts and never again."""
        services = _services()

        def explode(user_id: str) -> None:
            raise RuntimeError("database is locked")

        services.users.get = explode
        processor = self._processor(services, _queue_item(user_id="user-1", item_id=11))

        await processor._on_user_access_changed(self._event("user-1", is_active=False))

        services.session_queue.cancel_queue_item.assert_called_once_with(11)


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

    def test_an_unreadable_owner_stops_later_nodes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deactivation the database can no longer be asked about still stops the run."""
        monkeypatch.setattr(session_processor_default, "OWNER_LOOKUP_RETRY_SECONDS", 0)
        node1, node2 = SimpleNamespace(id="n1"), SimpleNamespace(id="n2")
        # Active for the first check; afterwards the record cannot be read at all.
        answered = []

        def flaky(user_id: str):
            answered.append(user_id)
            if len(answered) == 1:
                return _active("user-1")
            raise RuntimeError("database is locked")

        services = _services()
        services.users = SimpleNamespace(get=flaky)
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
