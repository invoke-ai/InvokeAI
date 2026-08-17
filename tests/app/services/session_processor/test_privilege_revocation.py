"""Tests that queued execution is revoked when the owning account is deactivated
or deleted.

Policy (see queue_owner_is_active):
- Pending items are rejected (canceled) at dequeue, before any invocation runs.
- Running items are stopped at the next node boundary; canceling also sets the
  processor's cancel event, which stops step-callback nodes mid-node.
- Single-user mode is exempt. The ``system`` user is not special-cased: it has a real,
  active database row and passes on its own merits.
"""

import time
from threading import BoundedSemaphore, Thread
from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.session_processor import session_processor_default
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
    _SessionWorker,
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


class TestRunnerStopsAfterTheLastNode:
    """The owner is re-checked once more after the graph's last node.

    Checking only on the way into a node leaves the last node of every graph — the only
    node of a one-node graph — unguarded: an account deleted while it runs would have its
    item recorded as completed. The node itself cannot be un-spent, but the item's status
    can still tell the truth, and `_on_after_run_session` will not complete an item it
    finds canceled.

    Scoped tightly, because after a node the trade is reversed: nothing is saved by
    refusing, and a completed result can be destroyed. So it does not fail closed, and it
    does not run for a session that ended in an error or was already canceled.
    """

    def _runner_with_services(self, services: SimpleNamespace) -> DefaultSessionRunner:
        runner = DefaultSessionRunner()
        runner.start(services=services, cancel_event=ThreadEvent(), profiler=None)
        return runner

    def _one_node_queue_item(
        self,
        node: SimpleNamespace,
        user_id: str = "user-1",
        *,
        has_error: bool = False,
        complete: bool = True,
    ) -> SimpleNamespace:
        """A queue item whose session has exactly one node and is complete once it has run.

        `complete=False` stands in for a session suspended on a workflow call, which is
        deliberately not `is_complete()` while its children run.
        """
        node_iter = iter([node, None])
        state = {"ran": False}

        def next_node():
            invocation = next(node_iter)
            state["ran"] = True
            return invocation

        session = SimpleNamespace(
            next=next_node,
            is_complete=lambda: state["ran"] and complete,
            has_error=lambda: has_error,
        )
        return SimpleNamespace(user_id=user_id, item_id=21, status="in_progress", session=session)

    def _runner_recording(self, services: SimpleNamespace, executed: list) -> DefaultSessionRunner:
        runner = self._runner_with_services(services)
        runner.run_node = lambda invocation, queue_item: executed.append(invocation.id)  # type: ignore[method-assign]
        return runner

    def test_deletion_during_the_only_node_cancels_the_item(self) -> None:
        """The gap this closes: no socket, so no event, and no next node to be stopped at."""
        answers = iter([_active("user-1"), None])
        services = _services()
        services.users = SimpleNamespace(get=lambda user_id: next(answers))
        executed: list[str] = []
        runner = self._runner_recording(services, executed)

        runner._run_session_loop(self._one_node_queue_item(SimpleNamespace(id="n1")))

        assert executed == ["n1"], "the node was already running; only its aftermath can change"
        services.session_queue.cancel_queue_item.assert_called_once_with(21)

    def test_deactivation_during_the_only_node_cancels_the_item(self) -> None:
        answers = iter([_active("user-1"), _inactive("user-1")])
        services = _services()
        services.users = SimpleNamespace(get=lambda user_id: next(answers))
        executed: list[str] = []
        runner = self._runner_recording(services, executed)

        runner._run_session_loop(self._one_node_queue_item(SimpleNamespace(id="n1")))

        services.session_queue.cancel_queue_item.assert_called_once_with(21)

    def test_an_active_owners_one_node_item_is_not_canceled(self) -> None:
        services = _services(users_by_id={"user-1": _active("user-1")})
        executed: list[str] = []
        runner = self._runner_recording(services, executed)

        runner._run_session_loop(self._one_node_queue_item(SimpleNamespace(id="n1")))

        assert executed == ["n1"]
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_single_user_mode_leaves_the_one_node_item_alone(self) -> None:
        services = _services(multiuser=False)
        executed: list[str] = []
        runner = self._runner_recording(services, executed)

        runner._run_session_loop(self._one_node_queue_item(SimpleNamespace(id="n1"), user_id="system"))

        assert executed == ["n1"]
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_an_unreadable_record_does_not_cancel_finished_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The one place this gate must NOT fail closed.

        Before a node, refusing costs a retryable cancellation and may save a GPU. After
        the graph's last node there is nothing left to refuse, so the same refusal would
        destroy a completed generation belonging to a user who is very likely still valid
        — a busy-timeout on the shared connection is the module's own stated example of a
        transient failure. A genuinely revoked owner keeps nothing by the difference: the
        `invocation_context` save gates re-read the record themselves and already refused
        every write.
        """
        monkeypatch.setattr(session_processor_default, "OWNER_LOOKUP_RETRY_SECONDS", 0)
        answered = []

        def flaky(user_id: str):
            answered.append(user_id)
            if len(answered) == 1:
                return _active("user-1")
            raise RuntimeError("database is locked")

        services = _services()
        services.users = SimpleNamespace(get=flaky)
        executed: list[str] = []
        runner = self._runner_recording(services, executed)

        runner._run_session_loop(self._one_node_queue_item(SimpleNamespace(id="n1")))

        assert executed == ["n1"]
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_a_session_that_errored_is_not_re_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed session is `is_complete()` too. Cancelling it would overwrite the error
        the user needs to see and, for a workflow-call child, transition the waiting parent
        to canceled — so `_fail_parent_from_failed_child` would find it terminal and drop
        the error entirely."""
        monkeypatch.setattr(session_processor_default, "OWNER_LOOKUP_RETRY_SECONDS", 0)
        answers = iter([_active("user-1"), _inactive("user-1")])
        services = _services()
        services.users = SimpleNamespace(get=lambda user_id: next(answers))
        executed: list[str] = []
        runner = self._runner_recording(services, executed)

        runner._run_session_loop(self._one_node_queue_item(SimpleNamespace(id="n1"), has_error=True))

        services.session_queue.cancel_queue_item.assert_not_called()

    def test_a_suspended_workflow_call_is_not_re_checked(self) -> None:
        """A parent suspended on its children is not `is_complete()`, so the post-node check
        does not see it. Cancelling there would take down the children it just enqueued."""
        answers = iter([_active("user-1"), _inactive("user-1")])
        services = _services()
        services.users = SimpleNamespace(get=lambda user_id: next(answers))
        executed: list[str] = []
        runner = self._runner_recording(services, executed)

        runner._run_session_loop(self._one_node_queue_item(SimpleNamespace(id="n1"), complete=False))

        assert executed == ["n1"]
        services.session_queue.cancel_queue_item.assert_not_called()

    def test_an_already_canceled_item_is_not_re_checked(self) -> None:
        """Nothing to protect and nothing to correct — the item is already terminal."""
        answers = iter([_active("user-1"), _inactive("user-1")])
        services = _services()
        services.users = SimpleNamespace(get=lambda user_id: next(answers))
        executed: list[str] = []
        runner = self._runner_recording(services, executed)
        queue_item = self._one_node_queue_item(SimpleNamespace(id="n1"))
        queue_item.status = "canceled"

        runner._run_session_loop(queue_item)

        assert executed == ["n1"]
        services.session_queue.cancel_queue_item.assert_not_called()


class TestRunningQueueItemOwners:
    """What the revalidation sweep uses to find work whose owner it must re-check."""

    def _processor(self, *queue_items: SimpleNamespace | None) -> DefaultSessionProcessor:
        processor = DefaultSessionProcessor.__new__(DefaultSessionProcessor)
        processor._workers = [SimpleNamespace(queue_item=item) for item in queue_items]
        return processor

    def test_owners_of_running_items_are_reported(self) -> None:
        processor = self._processor(
            _queue_item(user_id="user-1", item_id=11),
            _queue_item(user_id="user-2", item_id=12),
            None,
            _queue_item(user_id="user-1", item_id=13),
        )

        assert processor.get_running_queue_item_owners() == {"user-1", "user-2"}

    def test_an_idle_processor_reports_nothing(self) -> None:
        assert self._processor(None, None).get_running_queue_item_owners() == set()

    def test_a_processor_that_never_started_reports_nothing(self) -> None:
        """`_workers` is populated in `start()`; the sweep runs on a timer that may fire
        before it."""
        assert DefaultSessionProcessor().get_running_queue_item_owners() == set()

    def test_a_parked_worker_stops_reporting_the_item_it_finished(self) -> None:
        """`worker.queue_item` is only overwritten by the next dequeue, and pausing blocks
        before that. Without an explicit clear, a worker that finished an item and then
        parked would report its owner as running work forever — so the sweep would keep
        publishing revocations, and its "nothing is live" skip could never fire.
        """
        processor = DefaultSessionProcessor.__new__(DefaultSessionProcessor)
        processor._invoker = SimpleNamespace(services=_services())
        processor._thread_semaphore = BoundedSemaphore(1)
        processor._polling_interval = 1
        worker = _SessionWorker(device=None, runner=MagicMock())
        worker.queue_item = _queue_item()  # type: ignore[assignment]
        processor._workers = [worker]

        stop_event, poll_now_event, resume_event = ThreadEvent(), ThreadEvent(), ThreadEvent()
        # `resume_event` is left clear, so the worker parks exactly where a paused
        # processor parks it: at the top of the loop, before any dequeue.
        thread = Thread(target=processor._process, args=(worker, stop_event, poll_now_event, resume_event), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while worker.queue_item is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert processor.get_running_queue_item_owners() == set()
        finally:
            stop_event.set()
            resume_event.set()
            thread.join(timeout=5)
            assert not thread.is_alive()
