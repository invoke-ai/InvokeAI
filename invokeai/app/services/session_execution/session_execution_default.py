from typing import Optional

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event

from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_execution.session_execution_base import SessionExecutionServiceBase
from invokeai.app.services.session_execution.session_execution_common import SessionExecutionStatusResult
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem, SessionQueueItemNotFoundError


class DefaultSessionExecutionService(SessionExecutionServiceBase):
    def __init__(self) -> None:
        self._invoker: Invoker
        self._current: Optional[SessionQueueItem] = None
        self._started: bool = False
        self._stop_after_current = False

    def start_service(self, invoker: Invoker) -> None:
        self._invoker = invoker
        local_handler.register(event_name=EventServiceBase.session_event, _func=self._on_event)

    async def _on_event(self, event: Event) -> Event:
        event_name = event[1]["event"]
        match event_name:
            case "graph_execution_state_complete":
                await self._handle_complete_event(event, False)
            case "invocation_error":
                await self._handle_complete_event(event, True)
            case "session_retrieval_error":
                await self._handle_complete_event(event, True)
            case "invocation_retrieval_error":
                await self._handle_complete_event(event, True)
        return event

    async def _handle_complete_event(self, event: Event, err: bool) -> None:
        data = event[1]["data"]
        try:
            queue_item = self._invoker.services.session_queue.get_queue_item_by_session_id(
                data["graph_execution_state_id"]
            )
        except SessionQueueItemNotFoundError:
            # shouldn't happen - we should have a queue item for every session
            queue_item = None
        # Sessions are marked complete when they have an error, so we get an `invocation_error`
        # followed by a `graph_execution_state_complete`. Don't mark queue items complete if
        # they are already marked error.
        if queue_item is not None and queue_item.status != "failed":
            queue_item = self._invoker.services.session_queue.set_queue_item_status(
                queue_item.id, "failed" if err else "completed"
            )
            self._invoker.services.events.emit_queue_item_status_changed(queue_item)
        if self._stop_after_current:
            self._stop_after_current = False
            self._started = False
            self._emit_queue_status()
        self._current = None
        if self._started:
            self.invoke_next()

    def _emit_queue_status(self) -> None:
        self._invoker.services.events.emit_queue_status_changed(self._started, self._stop_after_current)

    def _emit_queue_item_status(self) -> None:
        if self._current is None:
            return
        self._invoker.services.events.emit_queue_item_status_changed(self._current)

    def invoke_next(self) -> None:
        # do not invoke if already invoking
        if self._current:
            return

        queue_item = self._invoker.services.session_queue.dequeue()

        if queue_item is None:
            # queue empty
            self._current = None
            self._started = False
            self._stop_after_current = False
            self._emit_queue_status()
            return

        self._current = queue_item
        self._invoker.services.graph_execution_manager.set(queue_item.session)
        self._emit_queue_item_status()
        self._invoker.invoke(self._current.session, invoke_all=True)

    def start(self) -> None:
        if not self._stop_after_current:
            self._started = True
            self._emit_queue_status()
            self.invoke_next()

    def stop(self) -> None:
        self._started = False
        self._stop_after_current = True
        self._emit_queue_status()

    def cancel(self) -> None:
        if self._current is not None:
            self._invoker.services.queue.cancel(self._current.session_id)
            self._current = self._invoker.services.session_queue.set_queue_item_status(self._current.id, "canceled")
            self._emit_queue_item_status()
            self._current = None
        self._started = False
        self._stop_after_current = False
        self._emit_queue_status()

    def get_current(self) -> Optional[SessionQueueItem]:
        return self._current

    def get_status(self) -> SessionExecutionStatusResult:
        return SessionExecutionStatusResult(started=self._started, stop_after_current=self._stop_after_current)
