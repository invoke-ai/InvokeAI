from threading import BoundedSemaphore
from threading import Event as ThreadEvent
from threading import Thread
from typing import Optional

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event as FastAPIEvent

from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem

from ..invoker import Invoker
from .session_processor_base import SessionProcessorBase
from .session_processor_common import SessionProcessorStatus

POLLING_INTERVAL = 1
THREAD_LIMIT = 1


class DefaultSessionProcessor(SessionProcessorBase):
    def start(self, invoker: Invoker) -> None:
        self.__invoker: Invoker = invoker
        self.__queue_item: Optional[SessionQueueItem] = None

        self.__stop_event = ThreadEvent()
        self.__poll_now_event = ThreadEvent()

        local_handler.register(event_name=EventServiceBase.session_event, _func=self._on_session_event)
        local_handler.register(event_name=EventServiceBase.queue_event, _func=self._on_queue_event)

        self.__threadLimit = BoundedSemaphore(THREAD_LIMIT)
        self._start_thread()

    def stop(self, *args, **kwargs) -> None:
        self.__stop_event.set()
        self._emit_status_changed()

    def _poll_now(self) -> None:
        self.__poll_now_event.set()

    def _start_thread(self) -> None:
        # threads only live once, so we need to create a new one whenever we start the session processor
        self.__thread = Thread(
            name="session_processor",
            target=self.__process,
            kwargs=dict(
                stop_event=self.__stop_event,
                poll_now_event=self.__poll_now_event,
            ),
        )
        self.__thread.start()
        self._emit_status_changed()

    async def _on_session_event(self, event: FastAPIEvent) -> None:
        event_name = event[1]["event"]
        if event_name in [
            "graph_execution_state_complete",
            "invocation_error",
            "session_retrieval_error",
            "invocation_retrieval_error",
        ] or (
            event_name == "session_canceled"
            and self.__queue_item is not None
            and self.__queue_item.session_id == event[1]["data"]["graph_execution_state_id"]
        ):
            self.__queue_item = None
            self._poll_now()

    async def _on_queue_event(self, event: FastAPIEvent) -> None:
        event_name = event[1]["event"]
        if event_name == "batch_enqueued":
            self._poll_now()
        if event_name == "queue_cleared":
            self.__queue_item = None
            self._poll_now()

    def _is_started(self) -> bool:
        return self.__thread.is_alive()

    def _is_processing(self) -> bool:
        return self.__queue_item is not None

    def _is_stop_pending(self) -> bool:
        return self.__stop_event.is_set()

    def _emit_status_changed(self) -> None:
        self.__invoker.services.events.emit_processor_status_changed(self.get_status())

    def get_status(self) -> SessionProcessorStatus:
        return SessionProcessorStatus(
            is_started=self._is_started(),
            is_processing=self._is_processing(),
            is_stop_pending=self._is_stop_pending(),
        )

    def resume(self) -> None:
        if self._is_started():
            return
        self.__stop_event.clear()
        self._emit_status_changed()
        self._start_thread()

    def pause(self) -> None:
        self.__stop_event.set()
        self._emit_status_changed()

    def __process(
        self,
        stop_event: ThreadEvent,
        poll_now_event: ThreadEvent,
    ):
        try:
            self.__threadLimit.acquire()
            queue_item: Optional[SessionQueueItem] = None
            self.__invoker.services.logger
            while not stop_event.is_set():
                poll_now_event.clear()

                # do not dequeue if there is already a session running
                if self.__queue_item is None:
                    queue_item = self.__invoker.services.session_queue.dequeue()

                    if queue_item is not None:
                        # TODO: Why isn't the log level specified in dependencies.py working?
                        # Within the thread, it is always INFO and `logger.debug()` doesn't display.
                        # self.__invoker.services.logger.debug(f"Executing queue item {queue_item.item_id}")
                        self.__queue_item = queue_item
                        self.__invoker.services.graph_execution_manager.set(queue_item.session)
                        self.__invoker.invoke(queue_item.session, invoke_all=True)
                        queue_item = None

                if queue_item is None:
                    # self.__invoker.services.logger.debug("Waiting for next polling interval or event")
                    poll_now_event.wait(POLLING_INTERVAL)
                    continue
        except Exception:
            pass
        finally:
            stop_event.clear()
            poll_now_event.clear()
            self.__queue_item = None
            self.__threadLimit.release()
            self._emit_status_changed()
