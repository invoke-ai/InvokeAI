import traceback
from threading import BoundedSemaphore, Thread
from threading import Event as ThreadEvent
from typing import Optional

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event as FastAPIEvent

from invokeai.app.services.events.events_base import EventServiceBase
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

        self.__resume_event = ThreadEvent()
        self.__stop_event = ThreadEvent()
        self.__poll_now_event = ThreadEvent()

        local_handler.register(event_name=EventServiceBase.queue_event, _func=self._on_queue_event)

        self.__threadLimit = BoundedSemaphore(THREAD_LIMIT)
        self.__thread = Thread(
            name="session_processor",
            target=self.__process,
            kwargs={
                "stop_event": self.__stop_event,
                "poll_now_event": self.__poll_now_event,
                "resume_event": self.__resume_event,
            },
        )
        self.__thread.start()

    def stop(self, *args, **kwargs) -> None:
        self.__stop_event.set()

    def _poll_now(self) -> None:
        self.__poll_now_event.set()

    async def _on_queue_event(self, event: FastAPIEvent) -> None:
        event_name = event[1]["event"]

        # This was a match statement, but match is not supported on python 3.9
        if event_name in [
            "graph_execution_state_complete",
            "invocation_error",
            "session_retrieval_error",
            "invocation_retrieval_error",
        ]:
            self.__queue_item = None
            self._poll_now()
        elif (
            event_name == "session_canceled"
            and self.__queue_item is not None
            and self.__queue_item.session_id == event[1]["data"]["graph_execution_state_id"]
        ):
            self.__queue_item = None
            self._poll_now()
        elif event_name == "batch_enqueued":
            self._poll_now()
        elif event_name == "queue_cleared":
            self.__queue_item = None
            self._poll_now()

    def resume(self) -> SessionProcessorStatus:
        if not self.__resume_event.is_set():
            self.__resume_event.set()
        return self.get_status()

    def pause(self) -> SessionProcessorStatus:
        if self.__resume_event.is_set():
            self.__resume_event.clear()
        return self.get_status()

    def get_status(self) -> SessionProcessorStatus:
        return SessionProcessorStatus(
            is_started=self.__resume_event.is_set(),
            is_processing=self.__queue_item is not None,
        )

    def __process(
        self,
        stop_event: ThreadEvent,
        poll_now_event: ThreadEvent,
        resume_event: ThreadEvent,
    ):
        try:
            stop_event.clear()
            resume_event.set()
            self.__threadLimit.acquire()
            queue_item: Optional[SessionQueueItem] = None
            while not stop_event.is_set():
                poll_now_event.clear()
                try:
                    # do not dequeue if there is already a session running
                    if self.__queue_item is None and resume_event.is_set():
                        queue_item = self.__invoker.services.session_queue.dequeue()

                        if queue_item is not None:
                            self.__invoker.services.logger.debug(f"Executing queue item {queue_item.item_id}")
                            self.__queue_item = queue_item
                            self.__invoker.services.graph_execution_manager.set(queue_item.session)
                            self.__invoker.invoke(
                                session_queue_batch_id=queue_item.batch_id,
                                session_queue_id=queue_item.queue_id,
                                session_queue_item_id=queue_item.item_id,
                                graph_execution_state=queue_item.session,
                                workflow=queue_item.workflow,
                                invoke_all=True,
                            )
                            queue_item = None

                    if queue_item is None:
                        self.__invoker.services.logger.debug("Waiting for next polling interval or event")
                        poll_now_event.wait(POLLING_INTERVAL)
                        continue
                except Exception as e:
                    self.__invoker.services.logger.error(f"Error in session processor: {e}")
                    if queue_item is not None:
                        self.__invoker.services.session_queue.cancel_queue_item(
                            queue_item.item_id, error=traceback.format_exc()
                        )
                    poll_now_event.wait(POLLING_INTERVAL)
                    continue
        except Exception as e:
            self.__invoker.services.logger.error(f"Fatal Error in session processor: {e}")
            pass
        finally:
            stop_event.clear()
            poll_now_event.clear()
            self.__queue_item = None
            self.__threadLimit.release()
