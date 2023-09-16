from threading import BoundedSemaphore, Event as ThreadEvent, Thread
from typing import Optional

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event as FastAPIEvent

from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem

from ..invoker import Invoker
from .session_processor_base import SessionProcessorABC
from .session_processor_common import FINISHED_SESSION_EVENTS, POLLING_INTERVAL, THREAD_LIMIT


class DefaultSessionProcessor(SessionProcessorABC):
    def start_service(self, invoker: Invoker) -> None:
        self.__invoker: Invoker = invoker
        self.__queue_item: Optional[SessionQueueItem] = None

        self.__stop_event = ThreadEvent()
        # when a session is finished, we need to poll the queue immediately.
        # because we need to wait for current item to finish and also wait
        # if the queue is empty, need two events.
        self.__poll_now_busy_event = ThreadEvent()
        self.__poll_now_queue_event = ThreadEvent()

        self.__threadLimit = BoundedSemaphore(THREAD_LIMIT)
        local_handler.register(event_name=EventServiceBase.session_event, _func=self._on_event)
        self._start_thread()

    def stop_service(self, *args, **kwargs) -> None:
        self.__stop_event.set()

    def _poll_now(self) -> None:
        self.__poll_now_busy_event.set()
        self.__poll_now_queue_event.set()

    def _start_thread(self) -> None:
        # threads only live once, so we need to create a new one whenever we start the processor
        self.__thread = Thread(
            name="session_processor",
            target=self.__process,
            kwargs=dict(
                stop_event=self.__stop_event,
                poll_now_busy_event=self.__poll_now_busy_event,
                poll_now_queue_event=self.__poll_now_queue_event,
            ),
        )
        self.__thread.start()

    async def _on_event(self, event: FastAPIEvent) -> None:
        event_name = event[1]["event"]
        if event_name in FINISHED_SESSION_EVENTS:
            self.__queue_item = None
            self._poll_now()

    def stop(self) -> None:
        self.__stop_event.set()

    def start(self) -> None:
        if self.__thread.is_alive():
            return
        self.__stop_event.clear()
        self._start_thread()

    def poll_now(self) -> None:
        self._poll_now()

    def get_current(self) -> Optional[SessionQueueItem]:
        return self.__queue_item

    def clear_current(self) -> None:
        self.__queue_item = None

    def __process(
        self,
        stop_event: ThreadEvent,
        poll_now_busy_event: ThreadEvent,
        poll_now_queue_event: ThreadEvent,
    ):
        try:
            self.__threadLimit.acquire()

            while not stop_event.is_set():
                poll_now_busy_event.clear()
                poll_now_queue_event.clear()

                # do not dequeue if there is already a session running
                if self.__queue_item is not None:
                    poll_now_busy_event.wait(POLLING_INTERVAL)
                    continue

                # get next queue item
                self.__queue_item = self.__invoker.services.session_queue.dequeue()

                if self.__queue_item is None:
                    poll_now_queue_event.wait(POLLING_INTERVAL)
                    continue

                self.__invoker.services.graph_execution_manager.set(self.__queue_item.session)
                self.__invoker.invoke(self.__queue_item.session, invoke_all=True)
        except Exception:
            pass
        finally:
            self.__queue_item = None
            self.__threadLimit.release()
