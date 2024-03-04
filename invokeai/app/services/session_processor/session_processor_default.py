import traceback
from threading import BoundedSemaphore, Thread, Event as ThreadEvent
from typing import Optional

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event as FastAPIEvent

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.shared.graph_processor import GraphProcessor
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.util.profiler import Profiler

from ..invoker import Invoker
from .session_processor_base import SessionProcessorBase
from .session_processor_common import SessionProcessorStatus


class DefaultSessionProcessor(SessionProcessorBase):
    def start(self, invoker: Invoker, thread_limit: int = 1, polling_interval: int = 1) -> None:
        self._invoker: Invoker = invoker
        self._queue_item: Optional[SessionQueueItem] = None
        self._invocation: Optional[BaseInvocation] = None

        self._resume_event = ThreadEvent()
        self._stop_event = ThreadEvent()
        self._poll_now_event = ThreadEvent()
        self._cancel_event = ThreadEvent()

        local_handler.register(event_name=EventServiceBase.queue_event, _func=self._on_queue_event)

        self._thread_limit = thread_limit
        self._thread_semaphore = BoundedSemaphore(thread_limit)
        self._polling_interval = polling_interval

        # If profiling is enabled, create a profiler. The same profiler will be used for all sessions. Internally,
        # the profiler will create a new profile for each session.
        self._profiler = (
            Profiler(
                logger=self._invoker.services.logger,
                output_dir=self._invoker.services.configuration.profiles_path,
                prefix=self._invoker.services.configuration.profile_prefix,
            )
            if self._invoker.services.configuration.profile_graphs
            else None
        )

        self.graph_processor = GraphProcessor(
            services=self._invoker.services,
            cancel_event=self._cancel_event,
            profiler=self._profiler,
        )

        self._thread = Thread(
            name="session_processor",
            target=self._process,
            kwargs={
                "stop_event": self._stop_event,
                "poll_now_event": self._poll_now_event,
                "resume_event": self._resume_event,
                "cancel_event": self._cancel_event,
            },
        )
        self._thread.start()

    def stop(self, *args, **kwargs) -> None:
        self._stop_event.set()

    def _poll_now(self) -> None:
        self._poll_now_event.set()

    async def _on_queue_event(self, event: FastAPIEvent) -> None:
        event_name = event[1]["event"]

        if event_name == "session_canceled" or event_name == "queue_cleared":
            # These both mean we should cancel the current session.
            self._cancel_event.set()
            self._poll_now()
        elif event_name == "batch_enqueued":
            self._poll_now()

    def resume(self) -> SessionProcessorStatus:
        if not self._resume_event.is_set():
            self._resume_event.set()
        return self.get_status()

    def pause(self) -> SessionProcessorStatus:
        if self._resume_event.is_set():
            self._resume_event.clear()
        return self.get_status()

    def get_status(self) -> SessionProcessorStatus:
        return SessionProcessorStatus(
            is_started=self._resume_event.is_set(),
            is_processing=self._queue_item is not None,
        )

    def _process(
        self,
        stop_event: ThreadEvent,
        poll_now_event: ThreadEvent,
        resume_event: ThreadEvent,
        cancel_event: ThreadEvent,
    ):
        # Outermost processor try block; any unhandled exception is a fatal processor error
        try:
            self._thread_semaphore.acquire()
            stop_event.clear()
            resume_event.set()
            cancel_event.clear()

            while not stop_event.is_set():
                poll_now_event.clear()
                # Middle processor try block; any unhandled exception is a non-fatal processor error
                try:
                    # Get the next session to process
                    self._queue_item = self._invoker.services.session_queue.dequeue()
                    if self._queue_item is not None and resume_event.is_set():
                        self._invoker.services.logger.debug(f"Executing queue item {self._queue_item.item_id}")
                        cancel_event.clear()

                        # Run the graph
                        self.graph_processor.run(queue_item=self._queue_item)

                        # The session is complete, immediately poll for next session
                        self._queue_item = None
                        poll_now_event.set()
                    else:
                        # The queue was empty, wait for next polling interval or event to try again
                        self._invoker.services.logger.debug("Waiting for next polling interval or event")
                        poll_now_event.wait(self._polling_interval)
                        continue
                except Exception:
                    # Non-fatal error in processor
                    self._invoker.services.logger.error(
                        f"Non-fatal error in session processor:\n{traceback.format_exc()}"
                    )
                    # Cancel the queue item
                    if self._queue_item is not None:
                        self._invoker.services.session_queue.cancel_queue_item(
                            self._queue_item.item_id, error=traceback.format_exc()
                        )
                    # Reset the invocation to None to prepare for the next session
                    self._invocation = None
                    # Immediately poll for next queue item
                    poll_now_event.wait(self._polling_interval)
                    continue
        except Exception:
            # Fatal error in processor, log and pass - we're done here
            self._invoker.services.logger.error(f"Fatal Error in session processor:\n{traceback.format_exc()}")
            pass
        finally:
            stop_event.clear()
            poll_now_event.clear()
            self._queue_item = None
            self._thread_semaphore.release()
