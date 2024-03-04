import traceback
from contextlib import suppress
from threading import BoundedSemaphore, Thread
from threading import Event as ThreadEvent
from typing import Callable, Optional, Union

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event as FastAPIEvent

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_stats.invocation_stats_common import GESStatsNotFoundError
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context
from invokeai.app.util.profiler import Profiler

from ..invoker import Invoker
from .session_processor_base import SessionProcessorBase
from .session_processor_common import SessionProcessorStatus


class SessionRunner:
    """Processes a single session's invocations"""

    def __init__(
        self,
        services: InvocationServices,
        cancel_event: ThreadEvent,
        profiler: Union[Profiler, None] = None,
        on_before_run_node: Union[Callable[[BaseInvocation, SessionQueueItem], bool], None] = None,
        on_after_run_node: Union[Callable[[BaseInvocation, SessionQueueItem], bool], None] = None,
    ):
        self.services = services
        self.profiler = profiler
        self.cancel_event = cancel_event
        self.on_before_run_node = on_before_run_node
        self.on_after_run_node = on_after_run_node

    def run(self, queue_item: SessionQueueItem):
        """Run the graph"""
        if not queue_item.session:
            raise ValueError("Queue item has no session")
        # If profiling is enabled, start the profiler
        if self.profiler is not None:
            self.profiler.start(profile_id=queue_item.session_id)
        # Loop over invocations until the session is complete or canceled
        while not (queue_item.session.is_complete() or self.cancel_event.is_set()):
            # Prepare the next node
            invocation = queue_item.session.next()
            if invocation is None:
                # If there are no more invocations, complete the graph
                break
            # Build invocation context (the node-facing API
            self.run_node(invocation, queue_item)
        self.complete(queue_item)

    def complete(self, queue_item: SessionQueueItem):
        """Complete the graph"""
        self.services.events.emit_graph_execution_complete(
            queue_batch_id=queue_item.batch_id,
            queue_item_id=queue_item.item_id,
            queue_id=queue_item.queue_id,
            graph_execution_state_id=queue_item.session.id,
        )
        # If we are profiling, stop the profiler and dump the profile & stats
        if self.profiler:
            profile_path = self.profiler.stop()
            stats_path = profile_path.with_suffix(".json")
            self.services.performance_statistics.dump_stats(
                graph_execution_state_id=queue_item.session.id, output_path=stats_path
            )
        # We'll get a GESStatsNotFoundError if we try to log stats for an untracked graph, but in the processor
        # we don't care about that - suppress the error.
        with suppress(GESStatsNotFoundError):
            self.services.performance_statistics.log_stats(queue_item.session.id)
            self.services.performance_statistics.reset_stats()

    def run_node(self, invocation: BaseInvocation, queue_item: SessionQueueItem):
        """Run a single node in the graph"""
        # If we have a on_before_run_node callback, call it
        if self.on_before_run_node is not None:
            self.on_before_run_node(invocation, queue_item)
        try:
            data = InvocationContextData(
                invocation=invocation,
                source_invocation_id=queue_item.session.prepared_source_mapping[invocation.id],
                queue_item=queue_item,
            )

            # Send starting event
            self.services.events.emit_invocation_started(
                queue_batch_id=queue_item.batch_id,
                queue_item_id=queue_item.item_id,
                queue_id=queue_item.queue_id,
                graph_execution_state_id=queue_item.session_id,
                node=invocation.model_dump(),
                source_node_id=data.source_invocation_id,
            )

            # Innermost processor try block; any unhandled exception is an invocation error & will fail the graph
            with self.services.performance_statistics.collect_stats(invocation, queue_item.session_id):
                context = build_invocation_context(
                    data=data,
                    services=self.services,
                    cancel_event=self.cancel_event,
                )

                # Invoke the node
                outputs = invocation.invoke_internal(context=context, services=self.services)

                # Save outputs and history
                queue_item.session.complete(invocation.id, outputs)

                # Send complete event
            self.services.events.emit_invocation_complete(
                queue_batch_id=queue_item.batch_id,
                queue_item_id=queue_item.item_id,
                queue_id=queue_item.queue_id,
                graph_execution_state_id=queue_item.session.id,
                node=invocation.model_dump(),
                source_node_id=data.source_invocation_id,
                result=outputs.model_dump(),
            )
        except KeyboardInterrupt:
            # TODO(MM2): Create an event for this
            pass
        except CanceledException:
            # When the user cancels the graph, we first set the cancel event. The event is checked
            # between invocations, in this loop. Some invocations are long-running, and we need to
            # be able to cancel them mid-execution.
            #
            # For example, denoising is a long-running invocation with many steps. A step callback
            # is executed after each step. This step callback checks if the canceled event is set,
            # then raises a CanceledException to stop execution immediately.
            #
            # When we get a CanceledException, we don't need to do anything - just pass and let the
            # loop go to its next iteration, and the cancel event will be handled correctly.
            pass
        except Exception as e:
            error = traceback.format_exc()

            # Save error
            queue_item.session.set_node_error(invocation.id, error)
            self.services.logger.error(
                f"Error while invoking session {queue_item.session_id}, invocation {invocation.id} ({invocation.get_type()}):\n{e}"
            )
            self.services.logger.error(error)

            # Send error event
            self.services.events.emit_invocation_error(
                queue_batch_id=queue_item.session_id,
                queue_item_id=queue_item.item_id,
                queue_id=queue_item.queue_id,
                graph_execution_state_id=queue_item.session.id,
                node=invocation.model_dump(),
                source_node_id=queue_item.session.prepared_source_mapping[invocation.id],
                error_type=e.__class__.__name__,
                error=error,
            )
            pass
        finally:
            # If we have a on_after_run_node callback, call it
            if self.on_after_run_node is not None:
                self.on_after_run_node(invocation, queue_item)


class DefaultSessionProcessor(SessionProcessorBase):
    """Processes sessions from the session queue"""

    def start(
        self,
        invoker: Invoker,
        thread_limit: int = 1,
        polling_interval: int = 1,
        on_before_run_node: Union[Callable[[BaseInvocation, SessionQueueItem], bool], None] = None,
        on_after_run_node: Union[Callable[[BaseInvocation, SessionQueueItem], bool], None] = None,
        on_before_run_session: Union[Callable[[SessionQueueItem], bool], None] = None,
        on_after_run_session: Union[Callable[[SessionQueueItem], bool], None] = None,
    ) -> None:
        self._invoker: Invoker = invoker
        self._queue_item: Optional[SessionQueueItem] = None
        self._invocation: Optional[BaseInvocation] = None
        self.on_before_run_session = on_before_run_session
        self.on_after_run_session = on_after_run_session

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

        self.session_runner = SessionRunner(
            services=self._invoker.services,
            cancel_event=self._cancel_event,
            profiler=self._profiler,
            on_before_run_node=on_before_run_node,
            on_after_run_node=on_after_run_node,
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
                        # If we have a on_before_run_session callback, call it
                        if self.on_before_run_session is not None:
                            self.on_before_run_session(self._queue_item)

                        self._invoker.services.logger.debug(f"Executing queue item {self._queue_item.item_id}")
                        cancel_event.clear()

                        # Run the graph
                        self.session_runner.run(queue_item=self._queue_item)

                        # If we have a on_after_run_session callback, call it
                        if self.on_after_run_session is not None:
                            self.on_after_run_session(self._queue_item)

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
            self._invoker.services.logger.debug("Session processor stopped")
