import traceback
from contextlib import suppress
from threading import BoundedSemaphore, Thread
from threading import Event as ThreadEvent
from typing import Optional

from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event as FastAPIEvent

from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invocation_stats.invocation_stats_common import GESStatsNotFoundError
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context
from invokeai.app.util.profiler import Profiler

from ..invoker import Invoker
from .session_processor_base import SessionProcessorBase
from .session_processor_common import SessionProcessorStatus

POLLING_INTERVAL = 1
THREAD_LIMIT = 1


class DefaultSessionProcessor(SessionProcessorBase):
    def start(self, invoker: Invoker) -> None:
        self._invoker: Invoker = invoker
        self._queue_item: Optional[SessionQueueItem] = None

        self._resume_event = ThreadEvent()
        self._stop_event = ThreadEvent()
        self._poll_now_event = ThreadEvent()
        self._cancel_event = ThreadEvent()

        local_handler.register(event_name=EventServiceBase.queue_event, _func=self._on_queue_event)

        self._thread_limit = BoundedSemaphore(THREAD_LIMIT)
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
            self._thread_limit.acquire()
            stop_event.clear()
            resume_event.set()
            cancel_event.clear()

            # If profiling is enabled, create a profiler. The same profiler will be used for all sessions. Internally,
            # the profiler will create a new profile for each session.
            profiler = (
                Profiler(
                    logger=self._invoker.services.logger,
                    output_dir=self._invoker.services.configuration.profiles_path,
                    prefix=self._invoker.services.configuration.profile_prefix,
                )
                if self._invoker.services.configuration.profile_graphs
                else None
            )

            # Helper function to stop the profiler and save the stats
            def stats_cleanup(graph_execution_state_id: str) -> None:
                if profiler:
                    profile_path = profiler.stop()
                    stats_path = profile_path.with_suffix(".json")
                    self._invoker.services.performance_statistics.dump_stats(
                        graph_execution_state_id=graph_execution_state_id, output_path=stats_path
                    )
                # We'll get a GESStatsNotFoundError if we try to log stats for an untracked graph, but in the processor
                # we don't care about that - suppress the error.
                with suppress(GESStatsNotFoundError):
                    self._invoker.services.performance_statistics.log_stats(graph_execution_state_id)
                    self._invoker.services.performance_statistics.reset_stats()

            while not stop_event.is_set():
                poll_now_event.clear()
                # Middle processor try block; any unhandled exception is a non-fatal processor error
                try:
                    # Get the next session to process
                    self._queue_item = self._invoker.services.session_queue.dequeue()
                    if self._queue_item is not None and resume_event.is_set():
                        self._invoker.services.logger.debug(f"Executing queue item {self._queue_item.item_id}")
                        cancel_event.clear()

                        # If profiling is enabled, start the profiler
                        if profiler is not None:
                            profiler.start(profile_id=self._queue_item.session_id)

                        # Prepare invocations and take the first
                        invocation = self._queue_item.session.next()

                        # Loop over invocations until the session is complete or canceled
                        while invocation is not None and not cancel_event.is_set():
                            # get the source node id to provide to clients (the prepared node id is not as useful)
                            source_invocation_id = self._queue_item.session.prepared_source_mapping[invocation.id]

                            # Send starting event
                            self._invoker.services.events.emit_invocation_started(
                                queue_batch_id=self._queue_item.batch_id,
                                queue_item_id=self._queue_item.item_id,
                                queue_id=self._queue_item.queue_id,
                                graph_execution_state_id=self._queue_item.session_id,
                                node=invocation.model_dump(),
                                source_node_id=source_invocation_id,
                            )

                            # Innermost processor try block; any unhandled exception is an invocation error & will fail the graph
                            try:
                                with self._invoker.services.performance_statistics.collect_stats(
                                    invocation, self._queue_item.session.id
                                ):
                                    # Build invocation context (the node-facing API)
                                    context_data = InvocationContextData(
                                        invocation=invocation,
                                        source_invocation_id=source_invocation_id,
                                        queue_item=self._queue_item,
                                    )
                                    context = build_invocation_context(
                                        context_data=context_data,
                                        services=self._invoker.services,
                                        cancel_event=self._cancel_event,
                                    )

                                    # Invoke the node
                                    outputs = invocation.invoke_internal(
                                        context=context, services=self._invoker.services
                                    )

                                    # Save outputs and history
                                    self._queue_item.session.complete(invocation.id, outputs)

                                    # Send complete event
                                    self._invoker.services.events.emit_invocation_complete(
                                        queue_batch_id=self._queue_item.batch_id,
                                        queue_item_id=self._queue_item.item_id,
                                        queue_id=self._queue_item.queue_id,
                                        graph_execution_state_id=self._queue_item.session.id,
                                        node=invocation.model_dump(),
                                        source_node_id=source_invocation_id,
                                        result=outputs.model_dump(),
                                    )

                            except KeyboardInterrupt:
                                # TODO(psyche): should we set the cancel event here and/or cancel the queue item?
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
                                self._queue_item.session.set_node_error(invocation.id, error)
                                self._invoker.services.logger.error(
                                    f"Error while invoking session {self._queue_item.session_id}, invocation {invocation.id} ({invocation.get_type()}):\n{e}"
                                )

                                # Send error event
                                self._invoker.services.events.emit_invocation_error(
                                    queue_batch_id=self._queue_item.session_id,
                                    queue_item_id=self._queue_item.item_id,
                                    queue_id=self._queue_item.queue_id,
                                    graph_execution_state_id=self._queue_item.session.id,
                                    node=invocation.model_dump(),
                                    source_node_id=source_invocation_id,
                                    error_type=e.__class__.__name__,
                                    error=error,
                                )
                                pass

                            if self._queue_item.session.is_complete() or cancel_event.is_set():
                                # Send complete event
                                self._invoker.services.events.emit_graph_execution_complete(
                                    queue_batch_id=self._queue_item.batch_id,
                                    queue_item_id=self._queue_item.item_id,
                                    queue_id=self._queue_item.queue_id,
                                    graph_execution_state_id=self._queue_item.session.id,
                                )
                                # Save the stats and stop the profiler if it's running
                                stats_cleanup(self._queue_item.session.id)
                                invocation = None
                            else:
                                # Prepare the next invocation
                                invocation = self._queue_item.session.next()

                        # The session is complete, immediately poll for next session
                        self._queue_item = None
                        poll_now_event.set()
                    else:
                        # The queue was empty, wait for next polling interval or event to try again
                        self._invoker.services.logger.debug("Waiting for next polling interval or event")
                        poll_now_event.wait(POLLING_INTERVAL)
                        continue
                except Exception as e:
                    # Non-fatal error in processor, cancel the queue item and wait for next polling interval or event
                    self._invoker.services.logger.error(f"Error in session processor: {e}")
                    if self._queue_item is not None:
                        self._invoker.services.session_queue.cancel_queue_item(
                            self._queue_item.item_id, error=traceback.format_exc()
                        )
                    poll_now_event.wait(POLLING_INTERVAL)
                    continue
        except Exception as e:
            # Fatal error in processor, log and pass - we're done here
            self._invoker.services.logger.error(f"Fatal Error in session processor: {e}")
            pass
        finally:
            stop_event.clear()
            poll_now_event.clear()
            self._queue_item = None
            self._thread_limit.release()
