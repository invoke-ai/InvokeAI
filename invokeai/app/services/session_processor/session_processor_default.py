import traceback
from contextlib import suppress
from threading import BoundedSemaphore, Thread
from threading import Event as ThreadEvent
from typing import Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent,
    FastAPIEvent,
    QueueClearedEvent,
    QueueEvent,
    QueueItemStatusChangedEvent,
    SessionCanceledEvent,
    register_events,
)
from invokeai.app.services.invocation_stats.invocation_stats_common import GESStatsNotFoundError
from invokeai.app.services.session_processor.session_processor_base import (
    OnAfterRunNode,
    OnAfterRunSession,
    OnBeforeRunNode,
    OnBeforeRunSession,
    OnNodeError,
    OnNonFatalProcessorError,
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem, SessionQueueItemNotFoundError
from invokeai.app.services.shared.graph import NodeInputError
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context
from invokeai.app.util.profiler import Profiler

from ..invoker import Invoker
from .session_processor_base import InvocationServices, SessionProcessorBase, SessionRunnerBase
from .session_processor_common import SessionProcessorStatus


class DefaultSessionRunner(SessionRunnerBase):
    """Processes a single session's invocations."""

    def __init__(
        self,
        on_before_run_session_callbacks: Optional[list[OnBeforeRunSession]] = None,
        on_before_run_node_callbacks: Optional[list[OnBeforeRunNode]] = None,
        on_after_run_node_callbacks: Optional[list[OnAfterRunNode]] = None,
        on_node_error_callbacks: Optional[list[OnNodeError]] = None,
        on_after_run_session_callbacks: Optional[list[OnAfterRunSession]] = None,
    ):
        """
        Args:
            on_before_run_session_callbacks: Callbacks to run before the session starts.
            on_before_run_node_callbacks: Callbacks to run before each node starts.
            on_after_run_node_callbacks: Callbacks to run after each node completes.
            on_node_error_callbacks: Callbacks to run when a node errors.
            on_after_run_session_callbacks: Callbacks to run after the session completes.
        """

        self._on_before_run_session_callbacks = on_before_run_session_callbacks or []
        self._on_before_run_node_callbacks = on_before_run_node_callbacks or []
        self._on_after_run_node_callbacks = on_after_run_node_callbacks or []
        self._on_node_error_callbacks = on_node_error_callbacks or []
        self._on_after_run_session_callbacks = on_after_run_session_callbacks or []

    def start(self, services: InvocationServices, cancel_event: ThreadEvent, profiler: Optional[Profiler] = None):
        self._services = services
        self._cancel_event = cancel_event
        self._profiler = profiler

    def run(self, queue_item: SessionQueueItem):
        # Exceptions raised outside `run_node` are handled by the processor. There is no need to catch them here.

        self._on_before_run_session(queue_item=queue_item)

        # Loop over invocations until the session is complete or canceled
        while True:
            try:
                invocation = queue_item.session.next()
            # Anything other than a `NodeInputError` is handled as a processor error
            except NodeInputError as e:
                error_type = e.__class__.__name__
                error_message = str(e)
                error_traceback = traceback.format_exc()
                self._on_node_error(
                    invocation=e.node,
                    queue_item=queue_item,
                    error_type=error_type,
                    error_message=error_message,
                    error_traceback=error_traceback,
                )
                break

            if invocation is None or self._cancel_event.is_set():
                break

            self.run_node(invocation, queue_item)

            # The session is complete if all invocations have been run or there is an error on the session.
            if queue_item.session.is_complete() or self._cancel_event.is_set():
                break

        self._on_after_run_session(queue_item=queue_item)

    def run_node(self, invocation: BaseInvocation, queue_item: SessionQueueItem):
        try:
            # Any unhandled exception in this scope is an invocation error & will fail the graph
            with self._services.performance_statistics.collect_stats(invocation, queue_item.session_id):
                self._on_before_run_node(invocation, queue_item)

                data = InvocationContextData(
                    invocation=invocation,
                    source_invocation_id=queue_item.session.prepared_source_mapping[invocation.id],
                    queue_item=queue_item,
                )
                context = build_invocation_context(
                    data=data,
                    services=self._services,
                    cancel_event=self._cancel_event,
                )

                # Invoke the node
                output = invocation.invoke_internal(context=context, services=self._services)
                # Save output and history
                queue_item.session.complete(invocation.id, output)

                self._on_after_run_node(invocation, queue_item, output)

        except KeyboardInterrupt:
            # TODO(psyche): This is expected to be caught in the main thread. Do we need to catch this here?
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
            error_type = e.__class__.__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            self._on_node_error(
                invocation=invocation,
                queue_item=queue_item,
                error_type=error_type,
                error_message=error_message,
                error_traceback=error_traceback,
            )

    def _on_before_run_session(self, queue_item: SessionQueueItem) -> None:
        """Run before a session is executed"""

        self._services.logger.debug(
            f"On before run session: queue item {queue_item.item_id}, session {queue_item.session_id}"
        )

        # If profiling is enabled, start the profiler
        if self._profiler is not None:
            self._profiler.start(profile_id=queue_item.session_id)

        for callback in self._on_before_run_session_callbacks:
            callback(queue_item=queue_item)

    def _on_after_run_session(self, queue_item: SessionQueueItem) -> None:
        """Run after a session is executed"""

        self._services.logger.debug(
            f"On after run session: queue item {queue_item.item_id}, session {queue_item.session_id}"
        )

        # If we are profiling, stop the profiler and dump the profile & stats
        if self._profiler is not None:
            profile_path = self._profiler.stop()
            stats_path = profile_path.with_suffix(".json")
            self._services.performance_statistics.dump_stats(
                graph_execution_state_id=queue_item.session.id, output_path=stats_path
            )

        try:
            # Update the queue item with the completed session. If the queue item has been removed from the queue,
            # we'll get a SessionQueueItemNotFoundError and we can ignore it. This can happen if the queue is cleared
            # while the session is running.
            queue_item = self._services.session_queue.set_queue_item_session(queue_item.item_id, queue_item.session)

            # TODO(psyche): This feels jumbled - we should review separation of concerns here.
            # Send complete event. The events service will receive this and update the queue item's status.
            self._services.events.emit_session_complete(queue_item=queue_item)

            # We'll get a GESStatsNotFoundError if we try to log stats for an untracked graph, but in the processor
            # we don't care about that - suppress the error.
            with suppress(GESStatsNotFoundError):
                self._services.performance_statistics.log_stats(queue_item.session.id)
                self._services.performance_statistics.reset_stats()

            for callback in self._on_after_run_session_callbacks:
                callback(queue_item=queue_item)
        except SessionQueueItemNotFoundError:
            pass

    def _on_before_run_node(self, invocation: BaseInvocation, queue_item: SessionQueueItem):
        """Run before a node is executed"""

        self._services.logger.debug(
            f"On before run node: queue item {queue_item.item_id}, session {queue_item.session_id}, node {invocation.id} ({invocation.get_type()})"
        )

        # Send starting event
        self._services.events.emit_invocation_started(queue_item=queue_item, invocation=invocation)

        for callback in self._on_before_run_node_callbacks:
            callback(invocation=invocation, queue_item=queue_item)

    def _on_after_run_node(
        self, invocation: BaseInvocation, queue_item: SessionQueueItem, output: BaseInvocationOutput
    ):
        """Run after a node is executed"""

        self._services.logger.debug(
            f"On after run node: queue item {queue_item.item_id}, session {queue_item.session_id}, node {invocation.id} ({invocation.get_type()})"
        )

        # Send complete event on successful runs
        self._services.events.emit_invocation_complete(invocation=invocation, queue_item=queue_item, output=output)

        for callback in self._on_after_run_node_callbacks:
            callback(invocation=invocation, queue_item=queue_item, output=output)

    def _on_node_error(
        self,
        invocation: BaseInvocation,
        queue_item: SessionQueueItem,
        error_type: str,
        error_message: str,
        error_traceback: str,
    ):
        """Run when a node errors"""

        self._services.logger.debug(
            f"On node error: queue item {queue_item.item_id}, session {queue_item.session_id}, node {invocation.id} ({invocation.get_type()})"
        )

        # Node errors do not get the full traceback. Only the queue item gets the full traceback.
        node_error = f"{error_type}: {error_message}"
        queue_item.session.set_node_error(invocation.id, node_error)
        self._services.logger.error(
            f"Error while invoking session {queue_item.session_id}, invocation {invocation.id} ({invocation.get_type()}): {error_message}"
        )
        self._services.logger.error(error_traceback)

        # Send error event
        self._services.events.emit_invocation_error(
            queue_item=queue_item,
            invocation=invocation,
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
        )

        for callback in self._on_node_error_callbacks:
            callback(
                invocation=invocation,
                queue_item=queue_item,
                error_type=error_type,
                error_message=error_message,
                error_traceback=error_traceback,
            )


class DefaultSessionProcessor(SessionProcessorBase):
    def __init__(
        self,
        session_runner: Optional[SessionRunnerBase] = None,
        on_non_fatal_processor_error_callbacks: Optional[list[OnNonFatalProcessorError]] = None,
        thread_limit: int = 1,
        polling_interval: int = 1,
    ) -> None:
        super().__init__()

        self.session_runner = session_runner if session_runner else DefaultSessionRunner()
        self._on_non_fatal_processor_error_callbacks = on_non_fatal_processor_error_callbacks or []
        self._thread_limit = thread_limit
        self._polling_interval = polling_interval

    def start(self, invoker: Invoker) -> None:
        self._invoker: Invoker = invoker
        self._queue_item: Optional[SessionQueueItem] = None
        self._invocation: Optional[BaseInvocation] = None

        self._resume_event = ThreadEvent()
        self._stop_event = ThreadEvent()
        self._poll_now_event = ThreadEvent()
        self._cancel_event = ThreadEvent()

        register_events(
            events={SessionCanceledEvent, QueueClearedEvent, BatchEnqueuedEvent, QueueItemStatusChangedEvent},
            func=self._on_queue_event,
        )

        self._thread_semaphore = BoundedSemaphore(self._thread_limit)

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

        self.session_runner.start(services=invoker.services, cancel_event=self._cancel_event, profiler=self._profiler)
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

    async def _on_queue_event(self, event: FastAPIEvent[QueueEvent]) -> None:
        _event_name, payload = event
        if (
            isinstance(payload, SessionCanceledEvent)
            and self._queue_item
            and self._queue_item.item_id == payload.item_id
        ):
            self._cancel_event.set()
            self._poll_now()
        elif (
            isinstance(payload, QueueClearedEvent)
            and self._queue_item
            and self._queue_item.queue_id == payload.queue_id
        ):
            self._cancel_event.set()
            self._poll_now()
        elif isinstance(payload, BatchEnqueuedEvent):
            self._poll_now()
        elif isinstance(payload, QueueItemStatusChangedEvent) and payload.status in ["completed", "failed", "canceled"]:
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
        try:
            # Any unhandled exception in this block is a fatal processor error and will stop the processor.
            self._thread_semaphore.acquire()
            stop_event.clear()
            resume_event.set()
            cancel_event.clear()

            while not stop_event.is_set():
                poll_now_event.clear()
                try:
                    # Any unhandled exception in this block is a nonfatal processor error and will be handled.
                    # If we are paused, wait for resume event
                    resume_event.wait()

                    # Get the next session to process
                    self._queue_item = self._invoker.services.session_queue.dequeue()

                    if self._queue_item is None:
                        # The queue was empty, wait for next polling interval or event to try again
                        self._invoker.services.logger.debug("Waiting for next polling interval or event")
                        poll_now_event.wait(self._polling_interval)
                        continue

                    self._invoker.services.events.emit_session_started(self._queue_item)
                    self._invoker.services.logger.debug(f"Executing queue item {self._queue_item.item_id}")
                    cancel_event.clear()

                    # Run the graph
                    self.session_runner.run(queue_item=self._queue_item)

                except Exception as e:
                    error_type = e.__class__.__name__
                    error_message = str(e)
                    error_traceback = traceback.format_exc()
                    self._on_non_fatal_processor_error(
                        queue_item=self._queue_item,
                        error_type=error_type,
                        error_message=error_message,
                        error_traceback=error_traceback,
                    )
                    # Wait for next polling interval or event to try again
                    poll_now_event.wait(self._polling_interval)
                    continue
        except Exception as e:
            # Fatal error in processor, log and pass - we're done here
            error_type = e.__class__.__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            self._invoker.services.logger.error(f"Fatal Error in session processor {error_type}: {error_message}")
            self._invoker.services.logger.error(error_traceback)
            pass
        finally:
            stop_event.clear()
            poll_now_event.clear()
            self._queue_item = None
            self._thread_semaphore.release()

    def _on_non_fatal_processor_error(
        self,
        queue_item: Optional[SessionQueueItem],
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> None:
        # Non-fatal error in processor
        self._invoker.services.logger.error(f"Non-fatal error in session processor {error_type}: {error_message}")
        self._invoker.services.logger.error(error_traceback)

        if queue_item is not None:
            # Update the queue item with the completed session
            self._invoker.services.session_queue.set_queue_item_session(queue_item.item_id, queue_item.session)
            # Fail the queue item
            self._invoker.services.session_queue.fail_queue_item(
                item_id=queue_item.item_id,
                error_type=error_type,
                error_message=error_message,
                error_traceback=error_traceback,
            )

        for callback in self._on_non_fatal_processor_error_callbacks:
            callback(
                queue_item=queue_item,
                error_type=error_type,
                error_message=error_message,
                error_traceback=error_traceback,
            )
