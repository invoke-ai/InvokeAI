import traceback

from contextlib import suppress
from threading import Event
from typing import Callable, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context
from invokeai.app.services.invocation_stats.invocation_stats_common import GESStatsNotFoundError
from invokeai.app.util.profiler import Profiler
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.session_processor.session_processor_common import CanceledException

class GraphProcessor:
    """Process a graph of invocations"""
    def __init__(
            self,
            services: InvocationServices,
            cancel_event: Event,
            profiler: Union[Profiler, None] = None,
            on_before_run_node: Union[Callable[[BaseInvocation,SessionQueueItem], bool], None] = None,
            on_after_run_node: Union[Callable[[BaseInvocation,SessionQueueItem], bool], None] = None,
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
            with self.services.performance_statistics.collect_stats(
                invocation, queue_item.session_id
            ):
                context = build_invocation_context(
                    data=data,
                    services=self.services,
                    cancel_event=self.cancel_event,
                )

                # Invoke the node
                outputs = invocation.invoke_internal(
                    context=context, services=self.services
                )

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
            self.cancel_event.set()
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
