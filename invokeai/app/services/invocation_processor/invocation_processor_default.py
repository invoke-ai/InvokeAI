import time
import traceback
from contextlib import suppress
from threading import BoundedSemaphore, Event, Thread
from typing import Optional

import invokeai.backend.util.logging as logger
from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.app.services.invocation_queue.invocation_queue_common import InvocationQueueItem
from invokeai.app.services.invocation_stats.invocation_stats_common import (
    GESStatsNotFoundError,
)
from invokeai.app.util.profiler import Profiler

from ..invoker import Invoker
from .invocation_processor_base import InvocationProcessorABC
from .invocation_processor_common import CanceledException


class DefaultInvocationProcessor(InvocationProcessorABC):
    __invoker_thread: Thread
    __stop_event: Event
    __invoker: Invoker
    __threadLimit: BoundedSemaphore

    def start(self, invoker: Invoker) -> None:
        # if we do want multithreading at some point, we could make this configurable
        self.__threadLimit = BoundedSemaphore(1)
        self.__invoker = invoker
        self.__stop_event = Event()
        self.__invoker_thread = Thread(
            name="invoker_processor",
            target=self.__process,
            kwargs={"stop_event": self.__stop_event},
        )
        self.__invoker_thread.daemon = True  # TODO: make async and do not use threads
        self.__invoker_thread.start()

    def stop(self, *args, **kwargs) -> None:
        self.__stop_event.set()

    def __process(self, stop_event: Event):
        try:
            self.__threadLimit.acquire()
            queue_item: Optional[InvocationQueueItem] = None

            profiler = (
                Profiler(
                    logger=self.__invoker.services.logger,
                    output_dir=self.__invoker.services.configuration.profiles_path,
                    prefix=self.__invoker.services.configuration.profile_prefix,
                )
                if self.__invoker.services.configuration.profile_graphs
                else None
            )

            def stats_cleanup(graph_execution_state_id: str) -> None:
                if profiler:
                    profile_path = profiler.stop()
                    stats_path = profile_path.with_suffix(".json")
                    self.__invoker.services.performance_statistics.dump_stats(
                        graph_execution_state_id=graph_execution_state_id, output_path=stats_path
                    )
                with suppress(GESStatsNotFoundError):
                    self.__invoker.services.performance_statistics.log_stats(graph_execution_state_id)
                    self.__invoker.services.performance_statistics.reset_stats(graph_execution_state_id)

            while not stop_event.is_set():
                try:
                    queue_item = self.__invoker.services.queue.get()
                except Exception as e:
                    self.__invoker.services.logger.error("Exception while getting from queue:\n%s" % e)

                if not queue_item:  # Probably stopping
                    # do not hammer the queue
                    time.sleep(0.5)
                    continue

                if profiler and profiler.profile_id != queue_item.graph_execution_state_id:
                    profiler.start(profile_id=queue_item.graph_execution_state_id)

                try:
                    graph_execution_state = self.__invoker.services.graph_execution_manager.get(
                        queue_item.graph_execution_state_id
                    )
                except Exception as e:
                    self.__invoker.services.logger.error("Exception while retrieving session:\n%s" % e)
                    self.__invoker.services.events.emit_session_retrieval_error(
                        queue_batch_id=queue_item.session_queue_batch_id,
                        queue_item_id=queue_item.session_queue_item_id,
                        queue_id=queue_item.session_queue_id,
                        graph_execution_state_id=queue_item.graph_execution_state_id,
                        error_type=e.__class__.__name__,
                        error=traceback.format_exc(),
                    )
                    continue

                try:
                    invocation = graph_execution_state.execution_graph.get_node(queue_item.invocation_id)
                except Exception as e:
                    self.__invoker.services.logger.error("Exception while retrieving invocation:\n%s" % e)
                    self.__invoker.services.events.emit_invocation_retrieval_error(
                        queue_batch_id=queue_item.session_queue_batch_id,
                        queue_item_id=queue_item.session_queue_item_id,
                        queue_id=queue_item.session_queue_id,
                        graph_execution_state_id=queue_item.graph_execution_state_id,
                        node_id=queue_item.invocation_id,
                        error_type=e.__class__.__name__,
                        error=traceback.format_exc(),
                    )
                    continue

                # get the source node id to provide to clients (the prepared node id is not as useful)
                source_node_id = graph_execution_state.prepared_source_mapping[invocation.id]

                # Send starting event
                self.__invoker.services.events.emit_invocation_started(
                    queue_batch_id=queue_item.session_queue_batch_id,
                    queue_item_id=queue_item.session_queue_item_id,
                    queue_id=queue_item.session_queue_id,
                    graph_execution_state_id=graph_execution_state.id,
                    node=invocation.model_dump(),
                    source_node_id=source_node_id,
                )

                # Invoke
                try:
                    graph_id = graph_execution_state.id
                    with self.__invoker.services.performance_statistics.collect_stats(invocation, graph_id):
                        # use the internal invoke_internal(), which wraps the node's invoke() method,
                        # which handles a few things:
                        # - nodes that require a value, but get it only from a connection
                        # - referencing the invocation cache instead of executing the node
                        outputs = invocation.invoke_internal(
                            InvocationContext(
                                services=self.__invoker.services,
                                graph_execution_state_id=graph_execution_state.id,
                                queue_item_id=queue_item.session_queue_item_id,
                                queue_id=queue_item.session_queue_id,
                                queue_batch_id=queue_item.session_queue_batch_id,
                                workflow=queue_item.workflow,
                            )
                        )

                        # Check queue to see if this is canceled, and skip if so
                        if self.__invoker.services.queue.is_canceled(graph_execution_state.id):
                            continue

                        # Save outputs and history
                        graph_execution_state.complete(invocation.id, outputs)

                        # Save the state changes
                        self.__invoker.services.graph_execution_manager.set(graph_execution_state)

                        # Send complete event
                        self.__invoker.services.events.emit_invocation_complete(
                            queue_batch_id=queue_item.session_queue_batch_id,
                            queue_item_id=queue_item.session_queue_item_id,
                            queue_id=queue_item.session_queue_id,
                            graph_execution_state_id=graph_execution_state.id,
                            node=invocation.model_dump(),
                            source_node_id=source_node_id,
                            result=outputs.model_dump(),
                        )

                except KeyboardInterrupt:
                    pass

                except CanceledException:
                    stats_cleanup(graph_execution_state.id)
                    pass

                except Exception as e:
                    error = traceback.format_exc()
                    logger.error(error)

                    # Save error
                    graph_execution_state.set_node_error(invocation.id, error)

                    # Save the state changes
                    self.__invoker.services.graph_execution_manager.set(graph_execution_state)

                    self.__invoker.services.logger.error("Error while invoking:\n%s" % e)
                    # Send error event
                    self.__invoker.services.events.emit_invocation_error(
                        queue_batch_id=queue_item.session_queue_batch_id,
                        queue_item_id=queue_item.session_queue_item_id,
                        queue_id=queue_item.session_queue_id,
                        graph_execution_state_id=graph_execution_state.id,
                        node=invocation.model_dump(),
                        source_node_id=source_node_id,
                        error_type=e.__class__.__name__,
                        error=error,
                    )
                    pass

                # Check queue to see if this is canceled, and skip if so
                if self.__invoker.services.queue.is_canceled(graph_execution_state.id):
                    continue

                # Queue any further commands if invoking all
                is_complete = graph_execution_state.is_complete()
                if queue_item.invoke_all and not is_complete:
                    try:
                        self.__invoker.invoke(
                            session_queue_batch_id=queue_item.session_queue_batch_id,
                            session_queue_item_id=queue_item.session_queue_item_id,
                            session_queue_id=queue_item.session_queue_id,
                            graph_execution_state=graph_execution_state,
                            workflow=queue_item.workflow,
                            invoke_all=True,
                        )
                    except Exception as e:
                        self.__invoker.services.logger.error("Error while invoking:\n%s" % e)
                        self.__invoker.services.events.emit_invocation_error(
                            queue_batch_id=queue_item.session_queue_batch_id,
                            queue_item_id=queue_item.session_queue_item_id,
                            queue_id=queue_item.session_queue_id,
                            graph_execution_state_id=graph_execution_state.id,
                            node=invocation.model_dump(),
                            source_node_id=source_node_id,
                            error_type=e.__class__.__name__,
                            error=traceback.format_exc(),
                        )
                elif is_complete:
                    self.__invoker.services.events.emit_graph_execution_complete(
                        queue_batch_id=queue_item.session_queue_batch_id,
                        queue_item_id=queue_item.session_queue_item_id,
                        queue_id=queue_item.session_queue_id,
                        graph_execution_state_id=graph_execution_state.id,
                    )
                    stats_cleanup(graph_execution_state.id)

        except KeyboardInterrupt:
            pass  # Log something? KeyboardInterrupt is probably not going to be seen by the processor
        finally:
            self.__threadLimit.release()
