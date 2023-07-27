import time
import traceback
from threading import Event, Thread, BoundedSemaphore

from ..invocations.baseinvocation import InvocationContext
from .invocation_queue import InvocationQueueItem
from .invoker import InvocationProcessorABC, Invoker
from ..models.exceptions import CanceledException

import invokeai.backend.util.logging as logger


class DefaultInvocationProcessor(InvocationProcessorABC):
    __invoker_thread: Thread
    __stop_event: Event
    __invoker: Invoker
    __threadLimit: BoundedSemaphore

    def start(self, invoker) -> None:
        # if we do want multithreading at some point, we could make this configurable
        self.__threadLimit = BoundedSemaphore(1)
        self.__invoker = invoker
        self.__stop_event = Event()
        self.__invoker_thread = Thread(
            name="invoker_processor",
            target=self.__process,
            kwargs=dict(stop_event=self.__stop_event),
        )
        self.__invoker_thread.daemon = True  # TODO: make async and do not use threads
        self.__invoker_thread.start()

    def stop(self, *args, **kwargs) -> None:
        self.__stop_event.set()

    def __process(self, stop_event: Event):
        try:
            self.__threadLimit.acquire()
            while not stop_event.is_set():
                try:
                    queue_item: InvocationQueueItem = self.__invoker.services.queue.get()
                except Exception as e:
                    self.__invoker.services.logger.error("Exception while getting from queue:\n%s" % e)

                if not queue_item:  # Probably stopping
                    # do not hammer the queue
                    time.sleep(0.5)
                    continue

                try:
                    graph_execution_state = self.__invoker.services.graph_execution_manager.get(
                        queue_item.graph_execution_state_id
                    )
                except Exception as e:
                    self.__invoker.services.logger.error("Exception while retrieving session:\n%s" % e)
                    self.__invoker.services.events.emit_session_retrieval_error(
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
                    graph_execution_state_id=graph_execution_state.id,
                    node=invocation.dict(),
                    source_node_id=source_node_id,
                )

                # Invoke
                try:
                    outputs = invocation.invoke(
                        InvocationContext(
                            services=self.__invoker.services,
                            graph_execution_state_id=graph_execution_state.id,
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
                        graph_execution_state_id=graph_execution_state.id,
                        node=invocation.dict(),
                        source_node_id=source_node_id,
                        result=outputs.dict(),
                    )

                except KeyboardInterrupt:
                    pass

                except CanceledException:
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
                        graph_execution_state_id=graph_execution_state.id,
                        node=invocation.dict(),
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
                        self.__invoker.invoke(graph_execution_state, invoke_all=True)
                    except Exception as e:
                        self.__invoker.services.logger.error("Error while invoking:\n%s" % e)
                        self.__invoker.services.events.emit_invocation_error(
                            graph_execution_state_id=graph_execution_state.id,
                            node=invocation.dict(),
                            source_node_id=source_node_id,
                            error_type=e.__class__.__name__,
                            error=traceback.format_exc(),
                        )
                elif is_complete:
                    self.__invoker.services.events.emit_graph_execution_complete(graph_execution_state.id)

        except KeyboardInterrupt:
            pass  # Log something? KeyboardInterrupt is probably not going to be seen by the processor
        finally:
            self.__threadLimit.release()
