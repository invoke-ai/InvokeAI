import traceback
from threading import Event, Thread

from ..invocations.baseinvocation import InvocationContext
from .invocation_queue import InvocationQueueItem
from .invoker import InvocationProcessorABC, Invoker


class DefaultInvocationProcessor(InvocationProcessorABC):
    __invoker_thread: Thread
    __stop_event: Event
    __invoker: Invoker

    def start(self, invoker) -> None:
        self.__invoker = invoker
        self.__stop_event = Event()
        self.__invoker_thread = Thread(
            name="invoker_processor",
            target=self.__process,
            kwargs=dict(stop_event=self.__stop_event),
        )
        self.__invoker_thread.daemon = (
            True  # TODO: probably better to just not use threads?
        )
        self.__invoker_thread.start()

    def stop(self, *args, **kwargs) -> None:
        self.__stop_event.set()

    def __process(self, stop_event: Event):
        try:
            while not stop_event.is_set():
                queue_item: InvocationQueueItem = self.__invoker.services.queue.get()
                if not queue_item:  # Probably stopping
                    continue

                graph_execution_state = (
                    self.__invoker.services.graph_execution_manager.get(
                        queue_item.graph_execution_state_id
                    )
                )
                invocation = graph_execution_state.execution_graph.get_node(
                    queue_item.invocation_id
                )

                # Send starting event
                self.__invoker.services.events.emit_invocation_started(
                    graph_execution_state_id=graph_execution_state.id,
                    invocation_id=invocation.id,
                )

                # Invoke
                try:
                    outputs = invocation.invoke(
                        InvocationContext(
                            services=self.__invoker.services,
                            graph_execution_state_id=graph_execution_state.id,
                        )
                    )

                    # Save outputs and history
                    graph_execution_state.complete(invocation.id, outputs)

                    # Save the state changes
                    self.__invoker.services.graph_execution_manager.set(
                        graph_execution_state
                    )

                    # Send complete event
                    self.__invoker.services.events.emit_invocation_complete(
                        graph_execution_state_id=graph_execution_state.id,
                        invocation_id=invocation.id,
                        result=outputs.dict(),
                    )

                except KeyboardInterrupt:
                    pass

                except Exception as e:
                    error = traceback.format_exc()

                    # Save error
                    graph_execution_state.set_node_error(invocation.id, error)

                    # Save the state changes
                    self.__invoker.services.graph_execution_manager.set(
                        graph_execution_state
                    )

                    # Send error event
                    self.__invoker.services.events.emit_invocation_error(
                        graph_execution_state_id=graph_execution_state.id,
                        invocation_id=invocation.id,
                        error=error,
                    )

                    pass

                # Queue any further commands if invoking all
                is_complete = graph_execution_state.is_complete()
                if queue_item.invoke_all and not is_complete:
                    self.__invoker.invoke(graph_execution_state, invoke_all=True)
                elif is_complete:
                    self.__invoker.services.events.emit_graph_execution_complete(
                        graph_execution_state.id
                    )

        except KeyboardInterrupt:
            ...  # Log something?
