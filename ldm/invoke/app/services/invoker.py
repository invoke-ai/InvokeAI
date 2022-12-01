# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC
from threading import Event, Thread
from .graph import Graph, GraphExecutionState
from .item_storage import ItemStorageABC
from ..invocations.baseinvocation import InvocationContext
from .invocation_services import InvocationServices
from .invocation_queue import InvocationQueueABC, InvocationQueueItem


class InvokerServices:
    """Services used by the Invoker for execution"""

    queue: InvocationQueueABC
    graph_execution_manager: ItemStorageABC[GraphExecutionState]
    processor: 'InvocationProcessorABC'

    def __init__(self,
        queue: InvocationQueueABC,
        graph_execution_manager: ItemStorageABC[GraphExecutionState],
        processor: 'InvocationProcessorABC'):
        self.queue           = queue
        self.graph_execution_manager = graph_execution_manager
        self.processor       = processor


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices
    invoker_services: InvokerServices

    def __init__(self,
        services: InvocationServices,      # Services used by nodes to perform invocations
        invoker_services: InvokerServices # Services used by the invoker for orchestration
    ):
        self.services = services
        self.invoker_services = invoker_services
        self._start()


    def invoke(self, graph_execution_state: GraphExecutionState, invoke_all: bool = False) -> str|None:
        """Determines the next node to invoke and returns the id of the invoked node, or None if there are no nodes to execute"""

        # Get the next invocation
        invocation = graph_execution_state.next()
        if not invocation:
            return None

        # Save the execution state
        self.invoker_services.graph_execution_manager.set(graph_execution_state)

        # Queue the invocation
        print(f'queueing item {invocation.id}')
        self.invoker_services.queue.put(InvocationQueueItem(
            #session_id    = session.id,
            graph_execution_state_id = graph_execution_state.id,
            invocation_id = invocation.id,
            invoke_all    = invoke_all
        ))

        return invocation.id


    def create_execution_state(self, graph: Graph|None = None) -> GraphExecutionState:
        """Creates a new execution state for the given graph"""
        new_state = GraphExecutionState(graph = Graph() if graph is None else graph)
        self.invoker_services.graph_execution_manager.set(new_state)
        return new_state


    def __start_service(self, service) -> None:
        # Call start() method on any services that have it
        start_op = getattr(service, 'start', None)
        if callable(start_op):
            start_op(self)


    def __stop_service(self, service) -> None:
        # Call stop() method on any services that have it
        stop_op = getattr(service, 'stop', None)
        if callable(stop_op):
            stop_op(self)

    
    def _start(self) -> None:
        """Starts the invoker. This is called automatically when the invoker is created."""
        for service in vars(self.invoker_services):
            self.__start_service(getattr(self.invoker_services, service))

        for service in vars(self.services):
            self.__start_service(getattr(self.services, service))


    def stop(self) -> None:
        """Stops the invoker. A new invoker will have to be created to execute further."""
        # First stop all services
        for service in vars(self.services):
            self.__stop_service(getattr(self.services, service))

        for service in vars(self.invoker_services):
            self.__stop_service(getattr(self.invoker_services, service))

        self.invoker_services.queue.put(None)


class InvocationProcessorABC(ABC):
    pass