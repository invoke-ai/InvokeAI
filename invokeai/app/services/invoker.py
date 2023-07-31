# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC
from typing import Optional

from .graph import Graph, GraphExecutionState
from .invocation_queue import InvocationQueueItem
from .invocation_services import InvocationServices


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices

    def __init__(self, services: InvocationServices):
        self.services = services
        self._start()

    def invoke(self, graph_execution_state: GraphExecutionState, invoke_all: bool = False) -> Optional[str]:
        """Determines the next node to invoke and enqueues it, preparing if needed.
        Returns the id of the queued node, or `None` if there are no nodes left to enqueue."""

        # Get the next invocation
        invocation = graph_execution_state.next()
        if not invocation:
            return None

        # Save the execution state
        self.services.graph_execution_manager.set(graph_execution_state)

        # Queue the invocation
        self.services.queue.put(
            InvocationQueueItem(
                # session_id    = session.id,
                graph_execution_state_id=graph_execution_state.id,
                invocation_id=invocation.id,
                invoke_all=invoke_all,
            )
        )

        return invocation.id

    def create_execution_state(self, graph: Optional[Graph] = None) -> GraphExecutionState:
        """Creates a new execution state for the given graph"""
        new_state = GraphExecutionState(graph=Graph() if graph is None else graph)
        self.services.graph_execution_manager.set(new_state)
        return new_state

    def cancel(self, graph_execution_state_id: str) -> None:
        """Cancels the given execution state"""
        self.services.queue.cancel(graph_execution_state_id)

    def __start_service(self, service) -> None:
        # Call start() method on any services that have it
        start_op = getattr(service, "start", None)
        if callable(start_op):
            start_op(self)

    def __stop_service(self, service) -> None:
        # Call stop() method on any services that have it
        stop_op = getattr(service, "stop", None)
        if callable(stop_op):
            stop_op(self)

    def _start(self) -> None:
        """Starts the invoker. This is called automatically when the invoker is created."""
        for service in vars(self.services):
            self.__start_service(getattr(self.services, service))

    def stop(self) -> None:
        """Stops the invoker. A new invoker will have to be created to execute further."""
        # First stop all services
        for service in vars(self.services):
            self.__stop_service(getattr(self.services, service))

        self.services.queue.put(None)


class InvocationProcessorABC(ABC):
    pass
