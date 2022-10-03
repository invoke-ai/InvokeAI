# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from queue import Queue
from threading import Event, Thread
from typing import Dict, List

from .invocation_graph import InvocationGraph
from .invocation_session import InvocationSession, InvocationFieldLink
from .invocation_services import InvocationServices
from .invocation_queue import InvocationQueueABC, InvocationQueueItem
from .session_manager import SessionManagerABC


class InvokerServices:
    queue: InvocationQueueABC
    session_manager: SessionManagerABC

    def __init__(self,
        queue: InvocationQueueABC,
        session_manager: SessionManagerABC):
        self.queue           = queue
        self.session_manager = session_manager


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices
    invoker_services: InvokerServices

    __invoker_thread: Thread
    __stop_event: Event

    def __init__(self,
        services: InvocationServices,      # Services used by nodes to perform invocations
        invoker_services: InvokerServices # Services used by the invoker for orchestration
    ):
        self.services = services
        self.invoker_services = invoker_services
        self.__stop_event = Event()
        self.__invoker_thread = Thread(
            name = "invoker",
            target = self.__process,
            kwargs = dict(stop_event = self.__stop_event)
        )
        self.__invoker_thread.start()


    def __ensure_alive(self):
        if self.__stop_event.isSet():
            raise Exception("Invoker has been stopped. Must create a new invoker.")


    def __process(self, stop_event: Event):
        try:
            while not stop_event.isSet():
                queue_item: InvocationQueueItem = self.invoker_services.queue.get()
                if not queue_item: # Probably stopping
                    continue

                session = self.invoker_services.session_manager.get(queue_item.session_id)
                invocation = session.invocations[queue_item.invocation_id]

                # Send starting event
                self.services.events.emit_invocation_started(
                    session.id, invocation.id
                )

                # Invoke
                outputs = invocation.invoke(self.services, session_id = session.id)

                # Save outputs and history
                session._complete_invocation(invocation, outputs)

                # Save the session changes
                self.invoker_services.session_manager.set(session)

                # Send complete event
                self.services.events.emit_invocation_complete(
                    session.id, invocation.id, outputs.dict()
                )

                # Queue any further commands if invoking all
                if queue_item.invoke_all and session.ready_to_invoke():
                    self.invoke(session, invoke_all = True)
                elif not session.ready_to_invoke():
                    self.services.events.emit_session_invocation_complete(session.id)

        except KeyboardInterrupt:
            ... # Log something?


    def invoke(self, session: InvocationSession, invoke_all: bool = False) -> str:
        """Determines the next node to invoke and returns the id of the invoked node"""
        self.__ensure_alive()

        invocation_id = session._get_next_invocation_id()
        if not invocation_id:
            return # TODO: raise an error?

        # Save session in case user changed it
        self.invoker_services.session_manager.set(session)

        # Get updated input values
        # TODO: consider using a copy to keep history separate from input configuration
        session._map_inputs(invocation_id)
        invocation = session.invocations[invocation_id]

        # Queue the invocation
        self.invoker_services.queue.put(InvocationQueueItem(
            session_id    = session.id,
            invocation_id = invocation.id,
            invoke_all    = invoke_all
        ))

        # Return the id of the invocation
        return invocation.id


    def create_session(self) -> InvocationSession:
        self.__ensure_alive()
        return self.invoker_services.session_manager.create()


    def create_session_from_graph(self, invocation_graph: InvocationGraph) -> InvocationSession:
        self.__ensure_alive()

        # Create a new session
        session = self.create_session()

        # Directly create nodes and links, since graph is already validated
        for node in invocation_graph.nodes:
            session.invocations[node.id] = node
        
        for link in invocation_graph.links:
            if not link.to_node.id in session.links:
                session.links[link.to_node.id] = list()
            
            session.links[link.to_node.id].append(InvocationFieldLink(
                from_node_id = link.from_node.id,
                from_field   = link.from_node.field,
                to_field     = link.to_node.field))

        return session


    def __stop_service(self, service) -> None:
        # Call stop() method on any services that have it
        stop_op = getattr(service, 'stop', None)
        if callable(stop_op):
            stop_op()


    def stop(self) -> None:
        """Stops the invoker. A new invoker will have to be created to execute further."""
        # First stop all services
        for service in vars(self.services):
            self.__stop_service(service)

        for service in vars(self.invoker_services):
            self.__stop_service(service)

        self.__stop_event.set()
        self.invoker_services.queue.put(None)