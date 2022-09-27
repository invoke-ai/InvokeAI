# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from queue import Queue
from threading import Event, Thread
from typing import Dict, List

from ..invocations.baseinvocation import BaseInvocation
from .invocation_graph import InvocationGraph
from .invocation_context import InvocationContext, InvocationFieldLink
from .invocation_services import InvocationServices
from .invocation_queue import InvocationQueueABC, InvocationQueueItem
from .context_manager import ContextManagerABC


class InvokerServices:
    queue: InvocationQueueABC
    context_manager: ContextManagerABC

    def __init__(self,
        queue: InvocationQueueABC,
        context_manager: ContextManagerABC):
        self.queue           = queue
        self.context_manager = context_manager


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices
    invoker_services: InvokerServices

    __invoker_thread: Thread

    def __init__(self,
        services: InvocationServices,     # Services used by nodes to perform invocations
        invoker_services: InvokerServices # Services used by the invoker for orchestration
    ):
        self.services = services
        self.invoker_services = invoker_services
        self.__invoker_thread = Thread(
            name = "invoker",
            target = self.__process
        )
        self.__invoker_thread.start()


    def __process(self):
        try:
            # TODO: Figure out how to gracefully shut down a thread at exit
            while True:
                queue_item: InvocationQueueItem = self.invoker_services.queue.get()
                context = self.invoker_services.context_manager.get(queue_item.context_id)
                invocation = context.invocations[queue_item.invocation_id]

                # Invoke
                outputs = invocation.invoke(self.services)

                # Save outputs and history
                context._complete_invocation(invocation, outputs)

                # Save the context changes
                self.invoker_services.context_manager.set(context)

                # Queue any further commands if invoking all
                if queue_item.invoke_all and context.ready_to_invoke():
                    self.invoke(context, invoke_all = True)

        except KeyboardInterrupt:
            ... # Log something?


    def invoke(self, context: InvocationContext, invoke_all: bool = False) -> str:
        """Determines the next node to invoke and returns the id of the invoked node"""
        invocation_id = context._get_next_invocation_id()
        if not invocation_id:
            return # TODO: raise an error?

        # Save context in case user changed it
        self.invoker_services.context_manager.set(context)

        # Get updated input values
        # TODO: consider using a copy to keep history separate from input configuration
        context._map_inputs(invocation_id)
        invocation = context.invocations[invocation_id]

        # Queue the invocation
        self.invoker_services.queue.put(InvocationQueueItem(
            context_id    = context.id,
            invocation_id = invocation.id,
            invoke_all    = invoke_all
        ))

        # Return the id of the invocation
        return invocation.id


    def create_context(self) -> InvocationContext:
        return self.invoker_services.context_manager.create()


    def create_context_from_graph(self, invocation_graph: InvocationGraph) -> InvocationContext:
        # Create a new context
        context = self.create_context()

        # Directly create nodes and links, since graph is already validated
        for node in invocation_graph.nodes:
            context.invocations[node.id] = node
        
        for link in invocation_graph.links:
            if not link.to_node.id in context.links:
                context.links[link.to_node.id] = list()
            
            context.links[link.to_node.id].append(InvocationFieldLink(link.from_node.id, link.from_node.field, link.to_node.field))

        return context


    # def invoke_graph(self, invocation_graph: InvocationGraph) -> InvocationContext:
    #     # Create a new context
    #     context = self.create_context()

    #     # Get and prepare the node graph for execution
    #     graph = invocation_graph.get_graph()
    #     graph.prepare()

    #     # Execute the graph until all nodes are completed
    #     while (graph.is_active()):
    #         for node_id in graph.get_ready():
    #             #print(f'Preparing invocation {node_id}')
    #             # TODO: consider cloning the node at execution time, to handle looping
    #             node = invocation_graph.get_node(node_id)

    #             # Overwrite node inputs with links
    #             input_links = invocation_graph.get_node_input_links(node)

    #             # Create invocation links
    #             invocation_links = list(map(lambda link: InvocationFieldLink(link.from_node.id, link.from_node.field, link.to_node.field), input_links))

    #             # Get old outputs as inputs
    #             context._map_outputs(node, invocation_links)

    #             # Invoke
    #             self.invoke(context, node, invocation_links)

    #             # Mark node as complete
    #             graph.done(node_id)

    #     return context