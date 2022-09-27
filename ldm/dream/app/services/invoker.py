# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from queue import Queue
from threading import Event, Thread
from typing import Dict, List
from ..invocations.baseinvocation import BaseInvocation
from .invocation_graph import InvocationGraph
from .invocation_context import InvocationContext, InvocationFieldLink
from .invocation_services import InvocationServices


# TODO: make this serializable
class InvocationQueueItem:
    context: InvocationContext
    invocation: BaseInvocation
    links: List[InvocationFieldLink]

    def __init__(self, context: InvocationContext, invocation: BaseInvocation, links: List[InvocationFieldLink]):
        self.context = context
        self.invocation = invocation
        self.links = links


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices

    # TODO: This is naive, and should be its own service so it can use a different backend
    __invocation_queue: Queue = Queue()
    __thread: Thread

    def __init__(self, services: InvocationServices):
        self.services = services
        self.__thread = Thread(
            name = "invoker",
            target = self.__process
        )
        self.__thread.start()


    def create_context(self) -> InvocationContext:
        return InvocationContext()


    def __process(self):
        try:
            while True: # TODO: I don't know the application lifetime event for Python to shut down a thread
                queue_item: InvocationQueueItem = self.__invocation_queue.get()

                # Get old outputs as inputs
                queue_item.context.map_outputs(queue_item.invocation, queue_item.links)
                outputs = queue_item.invocation.invoke(self.services)

                # TODO: save context after this?
                queue_item.context.complete_invocation(queue_item.invocation, outputs)
        
        except KeyboardInterrupt:
            ... # Log something?


    def invoke(self, context: InvocationContext, invocation: BaseInvocation, links: List[InvocationFieldLink]):
        self.__invocation_queue.put(InvocationQueueItem(
            context = context,
            invocation = invocation,
            links = links
        ))

        # For now just wait for the invocation to complete
        context.wait_for_invocation(invocation.id)


    def invoke_graph(self, invocation_graph: InvocationGraph):
        # Create a new context
        context = self.create_context()

        # Get and prepare the node graph for execution
        graph = invocation_graph.get_graph()
        graph.prepare()

        # Execute the graph until all nodes are completed
        while (graph.is_active()):
            for node_id in graph.get_ready():
                #print(f'Preparing invocation {node_id}')
                # TODO: consider cloning the node at execution time, to handle looping
                node = invocation_graph.get_node(node_id)

                # Overwrite node inputs with links
                input_links = invocation_graph.get_node_input_links(node)

                # Create invocation links
                invocation_links = list(map(lambda link: InvocationFieldLink(link.from_node.id, link.from_node.field, link.to_node.field), input_links))

                # Get old outputs as inputs
                context.map_outputs(node, invocation_links)

                # Invoke
                self.invoke(context, node, invocation_links)

                # Mark node as complete
                graph.done(node_id)
