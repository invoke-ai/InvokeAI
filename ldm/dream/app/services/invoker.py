# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import List
from ..invocations.baseinvocation import BaseInvocation
from .invocation_graph import InvocationGraph
from .invocation_context import InvocationContext, InvocationFieldLink
from .invocation_services import InvocationServices


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices

    def __init__(self, services: InvocationServices):
        self.services = services


    def create_context(self) -> InvocationContext:
        return InvocationContext()


    def invoke(self, context: InvocationContext, invocation: BaseInvocation, links: List[InvocationFieldLink]):
        # Get old outputs as inputs
        context.map_outputs(invocation, links)

        # TODO: consider just passing a "instance services" object with services and other context (like context id)?

        # TODO: clone invocation before invoking?
        outputs = invocation.invoke(self.services)

        # Store outputs
        context.add_history_entry(invocation, outputs)


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
