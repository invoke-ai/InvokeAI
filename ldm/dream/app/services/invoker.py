# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from ldm.dream.app.invocations.baseinvocation import InvocationContext, InvocationFieldLink, InvocationServices
from ldm.dream.app.services.invocation_graph import InvocationGraph


class Invoker:
    """The invoker, used to execute invocations"""

    services: InvocationServices

    def __init__(self, services: InvocationServices):
        self.services = services


    def create_context(self) -> InvocationContext:
        return InvocationContext(self.services)

    ...
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

                context.map_outputs(node, invocation_links)

                # for link in input_links:
                #     output_id = link.from_node.id
                #     output_field = link.from_node.field
                #     input_field = link.to_node.field
                #     # TODO: should these be deep copied in case they get transformed by another node?
                #     output_value = context.get_output(output_id, output_field)
                #     setattr(node, input_field, output_value)

                # Invoke
                #print(f'invoking {node_id} of type {node.type}')
                outputs = node.invoke(context)

                # Store outputs and mark node as complete
                context.add_history_entry(node, outputs)
                graph.done(node_id)
