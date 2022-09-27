# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import List, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, root_validator
from graphlib import TopologicalSorter, CycleError
from ..invocations import *
from ..invocations.baseinvocation import BaseInvocation
from .invocation_context import is_field_compatible


class Node(BaseModel):
    """An invocation as a node in a graph"""
    id: str = Field(description="The id of the node for this link")
    field: str = Field(description="The name of the field for this link")


class Link(BaseModel):
    """A link between two fields in different nodes"""
    from_node: Node = Field(description="The node field linked from")
    to_node: Node = Field(description="The node field linked to")


# TODO: a lot of the validation code in here is part of InvocationContext now.
#       See if there's a way to reduce code.
class InvocationGraph(BaseModel):
    """A map of invocations"""
    class Config:
        schema_extra = {
            "example": {
                "nodes":[
                    {"id":"1","type":"txt2img","prompt":"A photo of a cat eating sushi"},
                    {"id":"2","type":"show_image"}
                ],
                "links":[
                    {"from_node":{"id":"1","field":"image"},"to_node":{"id":"2","field":"image"}}
                ]
            }
        }


    nodes: List[Annotated[Union[BaseInvocation.get_invocations()], Field(discriminator="type")]] = Field(description="The nodes in this map")
    links: List[Link] = Field(description="The links in this map")

    @staticmethod
    def build_graph(nodes, links):
        graph_nodes = dict({nodes[i].id: set() for i in range(0, len(nodes))})

        # Link all predecessors
        for link in links:
            graph_nodes[link.to_node.id].add(link.from_node.id)

        graph = TopologicalSorter(graph_nodes)
        return graph
    
    def get_graph(self):
        return InvocationGraph.build_graph(self.nodes, self.links)

    def get_node_input_links(self, node: Node):
        return filter(lambda link: link.to_node.id == node.id, self.links)

    def get_node(self, id: str):
        for node in self.nodes:
            if (node.id == id):
                return node
        return None


    # validate_schema (validate node ids, validate all links and their types)
    @root_validator(skip_on_failure=True)
    def validate_nodes_and_links(cls, data):
        nodes = data.get('nodes')
        links = data.get('links')

        # Check for duplicate node ids
        ids_list = list(map(lambda node: node.id, nodes))
        ids_set = set(ids_list)
        if len(ids_list) != len(ids_set):
            raise ValueError('All node ids must be unique')

        nodes_dict = dict({nodes[i].id: nodes[i] for i in range(0, len(nodes))})

        # Validate all links
        # TODO: validate only one link TO a single field of a node
        errors = {}
        for i in range(len(links)):
            link = links[i]
            link_errors = False

            # Ensure node ids both exist
            from_id = link.from_node.id
            to_id = link.to_node.id

            if from_id not in ids_set:
                errors['links'] = [f'from_node.id {from_id} does not match any node id']
                link_errors = True

            if to_id not in ids_set:
                errors['links'] = [f'to_node.id {to_id} does not match any node id']
                link_errors = True

            if from_id == to_id:
                errors['links'] = [f'node {from_id} must not link to itself']
                link_errors = True
            
            if link_errors:
                continue

            # Get node types
            invoker_types = BaseInvocation.get_invocations_map()

            from_node = nodes_dict[from_id]
            to_node = nodes_dict[to_id]
            from_field = link.from_node.field
            to_field = link.to_node.field

            if not is_field_compatible(from_node, from_field, to_node, to_field):
                errors['links'] = [f'invalid link {from_node}[{from_field}] -> {to_node}[{to_field}]']

        if errors:
            raise ValueError(f'One or more errors validating map: {errors}')
        
        # Validate that this is a directed graph (no cycles)
        ts = InvocationGraph.build_graph(nodes, links)
        try:
            ts.prepare()
        except CycleError:
            raise ValueError('Node graph must not have any cycles (loops)')

        # Success, return
        return data
