# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from threading import Event
from typing_extensions import Annotated
from pydantic import BaseModel, PrivateAttr
from pydantic.fields import Field
from typing import Any, Callable, Dict, List, Union, get_args, get_type_hints
from graphlib import TopologicalSorter, CycleError
from ..invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from ..invocations import *


InvocationsUnion = Annotated[Union[BaseInvocation.get_invocations()], Field(discriminator="type")]
InvocationOutputsUnion = Annotated[Union[BaseInvocationOutput.get_all_subclasses_tuple()], Field(discriminator="type")]

class InvocationHistoryEntry(BaseModel):
    """The context of an invoked node"""
    invocation: InvocationsUnion
    outputs: InvocationOutputsUnion


class InvocationFieldLink(BaseModel):
    from_node_id: Union[str,int]
    from_field: str
    to_field: str


def is_field_compatible(
    from_invocation: BaseInvocation,
    from_field: str,
    to_invocation: BaseInvocation,
    to_field: str) -> bool:

    from_node_type = type(from_invocation)
    to_node_type = type(to_invocation)

    # Get field type hints
    from_node_outputs = get_type_hints(from_node_type.get_output_type())
    to_node_inputs = get_type_hints(to_node_type)

    from_node_field = from_node_outputs.get(from_field) or None
    to_node_field = to_node_inputs.get(to_field) or None

    if not from_node_field:
        return False # TODO: should this throw?
    if not to_node_field:
        return False # TODO: should this throw?
    
    if from_node_field and to_node_field:
        if from_node_field != to_node_field and from_node_field not in get_args(to_node_field):
            return False
    
    return True


def ContextConflict(Exception):
    pass


class InvocationContext(BaseModel):
    """The invocation context.
    Maintains the current invocation graph, history, and results.
    """
    id: str

    # Invocations
    invocations: Dict[str, InvocationsUnion] = Field(description = "All invocations")
    links: Dict[str, List[InvocationFieldLink]] = Field(description="All links between invocations")

    # Invocation history
    invocation_results: Dict[str, InvocationHistoryEntry] = Field(description = "The outputs of executed invocations")
    history: List[str] = Field(description = "The invocations that have been run, in order")

    _wait_events: Dict[str, Event] = PrivateAttr()
    _change_callback: Callable[['InvocationContext'], None] = PrivateAttr()

    def __init__(self,
        id: str,
        change_callback: Callable[['InvocationContext'], None] = None,
        invocations: Dict[str, InvocationsUnion] = dict(),
        links: Dict[str, List[InvocationFieldLink]] = dict(),
        invocation_results: Dict[str, InvocationHistoryEntry] = dict(),
        history: List[str] = list()
    ):
        super().__init__(
            id = id,
            invocations = invocations,
            links = links,
            invocation_results = invocation_results,
            history = history
        )
        self._wait_events = dict()
        self._change_callback = change_callback


    def ready_to_invoke(self) -> bool:
        """Returns true if there are uninvoked nodes left in the graph.
        Note that this assumes that only one node in a graph is being invoked at a time.
        """
        return (len(self.invocations) > len(self.history))


    def add_invocation(self, invocation: BaseInvocation, links: List[InvocationFieldLink]):
        """Add an invocation and links to it to the current context
        Links can use negative integers as their id to refer to previous history
        and can utilize * as their from_field to find all matching fields
        """
        if invocation.id in self.invocations:
            raise ContextConflict(f"An invocation with id {id} already exists in this context")
        
        # Validate links
        final_links = list()
        for link in links:
            node_id = link.from_node_id
            # Get the results from history
            if type(node_id) is int:
                if node_id < 0:
                    if abs(node_id) > len(self.history):
                        raise IndexError()
                    else:
                        node_id = self.history[len(self.history) + node_id]
                else:
                    raise IndexError() # Must use negative integers when using relative node ids
            
            # TODO: validate node_id is in invocations
            from_node = self.invocations[node_id]
            from_node_outputs_type = type(from_node).get_output_type()

            if link.from_field == '*':
                output_fields = from_node_outputs_type.__fields__
                to_fields = invocation.__fields__
                for field_name in output_fields:
                    if field_name in to_fields:
                        if is_field_compatible(from_node, field_name, invocation, field_name):
                            final_links.append(InvocationFieldLink(
                                from_node_id = node_id,
                                from_field   = field_name,
                                to_field     = field_name))
            else:
                if is_field_compatible(from_node, link.from_field, invocation, link.to_field):
                    final_links.append(InvocationFieldLink(
                        from_node_id = node_id,
                        from_field   = link.from_field,
                        to_field     = link.to_field))

        # Ensure the new invocation won't create a cycle
        if self._is_graph_addition_valid(invocation, final_links):
            self.invocations[invocation.id] = invocation
            self.links[invocation.id] = final_links
            self._change_callback(self)
        else:
            raise ContextConflict("New links would create a cycle")



    def _create_graph(self, exclude_completed: bool = False):
        graph_nodes: Dict[str, set] = dict()
        for node_id in self.invocations:
            if exclude_completed and node_id in self.history:
                continue

            graph_nodes[node_id] = set()
            if node_id in self.links:
                for link in self.links[node_id]:
                    graph_nodes[node_id].add(link.from_node_id)

        return graph_nodes

    
    def _get_next_invocation_id(self) -> str:
        graph_nodes = self._create_graph(exclude_completed = True)
        graph = TopologicalSorter(graph_nodes)
        order = graph.static_order()
        # Need to filter out any inferred nodes from the TopologicalSorter
        next_id = next((n for n in order if n not in self.history), None)
        return next_id


    def _is_graph_addition_valid(self, new_invocation: BaseInvocation, new_links: List[InvocationFieldLink]) -> True:
        graph_nodes: Dict[str, set] = self._create_graph()

        # Add new node
        graph_nodes[new_invocation.id] = set()
        if new_links:
            for link in new_links:
                graph_nodes[new_invocation.id].add(link.from_node_id)
        
        graph = TopologicalSorter(graph_nodes)

        try:
            graph.prepare()
            return True
        except CycleError:
            return False


    def wait_for_all(self) -> None:
        """Wait for all ready invocations to complete"""
        for invocation_id in self.invocations:
            if invocation_id not in self.history:
                self.wait_for_invocation(invocation_id)


    def wait_for_invocation(self, invocation_id: str) -> None:
        """Wait for a specific invocation to complete"""
        if invocation_id in self.invocation_results:
            return

        if not invocation_id in self._wait_events:
            self._wait_events[invocation_id] = Event()

            # Check if complete again, in case completion happened during event creation
            if invocation_id in self.invocation_results:
                self._wait_events[invocation_id].set()
        
        # Wait for completion
        self._wait_events[invocation_id].wait()

        # Delete the event, since all waiters have a reference, and we can remove it otherwise
        del self._wait_events[invocation_id]


    def get_output(self, invocation_id: str, field_name: str) -> Any:
        return getattr(self.invocation_results[invocation_id].outputs, field_name)
    
    
    def _complete_invocation(self, invocation: BaseInvocation, outputs: BaseInvocationOutput):
        self.invocation_results[invocation.id] = InvocationHistoryEntry(
            invocation = invocation,
            outputs = outputs)
        self.history.append(invocation.id)

        self._change_callback(self)

        # Unblock any waiting threads
        if self._wait_events and invocation.id in self._wait_events:
            self._wait_events[invocation.id].set()
    

    def _map_inputs(self, invocation_id: str):
        """Map all current outputs to the specified invocation's inputs"""
        invocation = self.invocations[invocation_id]
        if invocation_id in self.links:
            for link in self.links[invocation_id]:
                output_value = self.get_output(link.from_node_id, link.from_field)
                setattr(invocation, link.to_field, output_value)


    def _map_outputs(self, invocation: BaseInvocation, links: List[InvocationFieldLink]):
        """Assign previous outputs to the given node.
        Negative node_id in a link will load from history if available,
        with -1 representing the newest node in history.
        * as from_field will load all matching fields from the previous node's output
        (ignoring to_field).
        """
        for link in links:
            node_id = link.from_node_id
            # Get the results from history
            if type(node_id) is int and node_id < 0:
                if abs(node_id) > len(self.history):
                    raise IndexError()
                else:
                    node_id = self.history[len(self.history) + node_id]
            
            result = self.invocation_results[node_id]

            if link.from_field == '*':
                output_fields = result.outputs.__fields__
                to_fields = invocation.__fields__
                for field_name in output_fields:
                    if field_name in to_fields:
                        if is_field_compatible(result.invocation, field_name, invocation, field_name):
                            output_value = self.get_output(node_id, field_name)
                            setattr(invocation, field_name, output_value)

            else:
                if is_field_compatible(result.invocation, link.from_field, invocation, link.to_field):
                    output_value = self.get_output(node_id, link.from_field)
                    setattr(invocation, link.to_field, output_value)
