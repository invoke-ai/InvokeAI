# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, get_args, get_type_hints
from pydantic import BaseModel, Field
from pydantic.fields import ModelField

from ldm.generate import Generate


class BaseInvocationOutput(BaseModel):
    """Base class for all invocation outputs"""
    ...


class InvocationHistoryEntry():
    """The context of an invoked node"""
    invocation: 'BaseInvocation'
    outputs: BaseInvocationOutput

    def __init__(self, invocation: 'BaseInvocation', outputs: BaseInvocationOutput):
        self.invocation = invocation
        self.outputs = outputs


class InvocationServices():
    generate: Generate

    def __init__(self,
        generate: Generate
    ):
        self.generate = generate


class InvocationFieldLink():
    node_id: Union[str,int]
    from_field: str
    to_field: str

    def __init__(self, node_id: Union[str,int], from_field: str, to_field: str):
        self.node_id = node_id
        self.from_field = from_field
        self.to_field = to_field


class BaseInvocation(ABC, BaseModel):
    """A node to process inputs and produce outputs.
    May use dependency injection in __init__ to receive providers.
    """
    @classmethod
    def get_all_subclasses(cls):
        subclasses = []
        toprocess = [cls]
        while len(toprocess) > 0:
            next = toprocess.pop(0)
            next_subclasses = next.__subclasses__()
            subclasses.extend(next_subclasses)
            toprocess.extend(next_subclasses)
        return subclasses

    @classmethod
    def get_invocations(cls):
        return tuple(BaseInvocation.get_all_subclasses())

    @classmethod
    def get_invocations_map(cls):
        # Get the type strings out of the literals and into a dictionary
        return dict(map(lambda t: (get_args(get_type_hints(t)['type'])[0], t),BaseInvocation.get_all_subclasses()))
        
    @abstractmethod
    def invoke(self, context: 'InvocationContext') -> BaseInvocationOutput:
        """Invoke with provided context and return outputs."""
        pass

    id: str = Field(description="The id of this node. Must be unique among all nodes.")


def is_field_compatible(
    from_invocation: BaseInvocation,
    from_field: str,
    to_invocation: BaseInvocation,
    to_field: str) -> bool:

    from_node_type = type(from_invocation)
    to_node_type = type(to_invocation)

    # Get field type hints
    from_node_outputs = get_type_hints(from_node_type.Outputs)
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


class InvocationContext():
    """The invocation context."""

    # Invocation history
    invocation_results: Dict[str, InvocationHistoryEntry]
    history: List[str]

    # Services used by invocations
    services: InvocationServices

    def __init__(self, services: InvocationServices):
        self.invocation_results = dict()
        self.history = list()
        self.services = services
    
    def add_history_entry(self, invocation: BaseInvocation, outputs: BaseInvocationOutput):
        self.invocation_results[invocation.id] = InvocationHistoryEntry(invocation, outputs)
        self.history.append(invocation.id)
    
    def get_output(self, invocation_id: str, field_name: str) -> Any:
        return getattr(self.invocation_results[invocation_id].outputs, field_name)
    
    def map_outputs(self, invocation: BaseInvocation, links: List[InvocationFieldLink]):
        """Assign previous outputs to the given node.
        Negative node_id in a link will load from history if available,
        with -1 representing the newest node in history.
        * as from_field will load all matching fields from the previous node's output
        (ignoring to_field).
        """
        for link in links:
            node_id = link.node_id
            # Get the results from history
            if type(node_id) is int and node_id < 0:
                if abs(node_id) >= len(self.history):
                    raise IndexError()
                else:
                    node_id = self.history[len(self.history) + node_id]
            
            result = self.invocation_results[node_id]

            if link.from_field == '*':
                output_fields = result.outputs.__fields__()
                to_fields = invocation.__fields__()
                for field_name in output_fields:
                    if to_fields.has_key(field_name):
                        if is_field_compatible(result.invocation, field_name, invocation, field_name):
                            output_value = self.get_output(node_id, field_name)
                            setattr(invocation, field_name, output_value)

            else:
                if is_field_compatible(result.invocation, link.from_field, invocation, link.to_field):
                    output_value = self.get_output(node_id, link.from_field)
                    setattr(invocation, link.to_field, output_value)
