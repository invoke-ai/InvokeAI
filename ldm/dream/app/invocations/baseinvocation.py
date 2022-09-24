# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
from typing import Any, Dict, List, get_args, get_type_hints
from pydantic import BaseModel, Field

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


class InvocationContext():
    """The invocation context."""

    # Invocation history
    invocation_results: Dict[str, InvocationHistoryEntry]
    history: List[str]

    # Services used by invocations
    generate: Generate

    def __init__(self, generate: Generate):
        self.invocation_results = dict()
        self.history = list()
        self.generate = generate
    
    def add_history_entry(self, invocation: 'BaseInvocation', outputs: BaseInvocationOutput):
        self.invocation_results[invocation.id] = InvocationHistoryEntry(invocation, outputs)
        self.history.append(invocation.id)
    
    def get_output(self, invocation_id: str, field_name: str) -> Any:
        return getattr(self.invocation_results[invocation_id].outputs, field_name)


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
    def invoke(self, context: InvocationContext) -> BaseInvocationOutput:
        """Invoke with provided arguments and return outputs.
        **kwargs should be replaced with specific arguments on deriving classes.
        Deriving classes should additionally provide an InvocationSchema to define
        inputs and outputs.
        """
        pass

    id: str = Field(description="The id of this node. Must be unique among all nodes.")
