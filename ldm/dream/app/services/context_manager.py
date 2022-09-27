from abc import ABC, abstractmethod
from base64 import urlsafe_b64encode
from typing import Dict, Union
from uuid import uuid4
from .invocation_context import InvocationContext


class ContextManagerABC(ABC):
    """Base context manager class"""
    @abstractmethod
    def get(self, context_id: str) -> Union[InvocationContext,None]:
        pass

    @abstractmethod
    def set(self, context: InvocationContext) -> None:
        pass

    @abstractmethod
    def create(self) -> InvocationContext:
        pass


class MemoryContextManager(ContextManagerABC):
    """An in-memory context manager"""
    __contexts: Dict[str, InvocationContext]

    def __init__(self):
        self.__contexts = dict()

    def get(self, context_id: str) -> Union[InvocationContext,None]:
        return self.__contexts.get(context_id)
    
    def set(self, context: InvocationContext) -> None:
        self.__contexts[context.id] = context

    def create(self) -> InvocationContext:
        # TODO: consider using a provided id generator from services?
        context_id = urlsafe_b64encode(uuid4().bytes).decode("ascii")
        context = InvocationContext(context_id)
        self.set(context)
        return context
