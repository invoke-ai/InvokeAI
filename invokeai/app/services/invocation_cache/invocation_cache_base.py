from abc import ABC, abstractmethod
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput


class InvocationCacheBase(ABC):
    """Base class for invocation caches."""

    @abstractmethod
    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        """Retrieves and invocation output from the cache"""
        pass

    @abstractmethod
    def save(self, value: BaseInvocationOutput) -> None:
        """Stores an invocation output in the cache"""
        pass

    @abstractmethod
    def delete(self, key: Union[int, str]) -> None:
        """Deleted an invocation output from the cache"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears the cache"""
        pass

    @classmethod
    @abstractmethod
    def create_key(cls, value: BaseInvocation) -> Union[int, str]:
        """Creates the cache key for an invocation"""
        pass
