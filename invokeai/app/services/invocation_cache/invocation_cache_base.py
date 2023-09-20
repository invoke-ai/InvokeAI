from abc import ABC, abstractmethod
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput


class InvocationCacheBase(ABC):
    """
    Base class for invocation caches.
    When an invocation is executed, it is hashed and its output stored in the cache.
    When new invocations are executed, if they are flagged with `use_cache`, they
    will attempt to pull their value from the cache before executing.

    Implementations should register for the `on_deleted` event of the `images` and `latents`
    services, and delete any cached outputs that reference the deleted image or latent.

    See the memory implementation for an example.

    Implementations should respect the `node_cache_size` configuration value, and skip all
    cache logic if the value is set to 0.
    """

    @abstractmethod
    def get(self, key: Union[int, str]) -> Optional[BaseInvocationOutput]:
        """Retrieves an invocation output from the cache"""
        pass

    @abstractmethod
    def save(self, key: Union[int, str], invocation_output: BaseInvocationOutput) -> None:
        """Stores an invocation output in the cache"""
        pass

    @abstractmethod
    def delete(self, key: Union[int, str]) -> None:
        """Deleteds an invocation output from the cache"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears the cache"""
        pass

    @abstractmethod
    def create_key(self, invocation: BaseInvocation) -> int:
        """Gets the key for the invocation's cache item"""
        pass
