from abc import ABC, abstractmethod


class ClientStatePersistenceABC(ABC):
    """
    Base class for client persistence implementations.
    This class defines the interface for persisting client data.
    """

    @abstractmethod
    def set_by_key(self, queue_id: str, key: str, value: str) -> str:
        """
        Set a key-value pair for the client.

        Args:
            key (str): The key to set.
            value (str): The value to set for the key.

        Returns:
            str: The value that was set.
        """
        pass

    @abstractmethod
    def get_by_key(self, queue_id: str, key: str) -> str | None:
        """
        Get the value for a specific key of the client.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            str | None: The value associated with the key, or None if the key does not exist.
        """
        pass

    @abstractmethod
    def delete(self, queue_id: str) -> None:
        """
        Delete all client state.
        """
        pass
