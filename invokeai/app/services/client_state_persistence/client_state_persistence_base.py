from abc import ABC, abstractmethod

from pydantic import JsonValue


class ClientStatePersistenceABC(ABC):
    """
    Base class for client persistence implementations.
    This class defines the interface for persisting client data.
    """

    @abstractmethod
    def set_by_key(self, key: str, value: JsonValue) -> None:
        """
        Store the data for the client.

        :param data: The client data to be stored.
        """
        pass

    @abstractmethod
    def get_by_key(self, key: str) -> JsonValue | None:
        """
        Get the data for the client.

        :return: The client data.
        """
        pass

    @abstractmethod
    def delete(self) -> None:
        """
        Delete the data for the client.
        """
        pass
