from abc import ABC, abstractmethod


class ModelRelationshipRecordStorageBase(ABC):
    """Abstract base class for model-to-model relationship record storage."""

    @abstractmethod
    def add_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        """Creates a relationship between two models by keys."""
        pass

    @abstractmethod
    def remove_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        """Removes a relationship between two models by keys."""
        pass

    @abstractmethod
    def get_related_model_keys(self, model_key: str) -> list[str]:
        """Gets all models keys related to a given model key."""
        pass

    @abstractmethod
    def get_related_model_keys_batch(self, model_keys: list[str]) -> list[str]:
        """Get related model keys for multiple models given a list of keys."""
        pass
