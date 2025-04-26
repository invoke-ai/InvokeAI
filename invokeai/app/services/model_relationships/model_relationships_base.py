from abc import ABC, abstractmethod

from invokeai.backend.model_manager.config import AnyModelConfig


class ModelRelationshipsServiceABC(ABC):
    """High-level service for managing model-to-model relationships."""

    @abstractmethod
    def add_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        """Creates a relationship between two models keys."""
        pass

    @abstractmethod
    def remove_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        """Removes a relationship between two models keys."""
        pass

    @abstractmethod
    def get_related_model_keys(self, model_key: str) -> list[str]:
        """Gets all models keys related to a given model key."""
        pass
    
    @abstractmethod
    def get_related_model_keys_batch(self, model_keys: list[str]) -> list[str]:
        """Get related model keys for multiple models."""
        pass
    
    @abstractmethod
    def add_relationship_from_models(self, model_1: AnyModelConfig, model_2: AnyModelConfig) -> None:
        """Creates a relationship from model objects."""
        pass

    @abstractmethod
    def remove_relationship_from_models(self, model_1: AnyModelConfig, model_2: AnyModelConfig) -> None:
        """Removes a relationship from model objects."""
        pass

    @abstractmethod
    def get_related_keys_from_model(self, model: AnyModelConfig) -> list[str]:
        """Gets all model keys related to a given model object."""
        pass