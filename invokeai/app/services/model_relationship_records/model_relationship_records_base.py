from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from invokeai.backend.model_manager.config import AnyModelConfig

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
    
    @abstractmethod
    def get_related_model_key_count(self, model_key: str) -> int:
        """Gets the number of relations for a given model key."""
        pass

    """ Below are methods that use ModelConfigs instead of model keys, as convenience methods.
    These methods are not required to be implemented, but they are potentially useful for later development.
    They are not used in the current codebase."""

    @abstractmethod
    def add_relationship_from_models(self, model_1: "AnyModelConfig", model_2: "AnyModelConfig") -> None:
        """Creates a relationship between two models using ModelConfigs."""
        pass

    @abstractmethod
    def remove_relationship_from_models(self, model_1: "AnyModelConfig", model_2: "AnyModelConfig") -> None:
        """Removes a relationship between two models using ModelConfigs."""
        pass

    @abstractmethod
    def get_related_keys_from_model(self, model: "AnyModelConfig") -> list[str]:
        """Gets all model keys related to a given model using it's config."""
        pass
    
    @abstractmethod
    def get_related_model_key_count_from_model(self, model: "AnyModelConfig") -> int:
        """Gets the number of relations for a given model config."""
        pass