from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_relationships.model_relationships_base import ModelRelationshipsServiceABC
from invokeai.backend.model_manager.config import AnyModelConfig


class ModelRelationshipsService(ModelRelationshipsServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def add_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        self.__invoker.services.model_relationship_records.add_model_relationship(model_key_1, model_key_2)

    def remove_model_relationship(self, model_key_1: str, model_key_2: str) -> None:
        self.__invoker.services.model_relationship_records.remove_model_relationship(model_key_1, model_key_2)

    def get_related_model_keys(self, model_key: str) -> list[str]:
        return self.__invoker.services.model_relationship_records.get_related_model_keys(model_key)

    def add_relationship_from_models(self, model_1: AnyModelConfig, model_2: AnyModelConfig) -> None:
        self.add_model_relationship(model_1.key, model_2.key)

    def remove_relationship_from_models(self, model_1: AnyModelConfig, model_2: AnyModelConfig) -> None:
        self.remove_model_relationship(model_1.key, model_2.key)

    def get_related_keys_from_model(self, model: AnyModelConfig) -> list[str]:
        return self.get_related_model_keys(model.key)

    def get_related_model_keys_batch(self, model_keys: list[str]) -> list[str]:
        return self.__invoker.services.model_relationship_records.get_related_model_keys_batch(model_keys)
