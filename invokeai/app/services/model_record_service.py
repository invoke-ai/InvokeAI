# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

import shutil
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import Field
from typing import List, Optional, Union

from invokeai.backend.model_manager.storage import (
    AnyModelConfig,
    ModelConfigStore,
    ModelConfigStoreSQL,
    ModelConfigStoreYAML,
)
from invokeai.backend.model_manager import (
    ModelConfigBase,
    BaseModelType,
    ModelType,
    DuplicateModelException,
    UnknownModelException,
)


class ModelRecordServiceBase(ABC):
    """Responsible for managing model configuration records."""

    @abstractmethod
    def get_model(self, key: str) -> AnyModelConfig:
        """
        Retrieve the configuration for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        pass

    @abstractmethod
    def model_exists(
        self,
        key: str,
    ) -> bool:
        """Return true if the model configuration identified by key exists in the database."""
        pass

    @abstractmethod
    def model_info(self, key: str) -> ModelConfigBase:
        """
        Given a model name returns a dict-like (OmegaConf) object describing it.
        Uses the exact format as the omegaconf stanza.
        """
        pass

    @abstractmethod
    def list_models(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
    ) -> List[ModelConfigBase]:
        """
        Return a list of ModelConfigBases that match the base, type and name criteria.
        :param base_model: Filter by the base model type.
        :param model_type: Filter by the model type.
        :param model_name: Filter by the model name.
        """
        pass

    @abstractmethod
    def model_info_by_name(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> ModelConfigBase:
        """
        Return information about the model using the same format as list_models().

        If there are more than one model that match, raises a DuplicateModelException.
        If no model matches, raises an UnknownModelException
        """
        pass
        model_configs = self.list_models(model_name=model_name, base_model=base_model, model_type=model_type)
        if len(model_configs) > 1:
            raise DuplicateModelException(
                "More than one model share the same name and type: {base_model}/{model_type}/{model_name}"
            )
        if len(model_configs) == 0:
            raise UnknownModelException("No known model with name and type: {base_model}/{model_type}/{model_name}")
        return model_configs[0]

    @abstractmethod
    def all_models(self) -> List[ModelConfigBase]:
        """Return a list of all the models."""
        pass
        return self.list_models()

    @abstractmethod
    def add_model(self, key: str, config: Union[dict, ModelConfigBase]) -> None:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfig exceptions.
        """
        pass

    @abstractmethod
    def update_model(
        self,
        key: str,
        new_config: Union[dict, ModelConfigBase],
    ) -> ModelConfigBase:
        """
        Update the named model with a dictionary of attributes.

        Will fail with a
        UnknownModelException if the name does not already exist.

        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model key is unknown.
        """
        pass

    @abstractmethod
    def del_model(self, key: str, delete_files: bool = False):
        """
        Delete the named model from configuration. If delete_files
        is true, then the underlying file or directory will be
        deleted as well.
        """
        pass

    def rename_model(
        self,
        key: str,
        new_name: str,
    ) -> ModelConfigBase:
        """
        Rename the indicated model.
        """
        return self.update_model(key, {"name": new_name})


# implementation base class
class ModelRecordService(ModelRecordServiceBase):
    """Responsible for managing models on disk and in memory."""

    _store: ModelConfigStore = Field(description="Config record storage backend")

    @abstractmethod
    def __init__(self):
        """Initialize object -- abstract method."""
        pass

    def get_model(
        self,
        key: str,
    ) -> AnyModelConfig:
        """
        Retrieve the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        return self._store.get_model(key)

    def model_exists(
        self,
        key: str,
    ) -> bool:
        """
        Verify that a model with the given key exists.

        Given a model key, returns True if it is a valid
        identifier.
        """
        return self._store.exists(key)

    def model_info(self, key: str) -> ModelConfigBase:
        """
        Return configuration information about a model.

        Given a model key returns the ModelConfigBase describing it.
        """
        return self._store.get_model(key)

    def list_models(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
    ) -> List[ModelConfigBase]:
        """
        Return a ModelConfigBase object for each model in the database.
        """
        return self._store.search_by_name(model_name=model_name, base_model=base_model, model_type=model_type)

    def model_info_by_name(self, model_name: str, base_model: BaseModelType, model_type: ModelType) -> ModelConfigBase:
        """
        Return information about the model using the same format as list_models().

        If there are more than one model that match, raises a DuplicateModelException.
        If no model matches, raises an UnknownModelException
        """
        model_configs = self.list_models(model_name=model_name, base_model=base_model, model_type=model_type)
        if len(model_configs) > 1:
            raise DuplicateModelException(
                "More than one model share the same name and type: {base_model}/{model_type}/{model_name}"
            )
        if len(model_configs) == 0:
            raise UnknownModelException("No known model with name and type: {base_model}/{model_type}/{model_name}")
        return model_configs[0]

    def all_models(self) -> List[ModelConfigBase]:
        """Return a list of all the models."""
        return self.list_models()

    def add_model(self, key: str, config: Union[dict, ModelConfigBase]) -> None:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfig exceptions.
        """
        self._store.add_model(key, config)

    def update_model(
        self,
        key: str,
        new_config: Union[dict, ModelConfigBase],
    ) -> ModelConfigBase:
        """
        Update the named model with a dictionary of attributes.

        Will fail with a
        UnknownModelException if the name does not already exist.

        On a successful update, the config will be changed in memory. Will fail
        with an assertion error if provided attributes are incorrect or
        the model key is unknown.
        """
        new_info = self._store.update_model(key, new_config)
        print('FIX ME!!! need to call sync_model_path() somewhere... maybe router?')
        # self._loader.installer.sync_model_path(new_info.key)  Maybe this goes into the router call?
        return new_info

    def del_model(
        self,
        key: str,
    ):
        """
        Delete the named model from configuration.
        """
        model_info = self.model_info(key)
        self._store.del_model(key)

    def rename_model(
        self,
        key: str,
        new_name: str,
    ):
        """
        Rename the indicated model to the new name.

        :param key: Unique key for the model.
        :param new_name: New name for the model
        """
        return self.update_model(key, {"name": new_name})


class ModelRecordServiceSQL(ModelRecordService):
    """ModelRecordService that uses Sqlite for its backend."""

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock):
        """
        Initialize a ModelRecordService that uses a SQLITE3 database backend.

        :param conn: sqlite3 Connection object
        :param lock: Thread lock object
        """
        self._store = ModelConfigStoreSQL(conn, lock)


class ModelRecordServiceFile(ModelRecordService):
    """ModelRecordService that uses a YAML file for its backend."""

    def __init__(self, models_file: Path):
        """
        Initialize a ModelRecordService that uses a YAML file as the backend.

        :param models_file: Path to the YAML file backend.
        """
        self._store = ModelConfigStoreYAML(models_file)
