# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Implementation of ModelConfigStore using a YAML file.

Typical usage:

  from invokeai.backend.model_manager.storage.yaml import ModelConfigStoreYAML
  store = ModelConfigStoreYAML("./configs/models.yaml")
  config = dict(
        path='/tmp/pokemon.bin',
        name='old name',
        base_model='sd-1',
        model_type='embedding',
        model_format='embedding_file',
        author='Anonymous',
        tags=['sfw','cartoon']
     )

   # adding - the key becomes the model's "key" field
   store.add_model('key1', config)

   # updating
   config.name='new name'
   store.update_model('key1', config)

   # checking for existence
   if store.exists('key1'):
      print("yes")

   # fetching config
   new_config = store.get_model('key1')
   print(new_config.name, new_config.base_model)
   assert new_config.key == 'key1'

  # deleting
  store.del_model('key1')

  # searching
  configs = store.search_by_tag({'sfw','oss license'})
  configs = store.search_by_name(base_model='sd-2', model_type='main')
"""

import threading
import yaml
from enum import Enum
from pathlib import Path
from typing import Union, Set, List, Optional
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from ..config import (
    ModelConfigBase,
    ModelConfigFactory,
    BaseModelType,
    ModelType,
)

from .base import (
    DuplicateModelException,
    UnknownModelException,
    ModelConfigStore,
    CONFIG_FILE_VERSION,
)


class ModelConfigStoreYAML(ModelConfigStore):
    """Implementation of the ModelConfigStore ABC using a YAML file."""

    _filename: Path
    _config: DictConfig
    _lock: threading.Lock

    def __init__(self, config_file: Path):
        """Initialize ModelConfigStore object with a .yaml file."""
        super().__init__()
        self._filename = Path(config_file).absolute()  # don't let chdir mess us up!
        self._lock = threading.RLock()
        if not self._filename.exists():
            self._initialize_yaml()
        self._config = OmegaConf.load(self._filename)
        assert (
            self.version == CONFIG_FILE_VERSION
        ), f"Model config version {self.version} does not match expected version {CONFIG_FILE_VERSION}"

    def _initialize_yaml(self):
        try:
            self._lock.acquire()
            self._filename.parent.mkdir(parents=True, exist_ok=True)
            with open(self._filename, "w") as yaml_file:
                yaml_file.write(yaml.dump({"__metadata__": {"version": CONFIG_FILE_VERSION}}))
        finally:
            self._lock.release()

    def _commit(self):
        try:
            self._lock.acquire()
            newfile = Path(str(self._filename) + ".new")
            yaml_str = OmegaConf.to_yaml(self._config)
            with open(newfile, "w", encoding="utf-8") as outfile:
                outfile.write(yaml_str)
            newfile.replace(self._filename)
        finally:
            self._lock.release()

    @property
    def version(self) -> str:
        """Return version of this config file/database."""
        return self._config["__metadata__"].get("version")

    def add_model(self, key: str, config: Union[dict, ModelConfigBase]) -> None:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfig exceptions.
        """
        record = ModelConfigFactory.make_config(config, key)  # ensure it is a valid config obect
        dict_fields = record.dict()  # and back to a dict with valid fields
        try:
            self._lock.acquire()
            if key in self._config:
                existing_model = self.get_model(key)
                raise DuplicateModelException(
                    f"Can't save {record.name} because a model named '{existing_model.name}' is already stored with the same key '{key}'"
                )
            self._config[key] = self._fix_enums(dict_fields)
            self._commit()
        finally:
            self._lock.release()

    def _fix_enums(self, original: dict) -> dict:
        """In python 3.9, omegaconf stores incorrectly stringified enums"""
        fixed_dict = {}
        for key, value in original.items():
            fixed_dict[key] = value.value if isinstance(value, Enum) else value
        return fixed_dict

    def del_model(self, key: str) -> None:
        """
        Delete a model.

        :param key: Unique key for the model to be deleted

        Can raise an UnknownModelException
        """
        try:
            self._lock.acquire()
            if key not in self._config:
                raise UnknownModelException(f"Unknown key '{key}' for model config")
            self._config.pop(key)
            self._commit()
        finally:
            self._lock.release()

    def update_model(self, key: str, config: Union[dict, ModelConfigBase]) -> ModelConfigBase:
        """
        Update the model, returning the updated version.

        :param key: Unique key for the model to be updated
        :param config: Model configuration record. Either a dict with the
         required fields, or a ModelConfigBase instance.
        """
        record = ModelConfigFactory.make_config(config, key)  # ensure it is a valid config obect
        dict_fields = record.dict()  # and back to a dict with valid fields
        try:
            self._lock.acquire()
            if key not in self._config:
                raise UnknownModelException(f"Unknown key '{key}' for model config")
            self._config[key] = self._fix_enums(dict_fields)
            self._commit()
        finally:
            self._lock.release()

    def get_model(self, key: str) -> ModelConfigBase:
        """
        Retrieve the ModelConfigBase instance for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        try:
            record = self._config[key]
            return ModelConfigFactory.make_config(record, key)
        except KeyError as e:
            raise UnknownModelException(f"Unknown key '{key}' for model config") from e

    def exists(self, key: str) -> bool:
        """
        Return True if a model with the indicated key exists in the databse.

        :param key: Unique key for the model to be deleted
        """
        return key in self._config

    def search_by_tag(self, tags: Set[str]) -> List[ModelConfigBase]:
        """
        Return models containing all of the listed tags.

        :param tags: Set of tags to search on.
        """
        results = []
        tags = set(tags)
        try:
            self._lock.acquire()
            for config in self.all_models():
                config_tags = set(config.tags)
                if tags.difference(config_tags):  # not all tags in the model
                    continue
                results.append(config)
        finally:
            self._lock.release()
        return results

    def search_by_name(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
    ) -> List[ModelConfigBase]:
        """
        Return models matching name, base and/or type.

        :param model_name: Filter by name of model (optional)
        :param base_model: Filter by base model (optional)
        :param model_type: Filter by type of model (optional)

        If none of the optional filters are passed, will return all
        models in the database.
        """
        results = []
        try:
            self._lock.acquire()
            for key, record in self._config.items():
                if key == "__metadata__":
                    continue
                model = ModelConfigFactory.make_config(record, key)
                if model_name and model.name != model_name:
                    continue
                if base_model and model.base_model != base_model:
                    continue
                if model_type and model.model_type != model_type:
                    continue
                results.append(model)
        finally:
            self._lock.release()
        return results
