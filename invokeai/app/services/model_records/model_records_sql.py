"""
SQL implementation of the ModelRecordServiceBase API, on the database layer.

Typical usage:

  from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceSQL
  store = ModelRecordServiceSQL(database, logger)

  # adding - the config's key becomes the record's key
  store.add_model(config)

  # updating
  store.update_model(config.key, ModelRecordChanges(name="new name"))

  # checking for existence
  if store.exists(config.key):
      print("yes")

  # fetching a config
  config = store.get_model(config.key)

  # deleting
  store.del_model(config.key)

  # searching
  configs = store.search_by_path("/tmp/pokemon.bin")
  configs = store.search_by_hash("750a499f35e43b7e1b4d15c207aa2f01")
  configs = store.search_by_attr(base_model=BaseModelType.StableDiffusion2, model_type=ModelType.Main)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import pydantic
from pydantic import ValidationError

from invokeai.app.services.model_records.model_records_base import (
    DuplicateModelException,
    ModelRecordChanges,
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    UnknownModelException,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import UniqueViolation
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.models import MAX_KEY_LENGTH, MAX_PATH_LENGTH
from invokeai.app.services.shared.pagination import SQLiteDirection
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


def _construct_config_for_type(fields: dict, target_type: ModelType) -> AnyModelConfig:
    """Try every config class whose `type` default matches `target_type` and return the first that validates.

    Used when changing a model's type via the update endpoint: the existing record's `format`/`variant`
    fields belong to the old class and may not have a discriminator match in the new type space, so we
    fall back to constructing each candidate class directly with whatever fields it accepts.
    """
    last_error: Exception | None = None
    for candidate_class in Config_Base.CONFIG_CLASSES:
        type_field = candidate_class.model_fields.get("type")
        if type_field is None or type_field.default != target_type:
            continue
        try:
            return candidate_class(**fields)  # type: ignore[return-value]
        except ValidationError as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise ValidationError.from_exception_data(
        f"No model config class found for type={target_type!r}",
        line_errors=[],
    )


def _check_path_length(path: str) -> None:
    if len(path) > MAX_PATH_LENGTH:
        raise ValueError(f"A model's path can be at most {MAX_PATH_LENGTH} characters long")


def _parse(config: str) -> AnyModelConfig:
    return ModelConfigFactory.from_dict(json.loads(config))


class ModelRecordServiceSQL(ModelRecordServiceBase):
    """Implementation of the ModelConfigStore ABC using a SQL database."""

    def __init__(self, database: Database, logger: logging.Logger):
        super().__init__()
        self._queries = database.queries
        self._logger = logger

    def add_model(self, config: AnyModelConfig) -> AnyModelConfig:
        """
        Add a model to the database.

        :param config: Model configuration record; its key becomes the record's key.

        Raises DuplicateModelException when a model with the same path or key is installed, and ValueError when
        its key or path is longer than a record holds.
        """
        if len(config.key) > MAX_KEY_LENGTH:
            raise ValueError(f"A model's key can be at most {MAX_KEY_LENGTH} characters long")
        _check_path_length(config.path)
        stored = config.model_dump_json()
        try:
            self._queries.models.insert(config.key, stored)
        except UniqueViolation as error:
            if self._queries.models.at_path(str(config.path)):
                raise DuplicateModelException(f"A model with path '{config.path}' is already installed") from error
            raise DuplicateModelException(f"A model with key '{config.key}' is already installed") from error
        return _parse(stored)

    def del_model(self, key: str) -> None:
        """
        Delete a model.

        :param key: Unique key for the model to be deleted

        Can raise an UnknownModelException
        """
        if not self._queries.models.delete(key):
            raise UnknownModelException("model not found")

    def update_model(self, key: str, changes: ModelRecordChanges, allow_class_change: bool = False) -> AnyModelConfig:
        def apply(q: Queries) -> str:
            # Locked, so that a concurrent update applies its changes to this one's result, not beside it.
            stored = q.models.lock(key)
            if stored is None:
                raise UnknownModelException("model not found")
            record = _parse(stored)

            if allow_class_change:
                # The changes may cause the model config class to change. To handle this, we need to construct the new
                # class from scratch rather than trying to modify the existing instance in place.
                #
                # 1. Convert the existing record to a dict
                # 2. Apply the changes to the dict
                # 3. Attempt to create a new model config from the updated dict

                # 1. Convert the existing record to a dict
                record_as_dict = record.model_dump()

                # 2. Apply the changes to the dict
                for field_name in changes.model_fields_set:
                    record_as_dict[field_name] = getattr(changes, field_name)

                # 3. Attempt to create a new model config from the updated dict.
                #
                # When the model type is being changed, the previous record's `format` and `variant` likely
                # belong to the old config class and won't validate against the new one (e.g. switching a
                # Qwen3 encoder to a Text LLM keeps format=qwen3_encoder, which has no matching discriminator
                # under text_llm). If the initial validation fails and the type changed, retry with stale
                # format/variant fields stripped so the new class can apply its own defaults.
                type_changed = "type" in changes.model_fields_set and changes.type != record.type
                try:
                    record = ModelConfigFactory.from_dict(record_as_dict)
                except ValidationError:
                    if not type_changed:
                        raise
                    fallback_dict = dict(record_as_dict)
                    for stale_field in ("format", "variant"):
                        if stale_field not in changes.model_fields_set:
                            fallback_dict.pop(stale_field, None)
                    record = _construct_config_for_type(fallback_dict, changes.type)
            else:
                # We are not allowing the model config class to change, so we can just update the existing instance in
                # place. If the changes are invalid for the existing class, an exception will be raised by pydantic.
                for field_name in changes.model_fields_set:
                    setattr(record, field_name, getattr(changes, field_name))

            # If we get this far, the updated model config is valid, so we can save it to the database.
            if "path" in changes.model_fields_set:
                _check_path_length(record.path)
            updated = record.model_dump_json()
            q.models.save(key, updated)
            return updated

        return _parse(self._queries.run(apply))

    def replace_model(self, key: str, new_config: AnyModelConfig) -> AnyModelConfig:
        if key != new_config.key:
            raise ValueError("key does not match new_config.key")
        stored = new_config.model_dump_json()
        if not self._queries.models.save(key, stored):
            raise UnknownModelException("model not found")
        return _parse(stored)

    def get_model(self, key: str) -> AnyModelConfig:
        """
        Retrieve the ModelConfigBase instance for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        stored = self._queries.models.get(key)
        if stored is None:
            raise UnknownModelException("model not found")
        return _parse(stored)

    def exists(self, key: str) -> bool:
        """
        Return True if a model with the indicated key exists in the databse.

        :param key: Unique key for the model to be deleted
        """
        return self._queries.models.exists(key)

    def search_by_attr(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
        model_format: Optional[ModelFormat] = None,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
        direction: SQLiteDirection = SQLiteDirection.Ascending,
    ) -> List[AnyModelConfig]:
        """
        Return models matching name, base and/or type.

        :param model_name: Filter by name of model (optional)
        :param base_model: Filter by base model (optional)
        :param model_type: Filter by type of model (optional)
        :param model_format: Filter by model format (e.g. "diffusers") (optional)
        :param order_by: Result order
        :param direction: Result direction

        If none of the optional filters are passed, will return all
        models in the database.
        """
        assert isinstance(order_by, ModelRecordOrderBy)
        configs = self._queries.models.search(
            name=model_name,
            base=base_model,
            model_type=model_type,
            model_format=model_format,
            order_by=order_by.value,
            descending=direction == SQLiteDirection.Descending,
        )

        # Parse the model configs.
        results: list[AnyModelConfig] = []
        for config in configs:
            try:
                model_config = _parse(config)
            except pydantic.ValidationError as e:
                # We catch this error so that the app can still run if there are invalid model configs in the database.
                # One reason that an invalid model config might be in the database is if someone had to rollback from a
                # newer version of the app that added a new model type.
                row_data = f"{config[:64]}..." if len(config) > 64 else config
                try:
                    name = json.loads(config).get("name", "<unknown>")
                except Exception:
                    name = "<unknown>"
                self._logger.warning(
                    f"Skipping invalid model config in the database with name {name}. Ignoring this model. ({row_data})"
                )
                self._logger.warning(f"Validation error: {e}")
            else:
                results.append(model_config)

        return results

    def search_by_path(self, path: Union[str, Path]) -> List[AnyModelConfig]:
        """Return models with the indicated path."""
        return [_parse(config) for config in self._queries.models.at_path(str(path))]

    def search_by_hash(self, hash: str) -> List[AnyModelConfig]:
        """Return models with the indicated hash."""
        return [_parse(config) for config in self._queries.models.with_hash(hash)]

    def get_model_paths(self) -> list[str]:
        return self._queries.models.paths()
