# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
SQL Implementation of the ModelRecordServiceBase API

Typical usage:

  from invokeai.backend.model_manager import ModelConfigStoreSQL
  store = ModelConfigStoreSQL(sqlite_db)
  config = dict(
        path='/tmp/pokemon.bin',
        name='old name',
        base_model='sd-1',
        type='embedding',
        format='embedding_file',
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
   print(new_config.name, new_config.base)
   assert new_config.key == 'key1'

  # deleting
  store.del_model('key1')

  # searching
  configs = store.search_by_path(path='/tmp/pokemon.bin')
  configs = store.search_by_hash('750a499f35e43b7e1b4d15c207aa2f01')
  configs = store.search_by_attr(base_model='sd-2', model_type='main')
"""

import json
import logging
import sqlite3
from math import ceil
from pathlib import Path
from typing import List, Optional, Union

import pydantic
from pydantic import ValidationError

from invokeai.app.services.model_records.model_records_base import (
    DuplicateModelException,
    ModelRecordChanges,
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    ModelSummary,
    UnknownModelException,
)
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
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


class ModelRecordServiceSQL(ModelRecordServiceBase):
    """Implementation of the ModelConfigStore ABC using a SQL database."""

    def __init__(self, db: SqliteDatabase, logger: logging.Logger):
        """
        Initialize a new object from preexisting sqlite3 connection and threading lock objects.

        :param db: Sqlite connection object
        """
        super().__init__()
        self._db = db
        self._logger = logger

    def add_model(self, config: AnyModelConfig) -> AnyModelConfig:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfigException exceptions.
        """
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    INSERT INTO models (
                        id,
                        config
                        )
                    VALUES (?,?);
                    """,
                    (
                        config.key,
                        config.model_dump_json(),
                    ),
                )

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    if "models.path" in str(e):
                        msg = f"A model with path '{config.path}' is already installed"
                    elif "models.name" in str(e):
                        msg = f"A model with name='{config.name}', type='{config.type}', base='{config.base}' is already installed"
                    else:
                        msg = f"A model with key '{config.key}' is already installed"
                    raise DuplicateModelException(msg) from e
                else:
                    raise e

        return self.get_model(config.key)

    def del_model(self, key: str) -> None:
        """
        Delete a model.

        :param key: Unique key for the model to be deleted

        Can raise an UnknownModelException
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM models
                WHERE id=?;
                """,
                (key,),
            )
            if cursor.rowcount == 0:
                raise UnknownModelException("model not found")

    def update_model(self, key: str, changes: ModelRecordChanges, allow_class_change: bool = False) -> AnyModelConfig:
        with self._db.transaction() as cursor:
            record = self.get_model(key)

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
                type_changed = (
                    "type" in changes.model_fields_set
                    and changes.type != record.type
                )
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

                # If we get this far, the updated model config is valid, so we can save it to the database.
                json_serialized = record.model_dump_json()
            else:
                # We are not allowing the model config class to change, so we can just update the existing instance in
                # place. If the changes are invalid for the existing class, an exception will be raised by pydantic.
                for field_name in changes.model_fields_set:
                    setattr(record, field_name, getattr(changes, field_name))
                json_serialized = record.model_dump_json()

            cursor.execute(
                """--sql
                UPDATE models
                SET
                    config=?
                WHERE id=?;
                """,
                (json_serialized, key),
            )
            if cursor.rowcount == 0:
                raise UnknownModelException("model not found")

        return self.get_model(key)

    def replace_model(self, key: str, new_config: AnyModelConfig) -> AnyModelConfig:
        if key != new_config.key:
            raise ValueError("key does not match new_config.key")
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE models
                SET
                    config=?
                WHERE id=?;
                """,
                (new_config.model_dump_json(), key),
            )
            if cursor.rowcount == 0:
                raise UnknownModelException("model not found")
        return self.get_model(key)

    def get_model(self, key: str) -> AnyModelConfig:
        """
        Retrieve the ModelConfigBase instance for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT config FROM models
                WHERE id=?;
                """,
                (key,),
            )
            rows = cursor.fetchone()
        if not rows:
            raise UnknownModelException("model not found")
        model = ModelConfigFactory.from_dict(json.loads(rows[0]))
        return model

    def get_model_by_hash(self, hash: str) -> AnyModelConfig:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT config FROM models
                WHERE hash=?;
                """,
                (hash,),
            )
            rows = cursor.fetchone()
        if not rows:
            raise UnknownModelException("model not found")
        model = ModelConfigFactory.from_dict(json.loads(rows[0]))
        return model

    def exists(self, key: str) -> bool:
        """
        Return True if a model with the indicated key exists in the databse.

        :param key: Unique key for the model to be deleted
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                select count(*) FROM models
                WHERE id=?;
                """,
                (key,),
            )
            count = cursor.fetchone()[0]
        return count > 0

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
        with self._db.transaction() as cursor:
            assert isinstance(order_by, ModelRecordOrderBy)
            order_dir = "DESC" if direction == SQLiteDirection.Descending else "ASC"
            ordering = {
                ModelRecordOrderBy.Default: f"type {order_dir}, base COLLATE NOCASE {order_dir}, name COLLATE NOCASE {order_dir}, format",
                ModelRecordOrderBy.Type: "type",
                ModelRecordOrderBy.Base: "base COLLATE NOCASE",
                ModelRecordOrderBy.Name: "name COLLATE NOCASE",
                ModelRecordOrderBy.Format: "format",
                ModelRecordOrderBy.Size: "IFNULL(json_extract(config, '$.file_size'), 0)",
                ModelRecordOrderBy.DateAdded: "created_at",
                ModelRecordOrderBy.DateModified: "updated_at",
                ModelRecordOrderBy.Path: "path",
            }

            where_clause: list[str] = []
            bindings: list[str] = []
            if model_name:
                where_clause.append("name=?")
                bindings.append(model_name)
            if base_model:
                where_clause.append("base=?")
                bindings.append(base_model)
            if model_type:
                where_clause.append("type=?")
                bindings.append(model_type)
            if model_format:
                where_clause.append("format=?")
                bindings.append(model_format)
            where = f"WHERE {' AND '.join(where_clause)}" if where_clause else ""

            cursor.execute(
                f"""--sql
                SELECT config
                FROM models
                {where}
                ORDER BY {ordering[order_by]} {order_dir} -- using ? to bind doesn't work here for some reason;
                """,
                tuple(bindings),
            )
            result = cursor.fetchall()

        # Parse the model configs.
        results: list[AnyModelConfig] = []
        for row in result:
            try:
                model_config = ModelConfigFactory.from_dict(json.loads(row[0]))
            except pydantic.ValidationError as e:
                # We catch this error so that the app can still run if there are invalid model configs in the database.
                # One reason that an invalid model config might be in the database is if someone had to rollback from a
                # newer version of the app that added a new model type.
                row_data = f"{row[0][:64]}..." if len(row[0]) > 64 else row[0]
                try:
                    name = json.loads(row[0]).get("name", "<unknown>")
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
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT config FROM models
                WHERE path=?;
                """,
                (str(path),),
            )
            results = [ModelConfigFactory.from_dict(json.loads(x[0])) for x in cursor.fetchall()]
        return results

    def search_by_hash(self, hash: str) -> List[AnyModelConfig]:
        """Return models with the indicated hash."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT config FROM models
                WHERE hash=?;
                """,
                (hash,),
            )
            results = [ModelConfigFactory.from_dict(json.loads(x[0])) for x in cursor.fetchall()]
        return results

    def list_models(
        self,
        page: int = 0,
        per_page: int = 10,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
        direction: SQLiteDirection = SQLiteDirection.Ascending,
    ) -> PaginatedResults[ModelSummary]:
        """Return a paginated summary listing of each model in the database."""
        with self._db.transaction() as cursor:
            assert isinstance(order_by, ModelRecordOrderBy)
            order_dir = "DESC" if direction == SQLiteDirection.Descending else "ASC"
            ordering = {
                ModelRecordOrderBy.Default: f"type {order_dir}, base COLLATE NOCASE {order_dir}, name COLLATE NOCASE {order_dir}, format",
                ModelRecordOrderBy.Type: "type",
                ModelRecordOrderBy.Base: "base COLLATE NOCASE",
                ModelRecordOrderBy.Name: "name COLLATE NOCASE",
                ModelRecordOrderBy.Format: "format",
                ModelRecordOrderBy.Size: "IFNULL(json_extract(config, '$.file_size'), 0)",
                ModelRecordOrderBy.DateAdded: "created_at",
                ModelRecordOrderBy.DateModified: "updated_at",
                ModelRecordOrderBy.Path: "path",
            }

            # Lock so that the database isn't updated while we're doing the two queries.
            # query1: get the total number of model configs
            cursor.execute(
                """--sql
                select count(*) from models;
                """,
                (),
            )
            total = int(cursor.fetchone()[0])

            # query2: fetch key fields
            cursor.execute(
                f"""--sql
                SELECT config
                FROM models
                ORDER BY {ordering[order_by]} {order_dir} -- using ? to bind doesn't work here for some reason
                LIMIT ?
                OFFSET ?;
                """,
                (
                    per_page,
                    page * per_page,
                ),
            )
            rows = cursor.fetchall()
        items = [ModelSummary.model_validate(dict(x)) for x in rows]
        return PaginatedResults(page=page, pages=ceil(total / per_page), per_page=per_page, total=total, items=items)
