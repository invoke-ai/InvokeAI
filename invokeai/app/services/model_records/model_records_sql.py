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

from invokeai.app.services.model_records.model_records_base import (
    DuplicateModelException,
    ModelRecordChanges,
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    ModelSummary,
    UnknownModelException,
)
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelConfigFactory,
    ModelFormat,
    ModelType,
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
        self._cursor = db.conn.cursor()
        self._logger = logger

    @property
    def db(self) -> SqliteDatabase:
        """Return the underlying database."""
        return self._db

    def add_model(self, config: AnyModelConfig) -> AnyModelConfig:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfigException exceptions.
        """
        with self._db.lock:
            try:
                self._cursor.execute(
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
                self._db.conn.commit()

            except sqlite3.IntegrityError as e:
                self._db.conn.rollback()
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
            except sqlite3.Error as e:
                self._db.conn.rollback()
                raise e

        return self.get_model(config.key)

    def del_model(self, key: str) -> None:
        """
        Delete a model.

        :param key: Unique key for the model to be deleted

        Can raise an UnknownModelException
        """
        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    DELETE FROM models
                    WHERE id=?;
                    """,
                    (key,),
                )
                if self._cursor.rowcount == 0:
                    raise UnknownModelException("model not found")
                self._db.conn.commit()
            except sqlite3.Error as e:
                self._db.conn.rollback()
                raise e

    def update_model(self, key: str, changes: ModelRecordChanges) -> AnyModelConfig:
        record = self.get_model(key)

        # Model configs use pydantic's `validate_assignment`, so each change is validated by pydantic.
        for field_name in changes.model_fields_set:
            setattr(record, field_name, getattr(changes, field_name))

        json_serialized = record.model_dump_json()

        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    UPDATE models
                    SET
                        config=?
                    WHERE id=?;
                    """,
                    (json_serialized, key),
                )
                if self._cursor.rowcount == 0:
                    raise UnknownModelException("model not found")
                self._db.conn.commit()
            except sqlite3.Error as e:
                self._db.conn.rollback()
                raise e

        return self.get_model(key)

    def get_model(self, key: str) -> AnyModelConfig:
        """
        Retrieve the ModelConfigBase instance for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config, strftime('%s',updated_at) FROM models
                WHERE id=?;
                """,
                (key,),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise UnknownModelException("model not found")
            model = ModelConfigFactory.make_config(json.loads(rows[0]), timestamp=rows[1])
        return model

    def get_model_by_hash(self, hash: str) -> AnyModelConfig:
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config, strftime('%s',updated_at) FROM models
                WHERE hash=?;
                """,
                (hash,),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise UnknownModelException("model not found")
            model = ModelConfigFactory.make_config(json.loads(rows[0]), timestamp=rows[1])
        return model

    def exists(self, key: str) -> bool:
        """
        Return True if a model with the indicated key exists in the databse.

        :param key: Unique key for the model to be deleted
        """
        count = 0
        with self._db.lock:
            self._cursor.execute(
                """--sql
                select count(*) FROM models
                WHERE id=?;
                """,
                (key,),
            )
            count = self._cursor.fetchone()[0]
        return count > 0

    def search_by_attr(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
        model_format: Optional[ModelFormat] = None,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
    ) -> List[AnyModelConfig]:
        """
        Return models matching name, base and/or type.

        :param model_name: Filter by name of model (optional)
        :param base_model: Filter by base model (optional)
        :param model_type: Filter by type of model (optional)
        :param model_format: Filter by model format (e.g. "diffusers") (optional)
        :param order_by: Result order

        If none of the optional filters are passed, will return all
        models in the database.
        """

        assert isinstance(order_by, ModelRecordOrderBy)
        ordering = {
            ModelRecordOrderBy.Default: "type, base, name, format",
            ModelRecordOrderBy.Type: "type",
            ModelRecordOrderBy.Base: "base",
            ModelRecordOrderBy.Name: "name",
            ModelRecordOrderBy.Format: "format",
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
        with self._db.lock:
            self._cursor.execute(
                f"""--sql
                SELECT config, strftime('%s',updated_at)
                FROM models
                {where}
                ORDER BY {ordering[order_by]} -- using ? to bind doesn't work here for some reason;
                """,
                tuple(bindings),
            )
            result = self._cursor.fetchall()

        # Parse the model configs.
        results: list[AnyModelConfig] = []
        for row in result:
            try:
                model_config = ModelConfigFactory.make_config(json.loads(row[0]), timestamp=row[1])
            except pydantic.ValidationError:
                # We catch this error so that the app can still run if there are invalid model configs in the database.
                # One reason that an invalid model config might be in the database is if someone had to rollback from a
                # newer version of the app that added a new model type.
                self._logger.warning(f"Found an invalid model config in the database. Ignoring this model. ({row[0]})")
            else:
                results.append(model_config)

        return results

    def search_by_path(self, path: Union[str, Path]) -> List[AnyModelConfig]:
        """Return models with the indicated path."""
        results = []
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config, strftime('%s',updated_at) FROM models
                WHERE path=?;
                """,
                (str(path),),
            )
            results = [
                ModelConfigFactory.make_config(json.loads(x[0]), timestamp=x[1]) for x in self._cursor.fetchall()
            ]
        return results

    def search_by_hash(self, hash: str) -> List[AnyModelConfig]:
        """Return models with the indicated hash."""
        results = []
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config, strftime('%s',updated_at) FROM models
                WHERE hash=?;
                """,
                (hash,),
            )
            results = [
                ModelConfigFactory.make_config(json.loads(x[0]), timestamp=x[1]) for x in self._cursor.fetchall()
            ]
        return results

    def list_models(
        self, page: int = 0, per_page: int = 10, order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default
    ) -> PaginatedResults[ModelSummary]:
        """Return a paginated summary listing of each model in the database."""
        assert isinstance(order_by, ModelRecordOrderBy)
        ordering = {
            ModelRecordOrderBy.Default: "type, base, name, format",
            ModelRecordOrderBy.Type: "type",
            ModelRecordOrderBy.Base: "base",
            ModelRecordOrderBy.Name: "name",
            ModelRecordOrderBy.Format: "format",
        }

        # Lock so that the database isn't updated while we're doing the two queries.
        with self._db.lock:
            # query1: get the total number of model configs
            self._cursor.execute(
                """--sql
                select count(*) from models;
                """,
                (),
            )
            total = int(self._cursor.fetchone()[0])

            # query2: fetch key fields
            self._cursor.execute(
                f"""--sql
                SELECT config
                FROM models
                ORDER BY {ordering[order_by]} -- using ? to bind doesn't work here for some reason
                LIMIT ?
                OFFSET ?;
                """,
                (
                    per_page,
                    page * per_page,
                ),
            )
            rows = self._cursor.fetchall()
            items = [ModelSummary.model_validate(dict(x)) for x in rows]
            return PaginatedResults(
                page=page, pages=ceil(total / per_page), per_page=per_page, total=total, items=items
            )
