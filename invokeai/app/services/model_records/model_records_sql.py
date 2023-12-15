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
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelConfigFactory,
    ModelFormat,
    ModelType,
)

from ..shared.sqlite.sqlite_database import SqliteDatabase
from .model_records_base import (
    DuplicateModelException,
    ModelRecordServiceBase,
    UnknownModelException,
)


class ModelRecordServiceSQL(ModelRecordServiceBase):
    """Implementation of the ModelConfigStore ABC using a SQL database."""

    _db: SqliteDatabase
    _cursor: sqlite3.Cursor

    def __init__(self, db: SqliteDatabase):
        """
        Initialize a new object from preexisting sqlite3 connection and threading lock objects.

        :param conn: sqlite3 connection object
        :param lock: threading Lock object
        """
        super().__init__()
        self._db = db
        self._cursor = self._db.conn.cursor()

    def add_model(self, key: str, config: Union[dict, AnyModelConfig]) -> AnyModelConfig:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfigException exceptions.
        """
        record = ModelConfigFactory.make_config(config, key=key)  # ensure it is a valid config obect.
        json_serialized = record.model_dump_json()  # and turn it into a json string.
        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    INSERT INTO model_config (
                       id,
                       original_hash,
                       config
                      )
                    VALUES (?,?,?);
                    """,
                    (
                        key,
                        record.original_hash,
                        json_serialized,
                    ),
                )
                self._db.conn.commit()

            except sqlite3.IntegrityError as e:
                self._db.conn.rollback()
                if "UNIQUE constraint failed" in str(e):
                    if "model_config.path" in str(e):
                        msg = f"A model with path '{record.path}' is already installed"
                    elif "model_config.name" in str(e):
                        msg = f"A model with name='{record.name}', type='{record.type}', base='{record.base}' is already installed"
                    else:
                        msg = f"A model with key '{key}' is already installed"
                    raise DuplicateModelException(msg) from e
                else:
                    raise e
            except sqlite3.Error as e:
                self._db.conn.rollback()
                raise e

        return self.get_model(key)

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
                    DELETE FROM model_config
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

    def update_model(self, key: str, config: Union[dict, AnyModelConfig]) -> AnyModelConfig:
        """
        Update the model, returning the updated version.

        :param key: Unique key for the model to be updated
        :param config: Model configuration record. Either a dict with the
         required fields, or a ModelConfigBase instance.
        """
        record = ModelConfigFactory.make_config(config, key=key)  # ensure it is a valid config obect
        json_serialized = record.model_dump_json()  # and turn it into a json string.
        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    UPDATE model_config
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
                SELECT config FROM model_config
                WHERE id=?;
                """,
                (key,),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise UnknownModelException("model not found")
            model = ModelConfigFactory.make_config(json.loads(rows[0]))
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
                select count(*) FROM model_config
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
    ) -> List[AnyModelConfig]:
        """
        Return models matching name, base and/or type.

        :param model_name: Filter by name of model (optional)
        :param base_model: Filter by base model (optional)
        :param model_type: Filter by type of model (optional)
        :param model_format: Filter by model format (e.g. "diffusers") (optional)

        If none of the optional filters are passed, will return all
        models in the database.
        """
        results = []
        where_clause = []
        bindings = []
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
                select config FROM model_config
                {where};
                """,
                tuple(bindings),
            )
            results = [ModelConfigFactory.make_config(json.loads(x[0])) for x in self._cursor.fetchall()]
        return results

    def search_by_path(self, path: Union[str, Path]) -> List[AnyModelConfig]:
        """Return models with the indicated path."""
        results = []
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config FROM model_config
                WHERE path=?;
                """,
                (str(path),),
            )
            results = [ModelConfigFactory.make_config(json.loads(x[0])) for x in self._cursor.fetchall()]
        return results

    def search_by_hash(self, hash: str) -> List[AnyModelConfig]:
        """Return models with the indicated original_hash."""
        results = []
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config FROM model_config
                WHERE original_hash=?;
                """,
                (hash,),
            )
            results = [ModelConfigFactory.make_config(json.loads(x[0])) for x in self._cursor.fetchall()]
        return results
