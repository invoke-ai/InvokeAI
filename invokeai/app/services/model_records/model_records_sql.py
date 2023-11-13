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
    ModelConfigBase,
    ModelConfigFactory,
    ModelType,
)

from ..shared.sqlite import SqliteDatabase
from .model_records_base import (
    CONFIG_FILE_VERSION,
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

        with self._db.lock:
            # Enable foreign keys
            self._db.conn.execute("PRAGMA foreign_keys = ON;")
            self._create_tables()
            self._db.conn.commit()
        assert (
            str(self.version) == CONFIG_FILE_VERSION
        ), f"Model config version {self.version} does not match expected version {CONFIG_FILE_VERSION}"

    def _create_tables(self) -> None:
        """Create sqlite3 tables."""
        #  model_config table breaks out the fields that are common to all config objects
        # and puts class-specific ones in a serialized json object
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_config (
                id TEXT NOT NULL PRIMARY KEY,
                -- The next 3 fields are enums in python, unrestricted string here
                base TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                original_hash TEXT, -- could be null
                -- Serialized JSON representation of the whole config object,
                -- which will contain additional fields from subclasses
                config TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- unique constraint on combo of name, base and type
                UNIQUE(name, base, type)
            );
            """
        )

        #  metadata table
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_manager_metadata (
                metadata_key TEXT NOT NULL PRIMARY KEY,
                metadata_value TEXT NOT NULL
            );
            """
        )

        # Add trigger for `updated_at`.
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS model_config_updated_at
            AFTER UPDATE
            ON model_config FOR EACH ROW
            BEGIN
                UPDATE model_config SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE id = old.id;
            END;
            """
        )

        # Add indexes for searchable fields
        for stmt in [
            "CREATE INDEX IF NOT EXISTS base_index ON model_config(base);",
            "CREATE INDEX IF NOT EXISTS type_index ON model_config(type);",
            "CREATE INDEX IF NOT EXISTS name_index ON model_config(name);",
            "CREATE UNIQUE INDEX IF NOT EXISTS path_index ON model_config(path);",
        ]:
            self._cursor.execute(stmt)

        # Add our version to the metadata table
        self._cursor.execute(
            """--sql
            INSERT OR IGNORE into model_manager_metadata (
               metadata_key,
               metadata_value
            )
            VALUES (?,?);
            """,
            ("version", CONFIG_FILE_VERSION),
        )

    def add_model(self, key: str, config: Union[dict, ModelConfigBase]) -> AnyModelConfig:
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
                       base,
                       type,
                       name,
                       path,
                       original_hash,
                       config
                      )
                    VALUES (?,?,?,?,?,?,?);
                    """,
                    (
                        key,
                        record.base,
                        record.type,
                        record.name,
                        record.path,
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

    @property
    def version(self) -> str:
        """Return the version of the database schema."""
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT metadata_value FROM model_manager_metadata
                WHERE metadata_key=?;
                """,
                ("version",),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise KeyError("Models database does not have metadata key 'version'")
            return rows[0]

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

    def update_model(self, key: str, config: ModelConfigBase) -> AnyModelConfig:
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
                    SET base=?,
                        type=?,
                        name=?,
                        path=?,
                        config=?
                    WHERE id=?;
                    """,
                    (record.base, record.type, record.name, record.path, json_serialized, key),
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
    ) -> List[AnyModelConfig]:
        """
        Return models matching name, base and/or type.

        :param model_name: Filter by name of model (optional)
        :param base_model: Filter by base model (optional)
        :param model_type: Filter by type of model (optional)

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

    def search_by_path(self, path: Union[str, Path]) -> List[ModelConfigBase]:
        """Return models with the indicated path."""
        results = []
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config FROM model_config
                WHERE model_path=?;
                """,
                (str(path),),
            )
            results = [ModelConfigFactory.make_config(json.loads(x[0])) for x in self._cursor.fetchall()]
        return results

    def search_by_hash(self, hash: str) -> List[ModelConfigBase]:
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
