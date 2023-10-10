# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Implementation of ModelConfigStore using a SQLite3 database

Typical usage:

  from invokeai.backend.model_manager import ModelConfigStoreSQL
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

import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Set, Union

from ..config import AnyModelConfig, BaseModelType, ModelConfigBase, ModelConfigFactory, ModelType
from .base import CONFIG_FILE_VERSION, DuplicateModelException, ModelConfigStore, UnknownModelException


class ModelConfigStoreSQL(ModelConfigStore):
    """Implementation of the ModelConfigStore ABC using a YAML file."""

    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.Lock

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock):
        """
        Initialize a new object from preexisting sqlite3 connection and threading lock objects.

        :param conn: sqlite3 connection object
        :param lock: threading Lock object
        """

        super().__init__()
        self._conn = conn
        # Enable row factory to get rows as dictionaries (must be done before making the cursor!)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._lock = lock

        with self._lock:
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._create_tables()
            self._conn.commit()
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
                -- These 4 fields are enums in python, unrestricted string here
                base_model TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                -- Serialized JSON representation of the whole config object,
                -- which will contain additional fields from subclasses
                config TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
            );
            """
        )

        #  model_tag table 1:M relation between model key and tag(s)
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_tag (
                id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                FOREIGN KEY(id) REFERENCES model_config(id),
                FOREIGN KEY(tag_id) REFERENCES tags(tag_id),
                UNIQUE(id,tag_id)
            );
            """
        )

        #  tags table
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER NOT NULL PRIMARY KEY,
                tag_text TEXT NOT NULL UNIQUE
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

        # Add trigger to remove tags when model is deleted
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS model_deleted
            AFTER DELETE
            ON model_config
            BEGIN
                DELETE from model_tag WHERE id=old.id;
            END;
            """
        )

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

    def add_model(self, key: str, config: Union[dict, ModelConfigBase]) -> ModelConfigBase:
        """
        Add a model to the database.

        :param key: Unique key for the model
        :param config: Model configuration record, either a dict with the
         required fields or a ModelConfigBase instance.

        Can raise DuplicateModelException and InvalidModelConfigException exceptions.
        """
        record = ModelConfigFactory.make_config(config, key=key)  # ensure it is a valid config obect.
        json_serialized = json.dumps(record.dict())  # and turn it into a json string.
        with self._lock:
            try:
                self._cursor.execute(
                    """--sql
                    INSERT INTO model_config (
                       id,
                       base_model,
                       model_type,
                       model_name,
                       model_path,
                       config
                      )
                    VALUES (?,?,?,?,?,?);
                    """,
                    (
                        key,
                        record.base_model,
                        record.model_type,
                        record.name,
                        record.path,
                        json_serialized,
                    ),
                )
                if record.tags:
                    self._update_tags(key, record.tags)
                self._conn.commit()

            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                if "UNIQUE constraint failed" in str(e):
                    raise DuplicateModelException(f"A model with key '{key}' is already installed") from e
                else:
                    raise e
            except sqlite3.Error as e:
                self._conn.rollback()
                raise e

        return self.get_model(key)

    @property
    def version(self) -> str:
        """Return the version of the database schema."""
        with self._lock:
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

    def _update_tags(self, key: str, tags: List[str]) -> None:
        """Update tags for model with key."""
        # remove previous tags from this model
        self._cursor.execute(
            """--sql
            DELETE FROM model_tag
            WHERE id=?;
            """,
            (key,),
        )

        # NOTE: isn't there a more elegant way of doing this than one tag
        # at a time, with a select to get the tag ID?
        for tag in tags:
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO tags (
                  tag_text
                  )
                VALUES (?);
                """,
                (tag,),
            )
            self._cursor.execute(
                """--sql
                SELECT tag_id
                FROM tags
                WHERE tag_text = ?
                LIMIT 1;
                """,
                (tag,),
            )
            tag_id = self._cursor.fetchone()[0]
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO model_tag (
                   id,
                   tag_id
                  )
                VALUES (?,?);
                """,
                (key, tag_id),
            )

    def del_model(self, key: str) -> None:
        """
        Delete a model.

        :param key: Unique key for the model to be deleted

        Can raise an UnknownModelException
        """
        with self._lock:
            try:
                self._cursor.execute(
                    """--sql
                    DELETE FROM model_config
                    WHERE id=?;
                    """,
                    (key,),
                )
                if self._cursor.rowcount == 0:
                    raise UnknownModelException
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise e

    def update_model(self, key: str, config: Union[dict, ModelConfigBase]) -> ModelConfigBase:
        """
        Update the model, returning the updated version.

        :param key: Unique key for the model to be updated
        :param config: Model configuration record. Either a dict with the
         required fields, or a ModelConfigBase instance.
        """
        record = ModelConfigFactory.make_config(config, key=key)  # ensure it is a valid config obect
        json_serialized = json.dumps(record.dict())  # and turn it into a json string.
        with self._lock:
            try:
                self._cursor.execute(
                    """--sql
                    UPDATE model_config
                    SET base_model=?,
                        model_type=?,
                        model_name=?,
                        model_path=?,
                        config=?
                    WHERE id=?;
                    """,
                    (record.base_model, record.model_type, record.name, record.path, json_serialized, key),
                )
                if self._cursor.rowcount == 0:
                    raise UnknownModelException
                if record.tags:
                    self._update_tags(key, record.tags)
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise e

        return self.get_model(key)

    def get_model(self, key: str) -> AnyModelConfig:
        """
        Retrieve the ModelConfigBase instance for the indicated model.

        :param key: Key of model config to be fetched.

        Exceptions: UnknownModelException
        """
        with self._lock:
            self._cursor.execute(
                """--sql
                SELECT config FROM model_config
                WHERE id=?;
                """,
                (key,),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise UnknownModelException
            model = ModelConfigFactory.make_config(json.loads(rows[0]))
        return model

    def exists(self, key: str) -> bool:
        """
        Return True if a model with the indicated key exists in the databse.

        :param key: Unique key for the model to be deleted
        """
        count = 0
        with self._lock:
            try:
                self._cursor.execute(
                    """--sql
                    select count(*) FROM model_config
                    WHERE id=?;
                    """,
                    (key,),
                )
                count = self._cursor.fetchone()[0]
            except sqlite3.Error as e:
                raise e
        return count > 0

    def search_by_tag(self, tags: Set[str]) -> List[AnyModelConfig]:
        """Return models containing all of the listed tags."""
        # rather than create a hairy SQL cross-product, we intersect
        # tag results in a stepwise fashion at the python level.
        results = []
        with self._lock:
            try:
                matches: Set[str] = set()
                for tag in tags:
                    self._cursor.execute(
                        """--sql
                        SELECT a.id FROM model_tag AS a,
                                           tags AS b
                        WHERE a.tag_id=b.tag_id
                          AND b.tag_text=?;
                        """,
                        (tag,),
                    )
                    model_keys = {x[0] for x in self._cursor.fetchall()}
                    matches = matches.intersection(model_keys) if len(matches) > 0 else model_keys
                if matches:
                    self._cursor.execute(
                        f"""--sql
                        SELECT config FROM model_config
                        WHERE id IN ({','.join('?' * len(matches))});
                        """,
                        tuple(matches),
                    )
                    results = [ModelConfigFactory.make_config(json.loads(x[0])) for x in self._cursor.fetchall()]
            except sqlite3.Error as e:
                raise e
        return results

    def search_by_name(
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
            where_clause.append("model_name=?")
            bindings.append(model_name)
        if base_model:
            where_clause.append("base_model=?")
            bindings.append(base_model)
        if model_type:
            where_clause.append("model_type=?")
            bindings.append(model_type)
        where = f"WHERE {' AND '.join(where_clause)}" if where_clause else ""
        with self._lock:
            try:
                self._cursor.execute(
                    f"""--sql
                    select config FROM model_config
                    {where};
                    """,
                    tuple(bindings),
                )
                results = [ModelConfigFactory.make_config(json.loads(x[0])) for x in self._cursor.fetchall()]
            except sqlite3.Error as e:
                raise e
        return results

    def search_by_path(self, path: Union[str, Path]) -> Optional[ModelConfigBase]:
        """Return the model with the indicated path, or None."""
        raise NotImplementedError("search_by_path not implemented in storage.sql")
