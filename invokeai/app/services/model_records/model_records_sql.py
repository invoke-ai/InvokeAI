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
import time
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelConfigFactory,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata, ModelMetadataStore, UnknownMetadataException
from invokeai.backend.model_manager.load import AnyModelLoader, LoadedModel

from ..shared.sqlite.sqlite_database import SqliteDatabase
from .model_records_base import (
    DuplicateModelException,
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    ModelSummary,
    UnknownModelException,
)


class ModelRecordServiceSQL(ModelRecordServiceBase):
    """Implementation of the ModelConfigStore ABC using a SQL database."""

    def __init__(self, db: SqliteDatabase, loader: Optional[AnyModelLoader]=None):
        """
        Initialize a new object from preexisting sqlite3 connection and threading lock objects.

        :param db: Sqlite connection object
        :param loader: Initialized model loader object (optional)
        """
        super().__init__()
        self._db = db
        self._cursor = db.conn.cursor()
        self._loader = loader

    @property
    def db(self) -> SqliteDatabase:
        """Return the underlying database."""
        return self._db

    def add_model(self, key: str, config: Union[Dict[str, Any], AnyModelConfig]) -> AnyModelConfig:
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
                SELECT config, strftime('%s',updated_at) FROM model_config
                WHERE id=?;
                """,
                (key,),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise UnknownModelException("model not found")
            model = ModelConfigFactory.make_config(json.loads(rows[0]), timestamp=rows[1])
        return model

    def load_model(self, key: str, submodel_type: Optional[SubModelType]) -> LoadedModel:
        """
        Load the indicated model into memory and return a LoadedModel object.

        :param key: Key of model config to be fetched.
        :param submodel_type: For main (pipeline models), the submodel to fetch.

        Exceptions: UnknownModelException -- model with this key not known
                    NotImplementedException -- a model loader was not provided at initialization time
        """
        if not self._loader:
            raise NotImplementedError(f"Class {self.__class__} was not initialized with a model loader")
        model_config = self.get_model(key)
        return self._loader.load_model(model_config, submodel_type)

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
                select config, strftime('%s',updated_at) FROM model_config
                {where};
                """,
                tuple(bindings),
            )
            results = [ModelConfigFactory.make_config(json.loads(x[0]), timestamp=x[1]) for x in self._cursor.fetchall()]
        return results

    def search_by_path(self, path: Union[str, Path]) -> List[AnyModelConfig]:
        """Return models with the indicated path."""
        results = []
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config, strftime('%s',updated_at) FROM model_config
                WHERE path=?;
                """,
                (str(path),),
            )
            results = [ModelConfigFactory.make_config(json.loads(x[0]), timestamp=x[1]) for x in self._cursor.fetchall()]
        return results

    def search_by_hash(self, hash: str) -> List[AnyModelConfig]:
        """Return models with the indicated original_hash."""
        results = []
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT config, strftime('%s',updated_at) FROM model_config
                WHERE original_hash=?;
                """,
                (hash,),
            )
            results = [ModelConfigFactory.make_config(json.loads(x[0]), timestamp=x[1]) for x in self._cursor.fetchall()]
        return results

    @property
    def metadata_store(self) -> ModelMetadataStore:
        """Return a ModelMetadataStore initialized on the same database."""
        return ModelMetadataStore(self._db)

    def get_metadata(self, key: str) -> Optional[AnyModelRepoMetadata]:
        """
        Retrieve metadata (if any) from when model was downloaded from a repo.

        :param key: Model key
        """
        store = self.metadata_store
        try:
            metadata = store.get_metadata(key)
            return metadata
        except UnknownMetadataException:
            return None

    def search_by_metadata_tag(self, tags: Set[str]) -> List[AnyModelConfig]:
        """
        Search model metadata for ones with all listed tags and return their corresponding configs.

        :param tags: Set of tags to search for. All tags must be present.
        """
        store = ModelMetadataStore(self._db)
        keys = store.search_by_tag(tags)
        return [self.get_model(x) for x in keys]

    def list_tags(self) -> Set[str]:
        """Return a unique set of all the model tags in the metadata database."""
        store = ModelMetadataStore(self._db)
        return store.list_tags()

    def list_all_metadata(self) -> List[Tuple[str, AnyModelRepoMetadata]]:
        """List metadata for all models that have it."""
        store = ModelMetadataStore(self._db)
        return store.list_all_metadata()

    def list_models(
        self, page: int = 0, per_page: int = 10, order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default
    ) -> PaginatedResults[ModelSummary]:
        """Return a paginated summary listing of each model in the database."""
        ordering = {
            ModelRecordOrderBy.Default: "a.type, a.base, a.format, a.name",
            ModelRecordOrderBy.Type: "a.type",
            ModelRecordOrderBy.Base: "a.base",
            ModelRecordOrderBy.Name: "a.name",
            ModelRecordOrderBy.Format: "a.format",
        }

        def _fixup(summary: Dict[str, str]) -> Dict[str, Union[str, int, Set[str]]]:
            """Fix up results so that there are no null values."""
            result: Dict[str, Union[str, int, Set[str]]] = {}
            for key, item in summary.items():
                result[key] = item or ""
            result["tags"] = set(json.loads(summary["tags"] or "[]"))
            return result

        # Lock so that the database isn't updated while we're doing the two queries.
        with self._db.lock:
            # query1: get the total number of model configs
            self._cursor.execute(
                """--sql
                select count(*) from model_config;
                """,
                (),
            )
            total = int(self._cursor.fetchone()[0])

            # query2: fetch key fields from the join of model_config and model_metadata
            self._cursor.execute(
                f"""--sql
                SELECT a.id as key, a.type, a.base, a.format, a.name,
                       json_extract(a.config, '$.description') as description,
                       json_extract(b.metadata, '$.tags') as tags
                FROM model_config AS a
                LEFT JOIN model_metadata AS b on a.id=b.id
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
            items = [ModelSummary.model_validate(_fixup(dict(x))) for x in rows]
            return PaginatedResults(
                page=page, pages=ceil(total / per_page), per_page=per_page, total=total, items=items
            )
