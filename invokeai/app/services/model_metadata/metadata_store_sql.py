# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
SQL Storage for Model Metadata
"""

import sqlite3
from typing import List, Optional, Set, Tuple

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata, UnknownMetadataException
from invokeai.backend.model_manager.metadata.fetch import ModelMetadataFetchBase

from .metadata_store_base import ModelMetadataStoreBase


class ModelMetadataStoreSQL(ModelMetadataStoreBase):
    """Store, search and fetch model metadata retrieved from remote repositories."""

    def __init__(self, db: SqliteDatabase):
        """
        Initialize a new object from preexisting sqlite3 connection and threading lock objects.

        :param conn: sqlite3 connection object
        :param lock: threading Lock object
        """
        super().__init__()
        self._db = db
        self._cursor = self._db.conn.cursor()

    def add_metadata(self, model_key: str, metadata: AnyModelRepoMetadata) -> None:
        """
        Add a block of repo metadata to a model record.

        The model record config must already exist in the database with the
        same key. Otherwise a FOREIGN KEY constraint exception will be raised.

        :param model_key: Existing model key in the `model_config` table
        :param metadata: ModelRepoMetadata object to store
        """
        json_serialized = metadata.model_dump_json()
        print("json_serialized")
        print(json_serialized)
        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    INSERT INTO model_metadata(
                       id,
                       metadata
                    )
                    VALUES (?,?);
                    """,
                    (
                        model_key,
                        json_serialized,
                    ),
                )
                # self._update_tags(model_key, metadata.tags)
                self._db.conn.commit()
            except sqlite3.IntegrityError as excp:  # FOREIGN KEY error: the key was not in model_config table
                self._db.conn.rollback()
                raise UnknownMetadataException from excp
            except sqlite3.Error as excp:
                self._db.conn.rollback()
                raise excp
            except Exception as e:
                raise e

    def get_metadata(self, model_key: str) -> AnyModelRepoMetadata:
        """Retrieve the ModelRepoMetadata corresponding to model key."""
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT metadata FROM model_metadata
                WHERE id=?;
                """,
                (model_key,),
            )
            rows = self._cursor.fetchone()
            if not rows:
                raise UnknownMetadataException("model metadata not found")
            return ModelMetadataFetchBase.from_json(rows[0])

    def list_all_metadata(self) -> List[Tuple[str, AnyModelRepoMetadata]]:  # key, metadata
        """Dump out all the metadata."""
        with self._db.lock:
            self._cursor.execute(
                """--sql
                SELECT id,metadata FROM model_metadata;
                """,
                (),
            )
            rows = self._cursor.fetchall()
        return [(x[0], ModelMetadataFetchBase.from_json(x[1])) for x in rows]

    def update_metadata(self, model_key: str, metadata: AnyModelRepoMetadata) -> AnyModelRepoMetadata:
        """
        Update metadata corresponding to the model with the indicated key.

        :param model_key: Existing model key in the `model_config` table
        :param metadata: ModelRepoMetadata object to update
        """
        json_serialized = metadata.model_dump_json()  # turn it into a json string.
        with self._db.lock:
            try:
                self._cursor.execute(
                    """--sql
                    UPDATE model_metadata
                    SET
                        metadata=?
                    WHERE id=?;
                    """,
                    (json_serialized, model_key),
                )
                if self._cursor.rowcount == 0:
                    raise UnknownMetadataException("model metadata not found")
                self._update_tags(model_key, metadata.tags)
                self._db.conn.commit()
            except sqlite3.Error as e:
                self._db.conn.rollback()
                raise e
            except Exception as e:
                raise e

        return self.get_metadata(model_key)

    def list_tags(self) -> Set[str]:
        """Return all tags in the tags table."""
        self._cursor.execute(
            """--sql
            select tag_text from tags;
            """
        )
        return {x[0] for x in self._cursor.fetchall()}

    def search_by_tag(self, tags: Set[str]) -> Set[str]:
        """Return the keys of models containing all of the listed tags."""
        with self._db.lock:
            try:
                matches: Optional[Set[str]] = None
                for tag in tags:
                    self._cursor.execute(
                        """--sql
                        SELECT a.model_id FROM model_tags AS a,
                                                     tags AS b
                        WHERE a.tag_id=b.tag_id
                          AND b.tag_text=?;
                        """,
                        (tag,),
                    )
                    model_keys = {x[0] for x in self._cursor.fetchall()}
                    if matches is None:
                        matches = model_keys
                    matches = matches.intersection(model_keys)
            except sqlite3.Error as e:
                raise e
        return matches if matches else set()

    def search_by_author(self, author: str) -> Set[str]:
        """Return the keys of models authored by the indicated author."""
        self._cursor.execute(
            """--sql
            SELECT id FROM model_metadata
            WHERE author=?;
            """,
            (author,),
        )
        return {x[0] for x in self._cursor.fetchall()}

    def search_by_name(self, name: str) -> Set[str]:
        """
        Return the keys of models with the indicated name.

        Note that this is the name of the model given to it by
        the remote source. The user may have changed the local
        name. The local name will be located in the model config
        record object.
        """
        self._cursor.execute(
            """--sql
            SELECT id FROM model_metadata
            WHERE name=?;
            """,
            (name,),
        )
        return {x[0] for x in self._cursor.fetchall()}

    def _update_tags(self, model_key: str, tags: Optional[Set[str]]) -> None:
        """Update tags for the model referenced by model_key."""
        if tags:
        # remove previous tags from this model
            self._cursor.execute(
                """--sql
                DELETE FROM model_tags
                WHERE model_id=?;
                """,
                (model_key,),
            )

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
                    INSERT OR IGNORE INTO model_tags (
                    model_id,
                    tag_id
                    )
                    VALUES (?,?);
                    """,
                    (model_key, tag_id),
                )
