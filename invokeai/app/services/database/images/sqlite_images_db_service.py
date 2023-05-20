import sqlite3
import threading
from typing import Optional, Union
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.resources import ImageKind, ResourceOrigin, TensorKind
from invokeai.app.services.database.create_enum_table import create_enum_table
from invokeai.app.services.database.images.images_db_service_base import (
    ImagesDbServiceBase,
)
from invokeai.app.services.database.images.models import ImageEntity
from invokeai.app.services.database.images.deserialize_image_entity import (
    deserialize_image_entity,
)

from invokeai.app.services.item_storage import PaginatedResults


class SqliteImagesDbService(ImagesDbServiceBase):
    _filename: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.Lock

    def __init__(self, filename: str) -> None:
        super().__init__()

        self._filename = filename
        self._conn = sqlite3.connect(filename, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._lock = threading.Lock()

        try:
            self._lock.acquire()
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON;")
            # Enable row factory to get rows as dictionaries
            self._create_table()

            # TODO: Create these elsewhere
            create_enum_table(
                enum=ResourceOrigin,
                table_name="resource_origins",
                primary_key_name="origin_name",
                cursor=self._cursor,
            )

            create_enum_table(
                enum=ImageKind,
                table_name="image_kinds",
                primary_key_name="kind_name",
                cursor=self._cursor,
            )

            create_enum_table(
                enum=TensorKind,
                table_name="tensor_kinds",
                primary_key_name="kind_name",
                cursor=self._cursor,
            )

            self._conn.commit()
        finally:
            self._lock.release()

    def _create_table(self) -> None:
        # Create the `images` table and its indicies.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, -- The unique identifier for the image.
                origin TEXT,
                image_kind TEXT,
                session_id TEXT,
                node_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(origin) REFERENCES resource_origins(origin_name),
                FOREIGN KEY(image_kind) REFERENCES image_kinds(kind_name)
            );
            """
        )
        self._cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_images_id ON images(id);
            """
        )
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_origin ON images(origin);
            """
        )
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_image_kind ON images(image_kind);
            """
        )

    def get(self, id: str) -> Union[ImageEntity, None]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT * FROM images
                WHERE id = ?;
                """,
                (id,),
            )

            result = self._cursor.fetchone()
        finally:
            self._lock.release()

        if not result:
            return None

        return deserialize_image_entity(result)

    def get_many(
        self,
        origin: ResourceOrigin,
        image_kind: ImageKind,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageEntity]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT * FROM images
                WHERE origin = ? AND image_kind = ?
                LIMIT ? OFFSET ?;
                """,
                (origin.value, image_kind.value, per_page, page * per_page),
            )

            result = self._cursor.fetchall()

            images = list(map(lambda r: deserialize_image_entity(r), result))

            self._cursor.execute(
                """--sql
                SELECT count(*) FROM images
                WHERE origin = ? AND image_kind = ?
                """,
                (origin.value, image_kind.value),
            )

            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedResults(
            items=images, page=page, pages=pageCount, per_page=per_page, total=count
        )

    def delete(self, id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM images
                WHERE id = ?;
                """,
                (id,),
            )
            self._conn.commit()
        finally:
            self._lock.release()

    def set(
        self,
        id: str,
        origin: ResourceOrigin,
        image_kind: ImageKind,
        session_id: Optional[str],
        node_id: Optional[str],
        metadata: Union[
            GeneratedImageOrLatentsMetadata, UploadedImageOrLatentsMetadata, None
        ],
    ) -> None:
        try:
            metadata_json = (
                None if metadata is None else metadata.json(exclude_none=True)
            )
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO images (
                    id,
                    origin,
                    image_kind,
                    node_id,
                    session_id,
                    metadata
                    )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    id,
                    origin.value,
                    image_kind.value,
                    node_id,
                    session_id,
                    metadata_json,
                ),
            )
            self._conn.commit()
        finally:
            self._lock.release()
