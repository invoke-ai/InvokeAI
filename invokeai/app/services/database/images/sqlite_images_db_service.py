import sqlite3
import threading
from typing import Optional, Union
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.resources import ImageKind, ResourceOrigin
from invokeai.app.services.database.images.images_db_service_base import (
    ImagesDbServiceBase,
)
from invokeai.app.services.database.images.models import (
    GeneratedImageEntity,
    UploadedImageEntity,
)
from invokeai.app.services.database.images.sqlite_util import (
    create_images_tables,
    parse_image_result,
)

from invokeai.app.services.item_storage import PaginatedResults


BASE_IMAGES_JOINED_QUERY = """--sql
    SELECT
        images.id AS id,
        images.session_id AS session_id,
        images.node_id AS node_id,
        images.origin AS origin,
        images.image_kind AS image_kind,
        images.created_at AS created_at,
        images_metadata.metadata AS metadata
    FROM images
    JOIN images_metadata ON images.id = images_metadata.images_id
"""


class SqliteImagesDbService(ImagesDbServiceBase):
    _filename: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.Lock

    def __init__(self, filename: str) -> None:
        super().__init__()

        self._filename = filename
        self._conn = sqlite3.connect(filename, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._lock = threading.Lock()

        try:
            self._lock.acquire()
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.commit()
            # Enable row factory to get rows as dictionaries
            self._conn.row_factory = sqlite3.Row
            # Create tables
            create_images_tables(self._cursor)
            self._conn.commit()
        finally:
            self._lock.release()

    def get(self, id: str) -> Union[GeneratedImageEntity, UploadedImageEntity, None]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                {BASE_IMAGES_JOINED_QUERY}
                WHERE id = ?;
                """,
                (id,),
            )

            result = self._cursor.fetchone()
        finally:
            self._lock.release()

        if not result:
            return None

        return parse_image_result(result)

    def get_many(
        self,
        origin: ResourceOrigin,
        image_kind: ImageKind,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[Union[GeneratedImageEntity, UploadedImageEntity]]:
        try:
            self._lock.acquire()

            # Retrieve the page of images
            self._cursor.execute(
                f"""--sql
                {BASE_IMAGES_JOINED_QUERY}
                WHERE origin = ? AND image_kind = ?
                LIMIT ? OFFSET ?;
                """,
                (origin.value, image_kind.value, per_page, page * per_page),
            )

            result = self._cursor.fetchall()

            images = list(map(lambda r: parse_image_result(r[0]), result))

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
        metadata: GeneratedImageOrLatentsMetadata | UploadedImageOrLatentsMetadata,
    ) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO images (
                    id,
                    origin,
                    image_kind)
                VALUES (?, ?, ?);
                """,
                (id, origin.value, image_kind.value),
            )

            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO images_metadata (
                    images_id,
                    metadata)
                VALUES (?, ?);
                """,
                (id, metadata.json(exclude_none=True)),
            )

            if origin is ResourceOrigin.RESULTS:
                self._cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO images_results (
                        images_id,
                        session_id,
                        node_id)
                    VALUES (?, ?, ?);
                    """,
                    (id, session_id, node_id),
                )
            elif origin is ResourceOrigin.INTERMEDIATES:
                self._cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO images_intermediates (
                        images_id,
                        session_id,
                        node_id)
                    VALUES (?, ?, ?);
                    """,
                    (id, session_id, node_id),
                )

            self._conn.commit()
        finally:
            self._lock.release()
