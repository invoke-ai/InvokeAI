import sqlite3
import threading
from typing import Optional, Union
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.image import (
    ImageCategory,
    ImageType,
)
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
        # Enable row factory to get rows as dictionaries (must be done before making the cursor!)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._lock = threading.Lock()

        try:
            self._lock.acquire()
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._create_tables()
            self._conn.commit()
        finally:
            self._lock.release()

    def _create_tables(self) -> None:
        # Create the `images` table.
        self._cursor.execute(
            f"""--sql
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, -- The unique identifier for the image.
                image_type TEXT,
                image_category TEXT,
                session_id TEXT,
                node_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_type) REFERENCES image_types(type_name),
                FOREIGN KEY(image_category) REFERENCES image_categories(category_name)
            );
            """
        )

        # Create the `images` table indices.
        self._cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_images_id ON images(id);
            """
        )
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_image_type ON images(image_type);
            """
        )
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_image_category ON images(image_category);
            """
        )

        # Create the tables for image-related enums
        create_enum_table(
            enum=ImageType,
            table_name="image_types",
            primary_key_name="type_name",
            cursor=self._cursor,
        )

        create_enum_table(
            enum=ImageCategory,
            table_name="image_categories",
            primary_key_name="category_name",
            cursor=self._cursor,
        )

        # Create the `tags` table. TODO: do this elsewhere, shouldn't be in images db service
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT UNIQUE NOT NULL
            );
            """
        )

        # Create the `images_tags` junction table.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS images_tags (
                image_id TEXT,
                tag_id INTEGER,
                PRIMARY KEY (image_id, tag_id),
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );
            """
        )

        # Create the `images_favorites` table.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS images_favorites (
                image_id TEXT PRIMARY KEY,
                favorited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            );
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
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageEntity]:
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT * FROM images
                WHERE image_type = ? AND image_category = ?
                LIMIT ? OFFSET ?;
                """,
                (image_type.value, image_category.value, per_page, page * per_page),
            )

            result = self._cursor.fetchall()

            images = list(map(lambda r: deserialize_image_entity(r), result))

            self._cursor.execute(
                """--sql
                SELECT count(*) FROM images
                WHERE image_type = ? AND image_category = ?
                """,
                (image_type.value, image_category.value),
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
        image_type: ImageType,
        image_category: ImageCategory,
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
                    image_type,
                    image_category,
                    node_id,
                    session_id,
                    metadata
                    )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    id,
                    image_type.value,
                    image_category.value,
                    node_id,
                    session_id,
                    metadata_json,
                ),
            )
            self._conn.commit()
        finally:
            self._lock.release()
