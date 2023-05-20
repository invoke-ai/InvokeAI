import sqlite3
from typing import Union

from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.resources import ImageKind, ResourceOrigin
from invokeai.app.services.database.images.models import (
    GeneratedImageEntity,
    UploadedImageEntity,
)


def parse_image_result(image: dict) -> Union[GeneratedImageEntity, UploadedImageEntity]:
    """Parses the image query result from the database."""

    origin = ResourceOrigin(image["origin"])

    if origin is ResourceOrigin.UPLOADS:
        return UploadedImageEntity(
            id=image["id"],
            origin=ResourceOrigin.UPLOADS,
            image_kind=ImageKind(image["image_kind"]),
            created_at=image["created_at"],
            metadata=UploadedImageOrLatentsMetadata.parse_raw(image["metadata"]),
        )

    if origin is ResourceOrigin.INTERMEDIATES:
        return GeneratedImageEntity(
            id=image["id"],
            session_id=image["session_id"],
            node_id=image["node_id"],
            origin=ResourceOrigin.INTERMEDIATES,
            image_kind=ImageKind(image["image_kind"]),
            created_at=image["created_at"],
            metadata=GeneratedImageOrLatentsMetadata.parse_raw(image["metadata"]),
        )

    return GeneratedImageEntity(
        id=image["id"],
        session_id=image["session_id"],
        node_id=image["node_id"],
        origin=ResourceOrigin.RESULTS,
        image_kind=ImageKind(image["image_kind"]),
        created_at=image["created_at"],
        metadata=GeneratedImageOrLatentsMetadata.parse_raw(image["metadata"]),
    )


def create_images_tables(cursor: sqlite3.Cursor):
    """
    Creates `images`, `images_results`, `images_intermediates` and `images_metadata` tables and their indices.
    """

    # Create the `images` table and its indicies.
    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            origin TEXT,
            image_kind TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(origin) REFERENCES resource_origins(origin_name),
            FOREIGN KEY(image_kind) REFERENCES image_kinds(kind_name)
        );
        """
    )
    cursor.execute(
        """--sql
        CREATE UNIQUE INDEX IF NOT EXISTS idx_images_id ON images(id);
        """
    )
    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_images_origin ON images(origin);
        """
    )
    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_images_image_kind ON images(image_kind);
        """
    )

    # Create the `images_results` table and its indicies.
    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS images_results (
            images_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            FOREIGN KEY(images_id) REFERENCES images(id) ON DELETE CASCADE
        );
        """
    )
    cursor.execute(
        """--sql
        CREATE UNIQUE INDEX IF NOT EXISTS idx_images_results_images_id ON images_results(images_id);
        """
    )

    # Create the `images_intermediates` table and its indicies.
    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS images_intermediates (
            images_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            FOREIGN KEY(images_id) REFERENCES images(id) ON DELETE CASCADE
        );
        """
    )
    cursor.execute(
        """--sql
        CREATE UNIQUE INDEX IF NOT EXISTS idx_images_intermediates_images_id ON images_intermediates(images_id);
        """
    )

    # Create the `images_metadata` table and its indicies.
    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS images_metadata (
            images_id TEXT PRIMARY KEY,
            metadata TEXT,
            FOREIGN KEY(images_id) REFERENCES images(id) ON DELETE CASCADE
        );
        """
    )
    cursor.execute(
        """--sql
        CREATE UNIQUE INDEX IF NOT EXISTS idx_images_metadata_images_id ON images_metadata(images_id);
        """
    )
