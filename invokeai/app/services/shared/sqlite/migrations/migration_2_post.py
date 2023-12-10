import sqlite3
from logging import Logger

from tqdm import tqdm

from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase


def migrate_embedded_workflows(
    cursor: sqlite3.Cursor,
    logger: Logger,
    image_files: ImageFileStorageBase,
) -> None:
    """
    In the v3.5.0 release, InvokeAI changed how it handles embedded workflows. The `images` table in
    the database now has a `has_workflow` column, indicating if an image has a workflow embedded.

    This migrate callbakc checks each image for the presence of an embedded workflow, then updates its entry
    in the database accordingly.
    """
    # Get the total number of images and chunk it into pages
    cursor.execute("SELECT image_name FROM images")
    image_names: list[str] = [image[0] for image in cursor.fetchall()]
    total_image_names = len(image_names)

    if not total_image_names:
        return

    logger.info(f"Migrating workflows for {total_image_names} images")

    # Migrate the images
    to_migrate: list[tuple[bool, str]] = []
    pbar = tqdm(image_names)
    for idx, image_name in enumerate(pbar):
        pbar.set_description(f"Checking image {idx + 1}/{total_image_names} for workflow")
        pil_image = image_files.get(image_name)
        if "invokeai_workflow" in pil_image.info:
            to_migrate.append((True, image_name))

    logger.info(f"Adding {len(to_migrate)} embedded workflows to database")
    cursor.executemany("UPDATE images SET has_workflow = ? WHERE image_name = ?", to_migrate)
