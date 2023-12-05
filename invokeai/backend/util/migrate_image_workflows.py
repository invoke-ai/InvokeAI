import sqlite3
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from invokeai.app.services.config.config_default import InvokeAIAppConfig


def migrate_image_workflows(output_path: Path, database: Path, page_size=100):
    """
    In the v3.5.0 release, InvokeAI changed how it handles image workflows. The `images` table in
    the database now has a `has_workflow` column, indicating if an image has a workflow embedded.

    This script checks each image for the presence of an embedded workflow, then updates its entry
    in the database accordingly.

    1) Check if the database is updated to support image workflows. Aborts if it doesn't have the
    `has_workflow` column yet.
    2) Backs up the database.
    3) Opens each image in the `images` table via PIL
    4) Checks if the `"invokeai_workflow"` attribute its in the image's embedded metadata, indicating
    that it has a workflow.
    5) If it does, updates the `has_workflow` column for that image to `TRUE`.

    If there are any problems, the script immediately aborts. Because the processing happens in chunks,
    if there is a problem, it is suggested that you restore the database from the backup and try again.
    """
    output_path = output_path
    database = database
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # We can only migrate if the `images` table has the `has_workflow` column
    cursor.execute("PRAGMA table_info(images)")
    columns = [column[1] for column in cursor.fetchall()]
    if "has_workflow" not in columns:
        raise Exception("Database needs to be updated to support image workflows")

    # Back up the database before we start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = database.parent / f"{database.stem}_migrate-image-workflows_{timestamp}.db"
    print(f"Backing up database to {backup_path}")
    backup_conn = sqlite3.connect(backup_path)
    with backup_conn:
        conn.backup(backup_conn)
    backup_conn.close()

    # Get the total number of images and chunk it into pages
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]
    total_pages = (total_images + page_size - 1) // page_size
    print(f"Processing {total_images} images in chunks of {page_size} images...")

    # Migrate the images
    migrated_count = 0
    pbar = tqdm(range(total_pages))
    for page in pbar:
        pbar.set_description(f"Migrating page {page + 1}/{total_pages}")
        offset = page * page_size
        cursor.execute("SELECT image_name FROM images LIMIT ? OFFSET ?", (page_size, offset))
        images = cursor.fetchall()
        for image_name in images:
            image_path = output_path / "images" / image_name[0]
            with Image.open(image_path) as img:
                if "invokeai_workflow" in img.info:
                    cursor.execute("UPDATE images SET has_workflow = TRUE WHERE image_name = ?", (image_name[0],))
                    migrated_count += 1
        conn.commit()
    conn.close()

    print(f"Migrated workflows for {migrated_count} images.")


if __name__ == "__main__":
    config = InvokeAIAppConfig.get_config()
    output_path = config.output_path
    database = config.db_path

    assert output_path is not None
    assert output_path.exists()
    assert database.exists()
    migrate_image_workflows(output_path=output_path, database=database, page_size=100)
