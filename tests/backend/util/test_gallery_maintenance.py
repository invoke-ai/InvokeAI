"""The gallery maintenance script, run headless on an install of its own."""

import sqlite3
from contextlib import closing
from pathlib import Path

from PIL import Image

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.backend.util.gallery_maintenance import InvokeAIDatabaseMaintenanceApp, MaintenanceOperation
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.database import migrate_to_newest


def _install(root: Path) -> str:
    """An install whose database knows `present.png` (with its file) and `missing.png` (on a board, file gone), and
    whose outputs hold `orphan.png`, which the database does not know. The board's id."""
    (root / "invokeai.yaml").write_text('schema_version: "4.0.3"\n')
    images = root / "outputs" / "images"
    (images / "thumbnails").mkdir(parents=True)
    Image.new("RGB", (64, 64), "red").save(images / "present.png")
    Image.new("RGB", (64, 64), "blue").save(images / "orphan.png")
    Image.new("RGB", (8, 8), "blue").save(images / "thumbnails" / "orphan.webp")
    Image.new("RGB", (8, 8), "green").save(images / "thumbnails" / "missing.webp")

    database = Database.open_sqlite(root / "databases" / "invokeai.db", InvokeAILogger.get_logger("test"))
    try:
        migrate_to_newest(database, root)
        records = ImageRecordStorage(database)
        for name in ("present.png", "missing.png"):
            records.save(
                image_name=name,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                width=64,
                height=64,
                has_workflow=False,
            )
        board_id = BoardRecordStorage(database).save("Board", "system").board_id
        BoardImageRecordStorage(database).add_image_to_board(board_id=board_id, image_name="missing.png")
        return board_id
    finally:
        database.dispose()


def test_all_operations_clean_records_archive_files_and_regenerate_thumbnails(tmp_path: Path) -> None:
    board_id = _install(tmp_path)

    InvokeAIDatabaseMaintenanceApp(MaintenanceOperation.All, tmp_path).main()

    database = Database.open_sqlite(tmp_path / "databases" / "invokeai.db", InvokeAILogger.get_logger("test"))
    try:
        records = ImageRecordStorage(database)
        assert records.exists("present.png")
        assert not records.exists("missing.png")
        assert not records.exists("orphan.png")
        assert BoardImageRecordStorage(database).get_all_board_image_names_for_board(board_id, None, None, None) == []
    finally:
        database.dispose()

    outputs = tmp_path / "outputs"
    assert (outputs / "images" / "present.png").is_file()
    assert (outputs / "images" / "thumbnails" / "present.webp").is_file()
    assert sorted(path.name for path in (outputs / "images-archive").glob("*.png")) == ["orphan.png"]
    archived_thumbnails = sorted(path.name for path in (outputs / "images-archive" / "thumbnails").iterdir())
    assert archived_thumbnails == ["missing.webp", "orphan.webp"]

    [backup] = (tmp_path / "databases" / "backup").iterdir()
    with closing(sqlite3.connect(backup)) as copy:
        # Taken once, before the run changed anything.
        rows = copy.execute("SELECT image_name FROM images ORDER BY image_name").fetchall()
        assert rows == [("missing.png",), ("present.png",)]
