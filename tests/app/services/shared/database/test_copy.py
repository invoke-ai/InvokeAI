"""Copying rows between databases keeps every value, and keeps ids from being issued twice."""

import json
from collections.abc import Iterator

import pytest
from sqlalchemy import delete, insert, select

from invokeai.app.services.shared.database import copy as copy_module
from invokeai.app.services.shared.database.copy import copy_rows
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.database.schema.boards import board_images, boards
from invokeai.app.services.shared.database.schema.image_index import image_embeddings
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.models import models
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.backend.util.logging import InvokeAILogger

MODEL_CONFIG = {
    "hash": "blake3:0123",
    "base": "sdxl",
    "type": "main",
    "path": "sdxl/model.safetensors",
    "format": "checkpoint",
    "name": "Ünïcödé model",
    "source": "https://example.com/model",
    "source_type": "url",
    "file_size": 2**33,
}


@pytest.fixture
def source() -> Iterator[Database]:
    """A SQLite database with the application schema, the copy's source as the copy tool's will be."""
    database = Database.open_sqlite(None, InvokeAILogger.get_logger("test_copy"))
    with database.begin(write=True) as conn:
        metadata.create_all(conn)
    try:
        yield database
    finally:
        database.dispose()


def _create_schema(database: Database) -> None:
    with database.begin(write=True) as conn:
        metadata.create_all(conn)


def _rows(database: Database) -> dict[str, list[tuple[object, ...]]]:
    with database.begin(write=False) as conn:
        return {
            table.name: [tuple(row) for row in conn.execute(select(table).order_by(*table.primary_key.columns))]
            for table in metadata.sorted_tables
        }


def test_rows_keep_their_values_across_a_copy(source: Database, empty_database: Database) -> None:
    image = {"image_origin": "internal", "image_category": "general", "width": 8, "height": 8}
    with source.begin(write=True) as conn:
        conn.execute(
            insert(users).values(user_id="u1", email="zoë@example.com", password_hash="h", display_name="Zoë 🎨")
        )
        document = json.dumps({"x": "y" * 100_000})
        conn.execute(insert(images).values({**image, "image_name": "a.png", "starred": True, "metadata": document}))
        conn.execute(
            insert(images).values({**image, "image_name": "b.png", "is_intermediate": None, "file_size_bytes": 2**40})
        )
        conn.execute(insert(boards).values(board_id="b1", board_name="Board", cover_image_name="a.png", user_id="u1"))
        conn.execute(insert(board_images).values(board_id="b1", image_name="a.png"))
        conn.execute(
            insert(image_embeddings).values(image_name="a.png", model_id="clip", dim=2, embedding=bytes(range(256)))
        )
        conn.execute(
            insert(videos).values(
                video_name="v.mp4", video_origin="internal", video_category="general", width=8, height=8, duration=1 / 3
            )
        )
        conn.execute(insert(models).values(id="m1", config=json.dumps(MODEL_CONFIG)))
    _create_schema(empty_database)

    copied = copy_rows(source, empty_database)

    assert copied["images"] == 2 and copied["board_images"] == 1 and copied["models"] == 1
    assert _rows(empty_database) == _rows(source)


def test_ids_continue_after_the_highest_the_source_issued(source: Database, empty_database: Database) -> None:
    item = {"batch_id": "b", "queue_id": "default", "session": "{}"}
    with source.begin(write=True) as conn:
        conn.execute(insert(session_queue), [{**item, "session_id": f"s{n}"} for n in (1, 2, 3)])
        # Item 3 was issued and is gone; its id must not come back.
        conn.execute(delete(session_queue).where(session_queue.c.session_id == "s3"))
    _create_schema(empty_database)

    copy_rows(source, empty_database, tables=[session_queue])
    with empty_database.begin(write=True) as conn:
        inserted = conn.execute(insert(session_queue).values({**item, "session_id": "s4"})).inserted_primary_key

    assert inserted is not None
    assert inserted.item_id == 4


def test_a_table_is_copied_in_batches(
    source: Database, empty_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(copy_module, "_BATCH_ROWS", 2)
    item = {"batch_id": "b", "queue_id": "default", "session": "{}"}
    with source.begin(write=True) as conn:
        conn.execute(insert(session_queue), [{**item, "session_id": f"s{n}"} for n in range(5)])
    _create_schema(empty_database)

    copied = copy_rows(source, empty_database, tables=[session_queue])

    assert copied == {"session_queue": 5}
    assert _rows(empty_database)["session_queue"] == _rows(source)["session_queue"]
