"""Existing images and videos read as unmarked; a rating stays within 0-5 and nothing is both picked and rejected."""

from pathlib import Path

import pytest
from sqlalchemy import column, delete, insert, inspect, select, table, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import CheckViolation
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.database.schema.videos import videos
from tests.fixtures.database import migrate_to_newest
from tests.fixtures.migration_state import (
    alembic_operations,
    build_application_migrations,
    migrate_to_before,
    run_migrations,
)

MIGRATION_ID = "2026_11_24_add_media_marks"
_MARKS = ("rating", "rejected", "color_label")

# As they were before the migration, as far as a row needs.
_images = table(
    "images",
    column("image_name"),
    column("image_origin"),
    column("image_category"),
    column("width"),
    column("height"),
    column("starred"),
)
_videos = table(
    "videos", column("video_name"), column("video_origin"), column("video_category"), column("width"), column("height")
)


@pytest.fixture
def migrated(empty_database: Database, tmp_path: Path) -> Database:
    """A starred image and a video from before this migration, then the migration, twice."""
    migrations = build_application_migrations(tmp_path)
    migrate_to_before(empty_database, migrations, MIGRATION_ID)
    with empty_database.begin(write=True) as conn:
        conn.execute(
            insert(_images).values(
                image_name="a.png", image_origin="internal", image_category="general", width=1, height=1, starred=True
            )
        )
        conn.execute(
            insert(_videos).values(
                video_name="a.mp4", video_origin="internal", video_category="general", width=1, height=1
            )
        )
    assert run_migrations(empty_database, migrations)
    assert not run_migrations(empty_database, migrations)
    return empty_database


def test_existing_media_read_as_unmarked(migrated: Database) -> None:
    with migrated.begin(write=False) as conn:
        image = conn.execute(select(images.c.starred, images.c.rating, images.c.rejected, images.c.color_label)).one()
        video = conn.execute(select(videos.c.starred, videos.c.rating, videos.c.rejected, videos.c.color_label)).one()

    assert tuple(image) == (True, 0, False, None)
    assert tuple(video) == (False, 0, False, None)


@pytest.mark.parametrize("media", [images, videos], ids=["images", "videos"])
@pytest.mark.parametrize(
    "values", [{"rating": 6}, {"rating": -1}, {"starred": True, "rejected": True}], ids=["above", "below", "both"]
)
def test_the_checks_refuse_an_out_of_range_rating_and_a_rejected_pick(
    migrated: Database, media: object, values: dict[str, object]
) -> None:
    with pytest.raises(CheckViolation):
        with migrated.begin(write=True) as conn:
            conn.execute(update(media).values(**values))  # type: ignore[arg-type]


def test_a_run_that_stopped_halfway_completes_what_it_left(database: Database, tmp_path: Path) -> None:
    # On a server each DDL statement commits as it runs: a run that failed after the images leaves them done, the
    # videos undone and the migration unrecorded.
    with database.begin(write=True) as conn:
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == MIGRATION_ID))
        if conn.dialect.name == "sqlite":
            for name in _MARKS:
                conn.exec_driver_sql(f"ALTER TABLE videos DROP COLUMN {name}")
        else:
            operations = alembic_operations(conn)
            for check in ("rating", "flag"):
                operations.drop_constraint(f"ck_videos_{check}", "videos", type_="check")
            for name in _MARKS:
                operations.drop_column("videos", name)

    migrate_to_newest(database, tmp_path)

    with database.begin(write=False) as conn:
        assert set(_MARKS) <= {entry["name"] for entry in inspect(conn).get_columns("videos")}
        recorded = select(applied_migrations.c.migration_id).where(applied_migrations.c.migration_id == MIGRATION_ID)
        assert conn.execute(recorded).first() is not None
    with pytest.raises(CheckViolation):
        with database.begin(write=True) as conn:
            conn.execute(
                insert(videos).values(
                    video_name="a.mp4", video_origin="internal", video_category="general", width=1, height=1, rating=6
                )
            )
