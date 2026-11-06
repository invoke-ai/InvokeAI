"""`image_precise_assets` is added to databases that already hold images, which then have no asset."""

from pathlib import Path

import pytest
from sqlalchemy import insert, inspect, select

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import ForeignKeyViolation
from invokeai.app.services.shared.database.schema.image_precise_assets import image_precise_assets
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_11_03_add_image_precise_assets import (
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from tests.fixtures.migration_state import build_application_migrations, migrate_to_before, run_migrations

MIGRATION_ID = "2026_11_03_add_image_precise_assets"


def _image(name: str) -> dict[str, object]:
    return {"image_name": name, "image_origin": "internal", "image_category": "general", "width": 8, "height": 8}


def _asset(name: str) -> dict[str, object]:
    return {"image_name": name, "asset_format": "raster_safetensors", "format_version": 1}


@pytest.fixture
def migrations(tmp_path: Path) -> list[MigrationBase]:
    return build_application_migrations(tmp_path)


@pytest.fixture
def before(empty_database: Database, migrations: list[MigrationBase]) -> Database:
    """A database just before this migration, holding an image."""
    migrate_to_before(empty_database, migrations, MIGRATION_ID)
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(images).values(_image("a.png")))
    return empty_database


def test_the_builder_follows_the_previous_head() -> None:
    migration = build_migration()

    assert migration.id == MIGRATION_ID
    assert migration.depends_on == "2026_10_28_add_moss_image_enhancer_system_prompt"


def test_existing_images_get_the_table_and_have_no_asset(before: Database, migrations: list[MigrationBase]) -> None:
    assert run_migrations(before, migrations)

    with before.begin(write=False) as conn:
        columns = {entry["name"]: entry for entry in inspect(conn).get_columns("image_precise_assets")}
    assert set(columns) == {"image_name", "asset_format", "format_version", "created_at"}
    assert not any(column["nullable"] for column in columns.values())
    assert before.queries.image_precise_assets.get("a.png") is None
    assert not run_migrations(before, migrations)


def test_a_run_that_stopped_after_creating_the_table_finishes(
    before: Database, migrations: list[MigrationBase]
) -> None:
    """On a server the table commits as it is created: a run that failed afterwards runs again on it."""
    with before.begin(write=True) as conn:
        image_precise_assets.create(conn)

    assert run_migrations(before, migrations)

    with before.begin(write=False) as conn:
        assert inspect(conn).has_table("image_precise_assets")
    assert not run_migrations(before, migrations)


def test_the_migrated_table_checks_images_and_cascades(before: Database, migrations: list[MigrationBase]) -> None:
    assert run_migrations(before, migrations)

    with pytest.raises(ForeignKeyViolation):
        with before.begin(write=True) as conn:
            conn.execute(insert(image_precise_assets).values(_asset("missing.png")))

    with before.begin(write=True) as conn:
        conn.execute(insert(image_precise_assets).values(_asset("a.png")))
        conn.execute(images.delete().where(images.c.image_name == "a.png"))
    with before.begin(write=False) as conn:
        assert conn.execute(select(image_precise_assets.c.image_name)).all() == []
