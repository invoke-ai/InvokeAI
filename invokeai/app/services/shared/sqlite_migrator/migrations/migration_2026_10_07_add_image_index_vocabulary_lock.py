"""Add the `image_index_vocabulary` lock, which a replace of the image index's custom vocabulary takes first.

The replace deletes every term and inserts the new ones. On MySQL and MariaDB two replaces running side by side
would each delete what the other has not committed yet, and leave the terms of both.
"""

from sqlalchemy import column, insert, select, table

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_db_locks = table("db_locks", column("name"))

# Literal rather than imported: a migration keeps meaning what it meant when it was written.
_IMAGE_INDEX_VOCABULARY = "image_index_vocabulary"


def _add_image_index_vocabulary_lock(context: PortableMigrationContext) -> None:
    lock = select(_db_locks.c.name).where(_db_locks.c.name == _IMAGE_INDEX_VOCABULARY)
    if context.conn.execute(lock).first() is None:
        context.conn.execute(insert(_db_locks).values(name=_IMAGE_INDEX_VOCABULARY))


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_07_add_image_index_vocabulary_lock",
        depends_on="2026_10_06_add_intermediates_state",
        callback=_add_image_index_vocabulary_lock,
    )
