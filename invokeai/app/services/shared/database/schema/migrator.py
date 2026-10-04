"""The migrator's records of which migrations a database has had."""

from sqlalchemy import Column

from invokeai.app.services.shared.database.schema.metadata import inserted_at, table
from invokeai.app.services.shared.database.types import BigInt, Key

# The schema version of the numbered migrations, the last of which, 34, ended them.
migrations = table(
    "migrations",
    Column("version", BigInt(), primary_key=True, nullable=True, autoincrement=False),
    inserted_at("migrated_at"),
)

applied_migrations = table(
    "applied_migrations",
    Column("migration_id", Key(), primary_key=True, nullable=True),
    # The numbered version of a migration from before ids, for databases those versions migrated.
    Column("legacy_version", BigInt(), unique=True),
    inserted_at("migrated_at"),
)
