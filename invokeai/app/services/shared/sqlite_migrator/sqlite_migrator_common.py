import sqlite3
from typing import Callable, Optional, TypeAlias

from pydantic import BaseModel, Field, model_validator

MigrateCallback: TypeAlias = Callable[[sqlite3.Cursor], None]


class MigrationError(RuntimeError):
    """Raised when a migration fails."""


class MigrationVersionError(ValueError):
    """Raised when a migration version is invalid."""


class Migration(BaseModel):
    """
    Represents a migration for a SQLite database.

    Migration callbacks will be provided an open cursor to the database. They should not commit their
    transaction; this is handled by the migrator.

    Pre- and post-migration callback may be registered with :meth:`register_pre_callback` or
    :meth:`register_post_callback`.

    If a migration has additional dependencies, it is recommended to use functools.partial to provide
    the dependencies and register the partial as the migration callback.
    """

    from_version: int = Field(ge=0, strict=True, description="The database version on which this migration may be run")
    to_version: int = Field(ge=1, strict=True, description="The database version that results from this migration")
    migrate: MigrateCallback = Field(description="The callback to run to perform the migration")
    pre_migrate: list[MigrateCallback] = Field(
        default=[], description="A list of callbacks to run before the migration"
    )
    post_migrate: list[MigrateCallback] = Field(
        default=[], description="A list of callbacks to run after the migration"
    )

    @model_validator(mode="after")
    def validate_to_version(self) -> "Migration":
        if self.to_version <= self.from_version:
            raise ValueError("to_version must be greater than from_version")
        return self

    def __hash__(self) -> int:
        # Callables are not hashable, so we need to implement our own __hash__ function to use this class in a set.
        return hash((self.from_version, self.to_version))

    def register_pre_callback(self, callback: MigrateCallback) -> None:
        """Registers a pre-migration callback."""
        self.pre_migrate.append(callback)

    def register_post_callback(self, callback: MigrateCallback) -> None:
        """Registers a post-migration callback."""
        self.post_migrate.append(callback)


class MigrationSet:
    """A set of Migrations. Performs validation during migration registration and provides utility methods."""

    def __init__(self) -> None:
        self._migrations: set[Migration] = set()

    def register(self, migration: Migration) -> None:
        """Registers a migration."""
        if any(m.from_version == migration.from_version for m in self._migrations):
            raise MigrationVersionError(f"Migration from {migration.from_version} already registered")
        if any(m.to_version == migration.to_version for m in self._migrations):
            raise MigrationVersionError(f"Migration to {migration.to_version} already registered")
        self._migrations.add(migration)

    def get(self, from_version: int) -> Optional[Migration]:
        """Gets the migration that may be run on the given database version."""
        # register() ensures that there is only one migration with a given from_version, so this is safe.
        return next((m for m in self._migrations if m.from_version == from_version), None)

    def validate_migration_chain(self) -> None:
        """
        Validates that the migrations form a single chain of migrations from version 0 to the latest version.
        Raises a MigrationError if there is a problem.
        """
        if self.count == 0:
            return
        if self.latest_version == 0:
            return
        next_migration = self.get(from_version=0)
        if next_migration is None:
            raise MigrationError("Migration chain is fragmented")
        touched_count = 1
        while next_migration is not None:
            next_migration = self.get(next_migration.to_version)
            if next_migration is not None:
                touched_count += 1
        if touched_count != self.count:
            raise MigrationError("Migration chain is fragmented")

    @property
    def count(self) -> int:
        """The count of registered migrations."""
        return len(self._migrations)

    @property
    def latest_version(self) -> int:
        """Gets latest to_version among registered migrations. Returns 0 if there are no migrations registered."""
        if self.count == 0:
            return 0
        return sorted(self._migrations, key=lambda m: m.to_version)[-1].to_version
