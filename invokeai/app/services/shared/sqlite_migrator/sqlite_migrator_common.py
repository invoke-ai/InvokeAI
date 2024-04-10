import sqlite3
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


@runtime_checkable
class MigrateCallback(Protocol):
    """
    A callback that performs a migration.

    Migrate callbacks are provided an open cursor to the database. They should not commit their
    transaction; this is handled by the migrator.

    If the callback needs to access additional dependencies, will be provided to the callback at runtime.

    See :class:`Migration` for an example.
    """

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        ...


class MigrationError(RuntimeError):
    """Raised when a migration fails."""


class MigrationVersionError(ValueError):
    """Raised when a migration version is invalid."""


class Migration(BaseModel):
    """
    Represents a migration for a SQLite database.

    :param from_version: The database version on which this migration may be run
    :param to_version: The database version that results from this migration
    :param migrate_callback: The callback to run to perform the migration

    Migration callbacks will be provided an open cursor to the database. They should not commit their
    transaction; this is handled by the migrator.

    It is suggested to use a class to define the migration callback and a builder function to create
    the :class:`Migration`. This allows the callback to be provided with additional dependencies and
    keeps things tidy, as all migration logic is self-contained.

    Example:
    ```py
    # Define the migration callback class
    class Migration1Callback:
        # This migration needs a logger, so we define a class that accepts a logger in its constructor.
        def __init__(self, image_files: ImageFileStorageBase) -> None:
            self._image_files = ImageFileStorageBase

        # This dunder method allows the instance of the class to be called like a function.
        def __call__(self, cursor: sqlite3.Cursor) -> None:
            self._add_with_banana_column(cursor)
            self._do_something_with_images(cursor)

        def _add_with_banana_column(self, cursor: sqlite3.Cursor) -> None:
            \"""Adds the with_banana column to the sushi table.\"""
            # Execute SQL using the cursor, taking care to *not commit* a transaction
            cursor.execute('ALTER TABLE sushi ADD COLUMN with_banana BOOLEAN DEFAULT TRUE;')

        def _do_something_with_images(self, cursor: sqlite3.Cursor) -> None:
            \"""Does something with the image files service.\"""
            self._image_files.get(...)

    # Define the migration builder function. This function creates an instance of the migration callback
    # class and returns a Migration.
    def build_migration_1(image_files: ImageFileStorageBase) -> Migration:
        \"""Builds the migration from database version 0 to 1.
        Requires the image files service to...
        \"""

        migration_1 = Migration(
            from_version=0,
            to_version=1,
            migrate_callback=Migration1Callback(image_files=image_files),
        )

        return migration_1

    # Register the migration after all dependencies have been initialized
    db = SqliteDatabase(db_path, logger)
    migrator = SqliteMigrator(db)
    migrator.register_migration(build_migration_1(image_files))
    migrator.run_migrations()
    ```
    """

    from_version: int = Field(ge=0, strict=True, description="The database version on which this migration may be run")
    to_version: int = Field(ge=1, strict=True, description="The database version that results from this migration")
    callback: MigrateCallback = Field(description="The callback to run to perform the migration")

    @model_validator(mode="after")
    def validate_to_version(self) -> "Migration":
        """Validates that to_version is one greater than from_version."""
        if self.to_version != self.from_version + 1:
            raise MigrationVersionError("to_version must be one greater than from_version")
        return self

    def __hash__(self) -> int:
        # Callables are not hashable, so we need to implement our own __hash__ function to use this class in a set.
        return hash((self.from_version, self.to_version))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MigrationSet:
    """
    A set of Migrations. Performs validation during migration registration and provides utility methods.

    Migrations should be registered with `register()`. Once all are registered, `validate_migration_chain()`
    should be called to ensure that the migrations form a single chain of migrations from version 0 to the latest version.
    """

    def __init__(self) -> None:
        self._migrations: set[Migration] = set()

    def register(self, migration: Migration) -> None:
        """Registers a migration."""
        migration_from_already_registered = any(m.from_version == migration.from_version for m in self._migrations)
        migration_to_already_registered = any(m.to_version == migration.to_version for m in self._migrations)
        if migration_from_already_registered or migration_to_already_registered:
            raise MigrationVersionError("Migration with from_version or to_version already registered")
        self._migrations.add(migration)

    def get(self, from_version: int) -> Optional[Migration]:
        """Gets the migration that may be run on the given database version."""
        # register() ensures that there is only one migration with a given from_version, so this is safe.
        return next((m for m in self._migrations if m.from_version == from_version), None)

    def validate_migration_chain(self) -> None:
        """
        Validates that the migrations form a single chain of migrations from version 0 to the latest version,
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
