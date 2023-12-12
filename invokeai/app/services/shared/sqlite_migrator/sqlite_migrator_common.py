import sqlite3
from functools import partial
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


@runtime_checkable
class MigrateCallback(Protocol):
    """A callback that performs a migration."""

    def __call__(self, cursor: sqlite3.Cursor, **kwargs: Any) -> None:
        ...


class MigrationError(RuntimeError):
    """Raised when a migration fails."""


class MigrationVersionError(ValueError):
    """Raised when a migration version is invalid."""


class MigrationDependency:
    """Represents a dependency for a migration."""

    def __init__(
        self,
        name: str,
        dependency_type: Any,
    ) -> None:
        self.name = name
        self.dependency_type = dependency_type
        self.value = None

    def set_value(self, value: Any) -> None:
        """
        Sets the value of the dependency.
        If the value is not of the correct type, a TypeError is raised.
        """
        if not isinstance(value, self.dependency_type):
            raise TypeError(f"Dependency {self.name} must be of type {self.dependency_type}")
        self.value = value


class Migration(BaseModel):
    """
    Represents a migration for a SQLite database.

    :param from_version: The database version on which this migration may be run
    :param to_version: The database version that results from this migration
    :param migrate: The callback to run to perform the migration
    :param dependencies: A dict of dependencies that must be provided to the migration

    Migration callbacks will be provided an open cursor to the database. They should not commit their
    transaction; this is handled by the migrator.

    If a migration needs an additional dependency, it must be provided with :meth:`provide_dependency`
    before the migration is run.

    Example Usage:
    ```py
    # Define the migrate callback
    def migrate_callback(cursor: sqlite3.Cursor, **kwargs) -> None:
        some_dependency = kwargs["some_dependency"]
        ...

    # Instantiate the migration, declaring dependencies
    migration = Migration(
        from_version=0,
        to_version=1,
        migrate_callback=migrate_callback,
        dependencies={"some_dependency": MigrationDependency(name="some_dependency", dependency_type=SomeType)},
    )

    # Register the dependency before running the migration
    migration.provide_dependency(name="some_dependency", value=some_value)
    ```
    """

    from_version: int = Field(ge=0, strict=True, description="The database version on which this migration may be run")
    to_version: int = Field(ge=1, strict=True, description="The database version that results from this migration")
    migrate_callback: MigrateCallback = Field(description="The callback to run to perform the migration")
    dependencies: dict[str, MigrationDependency] = Field(
        default={}, description="A dict of dependencies that must be provided to the migration"
    )

    @model_validator(mode="after")
    def validate_to_version(self) -> "Migration":
        if self.to_version != self.from_version + 1:
            raise ValueError("to_version must be one greater than from_version")
        return self

    def __hash__(self) -> int:
        # Callables are not hashable, so we need to implement our own __hash__ function to use this class in a set.
        return hash((self.from_version, self.to_version))

    def provide_dependency(self, name: str, value: Any) -> None:
        """Provides a dependency for this migration."""
        if name not in self.dependencies:
            raise ValueError(f"{name} of type {type(value)} is not a dependency of this migration")
        self.dependencies[name].set_value(value)

    def run(self, cursor: sqlite3.Cursor) -> None:
        """
        Runs the migration.
        If any dependencies are missing, a MigrationError is raised.
        """
        missing_dependencies = [d.name for d in self.dependencies.values() if d.value is None]
        if missing_dependencies:
            raise MigrationError(f"Missing migration dependencies: {', '.join(missing_dependencies)}")
        self.migrate_callback = partial(self.migrate_callback, **{d.name: d.value for d in self.dependencies.values()})
        self.migrate_callback(cursor=cursor)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MigrationSet:
    """A set of Migrations. Performs validation during migration registration and provides utility methods."""

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
