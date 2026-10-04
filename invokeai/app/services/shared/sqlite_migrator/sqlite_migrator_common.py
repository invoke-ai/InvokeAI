import sqlite3
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Optional, Protocol, runtime_checkable

from alembic.operations import Operations
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import Connection, MetaData, Table
from sqlalchemy.schema import SchemaItem
from typing_extensions import Self

from invokeai.app.services.shared.database.schema.metadata import define_table
from invokeai.app.services.shared.database.schema.metadata import metadata as schema_metadata


@runtime_checkable
class MigrateCallback(Protocol):
    """
    A callback that performs a migration.

    Migrate callbacks are provided an open cursor to the database. They should not commit their
    transaction; this is handled by the migrator.

    If the callback needs to access additional dependencies, will be provided to the callback at runtime.

    See :class:`Migration` for an example.
    """

    def __call__(self, cursor: sqlite3.Cursor) -> None: ...


@dataclass(frozen=True)
class PortableMigrationContext:
    """What a portable migration works with.

    :param conn: The connection of the migration's transaction.
    :param op: Alembic's operations (`op.add_column`, `op.create_index`, ...), bound to `conn`.
    :param logger: The logger.
    :param metadata: The tables this migration creates. A foreign key names a table here: one created earlier
        in the migration, or an existing one loaded with `Table(name, context.metadata, autoload_with=conn)`.
    """

    conn: Connection
    op: Operations
    logger: Logger
    metadata: MetaData = field(default_factory=lambda: MetaData(naming_convention=schema_metadata.naming_convention))

    def create_table(self, name: str, *items: SchemaItem, **options: Any) -> Table:
        """Creates a table the way the schema metadata creates its tables. Use it instead of `op.create_table`.

        It gets the server table options (InnoDB, the binary collation, DYNAMIC rows, whatever the server's
        defaults) and each backend's rules for defaults, generated columns and per-backend indexes, which
        Alembic's operations know nothing of. Items are given as to `schema.metadata.table()`, with indexes
        naming their columns.
        """
        created = define_table(self.metadata, name, *items, **options)
        created.create(self.conn)
        return created


@runtime_checkable
class PortableMigrateCallback(Protocol):
    """A callback that performs a migration on any database backend, with Alembic's operations.

    It runs in a transaction the migrator commits. On MySQL and MariaDB every DDL statement commits on its own,
    though, so a portable migration must be idempotent: run again after it failed halfway, it finds part of its
    work done and completes the rest. On SQLite it runs with foreign keys off, so that rebuilding a table
    (Alembic's batch mode) does not cascade into the tables that reference it; they are checked before the
    migration commits, so rows it orphans fail it. The schema metadata in `shared/database/schema/` gets the
    same change in the same commit, because new server databases are created from it.
    """

    def __call__(self, context: PortableMigrationContext) -> None: ...


class MigrationError(RuntimeError):
    """Raised when a migration fails."""


class MigrationVersionError(ValueError):
    """Raised when a migration version is invalid."""


class MigrationBase(BaseModel):
    """
    What every migration has: a stable ID, and the migration that must run first.

    :param from_version: The legacy database version on which this migration may be run
    :param to_version: The legacy database version that results from this migration
    :param id: The stable migration ID. Legacy migrations default to ``migration_{to_version}``.
    :param depends_on: The stable ID of the migration that must run first.

    Migrations are executed according to their stable ID dependencies. Existing legacy migrations also keep
    ``from_version`` and ``to_version`` so older numeric migration state can be mapped to applied migration IDs.
    New graph-only migrations omit legacy versions, and must provide an explicit ``id``.
    """

    from_version: Optional[int] = Field(
        default=None, ge=0, strict=True, description="The database version on which this migration may be run"
    )
    to_version: Optional[int] = Field(
        default=None, ge=1, strict=True, description="The database version that results from this migration"
    )
    id: Optional[str] = Field(default=None, description="Stable migration ID")
    depends_on: Optional[str] = Field(default=None, description="Stable ID of the migration dependency")

    @model_validator(mode="after")
    def validate_versions_and_ids(self) -> Self:
        """Validates legacy versions and derives stable IDs for legacy migrations."""
        has_from_version = self.from_version is not None
        has_to_version = self.to_version is not None
        if has_from_version != has_to_version:
            raise MigrationVersionError("from_version and to_version must both be provided for legacy migrations")
        if self.from_version is not None and self.to_version is not None and self.to_version != self.from_version + 1:
            raise MigrationVersionError("to_version must be one greater than from_version")
        if self.id is None and self.to_version is not None:
            self.id = f"migration_{self.to_version}"
        if self.id is None:
            raise MigrationVersionError("id is required for graph-only migrations")
        if self.depends_on is None and self.from_version is not None and self.from_version > 0:
            self.depends_on = f"migration_{self.from_version}"
        if self.depends_on == self.id:
            raise MigrationVersionError("migration cannot depend on itself")
        return self

    def __hash__(self) -> int:
        # Callables are not hashable, so we need to implement our own __hash__ function to use this class in a set.
        if self.from_version is not None and self.to_version is not None:
            return hash((self.from_version, self.to_version))
        return hash(self.id)

    @property
    def sort_key(self) -> tuple[int, int, str]:
        """Deterministic sort key for runnable migrations."""
        if self.to_version is None:
            return (1, 0, self.id or "")
        return (0, self.to_version, self.id or "")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Migration(MigrationBase):
    """
    A migration that runs on SQLite only, given an open cursor. Every migration up to ``PORTABLE_CUTOVER`` is one;
    new migrations are portable (:class:`PortableMigration`).

    :param callback: The callback to run to perform the migration. It is provided an open cursor, and does not
        commit; the migrator does.
    """

    callback: MigrateCallback = Field(description="The callback to run to perform the migration")


class PortableMigration(MigrationBase):
    """
    A migration that runs on every database backend: a callback using Alembic's operations (see
    :class:`PortableMigrateCallback`). Every migration after ``PORTABLE_CUTOVER`` is one.

    Example:
    ```py
    def _add_bananas(context: PortableMigrationContext) -> None:
        # Idempotent: on MySQL and MariaDB, a failed run may have created the table already.
        if not inspect(context.conn).has_table("bananas"):
            context.create_table(
                "bananas",
                Column("banana_id", Key(), primary_key=True),
                Column("ripeness", BigInt(), nullable=False, server_default=default(0)),
                inserted_at(),
            )


    def build_migration() -> PortableMigration:
        return PortableMigration(
            id="2026_10_05_add_bananas",
            depends_on="2026_10_01_add_anima_variant",
            callback=_add_bananas,
        )
    ```
    """

    callback: PortableMigrateCallback = Field(description="The callback to run to perform the migration")

    @model_validator(mode="after")
    def validate_no_legacy_version(self) -> Self:
        """A portable migration is identified by its ID alone."""
        if self.from_version is not None or self.to_version is not None:
            raise MigrationVersionError("a portable migration has no legacy version")
        return self


class MigrationSet:
    """
    A set of Migrations. Performs validation during migration registration and provides utility methods.

    Migrations should be registered with `register()`. Once all are registered, `validate_dependency_graph()`
    should be called to ensure that dependencies are complete and acyclic. `validate_migration_chain()` is retained for
    legacy chain validation tests and compatibility checks.
    """

    def __init__(self) -> None:
        self._migrations: set[MigrationBase] = set()

    def register(self, migration: MigrationBase) -> None:
        """Registers a migration."""
        migration_from_already_registered = migration.from_version is not None and any(
            m.from_version == migration.from_version for m in self._migrations if m.from_version is not None
        )
        migration_to_already_registered = migration.to_version is not None and any(
            m.to_version == migration.to_version for m in self._migrations if m.to_version is not None
        )
        if migration_from_already_registered or migration_to_already_registered:
            raise MigrationVersionError("Migration with from_version or to_version already registered")
        migration_id_already_registered = any(m.id == migration.id for m in self._migrations)
        if migration_id_already_registered:
            raise MigrationVersionError("Migration with id already registered")
        self._migrations.add(migration)

    def get(self, from_version: int) -> Optional[MigrationBase]:
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

    def validate_dependency_graph(self) -> None:
        """Validates migration ID dependencies."""
        migration_ids = {migration.id for migration in self._migrations}
        migrations_by_id = self.migrations_by_id
        for migration in self._migrations:
            if migration.depends_on is not None and migration.depends_on not in migration_ids:
                raise MigrationError(
                    f"Migration '{migration.id}' depends on unknown migration '{migration.depends_on}'"
                )
            if migration.to_version is not None and migration.depends_on is not None:
                dependency = migrations_by_id[migration.depends_on]
                if dependency.to_version is None:
                    raise MigrationError(
                        f"Legacy migration '{migration.id}' cannot depend on graph-only migration '{dependency.id}'"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(migration: MigrationBase) -> None:
            migration_id = migration.id
            if migration_id is None:
                raise MigrationError("Migration is missing id")
            if migration_id in visited:
                return
            if migration_id in visiting:
                raise MigrationError("Migration dependency graph contains a cycle")
            visiting.add(migration_id)
            if migration.depends_on is not None:
                visit(migrations_by_id[migration.depends_on])
            visiting.remove(migration_id)
            visited.add(migration_id)

        for migration in self._migrations:
            visit(migration)

    def get_migration_plan(self, applied_migration_ids: set[str]) -> list[MigrationBase]:
        """Gets a deterministic migration plan from the set of applied migration IDs."""
        self.validate_dependency_graph()
        known_migration_ids = set(self.migrations_by_id)
        unknown_applied_ids = applied_migration_ids - known_migration_ids
        if unknown_applied_ids:
            unknown_ids = ", ".join(sorted(unknown_applied_ids))
            raise MigrationError(f"Database contains unknown applied migration IDs: {unknown_ids}")

        plan: list[MigrationBase] = []
        planned_or_applied_ids = set(applied_migration_ids)
        remaining = {
            migration.id: migration for migration in self._migrations if migration.id not in applied_migration_ids
        }

        while remaining:
            runnable = sorted(
                (
                    migration
                    for migration in remaining.values()
                    if migration.depends_on is None or migration.depends_on in planned_or_applied_ids
                ),
                key=lambda migration: migration.sort_key,
            )
            if not runnable:
                raise MigrationError("Migration dependency graph cannot be resolved")
            migration = runnable[0]
            plan.append(migration)
            planned_or_applied_ids.add(migration.id or "")
            del remaining[migration.id]
        return plan

    @property
    def count(self) -> int:
        """The count of registered migrations."""
        return len(self._migrations)

    @property
    def latest_version(self) -> int:
        """Gets latest to_version among registered migrations. Returns 0 if there are no migrations registered."""
        if self.count == 0:
            return 0
        legacy_migrations = [migration for migration in self._migrations if migration.to_version is not None]
        if len(legacy_migrations) == 0:
            return 0
        latest_version = sorted(legacy_migrations, key=lambda m: m.to_version or 0)[-1].to_version
        return latest_version or 0

    @property
    def migrations(self) -> tuple[MigrationBase, ...]:
        return tuple(sorted(self._migrations, key=lambda migration: migration.sort_key))

    @property
    def migrations_by_id(self) -> dict[str, MigrationBase]:
        return {migration.id or "": migration for migration in self._migrations}
