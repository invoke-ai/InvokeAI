"""Repair the `projects` table on databases that skipped this fork's migration_33.

Numeric migrations are keyed by `migration_<to_version>`, and a database records the ids it has
applied. This fork's `migration_33` creates `projects`; upstream's `migration_33` creates the
image-subfolder move tables (which this fork carries as `migration_34`). The two are different
migrations sharing one id.

So a database that was ever opened by an upstream build has `migration_33` recorded, and this
fork then treats its own `projects` migration as already applied and skips it forever.
`migration_34` is no help — its DDL is all `IF NOT EXISTS`, so it quietly no-ops. The user is
left with a working install whose `projects` queries fail with `no such table: projects`.

This migration has a dated id, so it is unaffected by that collision: it runs exactly once on
every database, repairs the ones that are missing the table, and no-ops on the ones that are
not. The DDL below is a verbatim copy of `migration_33`'s, with `IF NOT EXISTS` added — keep the
two in sync if the schema ever changes.

Note for future migrations: do not add more numeric migrations to this fork. Every new migration
should use a dated id, which cannot collide with an upstream migration number.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class RepairProjectsTableMigrationCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_projects_table(cursor)

    def _create_projects_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS projects (
                -- Client-generated identifier; unique per user, not globally.
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                -- Opaque client-owned project document (JSON).
                data TEXT NOT NULL,
                -- Incremented on every update; used for optimistic concurrency.
                revision INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (user_id, project_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_projects_updated_at
            AFTER UPDATE ON projects
            FOR EACH ROW
            BEGIN
              UPDATE projects
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
              WHERE user_id = OLD.user_id AND project_id = OLD.project_id;
            END;
            """
        )


def build_migration() -> Migration:
    """Build the migration that repairs a missing `projects` table."""
    return Migration(
        id="2026_07_30_repair_projects_table",
        # migration_33 is where `projects` should have been created. Depending on it is correct in
        # both cases: on a fork database it genuinely ran, and on an upstream-originated database
        # its id is recorded (by upstream's unrelated migration), so the dependency still resolves.
        #
        # It does *not* guarantee migration_27's `users` table, for exactly the reason this
        # migration exists: an upstream-originated database has migration_27's id recorded too,
        # against upstream's unrelated migration of that number, so the fork's never runs there.
        # The FK below survives that because SQLite does not resolve a foreign key until DML — but
        # anything that *reads* `users` has to check first, which is what
        # `2026_08_06_add_project_boards` does.
        depends_on="migration_33",
        callback=RepairProjectsTableMigrationCallback(),
    )
