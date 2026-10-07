# Application backend

## Ownership and contracts

- Routers validate requests, enforce authentication/authorization, and map service errors to HTTP. Services own domain policy; storage owns persistence.
- Before changing flows, inspect `api/dependencies.py`, affected routers, service base/default/storage implementations, and tests. Preserve dependency/start/stop lifecycles; avoid parallel global services.
- Server-authorize every account-owned resource, including queries, writes, events, exports, and recovery; UI guards are insufficient.
- Trace frontend/saved-workflow consumers before changing Pydantic models, invocation fields/versions, HTTP DTOs, or socket-event names, defaults, validation, or errors.
- API changes may require regenerating `frontend/api/openapi.json` and `frontend/api/schema.ts` under `invokeai/`, even for webv2. Follow shared API package generation guidance; never hand-edit output.

## Persistence and lifecycle

- Preserve atomicity, revision checks, idempotency, and explicit conflicts. Inspect migrations/old-record readers for stored-data changes; add needed migration/compatibility tests.
- SQL lives only in `services/shared/database/`, reached through `Database.queries`; `tests/app/test_database_access_guard.py` enforces this. Services not yet ported still use the transitional cursor facade in `services/shared/sqlite/`; do not add SQL to them. Query modules build their statements once, at module level with `bindparam()` (a statement built per call costs several times its execution), read rows by position, and build DTOs with `@mapped(...)` outside the transaction. A shape-dependent statement comes from a cached builder whose shapes a client cannot multiply (normalize its lists and flags, round its list lengths; see the guide).
- Use one transaction owner for multi-record writes: `with db.queries.transaction() as q:`, passing `q` (or, for unported services, the cursor) to helpers. A second `queries` transaction on the same thread raises `NestedTransactionError`; a nested cursor transaction of the facade joins the outer one. See `services/shared/database/database.py`.
- SQLite serialises everything behind one lock; MySQL/MariaDB do not. Put an invariant like "only if still pending" into the statement (a conditional `UPDATE` with a checked row count), not into a read followed by a write. An invariant that spans rows (the last active administrator) gets a `DatabaseLock`, which every transaction changing what it reads acquires as its first call (`q.locks.acquire(...)`); its row is added by a migration. A decision about one row (may this board be claimed?) locks that row before reading what it decides on (a `@locking` query method, `SELECT ... FOR UPDATE`); every transaction changing what it reads writes or locks the same row, and rows are locked in one order (a project's before its board's). Run a unit whose work has no effects outside the database with `db.queries.run(work)`, which retries it after a deadlock.
- Tables are defined in `services/shared/database/schema/`, which must match the migrated SQLite schema exactly (`test_schema_parity.py`). Change the schema with a migration and the same change to the metadata, in one change. No database has triggers: queries set `updated_at` (also in upserts) and the queue's `started_at`, `completed_at`, `session_revision`.
- New migrations are `PortableMigration`s (Alembic operations; tables via `context.create_table()`) and idempotent: MySQL/MariaDB commit DDL as it runs, so a migration that failed halfway runs again. On SQLite they run with foreign keys off and are checked before commit. Migrations up to `PORTABLE_CUTOVER` stay SQLite-only and unchanged. See the Database Migrations guide.
- Preserve project/account isolation, document bounds, queue admission/receipts, and lost-response retry safety. Never silently discard recovery data or resolve conflicts with last-write-wins.
- Bound concurrency; support cancellation, timeouts/retries, and file/process/listener/task cleanup. Keep CPU/blocking I/O off the async server loop.
- Preserve file/path validation, remote-fetch protections, and sanitized errors. Never modify real user models, databases, outputs, or credentials during tests/debugging.

## Efficiency and verification

- Inspect queries for repeated lookups, full scans/materialization, missing bounds, and serialization. Measure representative data/query counts before adding caches or indexes.
- Use real temporary SQLite for transaction, migration, revision, and ownership tests; cover affected malformed input, cross-user access, cancellation, repeated requests, and partial failures. Tests on the database fixtures also run against MySQL and MariaDB when `INVOKEAI_TEST_DB_URL` is set (see `tests/fixtures/database.py`).
- Read `tests/AGENTS.md`; use `tests/app/` or the existing owning suite. Run focused pytest and root Ruff; broaden to relevant service/router/invocation suites for milestones.
- For graph execution, consult `services/shared/README.md`; test scheduling and invocation lifecycles through owning interfaces.
- Report schema generation, migration validation, and unavailable integration/hardware checks.
