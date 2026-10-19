# Application backend

## Ownership and contracts

- Routers validate requests, enforce authentication/authorization, map service errors to HTTP. Services own domain policy. Storage owns persistence.
- Before changing flows, read `api/dependencies.py`, affected routers, service base/default/storage implementations, tests. Keep dependency/start/stop lifecycles. No parallel global services.
- Server-authorize every account-owned resource, including queries, writes, events, exports, recovery. UI guards are not enough.
- Before changing Pydantic models, invocation fields/versions, HTTP DTOs, or socket-event names/defaults/validation/errors: trace frontend and saved-workflow consumers.
- API changes may need regenerated `frontend/api/openapi.json` and `frontend/api/schema.ts` under `invokeai/`, webv2 included. Follow shared API package generation guidance. Never hand-edit output.

## Persistence and lifecycle

- Preserve atomicity, revision checks, idempotency, explicit conflicts. Stored-data change: read migrations and old-record readers; add needed migration/compatibility tests.
- SQL lives only in `services/shared/database/`, reached through `Database.queries`. `tests/app/test_database_access_guard.py` enforces this.
- Query modules build statements once, at module level, with `bindparam()` (per-call building costs several times execution). Read rows by position; build DTOs with `@mapped(...)` outside the transaction. Shape-dependent statement: cached builder whose shapes a client cannot multiply (normalize lists and flags, round list lengths; see the guide).
- Multi-record writes: one transaction owner, `with db.queries.transaction() as q:`, pass `q` to helpers. Second transaction on the same thread raises `NestedTransactionError`. See `services/shared/database/database.py`.
- Concurrency. SQLite serialises everything behind one lock; MySQL/MariaDB do not.
  - Single-statement invariant ("only if still pending"): put it in the statement, a conditional `UPDATE` with checked row count. Never a read then a write.
  - Invariant spanning rows (last active administrator): a `DatabaseLock`. Every transaction that changes rows the invariant reads acquires the lock as its first call (`q.locks.acquire(...)`). A migration adds its row.
  - Decision about one row (may this board be claimed?): lock that row before reading what the decision uses (`@locking` query method, `SELECT ... FOR UPDATE`). Every transaction changing those inputs writes or locks the same row. Lock rows in one order (e.g. project before its board).
  - Unit with no effects outside the database: run with `db.queries.run(work)`; it retries after a deadlock.
- Tables live in `services/shared/database/schema/` and must match the migrated SQLite schema exactly (`test_schema_parity.py`). Change schema with a migration plus the same metadata change, in one change.
- No database has triggers. Queries set `updated_at` (upserts too) and the queue's `started_at`, `completed_at`, `session_revision`.
- New migrations: `PortableMigration`s (Alembic operations; tables via `context.create_table()`), idempotent. MySQL/MariaDB commit DDL as it runs, so a half-failed migration runs again. On SQLite they run with foreign keys off and are checked before commit. Migrations up to `PORTABLE_CUTOVER` stay SQLite-only and unchanged. See the Database Migrations guide.
- Preserve project/account isolation, document bounds, queue admission/receipts, lost-response retry safety. Never silently discard recovery data. Never resolve conflicts with last-write-wins.
- Bound concurrency. Support cancellation, timeouts/retries, file/process/listener/task cleanup. Keep CPU and blocking I/O off the async server loop.
- Preserve file/path validation, remote-fetch protections, sanitized errors. Never modify real user models, databases, outputs, or credentials in tests or debugging.

## Efficiency and verification

- Check queries for repeated lookups, full scans/materialization, missing bounds, serialization. Measure representative data and query counts before adding caches or indexes.
- Transaction, migration, revision, ownership tests: real temporary SQLite. Cover affected malformed input, cross-user access, cancellation, repeated requests, partial failures. Database-fixture tests also run against MySQL and MariaDB when `INVOKEAI_TEST_DB_URL` is set (`tests/fixtures/database.py`).
- Read `tests/AGENTS.md`. Use `tests/app/` or the existing owning suite. Run focused pytest and root Ruff. Milestones: broaden to relevant service/router/invocation suites.
- Graph execution: read `services/shared/README.md`. Test scheduling and invocation lifecycles through owning interfaces.
- Report schema generation, migration validation, unavailable integration/hardware checks.
