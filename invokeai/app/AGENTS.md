# Application backend

## Ownership and contracts

- Routers validate requests, enforce authentication/authorization, and map service errors to HTTP. Services own domain policy; storage owns persistence.
- Before changing flows, inspect `api/dependencies.py`, affected routers, service base/default/storage implementations, and tests. Preserve dependency/start/stop lifecycles; avoid parallel global services.
- Server-authorize every account-owned resource, including queries, writes, events, exports, and recovery; UI guards are insufficient.
- Trace frontend/saved-workflow consumers before changing Pydantic models, invocation fields/versions, HTTP DTOs, or socket-event names, defaults, validation, or errors.
- API changes may require regenerating `frontend/api/openapi.json` and `frontend/api/schema.ts` under `invokeai/`, even for webv2. Follow shared API package generation guidance; never hand-edit output.

## Persistence and lifecycle

- Preserve atomicity, revision checks, idempotency, and explicit conflicts. Inspect migrations/old-record readers for stored-data changes; add needed migration/compatibility tests.
- Read `services/shared/sqlite/sqlite_database.py` and `services/project_records/project_records_sqlite.py`: nested transactions can prematurely commit outer operations. Use one transaction owner for multi-record writes, passing its cursor where supported.
- Preserve project/account isolation, document bounds, queue admission/receipts, and lost-response retry safety. Never silently discard recovery data or resolve conflicts with last-write-wins.
- Bound concurrency; support cancellation, timeouts/retries, and file/process/listener/task cleanup. Keep CPU/blocking I/O off the async server loop.
- Preserve file/path validation, remote-fetch protections, and sanitized errors. Never modify real user models, databases, outputs, or credentials during tests/debugging.

## Efficiency and verification

- Inspect queries for repeated lookups, full scans/materialization, missing bounds, and serialization. Measure representative data/query counts before adding caches or indexes.
- Use real temporary SQLite for transaction, migration, revision, and ownership tests; cover affected malformed input, cross-user access, cancellation, repeated requests, and partial failures.
- Read `tests/AGENTS.md`; use `tests/app/` or the existing owning suite. Run focused pytest and root Ruff; broaden to relevant service/router/invocation suites for milestones.
- For graph execution, consult `services/shared/README.md`; test scheduling and invocation lifecycles through owning interfaces.
- Report schema generation, migration validation, and unavailable integration/hardware checks.
