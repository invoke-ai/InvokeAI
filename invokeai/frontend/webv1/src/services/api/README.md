# API

The API client is a fairly standard Redux Toolkit Query (RTK-Query) setup.

It defines a simple base query with special handling for OpenAPI schema queries and endpoints: invokeai/frontend/webv1/src/services/api/index.ts

## Types

The shared `invokeai/frontend/api` package owns the generated `openapi.json`, `schema.ts`, and tooling. This package's `src/services/api/schema.ts` keeps the existing `services/api/schema` import path as a type-only re-export.

From the repository root in its activated Python environment, run `make frontend-api-install`, `make frontend-openapi`, and `make frontend-typegen`. See `invokeai/frontend/api/README.md` for ownership and generation details. Contract CI runs in that shared package; frontend CI runs webv2.
