# Shared API contracts

Owns `openapi.json`, generated `schema.ts`, and the generator that checks backend/frontend contract drift. No UI runtime. Frontend CI runs webv2; contract CI installs only this package.

- pnpm 10 and this package's lockfile. Change pinned generator, TypeScript 5 peer, and formatter versions only on purpose. Webv2's TypeScript version is independent.
- Never hand-edit generated output. From the repo root, in the activated Python environment: `make frontend-api-install`, `make frontend-openapi`, `make frontend-typegen`. Shell pipelines must keep generator failures.
- Run `pnpm lint` and `pnpm test` here. Generator tests compile independent consumer expectations against real generated types and exercise invalid input.
- Keep legacy `../webv1/src/services/api/schema.ts` type-only re-export compatible. Shared types or that boundary changed: validate legacy TypeScript.
- API shape changes follow `invokeai/app/AGENTS.md` and need owning backend tests. Moving artifact ownership must keep generated contracts intact.
