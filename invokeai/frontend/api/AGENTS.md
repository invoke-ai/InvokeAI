# Shared API contracts

This package owns `openapi.json`, generated `schema.ts`, and the generator used to check backend/frontend contract drift. It has no UI runtime. Frontend CI runs webv2; contract CI installs this package only.

- Use pnpm 10 and this package's lockfile. Keep the pinned generator, TypeScript 5 peer, and formatter versions intentional; webv2's TypeScript version is independent.
- Never hand-edit generated output. In the repository's activated Python environment, run `make frontend-api-install`, `make frontend-openapi`, and `make frontend-typegen` from the root. Shell pipelines must preserve generator failures.
- Run `pnpm lint` and `pnpm test` here. The generator tests compile independent consumer expectations against real generated types and exercise invalid input.
- Keep the legacy `../webv1/src/services/api/schema.ts` type-only re-export compatible. Validate legacy TypeScript when shared types or that boundary change.
- API shape changes follow `invokeai/app/AGENTS.md` and require owning backend tests. Moving artifact ownership must preserve the generated contracts.
