# Legacy frontend

Change legacy UI only when explicitly requested or required for shared compatibility. Ordinary frontend work belongs in `../webv2/`; read its guidance.

This package owns `openapi.json`, generated `src/services/api/schema.ts`, and generation tooling, including for backend contracts consumed by webv2.

## Commands

Run from this directory with this package's lockfile:

- `pnpm lint`: legacy TypeScript, dependency-cycle, ESLint, Prettier, and unused-dependency checks.
- `pnpm test:no-watch`: legacy Vitest; colocate tests following existing conventions.
- `pnpm build`: legacy production build.
- In the activated repository Python environment, regenerate schema: `python ../../../scripts/generate_openapi_schema.py > openapi.json`, then `pnpm exec prettier --write openapi.json`.
- Regenerate types in that environment: `python ../../../scripts/generate_openapi_schema.py | pnpm typegen`. Enable shell pipeline failure propagation and verify generator success; never hand-edit generated files.

Use root Ruff/pytest for Python changes. Do not apply webv2's Chakra version, @dnd-kit APIs, Oxc commands, or browser-test assumptions here. Preserve existing persisted-state migrations when changing legacy state.
