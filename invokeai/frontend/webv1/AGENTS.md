# Legacy frontend

Change legacy UI only when explicitly asked or needed for shared compatibility. Ordinary frontend work belongs in `../webv2/`; read its guidance.

Shared OpenAPI/type artifacts and generation tooling: [../api/AGENTS.md](../api/AGENTS.md). This package consumes those types through `src/services/api/schema.ts`; keep that type-only compatibility import path.

## Commands

Run from this directory, with this package's lockfile:

- `pnpm lint`: legacy TypeScript, dependency cycles, ESLint, Prettier, unused dependencies.
- `pnpm test:no-watch`: legacy Vitest. Colocate tests per existing conventions.
- `pnpm build`: legacy production build.
- Regenerate shared contracts from `../api/` with its commands and lockfile.

Python changes: root Ruff/pytest. Never apply webv2's Chakra version, @dnd-kit APIs, Oxc commands, or browser-test assumptions here. Changing legacy state: keep existing persisted-state migrations.
