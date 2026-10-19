# Webv2

Ownership or interface change: read [ARCHITECTURE.md](ARCHITECTURE.md). Its "Owner documents" table lists the spec to read before changing feature, window, settings, persistence, or Canvas behavior. Stack: React 19, Vite, Chakra UI 3, TanStack Query/Router, @dnd-kit. Legacy web differs. Served by default; `--webv2` is a compatibility alias, `--web-legacy` selects `../webv1/`.

Shared generated API contracts and their CI gates: [../api/AGENTS.md](../api/AGENTS.md).

## Ownership and state

- Direction `app -> workbench/features -> platform`, enforced by `src/architecture/`. App composes dependencies. Platform owns domain-neutral infrastructure. Features own domain behavior. Workbench owns project aggregate, shell, Canvas.
- Import Features only through registered top-level public entries. Never reach into another owner's internals, not even to avoid barrel cost. Pure cores import no React/UI/transport code, type-only imports included.
- Production Canvas callers use `src/workbench/canvas-engine/api.ts`. Engine changes follow `src/workbench/canvas-engine/ARCHITECTURE.md`. Contracts live with the owner of their invariants. Check the ownership manifest before adding Workbench modules.
- Map and serialize wire DTOs and persisted schemas explicitly. Never grow compatibility re-exports into implementations. Never add architecture exceptions to pass checks.
- Query owns backend read models. Non-React runtimes own orchestration and lifecycles. Use stable, narrowly selected external-store snapshots, not broad component-state mirrors.
- Preserve account/project isolation, stale-result fencing, revision conflicts, reconnects, start/dispose ownership. No process-wide mutable per-account/project caches.
- Persistence: read `src/workbench/projects/README.md`. Preserve backend authority, bounded recovery, cross-tab ownership, durable queue receipts, explicit conflicts.

## React and product quality

- No direct `useEffect`. Derive during render, act in event handlers, fetch through Query, subscribe via external-store adapters.
- `@platform/react/useMountEffect`: only mount/unmount registration with stable dependencies and complete cleanup. No changing-dependency refs, wrapper aliases, layout-effect evasions. Layout effects only for pre-paint DOM measurement.
- Changing an effect's behavior: migrate it. No unrelated sweeps. Identity changes use store/runtime lifecycles or deliberate keyed remounts, without accidental state loss.
- Reuse Platform controls, Chakra/theme tokens, icons, interactions. No second styling system, no generic Next.js/SWR patterns. Import controls by subpath (`@platform/ui/Button`): the `@platform/ui` barrel's importer budget is full.
- Typography scale: `xs` = 10px, `sm` = 11px. `architecture:check` (also in `lint`) rejects retired `2xs` literals in `fontSize`/`textStyle` JSX and style/recipe properties, conditional and responsive values included. Typegen cannot enforce this: Chakra accepts arbitrary CSS strings. Guard does not resolve variables or calls. `sm` stays valid, so review must catch uses still meaning its old 14px size (now `lg`). Stock-name mapping data allowed.
- Include disabled states. Keep keyboard operation, visible and restored focus, accessible names, pointer alternatives.
- Icon-only controls get the platform `Tooltip`, menu and popover triggers included (share `useTooltipTriggerIds` between root and Tooltip). Never native `title`.
- Dense layouts stay usable across window sizes and zoom. Check overflow, long/localized text, selected/hover/active states, themes. Follow localization conventions.
- Interactions immediate. Motion restrained, interruptible, reduced-motion compatible. Name transition properties. Animation never delays editing and is never the only state cue.
- Give visual evidence of browser-inspected flows. Browser tests and axe do not replace visual/interaction review.

## Browser verification

No Python backend: `pnpm dev:mock` serves the app against the in-repo mock backend, which picks its own free port. Arguments go to `vite dev`. Default Vite port 5174 taken: pick a spare, `pnpm dev:mock --port 5273 --strictPort`. `pnpm dev` alone proxies to a real backend at `http://127.0.0.1:9090`.

- `INVOKEAI_MOCK_BACKEND_PROFILE`: `representative` (default) or `empty`. `POST /__reset` on the mock origin printed at startup restores fixtures; `?profile=` switches them. Unimplemented endpoints return 404.
- Stop servers by PID (`ss -ltnp`), not `pkill -f`: its pattern can match the invoking shell. Restart after merges that add mock endpoints.
- Provider/context error from a `?t=` module after source edits: full reload before calling it a product bug. HMR can split context modules.

## Performance and validation

- Check rendering/subscriptions, galleries/lists, pointer handlers, Canvas resources, lazy loading, request scheduling. Use existing virtualization/lazy boundaries. Memoize only identified costs.
- Preserve disposal, bounded memory, account transitions, responsive input. Never update goldens/baselines to hide regressions.
- Run pnpm from this directory: Corepack picks pnpm's version from the current directory, so `pnpm -C` from the repo root can get the wrong one. Install: `pnpm install --frozen-lockfile`.

| Change/check                          | Command                                                               |
| ------------------------------------- | --------------------------------------------------------------------- |
| Formatting, lint, types, architecture | `pnpm lint`                                                           |
| Focused behavior                      | `pnpm test <test-path>` or `pnpm test:browser <browser-test-path>`    |
| Unit, Chromium, mock-backend fixtures | `pnpm test:all`                                                       |
| Architecture and browser performance  | `pnpm test:performance:architecture` ([guide](performance/README.md)) |
| Project-file integration              | `pnpm test:project-files`                                             |
| Accessibility in a fresh build        | `pnpm test:accessibility`                                             |
| Milestone/release completion          | `pnpm check:release`                                                  |

- Missing Chromium: `pnpm exec playwright install chromium` (`--with-deps` in Linux CI).
- Colocate tests: unit `*.test.ts`/`*.test.tsx`, browser `*.browser.test.ts`/`*.browser.test.tsx`.
- Use real IndexedDB, Web Locks, focus, pointer, rendering where mocks cannot prove correctness.
- Droppable enabled only after drag start: drag through dnd-kit mouse sensors with `dragWithMouse` (`@platform/browser/mouseDrag.testing`). Single-jump drags such as `userEvent.dragAndDrop` miss it.

Focused checks during development. `check:release` before code-milestone readiness: lint, all tests, architecture performance/build, project-file journeys, accessibility. Docs-only changes: formatting and link review, no application tests. Architecture inventories and performance reports stay in ignored `artifacts/`, never committed.
