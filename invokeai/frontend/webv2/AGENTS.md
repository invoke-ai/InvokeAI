# Webv2

Read [ARCHITECTURE.md](ARCHITECTURE.md) for ownership/interface changes; its "Owner documents" table lists the owner-local specifications to read before changing a feature, window, settings, persistence, or Canvas behavior. Stack: React 19, Vite, Chakra UI 3, TanStack Query/Router, @dnd-kit; legacy web differs. Webv2 is served by default; `--webv2` is a compatibility alias and `--web-legacy` selects `../webv1/`.

Shared generated API contracts and their CI gates belong to [../api/AGENTS.md](../api/AGENTS.md).

## Ownership and state

- Follow `app -> workbench/features -> platform` and `src/architecture/` policy. App composes dependencies; Platform owns domain-neutral infrastructure; Features own domain behavior; Workbench owns the project aggregate, shell, and Canvas.
- Import Features only through registered top-level public entries; never reach into another owner's internals, even to avoid barrel cost. Pure cores exclude React/UI/transport dependencies, including type-only imports.
- Production Canvas callers use `src/workbench/canvas-engine/api.ts`; engine changes follow `src/workbench/canvas-engine/ARCHITECTURE.md`. Contracts belong with their invariants' owner; consult the ownership manifest before adding Workbench modules.
- Map/serialize wire DTOs and persisted schemas explicitly. Do not turn compatibility re-exports into implementations or add architecture exceptions to pass checks.
- Query owns backend read models; non-React runtimes own orchestration/lifecycles. Use stable, narrowly selected external-store snapshots, not broad component-state mirrors.
- Preserve account/project isolation, stale-result fencing, revision conflicts, reconnects, and start/dispose ownership. No process-wide mutable per-account/project caches.
- For persistence, read `src/workbench/projects/README.md`: preserve backend authority, bounded recovery, cross-tab ownership, durable queue receipts, and explicit conflicts.

## React and product quality

- No direct `useEffect`: derive during render, act in event handlers, fetch through Query, subscribe via external-store adapters.
- `@platform/react/useMountEffect` is only for mount/unmount registration with stable dependencies and complete cleanup. No changing-dependency refs, wrapper aliases, or layout-effect evasions; layout effects require pre-paint DOM measurement.
- Migrate effects when changing their behavior, without unrelated sweeps. Identity changes use store/runtime lifecycles or deliberate keyed remounts without accidental state loss.
- Reuse Platform controls, Chakra/theme tokens, icons, and interactions; no second styling system or generic Next.js/SWR patterns. Import controls by subpath (`@platform/ui/Button`): the `@platform/ui` barrel's importer budget is full.
- Typography uses the workbench scale: `xs` is 10px and `sm` is 11px. `architecture:check` (also in `lint`) rejects retired `2xs` literals in `fontSize`/`textStyle` JSX and style/recipe properties, including conditional and responsive values. Chakra permits arbitrary CSS strings, so typegen cannot enforce this. The syntax guard does not resolve variables or calls; `sm` remains valid, so review must catch uses that still intend its old 14px size (`lg`). Stock-name mapping data is allowed.
- Include disabled states; preserve keyboard operation, visible/restored focus, accessible names, and pointer alternatives.
- Icon-only controls carry the platform `Tooltip`, including menu and popover triggers (share `useTooltipTriggerIds` between the root and the Tooltip); never a native `title`.
- Keep dense layouts usable across window sizes/zoom. Check overflow, long/localized text, selected/hover/active states, and themes; follow localization conventions.
- Keep interactions immediate and motion restrained, interruptible, and reduced-motion compatible. Specify transition properties; animation must neither delay editing nor be the sole state cue.
- Provide visual evidence of browser-inspected flows; browser tests and axe do not replace visual/interaction review.

## Browser verification

Without the Python backend, `pnpm dev:mock` serves the app against the in-repo mock backend, which takes a free port. Arguments go to `vite dev`; pick a spare Vite port when the default 5174 is taken: `pnpm dev:mock --port 5273 --strictPort`. `pnpm dev` alone proxies to a real backend at `http://127.0.0.1:9090`.

- `INVOKEAI_MOCK_BACKEND_PROFILE` selects `representative` (default) or `empty` fixtures; `POST /__reset` on the mock origin printed at startup restores fixtures, and `?profile=` switches them. Unimplemented endpoints return 404.
- Stop servers by PID (`ss -ltnp`), not `pkill -f`, whose pattern can match the invoking shell. Restart after merges that add mock endpoints.
- After source edits, fully reload before treating a provider/context error from a `?t=` module as a product bug; HMR can split context modules.

## Performance and validation

- Inspect rendering/subscriptions, galleries/lists, pointer handlers, Canvas resources, lazy loading, and request scheduling. Use existing virtualization/lazy boundaries; memoize only identified costs.
- Preserve disposal, bounded memory, account transitions, and responsive input. Never update goldens/baselines to hide regressions.
- Run pnpm from this directory; Corepack resolves pnpm's version from the current directory, so `pnpm -C` from the repository root can pick the wrong one. Install with `pnpm install --frozen-lockfile`.

| Change/check                          | Command                                                               |
| ------------------------------------- | --------------------------------------------------------------------- |
| Formatting, lint, types, architecture | `pnpm lint`                                                           |
| Focused behavior                      | `pnpm test <test-path>` or `pnpm test:browser <browser-test-path>`    |
| Unit, Chromium, mock-backend fixtures | `pnpm test:all`                                                       |
| Architecture and browser performance  | `pnpm test:performance:architecture` ([guide](performance/README.md)) |
| Project-file integration              | `pnpm test:project-files`                                             |
| Accessibility in a fresh build        | `pnpm test:accessibility`                                             |
| Milestone/release completion          | `pnpm check:release`                                                  |

Install missing Chromium: `pnpm exec playwright install chromium` (`--with-deps` in Linux CI). Colocate unit `*.test.ts`/`*.test.tsx` and browser `*.browser.test.ts`/`*.browser.test.tsx` tests. Use real IndexedDB, Web Locks, focus, pointer, and rendering where mocks cannot establish correctness. When a droppable enables only after a drag starts, drag through dnd-kit mouse sensors with `dragWithMouse` (`@platform/browser/mouseDrag.testing`); a single-jump drag such as `userEvent.dragAndDrop` misses it.

Run focused checks during development and `check:release` before code-milestone readiness: lint, all tests, architecture performance/build, project-file journeys, and accessibility. Docs-only changes need formatting/link review, not application tests. Keep architecture inventories/performance reports in ignored `artifacts/`, never committed.
