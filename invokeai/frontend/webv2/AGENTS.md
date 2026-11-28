# Webv2

Ownership or interface change: read [ARCHITECTURE.md](ARCHITECTURE.md). Its "Owner documents" table lists the spec to read before changing feature, window, settings, persistence, or Canvas behavior. [VERIFICATION.md](VERIFICATION.md) connects user-visible flows to fixtures, owner tests, and evidence limits. Stack: React 19, Vite, Chakra UI 3, TanStack Query/Router, @dnd-kit; legacy web differs.

Shared generated API contracts and their CI gates: [../api/AGENTS.md](../api/AGENTS.md).

## Enforced boundaries

- Direction `app -> workbench/features -> platform`, enforced by `src/architecture/`. App composes dependencies. Platform owns domain-neutral infrastructure. Features own domain behavior. Workbench owns project aggregate, shell, Canvas.
- Import Features only through registered top-level public entries. Never reach into another owner's internals, not even to avoid barrel cost. Pure cores import no React/UI/transport code, type-only imports included.
- Production Canvas callers use `src/workbench/canvas-engine/api.ts`. Engine changes follow `src/workbench/canvas-engine/ARCHITECTURE.md`. Contracts live with the owner of their invariants. Check the ownership manifest before adding Workbench modules.
- `pnpm architecture:check` enforces these import boundaries; failures name rule, source, target, and repair. Never add architecture exceptions to pass checks. Passing does not prove an interface has the right owner.
- Typography: `xs` = 10px, `sm` = 11px, `lg` = 14px. [The typography guard](src/architecture/typographyPolicy.test.ts) rejects retired `2xs` literals; review must still catch indirect values and `sm` used with its old 14px meaning.

## Ownership and state decisions

- Map and serialize wire DTOs and persisted schemas explicitly. Never grow compatibility re-exports into implementations.
- Query owns backend read models. Non-React runtimes own orchestration and lifecycles. Use stable, narrowly selected external-store snapshots, not broad component-state mirrors.
- Preserve account/project isolation, stale-result fencing, revision conflicts, reconnects, start/dispose ownership. No process-wide mutable per-account/project caches.
- Persistence: read `src/workbench/projects/README.md`. Preserve backend authority, bounded recovery, cross-tab ownership, durable queue receipts, explicit conflicts.

## React and product quality

- Polished desktop experience: deliberate layouts, responsive interactions, cohesive accessible controls, complete loading/empty/error/recovery states.
- No direct `useEffect`, adapters included; existing effects are not precedent. Derive during render, act in event handlers, fetch through Query, subscribe via external-store adapters; `@platform/react/useMountEffect` only for stable mount/unmount registration. Changing an effect's behavior: migrate it, without unrelated sweeps. Rationale, layout effects, identity resets, and what lint does not catch: [React adapters](ARCHITECTURE.md#react-adapters).
- Reuse Platform controls, Chakra/theme tokens, icons, interactions. No second styling system, no generic Next.js/SWR patterns. Import controls by subpath (`@platform/ui/Button`): the `@platform/ui` barrel's importer budget is full.
- Include disabled states. Keep keyboard operation, visible and restored focus, accessible names, pointer alternatives.
- Icon-only controls get the platform `Tooltip`, menu and popover triggers included (share `useTooltipTriggerIds` between root and Tooltip). Never native `title`.
- Dense layouts stay usable across window sizes and zoom. Check overflow, long/localized text, selected/hover/active states, themes. Follow localization conventions.
- Interactions immediate. Motion restrained, interruptible, reduced-motion compatible. Name transition properties. Animation never delays editing and is never the only state cue.
- Any UI change: verify changed interactions in the browser and give visual evidence. Browser tests and axe do not replace visual/interaction review.

## Browser verification

No Python backend: `pnpm dev:mock` serves the app against the in-repo mock backend, which picks its own free port. Arguments go to `vite dev`. Default Vite port 5174 taken: pick a spare, `pnpm dev:mock --port 5273 --strictPort`. `pnpm dev` alone proxies to a real backend at `http://127.0.0.1:9090`.

- `INVOKEAI_MOCK_BACKEND_PROFILE`: `representative` (default) or `empty`. `POST /__reset` on the mock origin printed at startup restores fixtures; `?profile=` switches them. Unimplemented endpoints return 404. Opt-in model seeds and image metadata: [VERIFICATION.md](VERIFICATION.md#start-from-a-known-state).
- Stop servers by PID (`ss -ltnp`), not `pkill -f`: its pattern can match the invoking shell. Restart after merges that add mock endpoints.
- Provider/context error from a `?t=` module after source edits: full reload before calling it a product bug. HMR can split context modules.

## Tests and comments

The package keeps a lean suite and lets code speak for itself. These rules tighten the root standard.

- Test each owner's core contract and its costliest failures: persisted-data compatibility, account/project isolation, security, destructive operations, recall contracts. Not every edge case, caller, past bug or layout detail. One test per behavior, at the cheapest level that proves it. Browser tests only for what needs a real browser: focus, pointer, storage, rendering. A marginal test is deleted, not kept just in case.
- Names, types and structure carry meaning. Comment only what a reader would otherwise get wrong or undo: a hidden constraint, compatibility rule, backend/external parity, or workaround. One line where possible. No doc comments that restate a name or type, no test narration, no history.

## Performance and validation

- Check rendering/subscriptions, galleries/lists, pointer handlers, Canvas resources, lazy loading, request scheduling. Use existing virtualization/lazy boundaries. Memoize only identified costs.
- Preserve disposal, bounded memory, account transitions, responsive input. Never update goldens/baselines to hide regressions.
- Install from this directory: `pnpm install --frozen-lockfile`.
- Shared development machine: use the [bounded local verification profile](VERIFICATION.md#bounded-local-verification). Same gates, fewer test workers.

| Change/check                            | Command                                                               |
| --------------------------------------- | --------------------------------------------------------------------- |
| Formatting, lint, types, architecture   | `pnpm lint`                                                           |
| Focused behavior                        | `pnpm test <test-path>` or `pnpm test:browser <browser-test-path>`    |
| Unit, Chromium, mock-backend fixtures   | `pnpm test:all`                                                       |
| Architecture and browser performance    | `pnpm test:performance:architecture` ([guide](performance/README.md)) |
| Project-file integration                | `pnpm test:project-files`                                             |
| Three-engine lifecycle in a fresh build | `pnpm test:browser:smoke`                                             |
| Accessibility in a fresh build          | `pnpm test:accessibility`                                             |
| Milestone/release completion            | `pnpm check:release`                                                  |

- Missing engines for completion: `pnpm exec playwright install chromium firefox webkit` (`--with-deps` in Linux CI).
- Colocate tests: unit `*.test.ts`/`*.test.tsx`, browser `*.browser.test.ts`/`*.browser.test.tsx`.
- Use real IndexedDB, Web Locks, focus, pointer, rendering where mocks cannot prove correctness.
- Every metadata key a graph builder writes needs an entry in `src/workbench/image-actions/imageRecallContract.test.ts` or `src/workbench/image-actions/videoRecallContract.test.ts`. They fail only for keys their coverage fixtures write: a key written under some setting needs that setting tuned in the fixtures (`recallTunedSettingsFor` for images). Component models reach the record from the graph's loader nodes through `RECORDED_COMPONENT_FIELDS` in `src/features/generation/core/graphBuilder.ts`; a new loader field needs an entry there (`src/workbench/image-actions/recordedComponents.test.ts` fails otherwise) and a `RECALLED_COMPONENTS` entry for its key (`vae`, `qwen_image_vae` and `refiner_model` have dedicated readers instead). `src/workbench/image-actions/imageRecordWriteGate.test.ts` fails when a setting changes the graph (Generate or any Canvas mode) without changing the record: record the setting, or add an `EXCEPTIONS` entry with its reason. A `VARIATIONS` entry must reach some fixture graph.
- Droppable enabled only after drag start: drag through dnd-kit mouse sensors with `dragWithMouse` (`@platform/browser/mouseDrag.testing`). Single-jump drags such as `userEvent.dragAndDrop` miss it.

Unit tests cache transformed modules between runs in ignored `node_modules/.vitest-cache`.

Focused checks during development. `check:release` before code-milestone readiness: lint, unit tests and fixtures, architecture performance/build, the Chromium browser suite, then project-file, accessibility and three-engine lifecycle journeys on the performance build. CI runs `lint:code` in the checks workflow and splits `check:release:tests` into parallel jobs (unit and fixtures, browser suite in shards, performance, journeys), each building what it needs; together they run each gate once, which [the workflow test](src/architecture/architectureWorkflow.test.ts) enforces. Routine browser performance checks take two resource observations per fixture; `test:performance:browser:profile` and baseline updates use full timing sampling. Docs-only changes: formatting and `node --test scripts/guidance-links.test.mjs` (Markdown links and anchors); no application tests. Performance reports and on-demand `pnpm architecture:inventory` output stay in ignored `artifacts/`, never committed.
