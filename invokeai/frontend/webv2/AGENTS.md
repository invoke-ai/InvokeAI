# Webv2

Ownership or interface change: read [ARCHITECTURE.md](ARCHITECTURE.md). Its "Owner documents" table lists the spec to read before changing feature, window, settings, persistence, or Canvas behavior. [VERIFICATION.md](VERIFICATION.md) connects user-visible flows to fixtures, owner tests, and evidence limits. Stack: React 19, Vite, Chakra UI 3, TanStack Query/Router, @dnd-kit; legacy web differs.

Shared generated API contracts and their CI gates: [../api/AGENTS.md](../api/AGENTS.md).

## Enforced boundaries

- Direction `app -> workbench/features -> platform`, enforced by `src/architecture/`. App composes dependencies. Platform owns domain-neutral infrastructure. Features own domain behavior. Workbench owns project aggregate, shell, Canvas.
- Import Features only through registered top-level public entries. Never reach into another owner's internals, not even to avoid barrel cost. Pure cores import no React/UI/transport code, type-only imports included.
- Production Canvas callers use `src/workbench/canvas-engine/api.ts`. Engine changes follow `src/workbench/canvas-engine/ARCHITECTURE.md`. Contracts live with the owner of their invariants. Check the ownership manifest before adding Workbench modules.
- `pnpm architecture:check` checks these import boundaries; failures name the rule, source, target, and repair direction. Never add architecture exceptions to pass checks. Passing the check does not establish that an interface belongs to the right owner.
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

- `INVOKEAI_MOCK_BACKEND_PROFILE`: `representative` (default) or `empty`. `POST /__reset` on the mock origin printed at startup restores fixtures; `?profile=` switches them. Unimplemented endpoints return 404.
- `POST /__models` with `{"set": "anima-lllite"}` or `{"set": "spandrel"}` adds opt-in model seeds (adapters and upscalers installed before their channels or scale were recorded); re-identify fills those fields. `{"set": "control-sdxl"}` adds SDXL ControlNets naming their preprocessor in both id forms, a Union, an unpurposed model and a T2I-Adapter; `{"set": "control-flux"}` adds FLUX dev with a depth Control LoRA and a Union ControlNet. `/__reset` drops them.
- Stop servers by PID (`ss -ltnp`), not `pkill -f`: its pattern can match the invoking shell. Restart after merges that add mock endpoints.
- Provider/context error from a `?t=` module after source edits: full reload before calling it a product bug. HMR can split context modules.

## Performance and validation

- Check rendering/subscriptions, galleries/lists, pointer handlers, Canvas resources, lazy loading, request scheduling. Use existing virtualization/lazy boundaries. Memoize only identified costs.
- Preserve disposal, bounded memory, account transitions, responsive input. Never update goldens/baselines to hide regressions.
- Run pnpm from this package directory. Install: `pnpm install --frozen-lockfile`.
- Shared development machines can opt into the [bounded local verification profile](VERIFICATION.md#bounded-local-verification). It preserves the gates and documents setup, resource ownership, and failure reporting.

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
- Every metadata key a graph builder writes needs an entry in `src/workbench/image-actions/imageRecallContract.test.ts` or `src/workbench/image-actions/videoRecallContract.test.ts`. They fail only for keys their coverage fixtures write: a key written under some setting needs that setting tuned in the fixtures (`recallTunedSettingsFor` for images).
- Droppable enabled only after drag start: drag through dnd-kit mouse sensors with `dragWithMouse` (`@platform/browser/mouseDrag.testing`). Single-jump drags such as `userEvent.dragAndDrop` miss it.

Unit tests cache transformed modules between runs in ignored `node_modules/.vitest-cache`.

Focused checks during development. `check:release` before code-milestone readiness: lint, all tests, architecture performance/build, project-file journeys, accessibility, three-engine lifecycle smoke. CI splits completion between `lint:code` and `check:release:tests`; together they run each gate once. Routine browser performance checks take two resource observations per fixture; `test:performance:browser:profile` and baseline updates use full timing sampling. Docs-only changes: formatting and `node --test scripts/guidance-links.test.mjs` (part of `test:fixtures`), which resolves relative links and heading anchors in this package's Markdown; no application tests. Architecture inventories and performance reports stay in ignored `artifacts/`, never committed.
