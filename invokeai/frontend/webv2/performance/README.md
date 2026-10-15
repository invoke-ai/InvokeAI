# Performance gates

Paths relative to the webv2 package. `pnpm test:performance:architecture` runs two gates against baselines in `performance/`. `pnpm check:release` includes both.

| Gate    | Command                         | Baseline                                 | Checks                                                                                                                                  |
| ------- | ------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Build   | `pnpm test:performance:build`   | `performance/architecture-baseline.json` | Per-route bytes, request counts, source-owner sets; sources a route must not load eagerly; direct-importer budgets for barrels and hubs |
| Browser | `pnpm test:performance:browser` | `performance/browser-baseline.json`      | Each route in Chromium against the mock backend: script requests, source-owner sets, required widgets, timing samples                   |

## Byte budgets

Baseline byte budgets record measurements only. Each byte ceiling is the lower of `max(1%, 4 KB)` over the **higher of base reference and committed measurement**, and `max(10%, 32 KB)` over the committed measurement (`deriveLimit` in `scripts/performance-budgets.mjs`).

- Re-recording a higher baseline on purpose accepts reviewed growth (e.g. a dependency upgrade) in the same PR. Growth past the allowance still fails.
- Earlier higher measurements stay accepted until a downward re-record locks in the savings. The committed hard ceiling still bounds cumulative growth on the base branch.
- Re-record: `pnpm test:performance:build:update-baseline` and `pnpm test:performance:browser:update-baseline`. Put the measured cost or savings in review.

Reference modes:

- PRs into `main` and merge groups targeting `main`: CI restores `main`'s own measurements from the Actions cache (`WEBV2_PERF_REFERENCE_DIR`). Logs and reports name the restored reference.
- Local runs, pushes to `main`, stacked PRs without a reference: compare against committed measurements directly.
- Every mode pins source-owner sets exactly: a module entering a route's initial graph fails at zero bytes.
- Request counts have no allowance; cap is the lower of reference and committed counts.
- A byte rebaseline never relaxes request counts, source ownership, inactive-widget loading, or other structural checks.

## Running

- `check:release` stops at the first failing gate; later gates did not run. In CI logs the failure sits at the tail, after passing test summaries.
- Browser gate measures timing and binds ports 4176 and 4177 (`INVOKEAI_PERFORMANCE_PORT`, `INVOKEAI_PERFORMANCE_BACKEND_PORT`). One performance gate per machine, nothing else building or testing.
- One failing sample on a busy machine can be noise, "did not request required widgets" included. Rerun once on a quiet machine. Fails again: regression.

## Fixing failures

Never raise a budget or re-record a baseline to make a change pass. Re-record only intended, reviewed growth. Then check the baseline diff adds only intended sources and leaves request counts unchanged.

### `<metric> has N direct importers of <specifier> (budget M)`

New file imports the counted barrel or hub. Import the subpath or owning module instead: `@platform/ui/Button`, not `@platform/ui`. These budgets only go down.

### `… eagerly includes editor runtime …`, `launchpad eagerly includes …`, `editor eagerly includes …`

An eager module now reaches lazy code. Usual causes:

- Module-scope expression reading a global (`Number.EPSILON * 100`) makes its module side-effectful. Every `export *` barrel through it stops tree-shaking. Move the expression into the function.
- Importing a value from a heavy module pulls in that module's graph. Move the value to a leaf module.
- A module the Launchpad palette reaches was added to `EDITOR_BOOT_SHARED_MODULES`.

### Request count grew, `initial source-owner graph changed`, or `script source-owner set changed`

A module shared by eager and lazy code got its own chunk, or one owner imports it lazily while another imports it statically. Group it in `vite.config.mts` (`ROUTE_SHARED_MODULES` for eager modules both routes share, `EDITOR_BOOT_SHARED_MODULES` for modules the editor shell and lazy widgets share, or the matching feature group), or fold it into the module already split. Prefer extending an existing module over adding a leaf file. Module legitimately becomes eager: re-record both baselines.

### Route bytes grew

Locale strings count: `en.json` is part of each route's `otherAssetRawBytes`. New strings need a reviewed re-record.

### Finding a new startup request

Build `main` in a separate worktree; compare hash-stripped `dist/assets/*.js` names with the branch build. Names only in the branch build that also appear in the route trace (`artifacts/architecture-performance/traces/<fixture>-<profile>.zip`, `trace.network` URLs) are the extra requests. `dist/.vite/chunk-sources.json` maps chunks to sources; `vite build --minify false` keeps `//#region` module markers.

## Remediation tickets

Failures name an owner and a remediation ticket: structural work that would retire a budget. Context for the budget, not the fix for a failing change.

| Ticket                           | Budgets                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `set-performance-budgets`        | Launchpad route budgets                                                       |
| `deepen-widget-registry-loading` | Editor route budgets and the inactive-widget check                            |
| `return-contracts-to-owners`     | Importers of the `@platform/ui` barrel and the retired `@workbench/types` hub |
| `seal-canvas-engine-seam`        | Importers of `@workbench/canvas-engine/contracts`                             |
| `finish-queue-ownership`         | Importers of `@features/queue`                                                |
| `define-gallery-feature`         | Importers of `./GalleryWidgetView`                                            |
