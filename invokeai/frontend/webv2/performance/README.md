# Performance gates

Paths are relative to the webv2 package. `pnpm test:performance:architecture` runs two gates against the baselines in `performance/`. `pnpm check:release` includes both.

| Gate    | Command                         | Baseline                                 | Checks                                                                                                                                         |
| ------- | ------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Build   | `pnpm test:performance:build`   | `performance/architecture-baseline.json` | Per-route bytes, request counts, and source-owner sets; sources each route must not load eagerly; direct-importer budgets for barrels and hubs |
| Browser | `pnpm test:performance:browser` | `performance/browser-baseline.json`      | Each route in Chromium against the mock backend: script requests, source-owner sets, required widgets, and timing samples                      |

Byte ceilings and when re-recording accepts growth are defined under [Byte budgets](#byte-budgets).

## Byte budgets

The byte budgets in `performance/architecture-baseline.json` and `performance/browser-baseline.json` record measurements only. Each byte ceiling is the lower of `max(1%, 4 KB)` over the **higher of the base reference and committed measurement**, and `max(10%, 32 KB)` over the committed measurement (`deriveLimit` in `scripts/performance-budgets.mjs`). Deliberately re-recording a higher baseline therefore accepts reviewed growth, such as a dependency upgrade, in the same PR. Growth beyond that allowance still fails. Previously recorded higher measurements remain accepted until a downward re-record protects the savings; the committed hard ceiling continues to bound cumulative growth in the base branch. Re-record with `pnpm test:performance:build:update-baseline` and `pnpm test:performance:browser:update-baseline`, and include the measured cost or savings in review.

CI restores `main`'s own measurements from the Actions cache (`WEBV2_PERF_REFERENCE_DIR`) for pull requests into `main` and merge groups targeting `main`. Logs and reports name the restored reference. Local runs, pushes to `main`, and stacked PRs without a reference compare against committed measurements directly. Source-owner sets stay pinned exactly in every mode: a module entering a route's initial graph fails at zero bytes. Request counts carry no allowance and remain capped at the lower of the reference and committed counts. A byte rebaseline does not relax request counts, source ownership, inactive-widget loading, or other structural checks.

## Running

- `check:release` stops at the first failing gate, so later gates have not run. In CI logs the failure sits at the tail, after the passing test summaries.
- The browser gate measures timing and binds ports 4176 and 4177 (`INVOKEAI_PERFORMANCE_PORT`, `INVOKEAI_PERFORMANCE_BACKEND_PORT`). Run one performance gate per machine, with nothing else building or testing.
- One failing sample on a busy machine can be noise, including "did not request required widgets". Rerun once on a quiet machine; a repeated failure is a regression.

## Fixing failures

Do not raise a budget or re-record a baseline to make a change pass. Re-record only intended, reviewed growth, then check that the baseline diff adds only the intended sources and leaves request counts unchanged.

| Failure                                                                                            | Usual cause and fix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `<metric> has N direct importers of <specifier> (budget M)`                                        | A new file imports the counted barrel or hub. Import the subpath or owning module instead, for example `@platform/ui/Button` rather than `@platform/ui`. These budgets only go down.                                                                                                                                                                                                                                                                                                                                                 |
| `… eagerly includes editor runtime …`, `launchpad eagerly includes …`, `editor eagerly includes …` | An eager module now reaches lazy code. A module-scope expression that reads a global (`Number.EPSILON * 100`) makes its module side-effectful, so every `export *` barrel through it stops tree-shaking: move the expression into the function. Importing a value from a heavy module pulls in that module's graph: move the value to a leaf module. Adding a module the Launchpad palette reaches to `EDITOR_BOOT_SHARED_MODULES` also causes this.                                                                                 |
| Request count grew, `initial source-owner graph changed`, or `script source-owner set changed`     | A module shared by eager and lazy code got its own chunk, or one owner imports it lazily while another imports it statically. Group it in `vite.config.mts` (`ROUTE_SHARED_MODULES` for eager modules both routes share, `EDITOR_BOOT_SHARED_MODULES` for modules the editor shell and lazy widgets share, or the matching feature group), or fold it into the module that is already split. Prefer extending an existing module over adding a leaf file. A module that legitimately becomes eager needs both baselines re-recorded. |
| Route bytes grew                                                                                   | Locale strings count: `en.json` is part of each route's `otherAssetRawBytes`, so new strings need a reviewed re-record.                                                                                                                                                                                                                                                                                                                                                                                                              |

To find a new startup request, build `main` in a separate worktree and compare hash-stripped `dist/assets/*.js` names with the branch build. Names only in the branch build that also appear in the route trace (`artifacts/architecture-performance/traces/<fixture>-<profile>.zip`, `trace.network` URLs) are the extra requests. `dist/.vite/chunk-sources.json` maps chunks to sources, and `vite build --minify false` keeps `//#region` module markers.

## Remediation tickets

Failures name an owner and a remediation ticket: the structural work that would retire a budget. A ticket is context for the budget, not the fix for a failing change.

| Ticket                           | Budgets                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `set-performance-budgets`        | Launchpad route budgets                                                       |
| `deepen-widget-registry-loading` | Editor route budgets and the inactive-widget check                            |
| `return-contracts-to-owners`     | Importers of the `@platform/ui` barrel and the retired `@workbench/types` hub |
| `seal-canvas-engine-seam`        | Importers of `@workbench/canvas-engine/contracts`                             |
| `finish-queue-ownership`         | Importers of `@features/queue`                                                |
| `define-gallery-feature`         | Importers of `./GalleryWidgetView`                                            |
