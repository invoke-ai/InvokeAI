# Performance gates

Paths relative to the webv2 package. `pnpm test:performance:architecture` runs two gates against baselines in `performance/`. `pnpm check:release` includes both.

| Gate    | Command                         | Baseline                                 | Checks                                                                                                                                  |
| ------- | ------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Build   | `pnpm test:performance:build`   | `performance/architecture-baseline.json` | Per-route bytes, request counts, source-owner sets; sources a route must not load eagerly; direct-importer budgets for barrels and hubs |
| Browser | `pnpm test:performance:browser` | `performance/browser-baseline.json`      | Each route in Chromium against the mock backend: script requests, source-owner sets, required widgets, repeated resource observations   |

## Resource checks and timing profiles

The routine browser gate visits each fixture twice in fresh browser contexts. Both observations run the complete journey, required/inactive-widget checks, and semantic-ready waits. Resource budgets use the maximum of each metric; fixture counts and script source owners must agree between observations. Successful routine runs do not record traces or report timing statistics. On failure, the gate attempts one diagnostic trace for the failing fixture and retains the original failure.

`pnpm test:performance:browser:profile` runs the full sampling plan from `performance/architecture-fixtures.json`: one warmup, one unscored trace, and five scored observations per fixture. Use it after the performance build for timing investigation and traces of every route. Baseline updates always use this plan, as does the routine command when the checked-in timing policy enables enforcement. Full profiles report timing medians and variance; routine reports identify `mode: resources` and omit those fields. Resource reference files contain only resource measurements and work with either mode.

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

- CI splits `check:release` between `lint:code` in the checks job and `check:release:tests` in the tests job. Architecture tests run in the complete unit suite; inventories are generated once and the performance build serves all subsequent browser journeys.
- `check:release` stops at the first failing gate; later gates did not run. In CI logs the failure sits at the tail, after passing test summaries.
- Browser gate binds ports 4176 and 4177 (`INVOKEAI_PERFORMANCE_PORT`, `INVOKEAI_PERFORMANCE_BACKEND_PORT`). One performance gate per machine, nothing else building or testing.
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

Locale strings count in the browser gate: `en.json` is part of each route's `otherRawBytes` (locale plus `index.html`). The build gate's `otherAssetRawBytes` does not include it. New strings need a reviewed browser re-record.

### Finding a new startup request

Build `main` in a separate worktree; compare hash-stripped `dist/assets/*.js` names with the branch build. Names only in the branch build that also appear in the route trace (`artifacts/architecture-performance/traces/<fixture>-<profile>.zip`, `trace.network` URLs) are the extra requests. Run `pnpm test:performance:browser:profile` to collect traces for passing routes. `dist/.vite/chunk-sources.json` maps each chunk to its source modules and packages; diff it between the two builds to see which modules moved between chunks. Unminified builds no longer carry `//#region` module markers.

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

## Render validity and sustained interaction evidence

Nominal browser journeys distinguish implementation mount marks from usable content. Visible images must finish
`decode()`; the representative Canvas journey also waits for fixture pixels on the Canvas surface and a painted
layer thumbnail. Mock full images are deterministic PNGs matching their DTO dimensions; thumbnails preserve aspect
ratio within 256 pixels, without upscaling. Their encoding differs from the backend's WEBP thumbnails. Generation
is capped at 4096 pixels per dimension and the shared dimension cache at 32 entries. Fixture tests verify PNG CRCs,
inflated scanlines and dimensions, including a 4096-square image. Corrupt bytes are used only in negative tests.

The shared journey diagnostics reject uncaught errors, console errors, HTTP failures and failed requests. Only the
mock's `/ws/socket.io/?…transport=polling` 404 is expected. Canceled Queue status polls are normal owner teardown;
HTTP failures for that endpoint still fail. Lifecycle/fault journeys declare their other expected failures locally.
Browser reports keep API media requests and decoded dimensions separate from static-resource byte accounting.
Failures retain diagnostics, screenshots and the existing diagnostic traces under `artifacts/architecture-performance`.

Full profiles add a bounded Gallery quick-settings workload after startup and return-switch measurements on the
representative Compose fixture: one warmup and six open/change/close cycles per scored sample. Every cycle waits for
the popup portal to retire. Retained popover count must be zero; monotonically growing retained counts also fail.
DOM nodes, Canvas elements and request counts are diagnostic observations, not universal lifetime bounds. The
report includes every cycle, interaction min/median/p95/max and observed long-task distributions. Long tasks cover
the interaction phase, including its warmup, and interaction latency includes automation overhead; neither is an INP measurement. Failure
artifacts preserve the workload context. Other owners can add scenarios through this shared runner plumbing.

Sustained reports name the runner, engine version, platform, workload and sample policy. They remain report-only:
`browser-baseline.json` still requires 20 stable runs on a designated runner before timing enforcement. Local runs
cannot satisfy that calibration policy. Existing byte ceilings, timing sampling and goldens are unchanged.

## Browser lifecycle smoke

`pnpm test:browser:smoke` uses the existing project-file preview/backend lifecycle and production `dist/` build.
Run after `pnpm test:performance:build`. Install engines using the locked package:
`pnpm exec playwright install --with-deps chromium firefox webkit`. Ports default to 4180/4181; override
`INVOKEAI_PROJECT_FILE_PORT` and `INVOKEAI_PROJECT_FILE_BACKEND_PORT` for an isolated focused run.

The required localhost capabilities are IndexedDB and Web Locks. Each engine must recover a held prompt immediately
after reload, give duplicated tabs separate live editor sessions, release a closed tab's session lock, preserve edits
across route re-entry, advance keyboard focus, and disable copy controls with an explanation when clipboard is
unavailable. This is targeted current-engine coverage, with no minimum-version support promise. Missing engines or
required capabilities fail the command and are reported, rather than skipped. Every engine is attempted, even after
another fails. Reports, failure screenshots and traces live in `artifacts/browser-smoke` and CI uploads them.

History return records the real `pageshow.persisted` result and, where exposed, `notRestoredReasons`. A reload is
reported as such; it does not establish BFCache restoration. Playwright WebKit is not actual Safari verification,
and says nothing about every Apple platform. The documented Chromium unload guarantee remains scoped to Chromium;
other engine results are evidence for the tested engine/build/platform only, not a broader durability guarantee.

The smoke context blocks service workers so its same-origin idle HTML document and fault probes cannot be bypassed
by the PWA navigation cache. This suite exercises real IndexedDB and Web Locks; service-worker cache behavior is
outside its coverage. The noncanceling unload checkpoint runs before `pagehide` because WebKit can discard an
IndexedDB journal transaction started in `pagehide`, even after explicit commit. A `beforeunload` listener can reduce
Firefox BFCache eligibility; restoring from the cache is reported only when the engine actually does so.

During the deliberate WebKit reload, native `Fetch API`/`XMLHttpRequest` access-control diagnostics can describe
old-document requests denied before dispatch. Only native messages for the fixture workspace-save, the finite Queue read-model pipeline, and mock Socket.IO polling, from the same origin while document navigation is active, are expected by that scenario. Other runtime
errors and every real HTTP failure remain critical. Queue reader cancellations during route re-entry are limited
to the same Queue read interface whose query/owner signals are retired: status, current, next and item IDs (GET),
then bounded item hydration (`items_by_ids`, read-only POST); the fault probes use independent strict diagnostics.

Accessibility runner failures retain nested aggregate diagnostics in `artifacts/accessibility/failure.txt`; the
settings journey also captures its failing page as text and a screenshot before the browser context closes.
