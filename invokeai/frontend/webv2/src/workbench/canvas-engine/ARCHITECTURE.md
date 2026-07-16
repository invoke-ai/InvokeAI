# Canvas Engine Architecture

## Ownership invariant

The workbench reducer owns the serializable canvas contract. A per-project canvas engine owns pixels, decoded images, rendering, interaction, history, transient editing state, and caller-owned raster snapshots. Engine-owned objects never enter reducer actions or persisted project state.

Canvas mutations are explicitly project-addressed. `CanvasProjectMutation` is the complete reducer mutation vocabulary, and the only `WorkbenchAction` that carries one is:

```ts
{
  type: 'applyCanvasProjectMutation';
  projectId: string;
  mutation: CanvasProjectMutation;
}
```

The workbench reducer handles this envelope only through `updateProjectById`. There are no canvas actions whose target is inferred from the active project.

Each engine receives a `CanvasProjectMutationPort` bound to its immutable `projectId`. The port exposes only `getCanvasState()`, `subscribe()`, and `dispatch(mutation)`. It neither exposes the global workbench dispatcher nor follows active-project changes. Widgets use `useCanvasProjectMutationDispatch`, which captures the current project ID and emits the same envelope. Consequently, project switches and colliding layer IDs cannot redirect a delayed engine mutation to another project.

The port's `dispatch()` returns whether the named project's canvas identity changed. Paint persistence clears a dirty result only after the intended layer accepts the bitmap reference. A rejected update retains dirty pixels for retry; deletion of the layer or project is terminal and discards obsolete persistence work. This applies equally to raster/control paint sources and paint-backed masks.

## Package and capability boundaries

`canvas-engine/` is the rendering and editing core. It may import engine-internal modules and shared canvas contracts, mutations, and the project mutation port. It must not import React, widgets, application canvas operations, generation graphs, socket infrastructure, or backend networking. `importBoundaries.test.ts` enforces that rule.

`canvas-operations/` owns application workflows such as Filter and Select Object. Coordinators may depend on queues, uploads, model selection, and narrow engine capabilities. Their stores are React-free and compatible with `useSyncExternalStore`; React adapters remain under `widgets/`.

The public engine handle is divided into capability groups such as `CanvasDocumentCapability`, `CanvasExportCapability`, `CanvasLayerCapability`, `CanvasLifecycleCapability`, and the interaction/store capabilities. Production consumers request the smallest aggregate they need. The full `CanvasEngine` type is limited by `fullEngineImports.test.ts` to the registry and composition roots; tests may construct full-engine fakes. The unused `fflate` dependency has been removed.

`CanvasCoreStores` contains interaction state only. Filter and Select Object snapshots are created and owned by `canvas-operations`; application-session state cannot leak into the core store contract.

Controllers communicate through constructor-injected ports or public capabilities. They do not reach into another controller's private maps, and the core engine never receives the global workbench dispatcher.

## Controllers

`engine.ts` is the core composition root and document-event router. New behavior belongs in a focused controller with explicit dependencies and an idempotent `dispose()` method.

Current extracted ownership includes:

- `RasterController`: base layer caches, decoded bitmap leases, derived and adjusted surfaces, rasterization jobs, invalidation, and the unified raster-memory controller.
- `RasterMemoryBudgetController`: base, derived, decoded, detached, and reserved byte accounting plus generation-scoped and operation-owned reservations and pins.
- `RenderController`: render targets, scheduler construction, previews, and render lifecycle.
- `HistoryController`: engine-owned history and inactive byte trimming.
- `PersistenceController`: dirty-bitmap flush barriers and disposal.
- `EditingController`: selection state and exclusive edit-lease lifecycle.
- `InteractionController`: public tool commands and their disposal boundary.
- `LayerController`: guarded layer mutations and thumbnail capability ports.
- `RasterExportController`: guarded layer rasterization, transformed copies, encoding, reservations, and pin leases.
- `PsdExportController`: immutable PSD planning, snapshot capture, reservation, execution, and cancellation.
- `StagedResultController`: guarded staged-candidate acceptance and its history entry.
- `ControlPixelController`, `SelectionImageController`, and `LayerMutationController`: atomic pixel/document publication at their respective boundaries.
- `MaskResultController`, `FilterResultController`, and `GeneratedResultController`: guarded application-result adoption through narrow host ports.

Controller-local tests instantiate these boundaries with fakes. Integration tests in `engine.test.ts` cover composition, document routing, and cross-controller invariants.

## Immutable document and raster snapshots

`CanvasDocumentCapability.captureSnapshot()` returns a `CanvasDocumentSnapshot` or `null`. A snapshot contains a structured clone of the exact `CanvasStateContractV2` and the engine's `documentGeneration`. The engine privately associates it with the reducer canvas identity from which it was captured, so a caller cannot manufacture a current snapshot by copying fields.

`CanvasExportCapability.captureRasterSnapshot(snapshot, layerIds, options)` requests exactly the required layer caches, temporarily pins them, reserves detached-surface capacity, and copies their pixels into independent surfaces. It returns a typed `ok`, `stale`, `aborted`, `not-ready`, or `over-budget` result. Freshness is checked before and after every asynchronous rasterization boundary and once after all copies are detached.

An `ok` result contains a `CanvasRasterSnapshot`: the cloned canvas contract, its document generation, a map of detached layer surfaces, and an idempotent `release()`. The snapshot is caller-owned. Its surfaces do not read live caches and remain valid while the engine cools down; cooldown releases generation-scoped reservations and pins but does not invalidate detached snapshots. The caller must release the snapshot in `finally`. Engine disposal releases any snapshots still outstanding.

This creates a deliberate transaction boundary:

1. Before detachment completes, a document change makes the capture `stale`; no downstream submission may use a mixture of generations.
2. After a successful invocation detachment, ordinary editing may continue. Composite, upload, graph compilation, and queue submission read only the frozen contract and detached surfaces.
3. A wholesale canvas replacement remains a separate session, identified by `documentRevision`; completed results from an older session are dropped.

Canvas invocation first crosses the paint-flush barrier, then captures the post-flush document, plans every raster/control/regional composite, and detaches exactly the contributing layers. Composite dedupe writes are staged in transaction-local maps. They are published to the persistent per-engine dedupe cache only after all frozen compositing, uploads, compilation, and queue dispatch complete; stale or failed captures leave no persistent dedupe entries.

`submitCanvasInvocationSnapshot` carries the exact cloned canvas state used for compilation. `enqueueCompiledSnapshot` clones that supplied state rather than reading the live project canvas. Generated candidates are placed from `queueItem.snapshot.canvas.document.bbox`, so later bbox edits do not move the result. `documentRevision` still prevents results from a replaced canvas session from entering current staging.

## Frame demand before allocation

`calculateActiveFrameLayerIds` is a pure pre-allocation pass. For each enabled renderable layer it combines persisted source bounds with any larger live cache rect, applies the committed or transient transform (including rotation), respects isolation, and intersects the result with the document-space viewport.

The render path computes this set before `ensureLayerCaches`. Only demanded layers are rasterized for the frame. Offscreen enabled layers are not eagerly allocated; they remain on-demand and LRU-evictable, then rasterize again when panning or a transform reveals them.

Budget enforcement protects only:

- layers demanded by the active frame; and
- layers explicitly pinned by an in-flight thumbnail, export, snapshot, or other background operation.

Derived surfaces are evicted first, followed by least-recently-used unprotected base caches. Export and thumbnail callers request and pin their own caches, preserving their behavior without treating all enabled layers as visible. Deterministic tests assert allocation counts and bytes for pan/reveal, rotated bounds, unflushed paint bounds, isolation, transient transforms, eviction, and re-rasterization.

## Raster resource ownership and memory policy

The raster soft limit is 512 MiB. `RasterMemoryBudgetController` maintains one snapshot across all material allocation classes:

- `baseBytes`: layer cache surfaces;
- `derivedBytes`: derived and adjusted display surfaces;
- `decodedBytes`: leased `ImageBitmap` instances;
- `detachedBytes`: caller-owned raster snapshot surfaces; and
- `reservedBytes`: capacity promised to background work but not yet represented by another class.

Active-frame base surfaces are allowed to exceed the soft limit so the visible frame can render; diagnostics report the overage. Background snapshots, thumbnails, raster exports, transformed/adjusted copies, and PSD exports must reserve sufficient available bytes before allocating and return typed `over-budget` outcomes when they cannot. Surface-cache enforcement subtracts decoded, detached, and reserved bytes before calculating the base/derived allowance.

Reservations and cache pins are idempotent leases. Cancellable preparation work uses lifecycle-generation leases, which activation, cooldown, and disposal release if the normal `finally` path has not already done so. Once an asynchronous raster composite is in flight, it uses operation-owned reservations and pins instead: lifecycle changes may make its result stale, but cannot erase its accounting while a yielded allocation is still able to resume. The operation releases those leases in `finally`. Detached snapshot accounting is also caller-owned: it follows the snapshot and is released only by `CanvasRasterSnapshot.release()` or engine disposal.

`DecodedBitmapPool` replaces permanent decoded-image caching. `acquire(imageName, decode, signal)` coalesces concurrent decodes for the same image and returns a short-lived `DecodedBitmapLease`. Each rasterizer releases its lease in `finally`. The bitmap closes after the final lease, a pending decode is aborted when all interested callers cancel, and disposal aborts pending work and closes every resolved bitmap. Pool byte changes feed `decodedBytes`, so decoded images participate in the same budget as surfaces.

## Derived surfaces and invalidation

Every display-only pixel effect uses `DerivedSurfaceCache`. A slot is identified by layer ID and effect kind, and guarded by source surface identity, monotonic source version, and a deterministic parameter key. Reuse is safe only when every guard matches. Source replacement therefore cannot publish or reuse a surface produced for an older preview.

Invalidate only the affected effect when parameters change. Delete every derived slot when a layer is removed, replaced, or its base cache is evicted. Frame-demand culling happens before derived lookup or construction.

## Edit leases and staged-result acceptance

Application operations acquire one exclusive lease from `CanvasEditGate`. A lease carries an `AbortSignal`, has an explicit `isCurrent()` freshness check, and is idempotently released. Release, document/project invalidation, cooldown, or disposal aborts and stales the lease. Expected cancellation and stale results use status unions; exceptions represent unexpected failures.

Staged-result acceptance is engine-owned. The UI enables Accept from interaction capabilities, but `StagedResultController` remains authoritative. It captures a document-edit permit, rejects an active gesture, and verifies the selected candidate's stable key immediately before dispatch.

The initial `commitStagedImage` mutation atomically inserts the new raster layer, selects it, clears staging, and records the project event. The controller verifies both the reducer postcondition and the document mirror before pushing exactly one engine-history entry. It returns `busy`, `stale`, or `missing` without changing staging or history when any guard fails.

Undo removes the accepted layer and restores the prior selection. Redo restores the identical layer and selection without recreating the event or staging candidate. The history entry uses failure-atomic replay, and a successful commit uses the normal history `push`, which clears any previous redo stack. Undo intentionally does not return the accepted candidate to staging.

## PSD export

PSD export is a planner/executor transaction over immutable snapshots. `PsdExportController` captures one `CanvasDocumentSnapshot`, plans exportable raster layers and their transformed world-space bounds, then obtains a `CanvasRasterSnapshot` including disabled layers. Execution reads only those detached surfaces; it cannot mix live layer caches from different document generations.

The controller retains the 30,000-pixel side cap and also derives a pixel-area limit from currently available reserved bytes. It accounts for the flattened composite plus each transformed layer surface, budgets eight bytes per required pixel, and reserves the full execution allocation before creating PSD surfaces. A final freshness check occurs before that reservation; once execution starts it uses only frozen pixels. Capture and execution reservations are released in `finally`, and cancellation aborts before the next allocation or side effect.

The capability returns `exported`, `nothing`, `too-large`, `not-ready`, `over-budget`, `stale`, or `aborted`. The layers widget awaits the promise and maps every non-success result to a user-visible notice.

## Lifecycle

Lifecycle states are `active`, `cooling`, `cool`, and `disposed`. Losing the final registry reference immediately detaches input/render targets, cancels background PSD work and rasterization, releases the old generation's reservations and pins, and starts a dirty-pixel flush. Successful, still-current flushes release reconstructible raster caches and trim history. Failed flushes retain dirty pixels and remain retryable; the registry retains the engine and retries after another grace period, disposing it only after a `cooled` result. Reacquisition increments the lifecycle generation so late flush or grace-period work cannot clean up or dispose a live engine. Operation-owned composite leases survive these generation transitions until their owning operation settles.

Caller-owned `CanvasRasterSnapshot` objects are not reconstructible lifecycle caches. They remain usable through cooldown and continue to count against detached bytes until released. Final engine disposal force-releases them.

## Browser acceptance

Node and browser suites are separated by filename and config. `vitest.config.mts` excludes `*.browser.test.ts(x)`; `vitest.browser.config.mts` runs those files through `@vitest/browser-playwright` 4.1.9 in headless Chromium. `test:browser` runs the browser suite and `test:all` runs Node followed by browser tests. CI installs Chromium with `pnpm exec playwright install chromium` before the browser job.

The real-Canvas2D suite verifies transformed composition, clipping, alpha and blend modes, browser text metrics, PNG blob encoding, `createImageBitmap` decoding, pixel undo/redo, and a small PSD write/read round trip. A React `StrictMode` harness verifies surface attachment, `ResizeObserver` sizing, detachment, registry cooldown/reacquisition, grace-period safety, and project switching.

## Adding functionality

- New tool: implement the tool interface under `tools/`, keep durable state in `CanvasProjectMutation`, and expose only required tool capability methods.
- New derived effect: add a stable effect kind/parameter key, construct through `DerivedSurfaceCache`, and test warm hits, precise invalidation, stale source guards, culling, and byte accounting.
- New layer operation: put pure pixel/document math in the core and orchestration in a controller; accept narrow ports, use project-bound mutations, reserve background allocations, and return status unions for expected no-op/stale cases.
- New background raster workflow: capture a document snapshot, plan required layers, reserve and pin before allocation, detach immutable surfaces, release every lease in `finally`, and define whether edits after detachment are allowed.
- New application coordinator: place it under `canvas-operations/`, acquire an edit lease, capture through narrow document/export capabilities, own its session store and cancellation, and publish through preview/layer capabilities.

Deterministic CI assertions use operation counters and byte totals. Wall-clock timing is informational only.
