# Canvas Engine Architecture

## Ownership invariant

The workbench reducer owns the serializable canvas document. The canvas engine owns pixels, decoded images, rendering, interaction, history, and transient editing state. Engine-owned objects must never be placed in reducer actions or persisted document contracts.

## Boundaries

`canvas-engine/` is the rendering and editing core. It may import engine-internal modules and shared canvas document/action contracts. It must not import React, widgets, generation graphs, socket infrastructure, backend HTTP clients, or `canvas-operations/`.

`canvas-operations/` owns application workflows such as Filter and Select Object. Coordinators may depend on generation graphs, queues, uploads, model selection, and narrow engine capabilities. Their stores are React-free and compatible with `useSyncExternalStore`; React adapters remain under `widgets/`.

`CanvasCoreStores` contains interaction state only. Filter and Select Object snapshots are created and owned by `canvas-operations`; widgets subscribe through the application operation capability, so application-session state cannot leak back into the core store contract.

The reducer and engine communicate through document notifications and actions. Controllers communicate through constructor-injected ports or public capabilities, never by reaching into another controller's maps.

## Controllers and capabilities

`engine.ts` is the core composition root and document-event router. New behavior belongs in a focused controller with explicit dependencies and an idempotent `dispose()` method. The public engine handle exposes grouped capabilities only; the deprecated flat facade has been removed. Consumers request the smallest capability they need, and the full-engine import audit prevents UI modules from importing the core composition type.

Current extracted ownership:

- `RasterController`: base layer caches, decoded bitmaps, derived surfaces, adjustment surfaces, invalidation, and byte accounting.
- `RenderController`: render scheduler construction and lifecycle.
- `HistoryController`: active history ownership and inactive byte trimming.
- `PersistenceController`: dirty-bitmap flush barriers and disposal.
- `EditingController`: selection state and exclusive edit-lease lifecycle.
- `InteractionController`: public tool commands and their disposal boundary.
- `LayerController`: guarded layer mutation and thumbnail capability ports.
- `RasterExportController`: guarded layer rasterization and export preparation.
- `ControlPixelController`: transient control-pixel editing and commit ownership.
- `SelectionImageController`: guarded image-to-selection replacement.
- `LayerMutationController`: atomic layer-cache/document publication.
- `MaskResultController`, `FilterResultController`, and `GeneratedResultController`: guarded application-result adoption through narrow host ports.

Controller-local tests instantiate controller boundaries with fakes. Integration tests in `engine.test.ts` cover composition, document routing, and cross-controller invariants.

## Derived surfaces and invalidation

Every display-only pixel effect uses `DerivedSurfaceCache`. A slot is identified by layer ID and effect kind, and guarded by source surface identity, monotonic source version, and a deterministic parameter key. Reuse is safe only when every guard matches. Source replacement therefore cannot publish or reuse a surface produced for an older preview.

Invalidate only the affected effect when parameters change. Delete every derived slot when a layer is removed or replaced. Viewport culling happens before derived lookup or construction.

## Memory policy

The active engine has one 512 MB budget for base and derived surfaces. Enforcement is deterministic:

1. Evict least-recently-used derived surfaces.
2. Evict least-recently-used hidden base caches.
3. Never evict a visible base cache needed by the active frame.
4. Render over-budget visible bases normally and report the overage through diagnostics.

History retains at most 256 MB while active and is trimmed by bytes to 64 MB during cooldown.

## Edit leases

Application operations acquire one exclusive lease from `CanvasEditGate`. A lease carries an `AbortSignal`, has an explicit `isCurrent()` freshness check, and is idempotently released. Release, document/project invalidation, cooldown, or disposal aborts and stales the lease. Expected cancellation and stale results use status unions; exceptions represent unexpected failures.

## Lifecycle

Lifecycle states are `active`, `cooling`, `cool`, and `disposed`. Losing the final registry reference immediately detaches input/render targets and starts a dirty-pixel flush. Successful, still-current flushes release reconstructible raster resources and trim history. Failed flushes retain base pixels. Reacquisition increments the lifecycle generation so late flush or grace-period work cannot clean up or dispose a live engine.

## Adding functionality

- New tool: implement the tool interface under `tools/`, keep durable state in document actions, and expose only required tool capability methods.
- New derived effect: add a stable effect kind/parameter key, construct through `DerivedSurfaceCache`, and test warm hits, precise invalidation, stale source guards, culling, and byte accounting.
- New layer operation: place pure pixel/document math in the core and orchestration in a controller; accept narrow ports and return status unions for expected no-op/stale cases.
- New application coordinator: place it under `canvas-operations/`, acquire an edit lease, capture an export guard, own its session store and cancellation, and publish/commit through preview/layer capabilities.

Deterministic CI assertions use operation counters and byte totals. Wall-clock timing is informational only.
