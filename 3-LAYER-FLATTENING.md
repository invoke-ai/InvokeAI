# 3-Layer Flattening System — Design Document

## Problem

Each canvas entity (raster layer, control layer, inpaint mask, regional guidance) creates its own `Konva.Layer`, which in turn creates a separate HTML `<canvas>` element in the DOM. With many layers, the browser must composite all of these canvas elements on every frame, leading to significant GPU/CPU overhead and sluggish interactions.

## Goal

Reduce the number of active `<canvas>` elements from **N** (one per entity) to **3** (constant), regardless of how many entities exist. This provides a dramatic performance improvement especially on lower-end devices.

## Architecture

### The 3-Layer System

```
┌─────────────────────────────────────┐
│       Konva.Layer: "behind"         │  ← All entities below the active one,
│  (flattened composite of layers     │     flattened into a single canvas
│   behind the active entity)         │
├─────────────────────────────────────┤
│       Konva.Layer: "active"         │  ← The currently selected entity,
│  (the entity being edited)          │     fully interactive with all
│                                     │     sub-modules (transformer, etc.)
├─────────────────────────────────────┤
│       Konva.Layer: "ahead"          │  ← All entities above the active one,
│  (flattened composite of layers     │     flattened into a single canvas
│   above the active entity)          │
└─────────────────────────────────────┘
```

Plus the existing **background layer** and **preview layer** (bbox, staging area, tool) which are unchanged.

### Key Concepts

**Flattening**: Render multiple entity layers into a single off-screen canvas, then display that canvas as a single `Konva.Image` node on the composite `Konva.Layer`. This is similar to what `CanvasCompositorModule.getCompositeCanvas()` already does for generation.

**Active Entity**: The entity currently selected by the user. This entity keeps its own dedicated `Konva.Layer` so it can be interactively edited (brush strokes, transforms, filters, SAM segmentation, etc.).

**Re-flattening**: When the user switches the active entity, the "behind" and "ahead" composites must be regenerated. This can be done incrementally (add/remove one entity from composite) or fully (re-render all).

## Implementation Plan

### Phase 1: New Module — `CanvasLayerFlatteningModule`

Create a new module at `konva/CanvasLayerFlatteningModule.ts`:

```typescript
class CanvasLayerFlatteningModule extends CanvasModuleBase {
  // The two composite Konva.Layers
  behindLayer: Konva.Layer;
  aheadLayer: Konva.Layer;

  // Cached composite canvases
  behindCanvas: HTMLCanvasElement | null;
  aheadCanvas: HTMLCanvasElement | null;

  // Konva.Image nodes to display the composites
  behindImage: Konva.Image;
  aheadImage: Konva.Image;

  // Track which entity is active
  activeEntityId: string | null;
}
```

**Responsibilities:**

1. Subscribe to `selectedEntityIdentifier` changes
2. On entity selection change, re-flatten the "behind" and "ahead" composites
3. Manage the two composite `Konva.Layer` nodes on the stage
4. Ensure individual entity adapters DON'T add their own `Konva.Layer` to the stage (except the active one)

### Phase 2: Modify Entity Adapter Lifecycle

**Current flow** (in `CanvasEntityAdapterBase` constructor):

```typescript
this.konva = {
  layer: new Konva.Layer({ ... }),
};
this.manager.stage.addLayer(this.konva.layer);
```

**New flow:**

- Entity adapters still create a `Konva.Layer` (needed for rendering to off-screen canvas via `getCanvas()`)
- But they do NOT add it to the stage by default
- Only the **active entity** has its layer added to the stage
- The flattening module manages which entity is "live" on stage

### Phase 3: Composite Rendering

Reuse the existing compositing logic from `CanvasCompositorModule.getCompositeCanvas()`:

```typescript
flattenBehind(activeIndex: number): HTMLCanvasElement {
  const behindAdapters = this.getOrderedAdapters().slice(0, activeIndex);
  // Filter to only enabled/visible adapters
  const canvas = document.createElement('canvas');
  // ... render each adapter's getCanvas() onto the composite
  return canvas;
}
```

**Blend modes**: Each raster layer can have a `globalCompositeOperation`. When flattening, these must be applied in order during compositing (same as `getCompositeCanvas` already does).

**Opacity**: Each layer's opacity must be respected during compositing.

**Adjustments**: Per-layer adjustments (brightness, contrast, curves) must be baked into the flattened result.

### Phase 4: Incremental Updates

When only the active entity changes content (brush strokes, image generation), the "behind" and "ahead" composites don't need to change. This is the common case and should be fast.

When a non-active entity changes (rare during editing), the affected composite must be regenerated. This can be detected via entity state subscriptions.

**Cache invalidation strategy:**

- Hash the state of all entities in each composite (similar to `getCompositeHash()` in compositor)
- Only re-flatten when the hash changes
- Cache the flattened canvas in the `CanvasCacheModule`

### Phase 5: Entity Selection Switch

When the user selects a different entity:

1. Remove the previously active entity's `Konva.Layer` from the stage
2. Render the previously active entity into the appropriate composite (behind or ahead)
3. Extract the newly active entity from its composite
4. Add the newly active entity's `Konva.Layer` to the stage
5. Re-render both composites without the newly active entity
6. Restore z-order: behind → active → ahead

**Optimization**: If the new selection is adjacent to the old one, only one composite needs to change by adding/removing one entity.

### Phase 6: Handle Edge Cases

**Entity types across composites:**
The draw order is: raster layers → control layers → regions → inpaint masks. All entity types participate in flattening. The "behind" composite includes all entities below the active one regardless of type, and "ahead" includes all above.

**Isolated preview modes:**
When filtering/transforming/segmenting, only the active entity should be visible. The composites should be hidden (same as current behavior).

**Staging preview:**
During generation staging with `isolatedStagingPreview`, only raster layers should be visible. The composites need to only include raster layer content.

**Disabled entities:**
Disabled entities are skipped during flattening (not rendered into composites).

**Entity type visibility:**
If a type is globally hidden (e.g., all control layers hidden), those entities are excluded from composites.

## Files to Modify

| File                                            | Change                                                  |
| ----------------------------------------------- | ------------------------------------------------------- |
| `konva/CanvasLayerFlatteningModule.ts`          | **NEW** — Core flattening logic                         |
| `konva/CanvasManager.ts`                        | Register new module, integrate into lifecycle           |
| `konva/CanvasEntity/CanvasEntityAdapterBase.ts` | Don't auto-add layer to stage; expose attach/detach API |
| `konva/CanvasEntityRendererModule.ts`           | Delegate layer arrangement to flattening module         |
| `konva/CanvasCompositorModule.ts`               | Reuse/share compositing utilities                       |
| `konva/CanvasStageModule.ts`                    | No change needed (addLayer/stage management stays)      |

## Performance Expectations

| Metric                 | Before                       | After                                              |
| ---------------------- | ---------------------------- | -------------------------------------------------- |
| Canvas elements in DOM | N + 2 (background + preview) | 5 (background + behind + active + ahead + preview) |
| Browser composite cost | O(N) per frame               | O(1) per frame                                     |
| Layer switch cost      | O(1)                         | O(N) one-time re-flatten                           |
| Active layer edit cost | O(1)                         | O(1) unchanged                                     |

The trade-off is that switching the selected entity requires a one-time re-flatten, but this can be made fast with caching and incremental updates. The per-frame rendering cost drops from O(N) to O(1), which is the dominant performance factor.

## Risks and Mitigations

1. **Visual fidelity**: Flattened composites must exactly match the per-layer rendering. Use the same compositing pipeline (`getCompositeCanvas`) to ensure consistency.

2. **Blend mode accuracy**: CSS `mix-blend-mode` on individual canvas elements may differ slightly from `globalCompositeOperation` during canvas compositing. Test thoroughly with all blend modes.

3. **Re-flatten latency**: For 50+ layers with complex content, flattening may take 50-100ms. Mitigate with:
   - Async flattening in a Web Worker (see README: "Perf: Konva in a web worker")
   - Show a brief transition indicator during re-flatten
   - Incremental flattening (only re-render changed entities)

4. **Memory**: Two extra composite canvases at full resolution. For a 4K canvas, this is ~32MB per composite. Acceptable for modern systems.

## Dependencies

- Konva's `layer.toCanvas()` or manual canvas rendering via `getCanvas()` on each adapter
- `CanvasCompositorModule` compositing utilities (hash computation, canvas compositing)
- `CanvasCacheModule` for caching flattened results

## References

- Current compositing: `konva/CanvasCompositorModule.ts` lines 204-247
- README future enhancement: `controlLayers/README.md` lines 196-206
- Entity adapter rendering: `konva/CanvasEntity/CanvasEntityAdapterBase.ts`
- Entity z-order management: `konva/CanvasEntityRendererModule.ts` lines 105-146
