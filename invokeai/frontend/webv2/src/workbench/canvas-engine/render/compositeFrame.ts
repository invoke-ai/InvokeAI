import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type {
  PreviewStateController,
  SamPreviewState,
} from '@workbench/canvas-engine/controllers/previewStateController';
import type { RasterMemoryBudgetController } from '@workbench/canvas-engine/controllers/rasterMemoryBudgetController';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { EngineStores } from '@workbench/canvas-engine/engineStores';
import type { DerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { getSourceContentRect, isRenderableLayer, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { compositeDocument, shouldSmoothAtZoom } from '@workbench/canvas-engine/render/compositor';
import { calculateActiveFrameLayerIds } from '@workbench/canvas-engine/render/frameDemand';
import { enforceSurfaceBudget } from '@workbench/canvas-engine/render/surfaceBudget';

import type { FloatingSelectionFrame } from './floatingSelectionFrame';
import type { LayerTransformOverrides } from './overlayFrame';

export interface CreateCompositeFrameDeps {
  readonly layerCache: LayerCacheStore;
  readonly derivedSurfaceCache: DerivedSurfaceCache;
  readonly backend: RasterBackend;
  readonly diagnostics: CanvasDiagnostics;
  readonly memory: RasterMemoryBudgetController;
  readonly previews: PreviewStateController;
  readonly stores: EngineStores;
  readonly viewport: Viewport;
  readonly transformOverrides: LayerTransformOverrides;
  readonly getAdjustedSurface: (layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null;
  readonly getMaskPatternTile: (style: string, color: string) => RasterSurface | null;
  readonly getCheckerboardTile: () => RasterSurface;
  /** Starts (or joins) the rasterization of a layer whose cache is stale. */
  readonly rasterizeLayer: (layer: CanvasLayerContract, doc: CanvasDocumentContractV2) => void;
  readonly syncMemoryBaselines: () => void;
  readonly deleteDerivedSurfaces: (layerId: string) => void;
}

export interface CompositeFrame {
  /** Composites the document onto the screen surface and enforces the surface budget. */
  draw(
    screen: RasterSurface,
    doc: CanvasDocumentContractV2,
    view: Mat2d,
    floatFrame: FloatingSelectionFrame | null,
    samPreview: SamPreviewState | null
  ): void;
}

/**
 * Draws the document onto the screen surface.
 *
 * This is the expensive half of a frame and runs only when pixels, layer order
 * or the viewport actually changed — an overlay-only invalidation reuses the
 * last composite. Before drawing it works out which layers the frame really
 * demands (visible, in view, not isolated away) and kicks off rasterization for
 * any of those whose cache is stale; after drawing it enforces the surface
 * budget against exactly that set, so a layer the user is looking at is never
 * the one evicted.
 *
 * SAM isolation makes this frame describe one layer instead of the document:
 * the composite is filtered down to the isolated layer and clipped to the
 * preview rect, and every other in-flight embellishment — filter previews, the
 * staged candidate, the floating selection, transform overrides — is suppressed,
 * because none of them describe what the user is being asked to judge.
 */
export const createCompositeFrame = (deps: CreateCompositeFrameDeps): CompositeFrame => {
  const { derivedSurfaceCache, diagnostics, layerCache, memory, previews, stores, transformOverrides, viewport } = deps;

  /**
   * Ensures every layer this frame demands has a cache entry, and starts a
   * rasterization for any that is stale.
   *
   * Creation must never resize an existing entry: a paint layer's live cache can
   * have grown past its persisted content rect (fresh unflushed strokes), and
   * shrinking it back to the contract size here would destroy those pixels. The
   * rasterizer owns sizing the surface and placing its content rect.
   */
  const ensureLayerCaches = (doc: CanvasDocumentContractV2, activeFrameLayerIds: ReadonlySet<string>): void => {
    for (const layer of doc.layers) {
      // The layer's rasterizable source: a raster/control `source`, or a mask
      // layer's alpha bitmap viewed as a paint source (colorized at composite).
      if (!layer.isEnabled || !renderableSourceOf(layer) || !activeFrameLayerIds.has(layer.id)) {
        continue;
      }
      if (!isRenderableLayer(layer)) {
        continue;
      }
      const entry = layerCache.getOrCreateRect(layer.id, getSourceContentRect(layer, doc));
      if (entry.stale) {
        deps.rasterizeLayer(layer, doc);
      }
    }
  };

  /** The document-space rect currently on screen, which bounds what the frame demands. */
  const visibleDocumentRect = (screen: RasterSurface): Rect => {
    const viewportSize = viewport.getViewportSize();
    const dpr = viewport.getDpr();
    const cssWidth = viewportSize.width > 0 ? viewportSize.width : screen.width / dpr;
    const cssHeight = viewportSize.height > 0 ? viewportSize.height : screen.height / dpr;
    const topLeft = viewport.screenToDocument({ x: 0, y: 0 });
    const bottomRight = viewport.screenToDocument({ x: cssWidth, y: cssHeight });
    return {
      height: Math.abs(bottomRight.y - topLeft.y),
      width: Math.abs(bottomRight.x - topLeft.x),
      x: Math.min(topLeft.x, bottomRight.x),
      y: Math.min(topLeft.y, bottomRight.y),
    };
  };

  /**
   * Trims cached surfaces back inside the budget, protecting the layers this
   * frame drew and anything pinned by in-flight background work. Runs after the
   * composite so the frame the user is looking at is never the eviction victim.
   */
  const enforceBudget = (activeFrameLayerIds: ReadonlySet<string>): void => {
    deps.syncMemoryBaselines();
    const snapshot = memory.snapshot();
    const surfaceBudgetBytes = Math.max(
      0,
      memory.budgetBytes - snapshot.decodedBytes - snapshot.detachedBytes - snapshot.reservedBytes
    );
    const { evictedBaseLayerIds } = enforceSurfaceBudget(
      layerCache,
      derivedSurfaceCache,
      new Set([...activeFrameLayerIds, ...memory.pinnedLayerIds()]),
      surfaceBudgetBytes,
      diagnostics
    );
    deps.syncMemoryBaselines();
    // Prune version-keyed dependents for every evicted id (mirrors dropLayer):
    // the evicted surface is gone, so its adjusted-surface memo and thumbnail
    // state must not linger keyed to a version the re-rasterized entry will
    // exceed (the cache floor keeps versions monotonic across the recreate).
    for (const id of evictedBaseLayerIds) {
      deps.deleteDerivedSurfaces(id);
      stores.thumbnailVersion.delete(id);
      stores.thumbnailStatus.delete(id);
    }
  };

  return {
    draw: (screen, doc, view, floatFrame, samPreview) => {
      const stagedPreview = previews.getStaged();
      const stagedPlacement = stagedPreview?.placement;
      const isolatedGuard = samPreview?.isolated ? samPreview.guard : null;
      const isolatedIds = isolatedGuard ? new Set([isolatedGuard.layerId]) : null;

      const liveCacheRects = new Map<string, Rect>();
      for (const layer of doc.layers) {
        const rect = layerCache.peek(layer.id)?.rect;
        if (rect) {
          liveCacheRects.set(layer.id, rect);
        }
      }
      const activeFrameLayerIds = calculateActiveFrameLayerIds({
        document: doc,
        isolationLayerIds: isolatedIds ?? undefined,
        liveCacheRects,
        transformOverrides: !isolatedGuard ? transformOverrides : undefined,
        viewport: visibleDocumentRect(screen),
      });
      ensureLayerCaches(doc, activeFrameLayerIds);

      const compositeDoc = isolatedIds
        ? { ...doc, layers: doc.layers.filter((layer) => isolatedIds.has(layer.id)) }
        : doc;
      compositeDocument(screen, compositeDoc, layerCache, view, {
        // Memoized adjusted surfaces for raster layers with brightness/contrast/
        // saturation/curves (not recomputed per frame — see adjustedSurfaceCache).
        adjustedSurface: deps.getAdjustedSurface,
        // The raster backend + mask fill tile resolver drive the mask colorize
        // path (alpha stencil → source-in fill colour/pattern, above all layers).
        backend: deps.backend,
        // Feed the cached checkerboard tile only while the toggle is ON; passing
        // `null` renders transparent documents without a checkerboard.
        checkerboardTile: stores.checkerboard.get() ? deps.getCheckerboardTile() : null,
        clipRect: isolatedGuard && samPreview ? samPreview.rect : null,
        derivedSurfaces: derivedSurfaceCache,
        diagnostics,
        // Pixels in flight, drawn directly above the layer they were cut from.
        floatingSelection: isolatedGuard ? null : (floatFrame?.composite ?? null),
        // Crisp + cheap when zoomed in (nearest-neighbor up-scale), smooth when
        // shrinking (bilinear down-scale). See `shouldSmoothAtZoom`.
        imageSmoothing: shouldSmoothAtZoom(viewport.getZoom()),
        // Non-destructive control-filter previews, drawn in place of the layer's
        // committed pixels. Snapshotted only when one is actually active — this
        // runs every composite frame, so an unconditional copy would allocate a
        // map per frame for the overwhelmingly common case of none.
        layerPreviews: !isolatedGuard && previews.filterCount() > 0 ? previews.filterSnapshot() : null,
        maskPatternTile: deps.getMaskPatternTile,
        // Candidate-specific placement wins for final images. Progress frames
        // and legacy image inputs continue to follow the CURRENT bbox origin.
        stagedPreview:
          !isolatedGuard && stagedPreview
            ? {
                opacity: stagedPlacement?.opacity ?? 1,
                rect: stagedPlacement
                  ? {
                      height: stagedPlacement.height,
                      width: stagedPlacement.width,
                      x: stagedPlacement.x,
                      y: stagedPlacement.y,
                    }
                  : { height: stagedPreview.height, width: stagedPreview.width, x: doc.bbox.x, y: doc.bbox.y },
                surface: stagedPreview.surface,
              }
            : null,
        // While a text-edit session is open on a layer, skip it in the composite —
        // the contenteditable portal shows its live text instead (avoids double-draw).
        skipLayerId: stores.textEditSession.get()?.layerId ?? null,
        transformOverrides: !isolatedGuard && transformOverrides.size > 0 ? transformOverrides : null,
      });

      enforceBudget(activeFrameLayerIds);
    },
  };
};
