import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type {
  PreviewStateController,
  SamPreviewState,
} from '@workbench/canvas-engine/controllers/previewStateController';
import type { RasterController } from '@workbench/canvas-engine/controllers/rasterController';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { EngineStores } from '@workbench/canvas-engine/engineStores';
import type { DerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { LayerDamage, Mat2d, Rect } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import { isLayerContributing } from '@workbench/canvas-engine/document/layerEligibility';
import { getSourceContentRect, isRenderableLayer, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import {
  compositeDocument,
  shouldSmoothAtZoom,
  type CompositeOptions,
} from '@workbench/canvas-engine/render/compositor';
import { calculateActiveFrameLayerIds } from '@workbench/canvas-engine/render/frameDemand';

import type { FloatingSelectionFrame } from './floatingSelectionFrame';
import type { LayerTransformOverrides } from './overlayFrame';

export interface CreateCompositeFrameDeps {
  readonly layerCache: LayerCacheStore;
  readonly derivedSurfaceCache: DerivedSurfaceCache;
  readonly backend: RasterBackend;
  readonly diagnostics: CanvasDiagnostics;
  readonly raster: Pick<RasterController, 'beginFrame' | 'enforceBudget'>;
  readonly previews: PreviewStateController;
  readonly stores: EngineStores;
  readonly viewport: Viewport;
  readonly transformOverrides: LayerTransformOverrides;
  readonly getAdjustedSurface: (layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null;
  readonly getGroupSurface: NonNullable<CompositeOptions['groupSurface']>;
  readonly getMaskPatternTile: (style: string, color: string) => RasterSurface | null;
  readonly getCheckerboardTile: () => RasterSurface;
  /** Starts (or joins) the rasterization of a layer whose cache is stale. */
  readonly rasterizeLayer: (layer: CanvasLayerContract, doc: CanvasDocumentContractV3) => void;
}

export interface CompositeFrame {
  /** Composites the document onto the screen surface and enforces the surface budget. */
  draw(
    screen: RasterSurface,
    doc: CanvasDocumentContractV3,
    view: Mat2d,
    floatFrame: FloatingSelectionFrame | null,
    samPreview: SamPreviewState | null,
    /**
     * Layer-local repaint regions; null means full repaint. Partial damage is valid only when every invalidation
     * named its region.
     */
    damage?: LayerDamage[] | null
  ): void;
}

/**
 * Composite only for pixel/order/view changes. Rasterize demanded visible layers and protect them during post-draw
 * eviction. SAM isolation restricts drawing and clipping to its layer/rect, suppressing unrelated previews, floats
 * and transform overrides.
 */
export const createCompositeFrame = (deps: CreateCompositeFrameDeps): CompositeFrame => {
  const { derivedSurfaceCache, diagnostics, layerCache, previews, raster, stores, transformOverrides, viewport } = deps;

  /**
   * Create missing caches and rasterize stale ones, but never resize existing entries here: unflushed paint may
   * exceed persisted bounds. Rasterization owns bounds changes.
   */
  const ensureLayerCaches = (doc: CanvasDocumentContractV3, activeFrameLayerIds: ReadonlySet<string>): void => {
    for (const layer of getDocumentLeaves(doc)) {
      if (!isLayerContributing(layer) || !renderableSourceOf(layer) || !activeFrameLayerIds.has(layer.id)) {
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

  /** Evict after drawing; the frame's demanded layers and the artifacts it read form the working set. */
  const enforceBudget = (
    activeFrameLayerIds: ReadonlySet<string>,
    usage: ReturnType<typeof raster.beginFrame>
  ): void => {
    const { evictedBaseLayerIds } = raster.enforceBudget(activeFrameLayerIds, usage);
    for (const id of evictedBaseLayerIds) {
      stores.thumbnailVersion.delete(id);
      stores.thumbnailStatus.delete(id);
    }
  };

  return {
    draw: (screen, doc, view, floatFrame, samPreview, damage) => {
      const stagedPreview = previews.getStaged();
      const stagedPlacement = stagedPreview?.placement;
      const isolatedGuard = samPreview?.isolated ? samPreview.guard : null;
      const isolatedIds = isolatedGuard ? new Set([isolatedGuard.layerId]) : null;

      const liveCacheRects = new Map<string, Rect>();
      for (const layer of getDocumentLeaves(doc)) {
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
      const usage = raster.beginFrame();

      compositeDocument(screen, doc, layerCache, view, {
        damage: isolatedGuard ? null : damage,
        adjustedSurface: deps.getAdjustedSurface,
        groupSurface: deps.getGroupSurface,
        backend: deps.backend,
        checkerboardTile: stores.checkerboard.get() ? deps.getCheckerboardTile() : null,
        clipRect: isolatedGuard && samPreview ? samPreview.rect : null,
        derivedSurfaces: derivedSurfaceCache,
        diagnostics,
        // Pixels in flight, drawn directly above the layer they were cut from.
        floatingSelection: isolatedGuard ? null : (floatFrame?.composite ?? null),
        imageSmoothing: shouldSmoothAtZoom(viewport.getZoom()),
        isolationLayerId: isolatedGuard?.layerId ?? null,
        // Snapshot filter previews only when active, avoiding an empty map allocation every frame.
        layerPreviews: !isolatedGuard && previews.filterCount() > 0 ? previews.filterSnapshot() : null,
        maskPatternTile: deps.getMaskPatternTile,
        // Screen frames are display-time: the region tint may draw.
        regionOverlays: true,
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
        // Skip the edited text layer because its portal renders live content.
        skipLayerId: stores.textEditSession.get()?.layerId ?? null,
        transformOverrides: !isolatedGuard && transformOverrides.size > 0 ? transformOverrides : null,
      });

      enforceBudget(activeFrameLayerIds, usage);
    },
  };
};
