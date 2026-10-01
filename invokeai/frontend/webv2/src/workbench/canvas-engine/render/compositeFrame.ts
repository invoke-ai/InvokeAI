import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type {
  PreviewStateController,
  SamPreviewState,
} from '@workbench/canvas-engine/controllers/previewStateController';
import type { RasterController } from '@workbench/canvas-engine/controllers/rasterController';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { EngineStores } from '@workbench/canvas-engine/engineStores';
import type { DerivedSurfaceCache } from '@workbench/canvas-engine/render/derivedSurfaceCache';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { FrameDamage, Mat2d, Rect } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { isLayerContributing } from '@workbench/canvas-engine/document/layerEligibility';
import { getSourceContentRect, isRenderableLayer, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import {
  compositeDocument,
  reusePreparation,
  shouldSmoothAtZoom,
  type CompositeOptions,
  type CompositePreparation,
} from '@workbench/canvas-engine/render/compositor';
import { calculateActiveFrameLayerIds, committedLeafBounds } from '@workbench/canvas-engine/render/frameDemand';
import { FULL_DAMAGE } from '@workbench/canvas-engine/types';

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
    /** What must be repainted; absent repaints everything. */
    damage?: FrameDamage
  ): void;
}

interface LeafBoundsMemo {
  readonly layer: SemanticLeaf['layer'];
  readonly matrix: Mat2d;
  readonly live: Rect | undefined;
  readonly bounds: Rect | null;
  // Legacy gradients without an extent are document-sized.
  readonly documentWidth: number;
  readonly documentHeight: number;
}

const sameOptionalRect = (a: Rect | undefined, b: Rect | undefined): boolean =>
  a === b || (!!a && !!b && a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height);

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
  const ensureLayerCaches = (
    doc: CanvasDocumentContractV3,
    frame: CompositePreparation,
    activeFrameLayerIds: ReadonlySet<string>
  ): void => {
    for (const { layer } of frame.leaves) {
      if (!activeFrameLayerIds.has(layer.id) || !isLayerContributing(layer) || !renderableSourceOf(layer)) {
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

  // The frame description is reused until the document, isolation or override contents change; each leaf's
  // bounds until its contract, placement, live cache extent or the document size does.
  let preparation: CompositePreparation | null = null;
  // A bounded or empty repaint reads only part of the working set; the usage window stays open from the last full
  // repaint so artifacts it skipped are not evicted as unused.
  let fullFrameUsage: ReturnType<typeof raster.beginFrame> | null = null;
  let boundsMemo = new Map<string, LeafBoundsMemo>();

  const describeFrame = (
    doc: CanvasDocumentContractV3,
    isolationLayerId: string | null,
    overrides: CompositeOptions['transformOverrides']
  ): CompositePreparation => {
    const plan = reusePreparation(preparation, doc, {
      backend: deps.backend,
      groupSurface: deps.getGroupSurface,
      isolationLayerId,
      transformOverrides: overrides,
    });
    preparation = plan;
    const nextMemo = new Map<string, LeafBoundsMemo>();
    const bounds = plan.leaves.map((leaf, index) => {
      const matrix = plan.matrices[index]!;
      const live = layerCache.peek(leaf.id)?.rect;
      const previous = boundsMemo.get(leaf.id);
      const memo =
        previous &&
        previous.layer === leaf.layer &&
        previous.matrix === matrix &&
        previous.documentWidth === doc.width &&
        previous.documentHeight === doc.height &&
        sameOptionalRect(previous.live, live)
          ? previous
          : {
              bounds: committedLeafBounds(leaf, matrix, doc, live),
              documentHeight: doc.height,
              documentWidth: doc.width,
              layer: leaf.layer,
              live: live && { ...live },
              matrix,
            };
      nextMemo.set(leaf.id, memo);
      return memo.bounds;
    });
    boundsMemo = nextMemo;
    return { ...plan, bounds };
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
      const overrides = !isolatedGuard && transformOverrides.size > 0 ? transformOverrides : null;
      const frame = describeFrame(doc, isolatedGuard?.layerId ?? null, overrides);
      // Snapshot filter previews only when active, avoiding an empty map allocation every frame.
      const layerPreviews = !isolatedGuard && previews.filterCount() > 0 ? previews.filterSnapshot() : null;
      const activeFrameLayerIds = calculateActiveFrameLayerIds({
        document: doc,
        preparation: frame,
        viewport: visibleDocumentRect(screen),
      });
      // A previewed layer draws through its cache entry even when only the preview is on screen.
      for (const layerId of layerPreviews?.keys() ?? []) {
        activeFrameLayerIds.add(layerId);
      }
      ensureLayerCaches(doc, frame, activeFrameLayerIds);
      const frameDamage = isolatedGuard ? FULL_DAMAGE : (damage ?? FULL_DAMAGE);
      const usage = raster.beginFrame();
      if (frameDamage.kind === 'full' || !fullFrameUsage) {
        fullFrameUsage = usage;
      }

      compositeDocument(screen, doc, layerCache, view, {
        damage: frameDamage,
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
        layerPreviews,
        maskPatternTile: deps.getMaskPatternTile,
        preparation: frame,
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
        transformOverrides: overrides,
      });

      enforceBudget(activeFrameLayerIds, fullFrameUsage);
    },
  };
};
