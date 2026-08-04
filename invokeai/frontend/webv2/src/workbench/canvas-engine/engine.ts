import type {
  CanvasHistoryCapability,
  CanvasDiagnosticsCapability,
  CanvasEngine,
  CanvasEngineExportCapability,
  CanvasEngineLayerCapability,
  CanvasEnginePreviewCapability,
  CanvasEngineToolCapability,
  CanvasInteractionState,
  CanvasInteractionStateCapability,
  CanvasDocumentCapability,
  CanvasDocumentSnapshot,
  CanvasLifecycleCapability,
  CanvasSelectionCapability,
  CanvasSurfaceCapability,
  CanvasViewportCapability,
  BooleanRasterOperation,
  BooleanRasterResult,
  CropLayerResult,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  ExtractMaskedAreaResult,
  MergeVisibleResult,
  NewRasterLayerResult,
  PsdExportResult,
} from '@workbench/canvas-engine/capabilities';
import type {
  CanvasCompositeExecutorDeps,
  CaptureRasterSnapshotResult,
} from '@workbench/canvas-engine/rasterTransactions';
export type {
  BooleanRasterResult,
  CanvasDiagnosticsCapability,
  CanvasEngine,
  CanvasEngineExportCapability,
  CanvasEngineLayerCapability,
  CanvasEnginePreviewCapability,
  CanvasEngineToolCapability,
  CommitGeneratedImageOptions,
  CommitGeneratedImageResult,
  CommitStagedImageOptions,
  CommitStagedImageResult,
  CropLayerResult,
  ExportBakedLayerBlobResult,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  ExtractMaskedAreaResult,
  FilterPreviewInput,
  GeneratedImageTarget,
  LayerExportGuard,
  LayerThumbnailRequestResult,
  MergeVisibleResult,
  PsdExportResult,
  ReplaceSelectionFromImageResult,
} from '@workbench/canvas-engine/capabilities';
export type {
  CommitMaskImageResult,
  CommitMaskImageResultOptions,
  MaskImageResultTarget,
} from '@workbench/canvas-engine/controllers/maskResultController';
export type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
} from '@workbench/canvas-engine/controllers/filterResultController';
import type { CanvasApplicationHost } from '@workbench/canvas-engine/applicationHost';
import type {
  CanvasImageRef,
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { FontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { OverlayCursor } from '@workbench/canvas-engine/render/overlayRenderer';
import type { RenderScheduler } from '@workbench/canvas-engine/render/scheduler';
import type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, RenderFlags, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { ControlPixelController } from '@workbench/canvas-engine/controllers/controlPixelController';
import { EditingController } from '@workbench/canvas-engine/controllers/editingController';
import { FilterResultController } from '@workbench/canvas-engine/controllers/filterResultController';
import { GeneratedResultController } from '@workbench/canvas-engine/controllers/generatedResultController';
import { HistoryController } from '@workbench/canvas-engine/controllers/historyController';
import { InteractionController } from '@workbench/canvas-engine/controllers/interactionController';
import { LayerController } from '@workbench/canvas-engine/controllers/layerController';
import { LayerMutationController } from '@workbench/canvas-engine/controllers/layerMutationController';
import { MaskResultController } from '@workbench/canvas-engine/controllers/maskResultController';
import {
  createCanvasMutationContext,
  type DocumentEditPermit,
} from '@workbench/canvas-engine/controllers/mutationContext';
import { PersistenceController } from '@workbench/canvas-engine/controllers/persistenceController';
import { PsdExportController } from '@workbench/canvas-engine/controllers/psdExportController';
import { RasterController } from '@workbench/canvas-engine/controllers/rasterController';
import {
  RasterExportController,
  type ExportLayerPixelsResult,
} from '@workbench/canvas-engine/controllers/rasterExportController';
import { RenderController } from '@workbench/canvas-engine/controllers/renderController';
import { StagedResultController } from '@workbench/canvas-engine/controllers/stagedResultController';
import { StructuralLayerController } from '@workbench/canvas-engine/controllers/structuralLayerController';
import { createCanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import {
  createEngineStores,
  type EngineStores,
  type ScalarStore,
  type TextToolOptions,
} from '@workbench/canvas-engine/engineStores';
import {
  exportRasterComposite as exportRasterCompositeWithDeps,
  RasterCompositeOverBudgetError,
  type RasterCompositeExportRequest,
  type RasterCompositeExportSnapshot,
} from '@workbench/canvas-engine/exportRasterComposite';
import { createPointerPipeline, type PointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createWheelHandler } from '@workbench/canvas-engine/input/wheel';
import { isEmpty, union } from '@workbench/canvas-engine/math/rect';
import { createCheckerboardTile } from '@workbench/canvas-engine/render/compositor';
import { createFontLoader, domFontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import { createMaskPatternTile } from '@workbench/canvas-engine/render/maskFill';
import { renderOverlay } from '@workbench/canvas-engine/render/overlayRenderer';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createDomRasterBackend, type RasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import { rasterizeSource, type ImageResolver, type RasterizeDeps } from '@workbench/canvas-engine/render/rasterizers';
import { getLayerThumbnailDisplayKey } from '@workbench/canvas-engine/render/thumbnail';
import { documentDeltaToLocal, liftSelectedPixels } from '@workbench/canvas-engine/selection/floatingSelection';
import { ANTS_STEP_PX, createAntsAnimator, type AntsAnimator } from '@workbench/canvas-engine/selection/marchingAnts';
import { createBboxTool } from '@workbench/canvas-engine/tools/bboxTool';
import { createBrushTool } from '@workbench/canvas-engine/tools/brushTool';
import { createColorPickerTool } from '@workbench/canvas-engine/tools/colorPickerTool';
import { createEraserTool } from '@workbench/canvas-engine/tools/eraserTool';
import { createGradientTool } from '@workbench/canvas-engine/tools/gradientTool';
import { createLassoTool } from '@workbench/canvas-engine/tools/lassoTool';
import { createMarqueeTool } from '@workbench/canvas-engine/tools/marqueeTool';
import { layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';
import { createMoveTool } from '@workbench/canvas-engine/tools/moveTool';
import { stepBrushSize } from '@workbench/canvas-engine/tools/paintConstants';
import { createSamTool } from '@workbench/canvas-engine/tools/samTool';
import { createShapeTool } from '@workbench/canvas-engine/tools/shapeTool';
import { createTextTool } from '@workbench/canvas-engine/tools/textTool';
import { createTransformTool } from '@workbench/canvas-engine/tools/transformTool';
import { createViewport, MAX_DPR, type Viewport } from '@workbench/canvas-engine/viewport';

import type { ImagePatchApply } from './history/imagePatch';
import type { CanvasProjectMutation } from './mutationContracts';
import type { StrokeCommittedEvent, Tool, ToolContext } from './tools/tool';

import { createBitmapStore, type BitmapStore } from './document/bitmapStore';
import { createDocumentMirror, type DocumentMirror } from './document/documentMirror';
import { decideLayerChange } from './document/layerChangeDecision';
import { getSourceBounds, getSourceContentRect, isRenderableLayer, renderableSourceOf } from './document/sources';
import { createLayerExportGuards, isSupportedExportSource } from './layerExportGuards';
import { createLayerRasterizer } from './layerRasterizer';
import { createPreviewPublisher } from './previewPublisher';
import { createRasterSnapshotCapture } from './rasterSnapshotCapture';
import { createCompositeFrame } from './render/compositeFrame';
import { floatingSelectionFrame } from './render/floatingSelectionFrame';
import { createOverlayFrame } from './render/overlayFrame';
import { createSelectObjectBridge } from './selectObjectBridge';
import { createStrokeCommit } from './strokeCommit';
import { createViewTool } from './tools/viewTool';

/**
 * Result of {@link CanvasEngineExportCapability.exportRasterLayersToPsd}: `'exported'` on
 * success, `'nothing'` when there are no raster layers with content, `'too-large'`
 * when the union bounds exceed the PSD dimension cap, and `'not-ready'` when a
 * participant's cache is still decoding (nothing exported — surface feedback).
 */
/**
 * Re-exported from the controller that produces it, so the many callers that
 * name it off `canvas-engine/engine` keep resolving while there is only one
 * definition to keep in step.
 */
export type { ExportLayerPixelsResult };

export type ExportBakedLayerPixelsResult = ExportLayerPixelsResult;

/** Runs every teardown step, then rethrows the first failure after cleanup is terminal. */
const createCleanupAccumulator = (): { run: (step: () => void) => void; throwIfFailed: () => void } => {
  let firstError: unknown;
  let hasFailed = false;
  return {
    run: (step) => {
      try {
        step();
      } catch (error) {
        if (!hasFailed) {
          firstError = error;
          hasFailed = true;
        }
      }
    },
    throwIfFailed: () => {
      if (hasFailed) {
        throw firstError instanceof Error ? firstError : new Error(String(firstError));
      }
    },
  };
};

export interface CanvasEngineErrorReport {
  area: 'canvas-engine';
  context: { error: string; layerId: string };
  message: 'Layer thumbnail rasterization failed' | 'Bitmap persistence failed';
  namespace: 'canvas';
  projectId: string;
}

/** Options for {@link createCanvasEngine}. */
export interface CanvasEngineOptions {
  projectId: string;
  mutationPort: CanvasProjectMutationPort;
  /**
   * Persists encoded engine-owned bitmaps DURABLY. A layer's document points at
   * the resulting image name, so the upload must not be garbage-collectable.
   * Application networking stays outside the core.
   */
  uploadImage(blob: Blob): Promise<{ height: number; imageName: string; width: number }>;
  /**
   * Uploads a TRANSIENT image — one no layer will reference, such as a
   * per-generation composite. Marked intermediate so it is reclaimable instead
   * of accumulating one durable image per invocation forever.
   */
  uploadIntermediateImage(blob: Blob): Promise<{ height: number; imageName: string; width: number }>;
  /** Supplies the currently selected model base for core-created control layer contracts. */
  getMainModelBase?: () => string | null;
  /** Supplies the default control model key for core-created control layer contracts. */
  getDefaultControlModel?: (base: string | null) => string | null;
  /** Reports structured engine failures without exposing the global workbench dispatcher. */
  reportError(report: CanvasEngineErrorReport): void;
  /** Raster surface/bitmap factory. Defaults to the DOM backend. */
  backend?: RasterBackend;
  /** Resolves persisted image assets to blobs for decoding. */
  imageResolver: ImageResolver;
  /**
   * Overrides the paint-persistence store. Defaults to a real {@link createBitmapStore}
   * wired to the layer cache and the upload backend. Tests inject a fake to
   * observe dirty-marking / avoid network uploads.
   */
  bitmapStore?: BitmapStore;
  /**
   * Overrides the web-font readiness api used to re-rasterize text layers once a
   * pending font loads. Defaults to the browser's `document.fonts` (or a no-op
   * in node). Tests inject a fake to drive the load without a real FontFaceSet.
   */
  fonts?: FontLoadApi | null;
  /** Enables deterministic raster/render counters. Disabled by default. */
  enableDiagnostics?: boolean;
}

export interface CanvasEngineSelectionCapability extends CanvasSelectionCapability {}

/**
 * The export surface the engine actually builds. It widens the public
 * {@link CanvasEngineExportCapability} with members that stay inside the Canvas
 * module — raster snapshots, layer-pixel exports, and the composite executor
 * deps are consumed by `canvas-operations`, never by `api.ts` callers.
 *
 * Declared as an `extends` rather than a parallel copy so the two can never
 * silently diverge: anything added to the public capability must be implemented
 * here, and anything added here is visibly internal.
 */
export interface CanvasEngineInternalExportCapability extends CanvasEngineExportCapability {
  captureRasterSnapshot(
    documentSnapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options?: { signal?: AbortSignal; includeDisabled?: boolean }
  ): Promise<CaptureRasterSnapshotResult>;
  exportBakedLayerPixels(
    layerId: string,
    options?: ExportBakedLayerPixelsOptions
  ): Promise<ExportBakedLayerPixelsResult>;
  exportLayerPixels(layerId: string, options?: ExportLayerPixelsOptions): Promise<ExportLayerPixelsResult>;
  getCompositeExecutorDeps(): CanvasCompositeExecutorDeps;
}

/** Private engine composition shape used only inside the Canvas implementation and its tests. */
export interface CanvasEngineImplementation extends CanvasEngine {
  readonly exports: CanvasEngineInternalExportCapability;
  readonly stores: EngineStores;
}

export interface CanvasEngineCoreComposition {
  readonly engine: CanvasEngineImplementation;
  readonly applicationHost: CanvasApplicationHost;
}

const sourceImageName = (source: CanvasLayerSourceContract): string | null => {
  if (source.type === 'image') {
    return source.image.imageName;
  }
  if (source.type === 'paint') {
    return source.bitmap?.imageName ?? null;
  }
  return null;
};

/** The image name a layer's source references, if any (raster/control source or mask bitmap). */
const layerImageName = (layer: CanvasLayerContract): string | null => {
  const source = renderableSourceOf(layer);
  return source ? sourceImageName(source) : null;
};

/** Mints a fresh layer id for engine-created paint layers. */
const createLayerId = (): string => `layer-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
const createEventId = (): string => `event-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const clearSurface = (surface: RasterSurface): void => {
  surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
  surface.ctx.clearRect(0, 0, surface.width, surface.height);
};

/** Creates a per-project canvas engine. */
export const createCanvasEngine = (opts: CanvasEngineOptions): CanvasEngineCoreComposition => {
  const { imageResolver, mutationPort, projectId } = opts;
  const reportError = (message: CanvasEngineErrorReport['message'], layerId: string, error: unknown): void =>
    opts.reportError({
      area: 'canvas-engine',
      context: { error: error instanceof Error ? error.message : String(error), layerId },
      message,
      namespace: 'canvas',
      projectId,
    });
  const backend = opts.backend ?? createDomRasterBackend();
  const diagnostics = createCanvasDiagnostics(opts.enableDiagnostics);

  const viewport = createViewport();
  const rasterController = new RasterController({
    backend,
    diagnostics,
    getDocument: () => mirror.getDocument(),
    getLayerImageName: layerImageName,
    imageResolver,
    onVersionChange: (layerId) => editingController?.invalidateLayer(layerId),
  });
  const layerCache = rasterController.layers;
  const stores = createEngineStores();
  const interactionStores: { [K in keyof CanvasInteractionState]: ScalarStore<CanvasInteractionState[K]> } = {
    activeTool: stores.activeTool,
    bboxGrid: stores.bboxGrid,
    bboxOptions: stores.bboxOptions,
    bboxOverlay: stores.bboxOverlay,
    brushOptions: stores.brushOptions,
    canRedo: stores.canRedo,
    canUndo: stores.canUndo,
    checkerboard: stores.checkerboard,
    checkerColors: stores.checkerColors,
    clipToBbox: stores.clipToBbox,
    documentEditingLocked: stores.documentEditingLocked,
    eraserOptions: stores.eraserOptions,
    gradientOptions: stores.gradientOptions,
    hasFloatingSelection: stores.hasFloatingSelection,
    hasSelection: stores.hasSelection,
    invertBrushSizeScroll: stores.invertBrushSizeScroll,
    lassoOptions: stores.lassoOptions,
    marqueeOptions: stores.marqueeOptions,
    ruleOfThirds: stores.ruleOfThirds,
    shapeOptions: stores.shapeOptions,
    showBbox: stores.showBbox,
    showGrid: stores.showGrid,
    snapToGrid: stores.snapToGrid,
    textEditSession: stores.textEditSession,
    textOptions: stores.textOptions,
    transformSession: stores.transformSession,
    viewportReady: stores.viewportReady,
    zoom: stores.zoom,
  };
  const interaction: CanvasInteractionStateCapability = {
    get: (key) => interactionStores[key].get(),
    getLayerThumbnailStatus: (layerId) => stores.thumbnailStatus.get(layerId) ?? 'idle',
    getLayerThumbnailVersion: (layerId) => stores.thumbnailVersion.get(layerId),
    set: (key, value) => interactionStores[key].set(value),
    subscribe: (key, listener) => interactionStores[key].subscribe(listener),
    subscribeLayerThumbnailStatus: (layerId, listener) => stores.thumbnailStatus.subscribeKey(layerId, listener),
    subscribeLayerThumbnailVersion: (layerId, listener) => stores.thumbnailVersion.subscribeKey(layerId, listener),
  };
  // Web-font readiness for text layers: re-rasterizes a text layer once its font
  // resolves (a no-op in node / when `document.fonts` is absent). `undefined`
  // opts.fonts falls back to the DOM api; an explicit `null` forces the no-op.
  const fontLoader = createFontLoader(opts.fonts === undefined ? domFontLoadApi() : opts.fonts);

  const tools = new Map<ToolId, Tool>([
    ['view', createViewTool()],
    ['brush', createBrushTool()],
    ['eraser', createEraserTool()],
    ['move', createMoveTool()],
    ['transform', createTransformTool()],
    ['bbox', createBboxTool()],
    ['colorPicker', createColorPickerTool()],
    ['lasso', createLassoTool()],
    ['marquee', createMarqueeTool()],
    ['shape', createShapeTool()],
    ['gradient', createGradientTool()],
    ['text', createTextTool()],
    ['sam', createSamTool()],
  ]);
  let interactionLocked = false;

  // Transient per-layer transform overrides driving the move/transform drag
  // preview (compositor + overlay read at render time; the mirror stays untouched).
  // The move tool sets only x/y; the transform tool sets the full transform.
  const transformOverrides = new Map<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  >();

  const cancelLayerRasterization = (layerId: string): void => rasterController.cancelRasterization(layerId);
  const cancelAllLayerRasterizations = (): void => rasterController.cancelAllRasterization();

  let disposed = false;
  let lifecycleState: 'active' | 'cooling' | 'cool' | 'disposed' = 'active';
  let lifecycleGeneration = 0;
  let cooldownPromise: Promise<'cooled' | 'dirty'> | null = null;

  // The brush/eraser cursor ring, drawn on the overlay (set by the active tool).
  let overlayCursor: OverlayCursor | null = null;

  // The transparency checkerboard pattern tile, built once (lazily) through the
  // raster backend and reused each frame (see `createCheckerboardTile`). It is
  // rebuilt only when the fed checker colors change (theme/color-mode switch),
  // signalled by nulling it in the `checkerColors` subscription below.
  let checkerboardTile: RasterSurface | null = null;
  const getCheckerboardTile = (): RasterSurface => {
    checkerboardTile ??= createCheckerboardTile(backend, stores.checkerColors.get());
    return checkerboardTile;
  };

  // Cached mask fill pattern tiles, keyed by `style:color` (a solid style has no
  // tile → cached as `null`). Built lazily through the backend seam like the
  // checkerboard and reused each frame by the compositor's mask colorize path.
  const maskPatternTiles = new Map<string, RasterSurface | null>();
  const getMaskPatternTile = (style: string, color: string): RasterSurface | null => {
    const key = `${style}:${color}`;
    if (!maskPatternTiles.has(key)) {
      maskPatternTiles.set(
        key,
        createMaskPatternTile(backend, style as Parameters<typeof createMaskPatternTile>[1], color)
      );
    }
    return maskPatternTiles.get(key) ?? null;
  };

  // Memoized adjusted surfaces for raster layers carrying brightness/contrast/
  // saturation/curves — rebuilt only when a layer's cache version or its
  // adjustments change (never per frame). Reused each frame by the compositor.
  const derivedSurfaceCache = rasterController.derived;
  const deleteDerivedSurfaces = (layerId: string): void => rasterController.deleteDerivedSurfaces(layerId);
  const getAdjustedSurface = (layer: CanvasLayerContract, entry: LayerCacheEntry): RasterSurface | null =>
    rasterController.getAdjustedSurface(layer, entry);

  /**
   * Re-reads the live cache sizes into the memory budget. Both caches change
   * outside the budget's knowledge (a rasterize grows one, an eviction shrinks
   * the other), so every allocation decision has to re-sync first or it reserves
   * against a stale total.
   */
  const syncMemoryBaselines = (): void => {
    rasterController.memory.setBaseBytes(layerCache.byteSize());
    rasterController.memory.setDerivedBytes(derivedSurfaceCache.byteSize());
  };

  // Completed-stroke subscribers (persistence P2.2, history P2.3).
  const strokeListeners = new Set<(event: StrokeCommittedEvent) => void>();
  const toolChangeListeners = new Set<(change: { from: string; to: string; temporary: boolean }) => void>();
  let samInputHandler: ((input: SamVisualInput) => void) | null = null;
  let applicationEscapeHandler: ((gestureWasActive: boolean) => boolean) | null = null;

  /**
   * A layer's CURRENT document source (raster/control layers only), or `null`
   * if the layer doesn't exist / isn't a source-bearing layer. Shared by the
   * bitmap store's source-type flush guard and `onLayersChanged`'s self-echo
   * check below — both need the same "what does this id currently point at"
   * lookup. Reads `mirror` by closure; safe because neither caller invokes it
   * before `mirror` is assigned further down.
   */
  const getLayerSourceById = (layerId: string): CanvasLayerSourceContract | null => {
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    // Masks expose their alpha bitmap as a synthetic `paint` source so the bitmap
    // store's source-type/redundant-dispatch guards and the mirror's self-echo
    // check work uniformly across paint layers and masks.
    return layer ? renderableSourceOf(layer) : null;
  };

  const getAuthoritativeLayerSourceById = (layerId: string): CanvasLayerSourceContract | null => {
    const layer = mutationPort.getCanvasState()?.document.layers.find((candidate) => candidate.id === layerId);
    return layer ? renderableSourceOf(layer) : null;
  };

  /**
   * Applies a persisted bitmap ref + offset to a layer's document contract — the
   * bitmap store's single swap-on-success dispatch. Raster/control layers take a
   * `paint` source (`updateCanvasLayerSource`); mask layers take their `mask`
   * bitmap + offset (`updateCanvasLayerConfig`, preserving the fill). The self-echo
   * `lastApplied` name the store records covers both, so a mask flush round-tripping
   * back through the mirror is skipped for re-rasterization exactly like a paint one.
   */
  const dispatchLayerBitmap = (layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }): boolean => {
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!layer) {
      return false;
    }
    if (layer.type === 'raster' || layer.type === 'control') {
      return mutationPort.dispatch(
        {
          id: layerId,
          source: { bitmap, offset, type: 'paint' },
          type: 'updateCanvasLayerSource',
        },
        'system'
      );
    } else if (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance') {
      return mutationPort.dispatch(
        {
          config: { layerType: layer.type, mask: { bitmap, offset } },
          id: layerId,
          type: 'updateCanvasLayerConfig',
        },
        'system'
      );
    }
    return false;
  };

  /**
   * Clears a layer's persisted bitmap, for a layer the paint-cache trim found empty.
   * The counterpart to {@link dispatchLayerBitmap}, routed the same way per layer
   * type. The resulting source is byte-identical to a brand-new layer's, so an
   * emptied layer is indistinguishable from a fresh one downstream.
   */
  const clearLayerBitmap = (layerId: string): boolean => {
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!layer) {
      return false;
    }
    if (layer.type === 'raster' || layer.type === 'control') {
      return mutationPort.dispatch(
        { id: layerId, source: { bitmap: null, type: 'paint' }, type: 'updateCanvasLayerSource' },
        'system'
      );
    } else if (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance') {
      // `patchLayerConfig` shallow-merges, so the mask's `fill` survives.
      return mutationPort.dispatch(
        {
          config: { layerType: layer.type, mask: { bitmap: null, offset: { x: 0, y: 0 } } },
          id: layerId,
          type: 'updateCanvasLayerConfig',
        },
        'system'
      );
    }
    return false;
  };

  /**
   * True while something other than persistence owns or frames a layer's pixels, so
   * the paint-cache trim defers rather than moving the extent underneath it. ANY new
   * session kind that reads a layer's cache rect belongs here — notably the transform
   * session, whose frame and bake are both expressed relative to that rect.
   */
  const isLayerBusyForTrim = (layerId: string): boolean => {
    if (pipeline.isGestureActive()) {
      return true;
    }
    if (stores.transformSession.get()?.layerId === layerId || stores.textEditSession.get()?.layerId === layerId) {
      return true;
    }
    if (floatingSelection.get()?.layerId === layerId) {
      return true;
    }
    if (controlPixelController?.isOpenFor([layerId])) {
      return true;
    }
    const layer = mirror.getDocument()?.layers.find((candidate) => candidate.id === layerId);
    return !!layer && isCurrentRasterizationJob(layer);
  };

  // Paint persistence: debounced PNG encode → SHA-256 dedupe → upload → a single
  // swap-on-success `updateCanvasLayerSource` (paint) / `updateCanvasLayerConfig`
  // (mask). Wired to committed strokes below.
  const bitmapStore: BitmapStore =
    opts.bitmapStore ??
    createBitmapStore({
      dispatch: (action) => mutationPort.dispatch(action, 'system'),
      clearBitmap: (layerId) => clearLayerBitmap(layerId),
      dispatchBitmap: (layerId, bitmap, offset) => dispatchLayerBitmap(layerId, bitmap, offset),
      encodeSurface: (surface) => backend.encodeSurface(surface),
      trimLayerPixels: (layerId) => {
        const result = trimPaintCacheToAlpha({ isLayerBusy: isLayerBusyForTrim, layers: layerCache }, layerId);
        if (result === 'emptied' || result === 'trimmed') {
          // Derived surfaces are keyed on the old extent; both calls are synchronous,
          // so they land before the clear dispatch and no frame sees a mismatch.
          deleteDerivedSurfaces(layerId);
          notifyLayerPainted(layerId);
        }
        return result;
      },
      getAuthoritativeLayerSource: getAuthoritativeLayerSourceById,
      getLayerSource: getLayerSourceById,
      getLayerSurface: (layerId) => {
        const entry = layerCache.get(layerId);
        // Content-sized: skip empty (zero-rect) caches — nothing to persist — and
        // carry the cache's content-rect origin as the paint source offset.
        if (!entry || entry.rect.width <= 0 || entry.rect.height <= 0) {
          return null;
        }
        return { offset: { x: entry.rect.x, y: entry.rect.y }, surface: entry.surface };
      },
      onError: (error, layerId) => reportError('Bitmap persistence failed', layerId, error),
      uploadImage: (blob) => opts.uploadImage(blob),
    });
  const persistenceController = new PersistenceController(bitmapStore);

  // Engine-owned canvas history (paint pixel patches + structural patches).
  // Project-level undo deliberately no longer covers the canvas (Phase 0).
  const historyController = new HistoryController({
    canEdit: () => canEditDocument(),
    canRedoStore: stores.canRedo,
    canUndoStore: stores.canUndo,
    endBurst: () => endNudgeBurst(),
    isGestureActive: () => pipeline.isGestureActive(),
  });
  const history = historyController.history;
  const dispatchCanvasMutation = (
    action: CanvasProjectMutation,
    origin: 'system' | 'user' = history.isApplying() ? 'system' : 'user'
  ): boolean => mutationPort.dispatch(action, origin);
  // Direct pixel writes do not replace the reducer canvas object. Snapshot
  // freshness therefore also binds to this engine-local content epoch.
  let rasterContentEpoch = 0;
  let controlPixelController: ControlPixelController | null = null;
  const cancelOpenControlPixelEdit = (): void => {
    controlPixelController?.cancel();
  };

  const structuralController = new StructuralLayerController({
    canEdit: () => canEditDocument(),
    dispatch: (action) => dispatchCanvasMutation(action),
    getDocument: () => mirror.getDocument(),
    history,
    isGestureActive: () => pipeline.isGestureActive(),
  });
  const endNudgeBurst = (): void => structuralController.endBurst();
  const commitStructural = (label: string, forward: CanvasProjectMutation, inverse: CanvasProjectMutation): boolean =>
    structuralController.commit(label, forward, inverse);
  const nudgeSelectedLayer = (dx: number, dy: number): void => structuralController.nudge(dx, dy);

  /**
   * The pixel-write bridge shared by undo and redo: put the patch's pixels back
   * into the layer's live cache surface, propagate the edit, and re-persist.
   *
   * ## Undo ↔ bitmap-store convergence (P2.2 reviewer note)
   *
   * Undo writes the OLD pixels into the cache and marks the layer dirty while an
   * upload of the NEW pixels may still be in flight. The sequence converges:
   *
   * 1. The in-flight upload finishes and dispatches `updateCanvasLayerSource`
   *    with the NEW bitmap ref. Because the store recorded that name in
   *    `lastApplied` before dispatching, the engine's mirror sees it as a
   *    self-echo ({@link BitmapStore.isSelfEcho}) and does NOT re-rasterize — so
   *    the cache keeps the OLD pixels this undo just wrote (never clobbered).
   *    The contract now *transiently* points at the NEW ref.
   * 2. This `markLayerDirty` schedules a follow-up flush. The bitmap store
   *    serializes per layer, so it runs after the in-flight upload settles. It
   *    encodes the cache (now the OLD pixels), hashes it, and the content-hash
   *    dedupe reuses the OLD pixels' already-uploaded image name — no re-upload.
   * 3. That flush dispatches the OLD ref, moving `lastApplied` back and pointing
   *    the contract at the OLD pixels: cache and contract have re-converged.
   */
  const applyImagePatch: ImagePatchApply = (layerId, rect, pixels) => {
    if (!layerCache.get(layerId)) {
      // The layer's cache is gone (removed/evicted); nothing to restore into.
      return;
    }
    // The patch `rect` is in LAYER-LOCAL coordinates (stable across cache growth).
    // Grow the cache to cover it before writing — an undo/redo whose region falls
    // outside the current (possibly shrunk-since) extent must re-expand the cache
    // rather than write out of bounds. `growToRect` preserves existing pixels.
    const entry = layerCache.growToRect(layerId, rect);
    // Match the paint hot path: write pixels straight into the live cache surface
    // (no re-rasterize from source), translated to the surface's local origin,
    // then bump version/thumbnail and recomposite.
    entry.surface.ctx.putImageData(pixels, rect.x - entry.rect.x, rect.y - entry.rect.y);
    notifyLayerPainted(layerId);
    // Re-persist the restored pixels (converges the contract ref; see above).
    bitmapStore.markLayerDirty(layerId);
  };

  /**
   * REPLACES a layer's whole cache with a fresh content-sized surface holding
   * `pixels` placed at `rect` (layer-local). Unlike {@link applyImagePatch} (which
   * grows + overlays a dirty region into a persistent cache), this swaps the entire
   * cache extent — used by the transform bake's undo/redo, where the pre- and
   * post-bake states occupy DIFFERENT rects (an overlay would leave stale pixels
   * outside the smaller rect). Shields the pixels from the async rasterize pass and
   * re-persists through the normal dirty path.
   */
  const restoreLayerCache = (layerId: string, rect: Rect, pixels: ImageData): void => {
    layerCache.delete(layerId);
    deleteDerivedSurfaces(layerId);
    const entry = layerCache.getOrCreateRect(layerId, rect);
    if (rect.width > 0 && rect.height > 0) {
      entry.surface.ctx.putImageData(pixels, 0, 0);
    }
    entry.stale = false;
    notifyLayerPainted(layerId);
    bitmapStore.markLayerDirty(layerId);
  };

  const createPath2DImpl: CreatePath2D = (d) => (d === undefined ? new Path2D() : new Path2D(d));

  /** Bumps a layer's cache version after a direct paint (pixels stay fresh) and recomposites. */
  const notifyLayerPainted = (layerId: string): void => {
    const entry = layerCache.publishPixels(layerId);
    if (entry) {
      rasterContentEpoch += 1;
      stores.thumbnailVersion.set(layerId, entry.version);
      stores.thumbnailStatus.set(layerId, 'ready');
    }
    if (renderController.previews.hasFilter(layerId)) {
      clearFilterPreview(layerId);
    }
    scheduler.invalidate({ layers: [layerId] });
  };

  /** Invalidates cached pixels and drops only previews tied to that exact cache version. */
  const invalidateLayerCache = (layerId: string): void => {
    cancelLayerRasterization(layerId);
    layerCache.invalidate(layerId);
    stores.thumbnailStatus.delete(layerId);
    if (renderController.previews.hasFilter(layerId)) {
      clearFilterPreview(layerId);
    }
  };

  /**
   * A composed history entry for a stroke that auto-created its paint layer.
   * Undo removes the created layer (its cache is dropped by the mirror, so no
   * pixel restore is needed); redo re-adds the layer, recreates a blank cache,
   * and re-applies the stroke's `after` pixels.
   */
  const { commitOrdinaryStroke } = createStrokeCommit({
    applyImagePatch,
    commitPaintEdit: () => mutationPort.commitEdit({ kind: 'paint' }),
    dispatchCanvasMutation,
    endNudgeBurst,
    history,
    layerCache,
    markLayerDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
    notifyLayerPainted,
    strokeListeners,
  });

  // ---- Selection (transient interaction state) + marching ants ------------
  //
  // The selection lives on the engine, never the reducer, and is not undoable
  // (legacy parity). The lasso tool commits paths through `commitSelection`; the
  // mask clips paint strokes and drives fill/erase. Marching ants animate on the
  // overlay only (never recomposite — Task-22 gate) while a selection exists and
  // the engine is attached.

  let antsPhase = 0;

  const onSelectionChanged = (): void => {
    // Selection state is already authoritative before this notification runs.
    // Keep each derived UI/render notification independent and best-effort so a
    // faulty observer cannot make an applied selection report false failure.
    try {
      stores.hasSelection.set(selection.hasSelection());
    } catch {
      // The scalar store commits before notifying observers.
    }
    try {
      updateAntsAnimation();
    } catch {
      // A later selection mutation/attach transition reconciles animation.
    }
    try {
      scheduler.invalidate({ overlay: true });
    } catch {
      // The next render invalidation will draw the authoritative selection.
    }
  };

  const editingController = new EditingController({
    floatingSelection: {
      applyImagePatch,
      backend,
      canEdit: () => canEditDocument(),
      endBurst: () => endNudgeBurst(),
      getDocument: () => mirror.getDocument(),
      history,
      invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
      onChange: () => stores.hasFloatingSelection.set(floatingSelection.has()),
    },
    getDocument: () => mirror.getDocument(),
    selection: {
      backend,
      createPath2D: createPath2DImpl,
      getDocumentSize: () => {
        const doc = mirror.getDocument();
        return doc ? { height: doc.height, width: doc.width } : null;
      },
      onChange: () => onSelectionChanged(),
    },
    selectionPixels: {
      applyImagePatch,
      backend,
      beginControlEdit: (layerId) => beginControlPixelEdit(layerId),
      canEdit: () => canEditDocument(),
      deleteDerived: deleteDerivedSurfaces,
      endBurst: () => endNudgeBurst(),
      getDocument: () => mirror.getDocument(),
      getFillColor: () => stores.brushOptions.get().color,
      history,
      invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
      isGestureActive: () => pipeline.isGestureActive(),
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
    },
    selectionImage: {
      capturePermit: (owner) => captureDocumentEditPermit(owner),
      decodeImage: (image, options) => rasterController.decodeImage(image, options),
      getDocument: () => mirror.getDocument(),
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: (guard) => isLayerExportGuardCurrent(guard),
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit),
    },
    text: {
      canEdit: () => canEditDocument(),
      commitStructural: (label, forward, inverse) => commitStructural(label, forward, inverse),
      createLayerId,
      getDocument: () => mirror.getDocument(),
      invalidate: (payload) => scheduler.invalidate(payload),
      isGestureActive: () => pipeline.isGestureActive(),
      options: stores.textOptions,
      session: stores.textEditSession,
    },
    transform: {
      backend,
      canEdit: () => canEditDocument(),
      dispatch: (action) => dispatchCanvasMutation(action),
      endBurst: () => endNudgeBurst(),
      getCache: (layerId) => layerCache.get(layerId) ?? null,
      getDocument: () => mirror.getDocument(),
      invalidate: (payload) => scheduler.invalidate(payload),
      isGestureActive: () => pipeline.isGestureActive(),
      pushHistory: (entry) => history.push(entry),
      replaceCache: (layerId, rect, surface) => {
        layerCache.delete(layerId);
        const target = layerCache.getOrCreateRect(layerId, rect);
        target.surface.ctx.drawImage(surface.canvas, 0, 0);
        target.stale = false;
        notifyLayerPainted(layerId);
        bitmapStore.markLayerDirty(layerId);
      },
      restoreCache: restoreLayerCache,
      session: stores.transformSession,
      setOverride: (layerId, transform) => {
        if (transform) {
          transformOverrides.set(layerId, transform);
        } else {
          transformOverrides.delete(layerId);
        }
      },
    },
  });
  const selection = editingController.selection;
  const floatingSelection = editingController.floatingSelection;

  /**
   * The float's render inputs for the current frame: the compositor's
   * layer-local placement, and the document-space matrix the ants ride through
   * so the outline tracks the pixels in flight.
   */

  const antsAnimator: AntsAnimator = createAntsAnimator({
    cancelFrame: (handle) => globalThis.cancelAnimationFrame(handle),
    now: () =>
      typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now(),
    onStep: () => {
      antsPhase += ANTS_STEP_PX;
      // Overlay-only: an ants tick never recomposites the document.
      scheduler.invalidate({ overlay: true });
    },
    requestFrame: (callback) => globalThis.requestAnimationFrame(callback),
  });

  /** Runs the ants loop only while a selection exists AND render targets are bound. */
  function updateAntsAnimation(): void {
    if (!disposed && selection.hasSelection() && renderController.getInputElement()) {
      antsAnimator.start();
    } else {
      antsAnimator.stop();
    }
  }

  /**
   * A one-shot color sample requested from outside the canvas — the color
   * picker's eyedropper button. While one is pending, the color-picker tool
   * hands its next sample here instead of writing the brush color, and the tool
   * that was active beforehand is restored. Cancelled (resolving `null`) by
   * Escape, by any other tool switch, and by teardown, so the promise the
   * caller is awaiting can never dangle.
   */
  let pendingColorSample: { previousToolId: ToolId; resolve: (hex: string | null) => void } | null = null;

  const toolContext: ToolContext = {
    applyTransform: () => applyTransform(),
    backend,
    beginControlPixelEdit: (layerId) => beginControlPixelEdit(layerId),
    beginTransformSession: (layerId) => beginTransformSession(layerId),
    cancelTextEdit: () => cancelTextEdit(),
    cancelTransform: () => cancelTransform(),
    cancelFloatingSelection: () => floatingSelection.cancel(),
    commitFloatingSelection: () => floatingSelection.commit(),
    commitSelection: (commit) => {
      // A new selection supersedes the float's own; land the pixels first so the
      // committed op applies to the document the user can see.
      floatingSelection.commit();
      selection.commit(commit);
    },
    commitStructural: (label, forward, inverse) => commitStructural(label, forward, inverse),
    documentDeltaToLayerLocal: (layerId, delta) => {
      const layer = mirror.getDocument()?.layers.find((candidate) => candidate.id === layerId);
      return layer ? documentDeltaToLocal(layerMatrix(layer.transform), delta) : delta;
    },
    getFloatingSelection: () => floatingSelection.get(),
    isPointInSelection: (point) => selection.containsPoint(point),
    liftFloatingSelection: (layerId) => floatingSelection.lift(layerId),
    setFloatingTransform: (transform) => floatingSelection.setTransform(transform),
    createLayerId,
    createPath2D: createPath2DImpl,
    dispatch: (action) => dispatchCanvasMutation(action),
    emitStrokeCommitted: (event) => commitOrdinaryStroke(event),
    getDocument: () => mirror.getDocument(),
    getSelectionMask: () => selection.mask(),
    getStrokeClipRect: () => {
      // Legacy "clip strokes to bbox". Read at gesture start, so moving the frame
      // mid-stroke cannot change where the stroke already landed.
      const doc = mirror.getDocument();
      return stores.clipToBbox.get() && doc ? { ...doc.bbox } : null;
    },
    invalidate: (payload) => scheduler.invalidate(payload),
    layers: layerCache,
    notifyLayerPainted,
    getSamInteraction: () => stores.samInteraction.get(),
    openTextCreate: (docPoint) => openTextCreate(docPoint),
    openTextEdit: (layerId) => openTextEdit(layerId),
    resolveColorSample: (hex) => {
      if (!pendingColorSample) {
        return false;
      }
      settleColorSample(hex, true);
      return true;
    },
    setLayerTransformOverride: (layerId, override) => {
      if (override) {
        transformOverrides.set(layerId, override);
      } else {
        transformOverrides.delete(layerId);
      }
      scheduler.invalidate({ layers: [layerId], overlay: true });
    },
    setOverlayCursor: (cursor) => {
      overlayCursor = cursor;
    },
    stores,
    updateCursor: () => updateCursor(),
    updateSamInput: (input) => samInputHandler?.(input),
    updateTransformSession: (transform) => updateTransformSession(transform),
    viewport,
  };

  const activeTool = (): Tool | undefined => tools.get(interactionController.getActiveToolId());

  /** Applies a CSS cursor to the input element, guarded for node stubs without `style`. */
  const applyCursorToInput = (cursor: string): void => {
    const style = renderController.getInputElement()?.style;
    if (style) {
      style.cursor = cursor;
    }
  };

  const updateCursor = (): void => {
    const cursor = activeTool()?.cursor?.(toolContext) ?? 'default';
    stores.cursor.set(cursor);
    // The store write alone never changes the pointer; apply to the DOM directly.
    applyCursorToInput(cursor);
  };

  /**
   * Resizes the brush/eraser cursor ring in place when the active tool's size
   * changes without a pointer event (`[`/`]` hotkeys, ctrl+wheel, or the
   * options-bar slider). The ring's radius otherwise stays stale until the next
   * pointermove; here we keep its last-known center and just refresh the radius,
   * then invalidate the overlay so it redraws immediately.
   */
  const refreshBrushCursorRadius = (): void => {
    if (!overlayCursor) {
      return;
    }
    let size: number | null = null;
    if (interactionController.getActiveToolId() === 'brush') {
      size = stores.brushOptions.get().size;
    } else if (interactionController.getActiveToolId() === 'eraser') {
      size = stores.eraserOptions.get().size;
    }
    if (size === null) {
      return;
    }
    overlayCursor = { point: overlayCursor.point, radiusDoc: size / 2 };
    scheduler.invalidate({ overlay: true });
  };

  // ---- Rasterization orchestration ---------------------------------------

  const rasterizeDeps = (doc: CanvasDocumentContractV2, signal?: AbortSignal): RasterizeDeps => ({
    backend,
    bitmapPool: rasterController.bitmaps,
    documentSize: { height: doc.height, width: doc.width },
    resolver: imageResolver,
    signal,
    store: layerCache,
  });

  const {
    captureCurrentLayerExportGuard,
    captureLayerExportGuard,
    hasExportableLayerContent,
    isCurrentRasterizationJob,
    isLayerExportGuardCurrent,
  } = createLayerExportGuards({
    getDocument: () => mirror.getDocument(),
    getDocumentGeneration: () => rasterController.getDocumentGeneration(),
    getRasterizationJob: (layerId) => rasterController.getRasterizationJob(layerId),
    hasCanvasState: () => mutationPort.getCanvasState() !== null,
    isDisposed: () => disposed,
    layerCache,
    projectId,
  });

  const { getOrStartLayerRasterization } = createLayerRasterizer({
    createSurface: (width, height) => backend.createSurface(width, height),
    fontLoader,
    getDocument: () => mirror.getDocument(),
    hasCanvasState: () => mutationPort.getCanvasState() !== null,
    invalidateLayerCache,
    invalidateLayerRender: (layerId) => scheduler.invalidate({ layers: [layerId] }),
    isDisposed: () => disposed,
    jobs: {
      cancel: cancelLayerRasterization,
      finish: (layerId, job) => rasterController.finishRasterizationJob(layerId, job),
      get: (layerId) => rasterController.getRasterizationJob(layerId),
      getDocumentGeneration: () => rasterController.getDocumentGeneration(),
      install: (layerId, job) => rasterController.installRasterizationJob(layerId, job),
    },
    layerCache,
    rasterize: (source, document, scratch, signal) => rasterizeSource(source, rasterizeDeps(document, signal), scratch),
    releaseBitmapIfUnreferenced: (imageName) => rasterController.releaseBitmapIfUnreferenced(imageName),
    reportError,
    thumbnails: {
      setStatus: (layerId, status) => stores.thumbnailStatus.set(layerId, status),
      setVersion: (layerId, version) => stores.thumbnailVersion.set(layerId, version),
    },
    trackPublishedLayerImage: (layer) => rasterController.trackPublishedLayerImage(layer),
  });

  const documentEditOwner = Symbol('canvas-operation-document-edit-owner');
  // Later-defined engine values (mirror, pipeline, prepared-cache helpers) are
  // passed as thunks: the context never invokes them during construction.
  const mutationContext = createCanvasMutationContext({
    commitEdit: (intent) => mutationPort.commitEdit(intent),
    createLayerId,
    dispatch: (action, origin) => dispatchCanvasMutation(action, origin),
    editOwner: documentEditOwner,
    editingLocked: stores.documentEditingLocked,
    endBurst: () => endNudgeBurst(),
    getDocument: () => mirror.getDocument(),
    getReducerDocument: () => mutationPort.getCanvasState()?.document ?? null,
    history,
    installPrepared: (prepared, persist) => installGeneratedPaintCache(prepared, persist),
    isGestureActive: () => pipeline.isGestureActive(),
    isGuardCurrent: (guard) => isLayerExportGuardCurrent(guard),
    preparePixels: (layerId, rect, pixels) => prepareGeneratedPaintCache(layerId, rect, pixels),
    refreshMirror: () => mirror.refresh(),
  });
  const canEditDocument = (owner?: symbol): boolean => mutationContext.canEdit(owner);
  const captureDocumentEditPermit = (owner?: symbol): DocumentEditPermit | null => mutationContext.capturePermit(owner);
  const isDocumentEditPermitCurrent = (permit: DocumentEditPermit): boolean => mutationContext.isPermitCurrent(permit);

  const rasterExportController = new RasterExportController({
    backend,
    captureGuard: captureLayerExportGuard,
    getDocument: () => mirror.getDocument(),
    getOrStartRasterization: getOrStartLayerRasterization,
    isGuardCurrent: isLayerExportGuardCurrent,
    isRasterizing: isCurrentRasterizationJob,
    isSupportedSource: isSupportedExportSource,
    layers: layerCache,
    pin: (layerId) => rasterController.memory.pin(layerId, lifecycleGeneration),
    reserve: (bytes) => {
      syncMemoryBaselines();
      return rasterController.memory.reserve(bytes, { generation: lifecycleGeneration, purpose: 'raster-export' });
    },
  });
  const rasterizeLayerPixels = rasterExportController.rasterize.bind(rasterExportController);
  const exportBakedLayerPixels = rasterExportController.baked.bind(rasterExportController);
  const exportBakedLayerBlob = rasterExportController.blob.bind(rasterExportController);
  type StructuralExportLayerPixelsResult =
    | Extract<ExportLayerPixelsResult, { status: 'ok' }>
    | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };
  const normalizeStructuralExport = async (
    result: Promise<ExportLayerPixelsResult>
  ): Promise<StructuralExportLayerPixelsResult> => {
    const resolved = await result;
    if (resolved.status === 'ok') {
      return resolved;
    }
    return { status: resolved.status === 'aborted' ? 'not-ready' : resolved.status };
  };
  const exportBakedLayerPixelsForStructural = (
    layerId: string,
    options?: ExportBakedLayerPixelsOptions
  ): Promise<StructuralExportLayerPixelsResult> => normalizeStructuralExport(exportBakedLayerPixels(layerId, options));
  const rasterizeLayerPixelsForStructural = (
    layerId: string,
    options?: ExportLayerPixelsOptions
  ): Promise<StructuralExportLayerPixelsResult> => normalizeStructuralExport(rasterizeLayerPixels(layerId, options));
  const rasterizeLayerForThumbnail = async (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV2
  ): Promise<'published' | 'stale' | 'error'> => {
    const result = await getOrStartLayerRasterization(layer, document);
    return result === 'aborted' ? 'stale' : result;
  };

  const cropLayerToBbox = (layerId: string): Promise<CropLayerResult> => layerController.crop.crop(layerId);

  const copyLayerToRaster = (layerId: string): Promise<string | null> => layerController.copy.copyToRaster(layerId);

  /**
   * Rasterizes a single layer on demand and returns its cache surface plus the
   * content rect (layer-local origin/size) those pixels occupy, for the
   * composite-for-generation executor. Rasterize-or-throw: a missing layer, a
   * non-raster/control layer, or an unsupported source throws a descriptive error
   * rather than returning a blank surface, so an invoke can never silently drop a
   * contributing layer. Only invoked for layers the pure planner already selected
   * (enabled image/paint rasters).
   */
  type LayerSurfaceForExportResult =
    | { status: 'ok'; surface: RasterSurface; rect: Rect }
    | { status: 'aborted' | 'not-ready' | 'over-budget' };
  const getLayerSurfaceForExport = async (
    layerId: string,
    signal?: AbortSignal
  ): Promise<LayerSurfaceForExportResult> => {
    const result = await rasterizeLayerPixels(layerId, { signal });
    if (result.status === 'ok') {
      return { rect: result.rect, status: 'ok', surface: result.surface };
    }
    if (result.status === 'over-budget') {
      return { status: 'over-budget' };
    }
    if (result.status === 'aborted') {
      return { status: 'aborted' };
    }
    return { status: 'not-ready' };
  };
  const requireLayerSurfaceForExport = async (layerId: string): Promise<{ surface: RasterSurface; rect: Rect }> => {
    const result = await getLayerSurfaceForExport(layerId);
    if (result.status === 'ok') {
      return result;
    }
    if (result.status === 'over-budget') {
      throw new RasterCompositeOverBudgetError();
    }
    throw new Error(`Cannot rasterize layer ${layerId} for generation: ${result.status}.`);
  };

  const releaseBitmapIfUnreferenced = (imageName: string): void =>
    rasterController.releaseBitmapIfUnreferenced(imageName);

  const dropLayer = (layerId: string): void => {
    // Generation-cancel persistence before the id can be restored by undo/redo.
    // A late upload from the removed incarnation must never target a recreated
    // paint layer with the same id.
    try {
      bitmapStore.discardLayer(layerId);
    } catch {
      // Keep authoritative removal cleanup observer-safe for injected stores.
    }
    rasterController.dropLayer(layerId);
    stores.thumbnailVersion.delete(layerId);
    stores.thumbnailStatus.delete(layerId);
  };

  // ---- Render loop --------------------------------------------------------

  const render = (flags: RenderFlags): void => {
    const screen = renderController.getScreen();
    const overlay = renderController.getOverlay();
    if (!screen || !overlay) {
      return;
    }
    const doc = mirror.getDocument();
    const view = viewport.viewMatrix(viewport.getDpr());

    if (!doc) {
      clearSurface(screen);
      clearSurface(overlay);
      return;
    }

    // The composited document only needs redrawing when pixels, layer order, or
    // the viewport transform changed. An overlay-ONLY invalidation (the common
    // hover case: a cursor-ring move dispatches `{ overlay: true }`) must NOT
    // recomposite: the screen canvas retains its last frame, and the overlay is
    // redrawn on top. Skipping the composite here is the single biggest zoom-lag
    // win — a full composite up-scales every doc-sized layer surface to fill the
    // screen, and that fill-rate grows with zoom, so recompositing on every hover
    // move at high zoom is exactly the reported "laggier the closer you zoom in".
    const samPreview = renderController.previews.getSam();
    // Resolved once: the composite draws the float's pixels, the overlay rides
    // the ants through the matching document-space transform.
    const floatRender = floatingSelectionFrame(floatingSelection.get(), doc);
    if (flags.all || flags.view || flags.layers.size > 0) {
      compositeFrame.draw(screen, doc, view, floatRender, samPreview, flags.damage);
    }

    // The overlay is cheap (a handful of screen-space strokes, independent of
    // zoom and document size) and shares the `view` transform with the composite,
    // so redraw it whenever any frame runs — including overlay-only frames.
    renderOverlay(overlay, overlayFrame.describe(doc, view, floatRender, samPreview));
  };

  const renderController = new RenderController({
    applyCursor: (value) => applyCursorToInput(value),
    clearPreview: () => clearStagedPreview(),
    getInputHandlers: () => ({ ...pipeline, onWheel, reset: () => pipeline.reset() }),
    isEngineDisposed: () => disposed,
    onPageHide: () => onPageHide(),
    onVisibilityChange: () => onVisibilityChange(),
    onWindowBlur: () => onWindowBlur(),
    render,
    setViewportReady: (ready) => stores.viewportReady.set(ready),
    updateAnimation: () => updateAntsAnimation(),
    updateCursor: () => updateCursor(),
  });
  const scheduler: RenderScheduler = renderController.scheduler;

  const compositeFrame = createCompositeFrame({
    backend,
    deleteDerivedSurfaces,
    derivedSurfaceCache,
    diagnostics,
    getAdjustedSurface,
    getCheckerboardTile,
    getMaskPatternTile,
    layerCache,
    memory: rasterController.memory,
    previews: renderController.previews,
    rasterizeLayer: (layer, doc) => void getOrStartLayerRasterization(layer, doc),
    stores,
    syncMemoryBaselines,
    transformOverrides,
    viewport,
  });

  const overlayFrame = createOverlayFrame({
    getActiveToolId: () => interactionController.getActiveToolId(),
    getAntsPhase: () => antsPhase,
    getFloatingSelection: () => floatingSelection.get(),
    getOverlayCursor: () => overlayCursor,
    selection,
    stores,
    transformOverrides,
  });
  // Stay paused until attached: invalidations accumulate but never request a
  // (DOM) frame, keeping the engine node-safe before it has render targets.
  scheduler.pause();

  // ---- Staged generation and filter previews ------------------------------

  const { clearAllFilterPreviews, clearFilterPreview, clearStagedPreview, setGuardedFilterPreview, setStagedPreview } =
    createPreviewPublisher({
      decodeBlob: (blob, dimensions) => rasterController.decodeBlob(blob, dimensions),
      getDocument: () => mirror.getDocument(),
      invalidateAll: () => scheduler.invalidate({ all: true }),
      invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
      isGuardCurrent: (guard) => isLayerExportGuardCurrent(guard),
      previews: renderController.previews,
      resolveImage: imageResolver,
    });

  // ---- Document mirror ----------------------------------------------------

  const mirror: DocumentMirror = createDocumentMirror(mutationPort, {
    // The bbox rectangle/handles are overlay chrome, so a bbox move is normally
    // overlay-only (no recomposite). The one exception: a legacy/progress staged
    // preview is drawn in the COMPOSITE at the current bbox origin, so it must
    // recomposite to follow the bbox. Explicitly placed candidates do not.
    onBboxChanged: () => {
      const staged = renderController.previews.getStaged();
      scheduler.invalidate(staged && !staged.placement ? { all: true } : { overlay: true });
    },
    onDocumentReplaced: () => {
      const cleanup = createCleanupAccumulator();
      cleanup.run(() => editingController.invalidateDocument());
      cleanup.run(() => pipeline.cancelActiveGesture());
      cleanup.run(cancelOpenControlPixelEdit);
      const previousImageNames = rasterController.mirroredImageNames();
      cleanup.run(() => rasterController.invalidateDocument());
      cleanup.run(() => stores.thumbnailStatus.clear());
      // A wholesale document swap — project switch, dims/background change, or a
      // snapshot restore that changes dims — invalidates the pixel history: its
      // entries reference layers/pixels that no longer describe the live document.
      //
      // Cancel any in-flight tool gesture FIRST: a swap mid-drag leaves stale tool
      // state (a bbox `startBbox`, a move drag anchor) whose pointer-up would
      // otherwise commit against the replaced document. Routing through the
      // pipeline clears `gestureActive` and runs the tool's `onPointerCancel`, so
      // the tool drops its own transient state.
      // Defensive: a non-bbox active tool won't have cleared a lingering preview.
      cleanup.run(() => stores.bboxPreview.set(null));
      cleanup.run(() => history.clear());
      cleanup.run(endNudgeBurst);
      // A transform session (which outlives individual gestures) belongs to the
      // outgoing document; tear it down alongside its preview override.
      cleanup.run(() => stores.transformSession.set(null));
      cleanup.run(() => transformOverrides.clear());
      // A text-edit session likewise belongs to the outgoing document; drop it.
      cleanup.run(() => stores.textEditSession.set(null));
      // A staged preview belongs to the outgoing document's bbox/candidates; a
      // wholesale swap (project switch, snapshot restore) invalidates it.
      cleanup.run(clearStagedPreview);
      // Per-layer control-filter previews likewise belong to the outgoing
      // document — a swap can reuse a layer id with different content, so
      // pruning only "missing" ids isn't enough; drop them all.
      cleanup.run(clearAllFilterPreviews);
      // The selection is document-scoped interaction state: a swap drops it (and
      // any in-progress lasso preview), stopping the ants loop via onChange.
      // A float belongs to the outgoing document's layer. Cancel (not commit):
      // the layer it would bake into is about to be replaced.
      cleanup.run(() => floatingSelection.cancel());
      cleanup.run(() => selection.clear());
      cleanup.run(() => stores.lassoPreview.set(null));
      cleanup.run(() => stores.marqueePreview.set(null));
      const doc = mirror.getDocument();
      const present = new Set(doc ? doc.layers.map((layer) => layer.id) : []);
      rasterController.clearMirroredImages();
      rasterController.clearThumbnailKeys();
      for (const layer of doc?.layers ?? []) {
        rasterController.setThumbnailKey(layer.id, getLayerThumbnailDisplayKey(layer));
        const imageName = layerImageName(layer);
        if (imageName) {
          rasterController.setMirroredImage(layer.id, imageName);
        }
      }
      const trackedIds = rasterController.trackedImageIds();
      for (const layerId of trackedIds) {
        if (!present.has(layerId)) {
          cleanup.run(() => dropLayer(layerId));
        } else {
          cleanup.run(() => rasterController.untrackLayerImage(layerId));
        }
      }
      // A wholesale replacement can reuse a layer id with a DIFFERENT source, so
      // a surviving cache entry may hold pixels from the outgoing document.
      // Invalidate EVERY id in the incoming document — not just ids whose
      // reference happened to change — to force a re-rasterize from the new
      // source; a diff can't be trusted across a full swap.
      for (const layerId of present) {
        cleanup.run(() => invalidateLayerCache(layerId));
      }
      for (const imageName of previousImageNames) {
        cleanup.run(() => releaseBitmapIfUnreferenced(imageName));
      }
      // Persistence bookkeeping (the self-echo `lastApplied` map and pending
      // debounced flushes) described the OLD document. Drop it so a reused layer
      // id can't have its next legit persistence dispatch suppressed as a stale
      // self-echo.
      cleanup.run(() => bitmapStore.reset());
      cleanup.run(() => scheduler.invalidate({ all: true }));
      cleanup.throwIfFailed();
    },
    onLayerOrderChanged: () => {
      scheduler.invalidate({ all: true });
    },
    onLayersChanged: (ids, sourceChangedIds) => {
      const cleanup = createCleanupAccumulator();
      const floatLayerId = floatingSelection.get()?.layerId;
      if (floatLayerId && ids.includes(floatLayerId)) {
        // The float's layer was replaced or removed. Drop the float rather than
        // baking pixels into a layer whose content just changed underneath them;
        // `cancel` is a no-op restore when the layer is already gone.
        cleanup.run(() => floatingSelection.cancel());
      }
      if (controlPixelController?.isOpenFor(ids)) {
        cleanup.run(() => pipeline.cancelActiveGesture());
        cleanup.run(cancelOpenControlPixelEdit);
      }
      const doc = mirror.getDocument();
      for (const id of sourceChangedIds) {
        cleanup.run(() => editingController.invalidateLayer(id));
      }
      for (const id of ids) {
        cleanup.run(() => editingController.invalidateLayer(id));
      }
      const sourceChanged = new Set(sourceChangedIds);
      const previousImageNames = new Map(ids.map((id) => [id, rasterController.getMirroredImage(id)]));
      for (const id of ids) {
        const layer = doc?.layers.find((candidate) => candidate.id === id);
        const imageName = layer ? layerImageName(layer) : null;
        if (imageName) {
          rasterController.setMirroredImage(id, imageName);
        } else {
          rasterController.deleteMirroredImage(id);
        }
      }
      // A transform session outlives individual gestures (and any tool switch,
      // including a temp modifier-hold), so it can easily outlive its own layer
      // being deleted out from under it — e.g. deleted via the layers panel
      // while the pointer is elsewhere, or while temp-switched to view/colorPicker.
      // Tear it down (session + preview override) the same way a document
      // replace does, rather than leaving a ghost session/override pointing at
      // a layer id that no longer exists.
      const session = stores.transformSession.get();
      const textSession = stores.textEditSession.get();
      for (const id of ids) {
        const decision = decideLayerChange({
          currentThumbnailKey: rasterController.getThumbnailKey(id),
          currentThumbnailVersion: stores.thumbnailVersion.get(id),
          hasTextEditSession: textSession?.layerId === id,
          hasTransformSession: session?.layerId === id,
          isSelfEcho: () => bitmapStore.isSelfEcho(id, getLayerSourceById(id)),
          layer: doc?.layers.find((candidate) => candidate.id === id),
          previousImageName: previousImageNames.get(id),
          sourceChanged: sourceChanged.has(id),
        });
        if (decision.kind === 'removed') {
          const { releaseImageName } = decision;
          rasterController.deleteThumbnailKey(id);
          cleanup.run(() => dropLayer(id));
          if (releaseImageName) {
            cleanup.run(() => releaseBitmapIfUnreferenced(releaseImageName));
          }
          // A control-filter preview (session + decoded surface) belongs to a
          // specific layer; a layer removed out from under an in-flight or
          // already-decoded preview (delete via the layers panel, or an undo
          // that removes it) must have its preview dropped and its decode
          // token bumped, or a late-resolving decode — or a later undo that
          // restores this same id — would repopulate a stale preview.
          cleanup.run(() => clearFilterPreview(id));
          if (decision.cancelTransformSession) {
            cleanup.run(cancelTransform);
          }
          if (decision.cancelTextEditSession) {
            cleanup.run(cancelTextEdit);
          }
          continue;
        }
        const preview = renderController.previews.getFilter(id);
        if (preview && !isLayerExportGuardCurrent(preview.guard)) {
          cleanup.run(() => clearFilterPreview(id));
        }
        if (decision.kind === 'appearance-only') {
          if (decision.thumbnailDisplay) {
            rasterController.setThumbnailKey(id, decision.thumbnailDisplay.key);
            stores.thumbnailVersion.set(id, decision.thumbnailDisplay.version);
          }
          continue;
        }
        const { releaseImageName } = decision;
        rasterController.setThumbnailKey(id, decision.thumbnailKey);
        cleanup.run(() => rasterController.untrackLayerImage(id));
        if (releaseImageName) {
          cleanup.run(() => releaseBitmapIfUnreferenced(releaseImageName));
        }
        if (decision.invalidateCache) {
          cleanup.run(() => invalidateLayerCache(id));
        }
      }
      cleanup.run(() => scheduler.invalidate({ layers: ids }));
      cleanup.throwIfFailed();
    },
    /**
     * The layers panel is the sole authority on which layer is active, so a
     * panel selection has to retarget the engine's per-layer transient state.
     * Nothing here dispatches — this only reconciles engine-side state.
     *
     * A selection-only document change reuses the `layers` array reference and
     * leaves the bbox equal, so none of the other callbacks fire for it: without
     * this the move-tool outline and the transform frame would keep framing the
     * previously selected layer until some unrelated edit invalidated the overlay.
     */
    onSelectionChanged: (selectedLayerId) => {
      const cleanup = createCleanupAccumulator();
      // A float belongs to exactly one layer; selecting another banks it rather
      // than leaving pixels in flight over a layer the user is no longer on.
      cleanup.run(() => floatingSelection.commit());
      const session = stores.transformSession.get();
      if (session && session.layerId !== selectedLayerId) {
        // The open session belongs to the layer that was just deselected; drop
        // its preview rather than leaving a frame on an inactive layer.
        cleanup.run(() => cancelTransform());
        if (selectedLayerId !== null && interactionController.getActiveToolId() === 'transform') {
          cleanup.run(() => beginTransformSession(selectedLayerId));
        }
      }
      cleanup.run(() => scheduler.invalidate({ overlay: true }));
      cleanup.throwIfFailed();
    },
    onStagingChanged: () => scheduler.invalidate({ overlay: true }),
  });

  for (const layer of mirror.getDocument()?.layers ?? []) {
    rasterController.setThumbnailKey(layer.id, getLayerThumbnailDisplayKey(layer));
    const imageName = layerImageName(layer);
    if (imageName) {
      rasterController.setMirroredImage(layer.id, imageName);
    }
  }

  // A guarded filter preview belongs to one continuous active-project epoch.
  // Switching away invalidates published and in-flight work, so returning to
  // this project cannot resurrect it.
  let projectWasPresent = mutationPort.getCanvasState() !== null;
  const unsubscribeProjectPreviewLifecycle = mutationPort.subscribe(() => {
    const projectIsPresent = mutationPort.getCanvasState() !== null;
    if (projectWasPresent && !projectIsPresent) {
      const cleanup = createCleanupAccumulator();
      cleanup.run(() => editingController.invalidateProject());
      cleanup.run(() => pipeline.cancelActiveGesture());
      cleanup.run(cancelOpenControlPixelEdit);
      cleanup.run(() => rasterController.invalidateDocument());
      cleanup.run(() => stores.thumbnailStatus.clear());
      const ids = new Set<string>(renderController.previews.filterLayerIds());
      for (const layerId of ids) {
        cleanup.run(() => clearFilterPreview(layerId));
      }
      projectWasPresent = false;
      cleanup.throwIfFailed();
      return;
    }
    projectWasPresent = projectIsPresent;
  });

  // ---- Viewport → stores/scheduler ---------------------------------------

  // Set while `resize` drives `setViewportSize` synchronously: the resize path
  // composites in the same task (the anti-strobe fix), so the viewport
  // subscription must NOT also schedule a `{ view: true }` frame — that pending
  // flag would recomposite identical content on the next rAF (a second full
  // composite per ResizeObserver event during a panel-drag resize).
  let suppressViewportInvalidate = false;

  const unsubscribeViewport = viewport.subscribe(() => {
    stores.zoom.set(viewport.getZoom());
    if (!suppressViewportInvalidate) {
      scheduler.invalidate({ view: true });
    }
  });

  // ---- Tool-option / setting stores → overlay + recomposite ---------------
  //
  // A brush/eraser size change must resize the cursor ring even with no pointer
  // event; toggling the checkerboard must recomposite the document.
  const unsubscribeBrushOptions = stores.brushOptions.subscribe(refreshBrushCursorRadius);
  const unsubscribeEraserOptions = stores.eraserOptions.subscribe(refreshBrushCursorRadius);
  const unsubscribeCheckerboard = stores.checkerboard.subscribe(() => scheduler.invalidate({ all: true }));
  // New checker colors (theme/color-mode switch): drop the cached tile so it
  // rebuilds with the fed colors on the next composite, then force a recomposite.
  const unsubscribeCheckerColors = stores.checkerColors.subscribe(() => {
    checkerboardTile = null;
    scheduler.invalidate({ all: true });
  });
  // The grid lives on the (cheap) overlay; toggling it or changing its snap size
  // only needs an overlay redraw, never a recomposite.
  const unsubscribeShowGrid = stores.showGrid.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeBboxGrid = stores.bboxGrid.subscribe(() => {
    if (stores.showGrid.get()) {
      scheduler.invalidate({ overlay: true });
    }
  });
  // The bbox frame, bbox overlay shade, and rule-of-thirds guides live on the
  // (cheap) overlay; toggling any only needs an overlay redraw, never a
  // recomposite. (`snapToGrid` is a pure interaction preference the bbox tool
  // reads on gesture — no render effect.)
  const unsubscribeShowBbox = stores.showBbox.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeBboxOverlay = stores.bboxOverlay.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeRuleOfThirds = stores.ruleOfThirds.subscribe(() => scheduler.invalidate({ overlay: true }));

  // ---- Pointer / wheel / key input ---------------------------------------
  //
  // Normalization, capture, coalescing, temp-tool holds, and gesture cancel live
  // in the pointer pipeline; wheel routing (zoom vs brush-size step) lives in the
  // wheel handler. The engine just supplies seams and wires the DOM listeners.

  /** Steps the active brush/eraser diameter by one notch (ctrl+wheel or the `[`/`]` hotkeys). */
  const stepActiveBrushSize = (direction: 1 | -1): void => {
    if (interactionController.getActiveToolId() === 'brush') {
      const opts = stores.brushOptions.get();
      stores.brushOptions.set({ ...opts, size: stepBrushSize(opts.size, direction) });
    } else if (interactionController.getActiveToolId() === 'eraser') {
      const opts = stores.eraserOptions.get();
      stores.eraserOptions.set({ ...opts, size: stepBrushSize(opts.size, direction) });
    }
  };

  const interactionController = new InteractionController({
    beforeSwitch: (from, to, switchOptions) => {
      // A REAL tool switch banks a float — the pixels are already cut, and
      // carrying them into an unrelated tool would strand them. A temporary
      // modifier-hold switch (space → view to pan) must not: the user has not
      // finished the move.
      if (!switchOptions?.temporary) {
        floatingSelection.commit();
      }
      for (const listener of toolChangeListeners) {
        listener({ from, temporary: switchOptions?.temporary === true, to });
      }
    },
    getTool: (toolId) => tools.get(toolId),
    getToolContext: () => toolContext,
    invalidateOverlay: () => scheduler.invalidate({ overlay: true }),
    isLocked: () => interactionLocked,
    publishActiveTool: (toolId) => stores.activeTool.set(toolId),
    stepBrushSize: stepActiveBrushSize,
    updateCursor,
  });
  /**
   * Settles a pending one-shot color sample. `restoreTool` puts the user back
   * on whatever they were holding before reaching for the eyedropper; callers
   * that are themselves switching tools skip it, so the switch they asked for
   * wins rather than being immediately undone.
   */
  const settleColorSample = (hex: string | null, restoreTool: boolean): void => {
    const pending = pendingColorSample;
    if (!pending) {
      return;
    }
    pendingColorSample = null;
    pending.resolve(hex);
    if (restoreTool) {
      interactionController.setTool(pending.previousToolId);
    }
  };

  const setTool = (toolId: ToolId, options?: { temporary?: boolean }): void => {
    // Any move off the eyedropper abandons the sample the caller is awaiting.
    if (toolId !== 'colorPicker') {
      settleColorSample(null, false);
    }
    interactionController.setTool(toolId, options);
  };

  /**
   * Arms the eyedropper for one sample and resolves with the picked `#rrggbb`,
   * or `null` if the user cancels. Backs the color picker's eyedropper button,
   * which reads the composited document rather than the screen (and so works
   * outside Chromium, and sees through the window chrome).
   */
  const requestColorSample = (): Promise<string | null> => {
    // A second request supersedes the first; the earlier caller gets a cancel.
    settleColorSample(null, false);

    return new Promise<string | null>((resolve) => {
      pendingColorSample = { previousToolId: interactionController.getActiveToolId(), resolve };
      interactionController.setTool('colorPicker');
    });
  };

  /**
   * The engine's Escape priority ladder, run by the pointer pipeline AFTER it
   * cancels any in-flight gesture, matching the planned chain "gesture → text
   * session → transform → deselect": cancel an open text-edit session, else an
   * open transform session, else deselect. A focused text portal consumes Escape
   * itself (stopPropagation), so this window-level handler only reaches a
   * defocused-but-open text session. Deselect is suppressed when a drag just
   * consumed the Escape (`gestureWasActive`), so a mid-lasso Escape drops only the
   * in-progress path, never the committed selection. Exposed for the pipeline
   * wiring and node tests (the real DOM keydown listener can't run in node-env).
   */
  const handleEscapePriority = ({ gestureWasActive }: { gestureWasActive: boolean }): void => {
    // An armed eyedropper is the most recent thing the user opted into, so it
    // is the first thing Escape takes back.
    if (pendingColorSample) {
      settleColorSample(null, true);
      return;
    }
    if (stores.textEditSession.get()) {
      cancelTextEdit();
      return;
    }
    if (stores.transformSession.get()) {
      cancelTransform();
      return;
    }
    if (floatingSelection.has()) {
      // Escape ABANDONS a float: the lifted pixels go back where they came from
      // and the selection stays, so the move can simply be redrawn. Committing
      // instead is what Enter / deselect / a tool switch do.
      floatingSelection.cancel();
      return;
    }
    if (applicationEscapeHandler?.(gestureWasActive)) {
      return;
    }
    if (!gestureWasActive && selection.hasSelection()) {
      selection.clear();
    }
  };

  const pipeline: PointerPipeline = createPointerPipeline({
    getActiveTool: activeTool,
    getActiveToolId: () => interactionController.getActiveToolId(),
    getInputElement: () => renderController.getInputElement(),
    getToolContext: () => toolContext,
    handleEscape: handleEscapePriority,
    hasTool: (id) => tools.has(id),
    // A primary-button pointerdown while a text-edit session is open commits it
    // (engine reads the live portal content). The pipeline swallows that press.
    maybeCommitModalSession: () => commitOpenTextSession(),
    setTool: (id, opts) => setTool(id, opts),
    updateCursor,
    viewport,
  });

  const onWheel = createWheelHandler({
    getActiveTool: activeTool,
    getInputElement: () => renderController.getInputElement(),
    getInvertBrushSizeScroll: () => stores.invertBrushSizeScroll.get(),
    getToolContext: () => toolContext,
    invalidate: (payload) => scheduler.invalidate(payload),
    stepActiveBrushSize,
    viewport,
  });

  // ---- Lifecycle: shrink the paint-loss window ---------------------------
  //
  // Unload cannot be reliably blocked, so these are fire-and-forget kicks that
  // narrow the gap between the last paint and its upload; the real barrier is
  // `flushPendingUploads()`, which invoke/export await. `blur` additionally
  // resets the pointer pipeline so a held space/alt temp tool doesn't strand
  // when the window loses focus mid-hold.
  const kickPendingFlush = (): void => {
    void persistenceController.flush();
  };
  const onPageHide = (): void => {
    kickPendingFlush();
  };
  const onVisibilityChange = (): void => {
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
      kickPendingFlush();
    }
  };
  const onWindowBlur = (): void => {
    pipeline.reset();
  };

  const clearSamPreview = (): void => {
    const previous = renderController.previews.clearSam();
    if (previous) {
      scheduler.invalidate(previous.isolated ? { all: true } : { overlay: true });
    }
  };

  const { decodeSelectObjectPreview, prepareSelectObjectStart } = createSelectObjectBridge({
    captureGuard: (layerId) => captureCurrentLayerExportGuard(layerId),
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    getDocument: () => mirror.getDocument(),
    layerCache,
  });

  // ---- Public API ---------------------------------------------------------

  const setInteractionLocked = (locked: boolean): void => {
    if (interactionLocked === locked) {
      return;
    }
    interactionLocked = locked;
    if (locked) {
      pipeline.cancelActiveGesture();
      setTool('view', { temporary: true });
    }
  };

  const attach = (screenCanvas: HTMLCanvasElement, overlayCanvas: HTMLCanvasElement): void =>
    renderController.attach(screenCanvas, overlayCanvas);
  const detach = (): void => renderController.detach();

  const activate = (): void => {
    if (disposed) {
      return;
    }
    rasterController.memory.releaseGeneration(lifecycleGeneration);
    lifecycleGeneration += 1;
    lifecycleState = 'active';
    editingController.activate();
    cooldownPromise = null;
  };

  const beginCooldown = (): Promise<'cooled' | 'dirty'> => {
    if (disposed) {
      return Promise.resolve('cooled');
    }
    if (lifecycleState === 'cooling' && cooldownPromise) {
      return cooldownPromise;
    }
    if (lifecycleState === 'cool') {
      return Promise.resolve('cooled');
    }
    psdExportController.cancel();
    rasterController.memory.releaseGeneration(lifecycleGeneration);
    lifecycleGeneration += 1;
    const generation = lifecycleGeneration;
    lifecycleState = 'cooling';
    editingController.cooldown();
    detach();
    cancelAllLayerRasterizations();
    cooldownPromise = persistenceController.flush().then(
      () => {
        if (disposed || lifecycleState !== 'cooling' || lifecycleGeneration !== generation) {
          return 'cooled';
        }
        layerCache.dispose();
        derivedSurfaceCache.dispose();
        renderController.previews.clearFilters();
        clearStagedPreview();
        checkerboardTile = null;
        maskPatternTiles.clear();
        stores.thumbnailStatus.clear();
        historyController.cooldown();
        lifecycleState = 'cool';
        return 'cooled';
      },
      () => {
        if (!disposed && lifecycleGeneration === generation) {
          // Retain the cooling state and live caches, but clear the completed
          // attempt so a zero-reference registry entry can retry persistence.
          cooldownPromise = null;
        }
        return 'dirty';
      }
    );
    return cooldownPromise;
  };

  const resize = (cssWidth: number, cssHeight: number, dpr: number): void => {
    // Suppress the viewport subscription's `{ view: true }` invalidate: the
    // synchronous `render` below already repaints this size change, so letting the
    // subscription schedule a frame would composite the identical result again on
    // the next rAF (two full composites per resize event).
    suppressViewportInvalidate = true;
    viewport.setViewportSize(cssWidth, cssHeight, dpr);
    suppressViewportInvalidate = false;
    const backingDpr = Math.min(dpr, MAX_DPR);
    const backingWidth = Math.round(cssWidth * backingDpr);
    const backingHeight = Math.round(cssHeight * backingDpr);
    renderController.resize(backingWidth, backingHeight);
    // Composite SYNCHRONOUSLY, in this same task, right after the backing-store
    // resize. Sizing a `<canvas>` backing store clears it, so deferring the
    // recomposite to the next rAF (the normal dirty-path) leaves a blank frame
    // on screen until then — during a continuous panel-drag resize that reads as
    // a flash/strobe. A same-task repaint lands before the browser paints, so the
    // canvas never shows empty. `all: true` forces the composite through the T22
    // dirty gate; `render` no-ops when detached (no surfaces).
    render({ all: true, damage: null, layers: new Set<string>(), overlay: true, view: true });
  };

  const fitToView = (): void => {
    const doc = mirror.getDocument();
    if (!doc) {
      return;
    }
    // The document rect is no longer a spatial boundary — fit content ∪ bbox. The
    // bbox (generation frame) is the primary anchor, so an empty canvas fits it;
    // any renderable layer beyond the bbox is unioned in so it lands in view.
    let bounds: Rect = { ...doc.bbox };
    for (const layer of doc.layers) {
      if (isRenderableLayer(layer)) {
        bounds = union(bounds, getSourceBounds(layer, doc));
      }
    }
    viewport.fitToView(bounds, viewport.getViewportSize());
  };

  const isLayerCacheReadyForOp = (layer: CanvasLayerContract, doc: CanvasDocumentContractV2): boolean => {
    if (isEmpty(getSourceContentRect(layer, doc))) {
      return true;
    }
    const entry = layerCache.get(layer.id);
    return !!entry && !entry.stale && !isCurrentRasterizationJob(layer);
  };

  const prepareGeneratedPaintCache = (layerId: string, rect: Rect, pixels: RasterSurface) =>
    layerCache.prepareReplacement(layerId, rect, pixels);

  const installGeneratedPaintCache = (
    prepared: ReturnType<LayerCacheStore['prepareReplacement']>,
    persist = true
  ): void => {
    const { layerId } = prepared;
    const target = layerCache.installReplacement(prepared);

    // Allocation and raster drawing happen in prepareGeneratedPaintCache().
    // Once a document mutation has been dispatched and this detached cache has
    // been installed, observer/scheduling/persistence hooks are notifications:
    // none may veto the already-applied document+cache transaction. In normal
    // production code these hooks do not throw; containment protects the
    // transaction from a faulty subscriber or host scheduling implementation.
    const notifyBestEffort = (notify: () => void): void => {
      try {
        notify();
      } catch {
        // The document and cache are already converged. A later render or dirty
        // mark can retry ancillary work without reporting a false failed commit.
      }
    };
    notifyBestEffort(() => deleteDerivedSurfaces(layerId));
    notifyBestEffort(() => stores.thumbnailVersion.set(layerId, target.version));
    if (renderController.previews.hasFilter(layerId)) {
      notifyBestEffort(() => clearFilterPreview(layerId));
    }
    notifyBestEffort(() => scheduler.invalidate({ layers: [layerId] }));
    if (persist) {
      notifyBestEffort(() => bitmapStore.markLayerDirty(layerId));
    }
  };

  const getReducerDocument = (): CanvasDocumentContractV2 | null => mutationPort.getCanvasState()?.document ?? null;
  const getMainModelBase = (): string | null => {
    return opts.getMainModelBase?.() ?? null;
  };
  const getDefaultControlModel = (base: string | null): string | null => {
    return opts.getDefaultControlModel?.(base) ?? null;
  };

  const dispatchPreparedMutation = mutationContext.dispatchPrepared;

  /** Conversion reducers clone contracts, so their publication postcondition compares by value. */
  const documentHasLayerContract = (
    document: CanvasDocumentContractV2 | null,
    expected: CanvasLayerContract
  ): boolean => {
    const current = document?.layers.find((candidate) => candidate.id === expected.id);
    return current !== undefined && areJsonValuesStructurallyEqual(current, expected);
  };

  controlPixelController = new ControlPixelController({
    applyImagePatch,
    backend,
    bitmapStore,
    canEdit: () => canEditDocument(),
    deleteDerived: deleteDerivedSurfaces,
    dispatchReplacement: (layer) =>
      dispatchPreparedMutation(
        { layer, layerId: layer.id, type: 'replaceCanvasLayer' },
        () => documentHasLayerContract(getReducerDocument(), layer),
        () => documentHasLayerContract(mirror.getDocument(), layer)
      ),
    endBurst: () => endNudgeBurst(),
    getActiveProjectId: () => projectId,
    getDocument: () => mirror.getDocument(),
    getTransformSession: () => stores.transformSession.get(),
    history,
    installPrepared: installGeneratedPaintCache,
    invalidate: (layerId, overlay) => scheduler.invalidate({ layers: [layerId], overlay: overlay || undefined }),
    isCacheReady: isLayerCacheReadyForOp,
    isOperationIdle: () => !stores.documentEditingLocked.get(),
    layers: layerCache,
    notifyPainted: notifyLayerPainted,
    preparePixels: prepareGeneratedPaintCache,
    projectId,
    publishStroke: (event) => {
      for (const listener of strokeListeners) {
        listener(event);
      }
    },
    setTransformOverride: (layerId, transform) => {
      if (transform) {
        transformOverrides.set(layerId, transform);
      } else {
        transformOverrides.delete(layerId);
      }
    },
  });
  const beginControlPixelEdit = controlPixelController.begin.bind(controlPixelController);

  const captureLayerCache = (
    layer: CanvasLayerContract,
    doc: CanvasDocumentContractV2
  ): { pixels: RasterSurface; rect: Rect } | null | 'not-ready' => {
    const entry = layerCache.get(layer.id);
    if (!entry || isEmpty(entry.rect)) {
      return null;
    }
    if (isCurrentRasterizationJob(layer) || (entry.stale && !isEmpty(getSourceContentRect(layer, doc)))) {
      return 'not-ready';
    }
    const pixels = backend.createSurface(entry.rect.width, entry.rect.height);
    pixels.ctx.drawImage(entry.surface.canvas, 0, 0);
    return { pixels, rect: { ...entry.rect } };
  };

  const layerNeedsPixelPersistence = (layer: CanvasLayerContract): boolean =>
    renderableSourceOf(layer)?.type === 'paint';

  const layerMutationController = new LayerMutationController({
    canEdit: () => canEditDocument(),
    captureCache: captureLayerCache,
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    dispatchPrepared: dispatchPreparedMutation,
    endBurst: () => endNudgeBurst(),
    getDocument: () => mirror.getDocument(),
    getReducerDocument,
    history,
    installPrepared: installGeneratedPaintCache,
    isGestureActive: () => pipeline.isGestureActive(),
    needsPixelPersistence: layerNeedsPixelPersistence,
    preparePixels: prepareGeneratedPaintCache,
    sameContract: documentHasLayerContract,
  });
  const commitLayerCopy = layerMutationController.copy.bind(layerMutationController);
  const commitLayerConversion = layerMutationController.convert.bind(layerMutationController);

  const replaceSelectionFromImage = editingController.selectionImage.replace.bind(editingController.selectionImage);

  const maskResultController = new MaskResultController({
    canEdit: (owner) => canEditDocument(owner),
    createLayerId,
    dispatchPrepared: dispatchPreparedMutation,
    endBurst: () => endNudgeBurst(),
    getDocument: () => mirror.getDocument(),
    getReducerDocument,
    history,
    isGestureActive: () => pipeline.isGestureActive(),
    isGuardCurrent: isLayerExportGuardCurrent,
  });
  const commitMaskImageResult = maskResultController.commit.bind(maskResultController);

  const filterResultController = new FilterResultController({
    captureCache: captureLayerCache,
    ctx: mutationContext,
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    getDefaultControlModel,
    getMainModelBase,
    needsPixelPersistence: layerNeedsPixelPersistence,
  });
  const commitRasterFilterResult = filterResultController.commit.bind(filterResultController);

  const generatedResultController = new GeneratedResultController({
    captureCache: captureLayerCache,
    clearPreview: clearFilterPreview,
    ctx: mutationContext,
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    getDefaultControlModel,
    getMainModelBase,
    needsPixelPersistence: layerNeedsPixelPersistence,
  });
  const commitGeneratedImageResult = generatedResultController.commit.bind(generatedResultController);

  const stagedResultController = new StagedResultController({
    capturePermit: (owner) => captureDocumentEditPermit(owner),
    createEventId,
    createLayerId,
    dispatchPrepared: dispatchPreparedMutation,
    endBurst: () => endNudgeBurst(),
    getCanvasState: () => mutationPort.getCanvasState(),
    getDocument: () => mirror.getDocument(),
    history,
    isGestureActive: () => pipeline.isGestureActive(),
    isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit),
    now: () => new Date().toISOString(),
  });
  const commitStagedImage = stagedResultController.commit.bind(stagedResultController);

  const booleanMergeRasterLayers = (
    upperLayerId: string,
    operation: BooleanRasterOperation
  ): Promise<BooleanRasterResult> => layerController.booleanMerge.merge(upperLayerId, operation);

  const extractMaskedArea = (maskLayerId: string): Promise<ExtractMaskedAreaResult> =>
    layerController.extractMaskedArea.extract(maskLayerId);

  const mergeLayerDown = (upperLayerId: string): boolean => layerController.merge.mergeDown(upperLayerId);
  const mergeVisibleRasterLayers = (): Promise<MergeVisibleResult> => layerController.merge.mergeVisible();

  const { captureDocumentSnapshot, captureRasterSnapshot, isDocumentSnapshotCurrent, releaseActiveSnapshots } =
    createRasterSnapshotCapture({
      createSurface: (width, height) => backend.createSurface(width, height),
      getCanvasState: () => mutationPort.getCanvasState(),
      getContentEpoch: () => rasterContentEpoch,
      getDocumentGeneration: () => rasterController.getDocumentGeneration(),
      getLifecycleGeneration: () => lifecycleGeneration,
      isDisposed: () => disposed,
      isGuardCurrent: isLayerExportGuardCurrent,
      memory: rasterController.memory,
      rasterizeLayerPixels,
      syncMemoryBaselines,
    });

  const psdExportController = new PsdExportController({
    backend,
    captureDocumentSnapshot,
    captureRasterSnapshot,
    getAvailableBytes: () => {
      syncMemoryBaselines();
      return rasterController.memory.getAvailableBytes();
    },
    isDocumentSnapshotCurrent,
    reserve: (bytes) => {
      syncMemoryBaselines();
      return rasterController.memory.reserve(bytes, { generation: lifecycleGeneration, purpose: 'psd-export' });
    },
  });
  const exportRasterLayersToPsd = (fileName: string): Promise<PsdExportResult> => psdExportController.export(fileName);

  const rasterizeLayer = (layerId: string): boolean => layerController.rasterize.rasterize(layerId);

  // ---- Transform session --------------------------------------------------
  //
  // The transform tool opens a session on one layer (start/live transform in
  // `stores.transformSession`, preview via `transformOverrides`) that outlives
  // individual pointer gestures. Apply commits — a param edit for image layers,
  // a pixel bake for paint layers — as ONE undoable entry; Cancel drops the
  // preview. The transform tool drives begin/update/cancel through the tool
  // context; React (numeric bar + Apply/Cancel buttons) drives the public API.

  const beginTransformSession = (layerId: string): void => editingController.transform.begin(layerId);
  const updateTransformSession = (transform: LayerTransform): void => editingController.transform.update(transform);
  // A framed float is the session — Apply banks it, Cancel abandons it. Only
  // with no float in flight do these reach the layer transform session.
  const cancelTransform = (): void => {
    if (floatingSelection.has()) {
      floatingSelection.cancel();
      return;
    }
    editingController.transform.cancel();
  };
  const applyTransform = (): void => {
    if (floatingSelection.has()) {
      floatingSelection.commit();
      return;
    }
    editingController.transform.apply();
  };

  // ---- Text editing session -----------------------------------------------
  //
  // The text tool opens a session (create or edit) exposed through
  // `stores.textEditSession`; React renders a contenteditable portal over it and
  // drives the commit (blur / mod+enter) — the engine never sees per-keystroke
  // content, so commit takes the final content from React. ONE commit per close:
  // create → `addCanvasLayer` (inverse removes), edit → `updateCanvasLayerSource`
  // (exact inverse). A no-change / empty-create commit dispatches nothing (cancel
  // semantics). The options bar restyles the live session via `updateTextEditStyle`.

  const setTextEditContentReader = (reader: (() => string) | null): void =>
    editingController.text.setContentReader(reader);
  const openTextCreate = (point: Vec2): void => editingController.text.openCreate(point);
  const openTextEdit = (layerId: string): void => editingController.text.openEdit(layerId);
  const updateTextEditStyle = (patch: Partial<TextToolOptions>): void => editingController.text.updateStyle(patch);
  const cancelTextEdit = (): void => editingController.text.cancel();
  const commitTextEdit = (content: string, styleChanges?: Partial<TextToolOptions>): void =>
    editingController.text.commit(content, styleChanges);
  const commitOpenTextSession = (): boolean => editingController.text.commitOpen();

  // ---- Selection public API -----------------------------------------------

  /**
   * The bounded domain selectAll/invert operate over now that the document rect is
   * retired: `content ∪ bbox` — the same union `fitToView` fits. The bbox anchors
   * an empty canvas; any renderable layer beyond it is unioned in. The closest
   * coherent analogue of legacy's bounded canvas for the complement in `invert`.
   */
  // Every selection-level operation banks a live float first: the pixels are
  // already cut, and the op that follows must see the document the user does.
  // (Escape is the one path that abandons instead — see `handleEscapePriority`.)
  const selectAll = (): void => {
    floatingSelection.commit();
    editingController.selectAll();
  };
  const deselect = (): void => {
    floatingSelection.commit();
    editingController.deselect();
  };
  const invertSelection = (): void => {
    floatingSelection.commit();
    editingController.invertSelection();
  };

  const fillSelection = (): void => {
    floatingSelection.commit();
    editingController.selectionPixels.run('fill');
  };
  const eraseSelection = (): void => {
    floatingSelection.commit();
    editingController.selectionPixels.run('erase');
  };

  /**
   * The selection's pixels on the active layer, encoded as a PNG. Reuses the
   * float's own masked-copy step (it never mutates the source — the cut is a
   * separate call), so Copy and a lift always take exactly the same pixels.
   */
  const exportSelectionBlob = (): Promise<Blob | null> => {
    floatingSelection.commit();
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === doc.selectedLayerId);
    const mask = selection.mask();
    const entry = layer ? layerCache.get(layer.id) : undefined;
    if (!doc || !layer || !mask || !entry || isEmpty(entry.rect)) {
      return Promise.resolve(null);
    }
    const lifted = liftSelectedPixels({
      backend,
      cache: { rect: entry.rect, surface: entry.surface },
      layerMatrix: layerMatrix(layer.transform),
      mask,
    });
    return lifted ? backend.encodeSurface(lifted.pixels.surface) : Promise.resolve(null);
  };

  const pasteImage = (pixels: ImageData, center?: Vec2): NewRasterLayerResult => {
    floatingSelection.commit();
    const doc = mirror.getDocument();
    // Centred on the generation frame by default — the part of an unbounded
    // canvas the user is actually composing in.
    const target =
      center ?? (doc ? { x: doc.bbox.x + doc.bbox.width / 2, y: doc.bbox.y + doc.bbox.height / 2 } : undefined);
    return layerController.newRasterLayer.pasteImage(pixels, 'Pasted', 'Paste', target);
  };

  const liftSelectionToLayer = (): NewRasterLayerResult => {
    floatingSelection.commit();
    return layerController.newRasterLayer.liftSelectionToLayer('Selection', 'Layer via copy');
  };

  const clearMask = (layerId: string): boolean => layerController.mask.clear(layerId);
  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    releaseActiveSnapshots();
    rasterController.memory.releaseGeneration(lifecycleGeneration);
    lifecycleGeneration += 1;
    lifecycleState = 'disposed';
    const cleanup = createCleanupAccumulator();
    cleanup.run(() => pipeline.cancelActiveGesture());
    cleanup.run(cancelOpenControlPixelEdit);
    cleanup.run(() => controlPixelController?.dispose());
    cleanup.run(() => filterResultController.dispose());
    cleanup.run(() => generatedResultController.dispose());
    cleanup.run(() => stagedResultController.dispose());
    cleanup.run(() => editingController.dispose());
    cleanup.run(() => layerController.dispose());
    cleanup.run(() => layerMutationController.dispose());
    cleanup.run(() => maskResultController.dispose());
    cleanup.run(() => interactionController.dispose());
    cleanup.run(() => psdExportController.dispose());
    cleanup.run(() => rasterExportController.dispose());
    cleanup.run(cancelAllLayerRasterizations);
    cleanup.run(detach);
    // Drop any open text-edit session (its layer belongs to a document this
    // engine no longer serves).
    cleanup.run(() => stores.textEditSession.set(null));
    // Drop any guarded filter previews outright — the engine is going away, so
    // there's no render loop left to invalidate for them.
    cleanup.run(() => antsAnimator.stop());
    // No render loop left to sample with; release any awaited eyedropper.
    cleanup.run(() => settleColorSample(null, false));
    cleanup.run(() => activeTool()?.onDeactivate?.(toolContext));
    cleanup.run(unsubscribeViewport);
    cleanup.run(unsubscribeBrushOptions);
    cleanup.run(unsubscribeEraserOptions);
    cleanup.run(unsubscribeCheckerboard);
    cleanup.run(unsubscribeCheckerColors);
    cleanup.run(unsubscribeShowGrid);
    cleanup.run(unsubscribeBboxGrid);
    cleanup.run(unsubscribeShowBbox);
    cleanup.run(unsubscribeBboxOverlay);
    cleanup.run(unsubscribeRuleOfThirds);
    cleanup.run(unsubscribeProjectPreviewLifecycle);
    cleanup.run(() => mutationContext.dispose());
    cleanup.run(() => historyController.dispose());
    cleanup.run(() => persistenceController.dispose());
    cleanup.run(() => mirror.dispose());
    cleanup.run(() => renderController.dispose());
    cleanup.run(() => rasterController.dispose());
    cleanup.run(() => stores.thumbnailStatus.clear());
    cleanup.run(() => strokeListeners.clear());
    cleanup.run(() => toolChangeListeners.clear());
    cleanup.run(() => {
      samInputHandler = null;
    });
    cleanup.throwIfFailed();
  };

  const onStrokeCommitted = (listener: (event: StrokeCommittedEvent) => void): (() => void) => {
    strokeListeners.add(listener);
    return () => {
      strokeListeners.delete(listener);
    };
  };

  const clearCaches = async (): Promise<void> => {
    // Flush pending paint-bitmap uploads FIRST: an unflushed stroke lives only in
    // the live `layerCache` until the debounced (1500ms) flush persists it. If we
    // invalidated the cache before flushing, that in-flight stroke would be
    // destroyed — the next composite re-rasterizes from the (older) source.
    await persistenceController.flush();
    const doc = mirror.getDocument();
    // Invalidate (mark stale → re-rasterize) every live layer cache and drop its
    // memoized adjusted surface; the next composite rebuilds them from source.
    for (const layer of doc?.layers ?? []) {
      invalidateLayerCache(layer.id);
      deleteDerivedSurfaces(layer.id);
    }
    // Drop the derived pattern tiles so they rebuild from the current fed colors.
    checkerboardTile = null;
    maskPatternTiles.clear();
    scheduler.invalidate({ all: true });
  };

  const clearHistory = (): void => historyController.clear();

  const logDebugInfo = (): void => {
    const doc = mirror.getDocument();
    // eslint-disable-next-line no-console
    console.info('[canvas-engine] debug info', {
      activeTool: interactionController.getActiveToolId(),
      bbox: doc?.bbox ?? null,
      canRedo: history.canRedo(),
      canUndo: history.canUndo(),
      document: doc ? { height: doc.height, layers: doc.layers.length, width: doc.width } : null,
      hasSelection: selection.hasSelection(),
      projectId,
      selectedLayerId: doc?.selectedLayerId ?? null,
      zoom: viewport.getZoom(),
    });
  };

  /**
   * Whether the canvas context menu may target a layer at all. It never picks a
   * layer by hit-testing — the menu acts on the document's selected layer, since
   * the layers panel is the sole authority on which layer is active. This only
   * suppresses the menu during an in-progress edit: a live paint/drag gesture, or
   * an open transform / text-edit session. Right-click during those belongs to
   * the interaction. Mirrors the mid-gesture guards on merge/nudge/undo above.
   */
  const canTargetLayerFromContextMenu = (): boolean =>
    !pipeline.isGestureActive() &&
    !stores.transformSession.get() &&
    !stores.textEditSession.get() &&
    mirror.getDocument() !== null;

  // A live float holds pixels that no history entry knows about, so replaying an
  // entry over them would write into a layer with a hole in it. Put them back
  // first; the float is not itself undoable until it commits.
  const undo = (): void => {
    floatingSelection.cancel();
    historyController.undo();
  };
  const redo = (): void => {
    floatingSelection.cancel();
    historyController.redo();
  };
  const setBboxGrid = (size: number): void => stores.bboxGrid.set(size > 0 ? size : 1);
  const getViewport = (): Viewport => viewport;
  const getCompositeExecutorDeps = (): CanvasCompositeExecutorDeps => ({
    backend,
    getLayerSurface: requireLayerSurfaceForExport,
    reserve: (bytes) => {
      syncMemoryBaselines();
      return rasterController.memory.reserveOperation(bytes, { purpose: 'invocation-composite' });
    },
    // Generation inputs, not layer pixels: nothing in the document will point at
    // these, so they upload as intermediates rather than durable images.
    uploadImage: (blob) => opts.uploadIntermediateImage(blob),
  });
  const exportRasterComposite = (request: RasterCompositeExportRequest) =>
    exportRasterCompositeWithDeps(request, {
      backend,
      captureSnapshot: (): RasterCompositeExportSnapshot => ({
        contentEpoch: rasterContentEpoch,
        document: mirror.getDocument(),
        documentGeneration: rasterController.getDocumentGeneration(),
        lifecycleGeneration,
      }),
      getLayerSurface: requireLayerSurfaceForExport,
      isSnapshotCurrent: (snapshot) =>
        !disposed &&
        mutationPort.getCanvasState() !== null &&
        snapshot.contentEpoch === rasterContentEpoch &&
        snapshot.document === mirror.getDocument() &&
        snapshot.documentGeneration === rasterController.getDocumentGeneration() &&
        snapshot.lifecycleGeneration === lifecycleGeneration,
      pin: (layerIds) => {
        const leases = layerIds.map((layerId) => rasterController.memory.pinOperation(layerId));
        let released = false;
        return {
          release: () => {
            if (released) {
              return;
            }
            released = true;
            for (const lease of leases) {
              lease.release();
            }
          },
        };
      },
      reserve: (bytes) => {
        syncMemoryBaselines();
        return rasterController.memory.reserveOperation(bytes, { purpose: 'background-snapshot' });
      },
    });
  const surface: CanvasSurfaceCapability = { attach, detach, resize };
  const viewportCapability: CanvasViewportCapability = { fitToView, getViewport, setBboxGrid };
  const historyCapability: CanvasHistoryCapability = { clearHistory, redo, undo };
  const lifecycle: CanvasLifecycleCapability = {
    activate,
    beginCooldown,
    dispose,
    flushPendingUploads: () => persistenceController.flush(),
    getLifecycleState: () => lifecycleState,
  };
  const layerController = new LayerController({
    booleanMerge: {
      backend,
      capturePermit: () => captureDocumentEditPermit(),
      createLayerId,
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      history,
      installPrepared: (prepared) => installGeneratedPaintCache(prepared),
      isCacheReady: isLayerCacheReadyForOp,
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit),
      preparePixels: prepareGeneratedPaintCache,
    },
    crop: {
      backend,
      captureCache: captureLayerCache,
      capturePermit: () => captureDocumentEditPermit(),
      discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled: true }),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      history,
      installPrepared: (prepared) => installGeneratedPaintCache(prepared),
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit),
      isSupportedSource: isSupportedExportSource,
      preparePixels: prepareGeneratedPaintCache,
    },
    copy: {
      capturePermit: () => captureDocumentEditPermit(),
      createLayerId,
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled: true }),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      history,
      installPrepared: (prepared) => installGeneratedPaintCache(prepared),
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit),
      preparePixels: prepareGeneratedPaintCache,
    },
    extractMaskedArea: {
      backend,
      capturePermit: () => captureDocumentEditPermit(),
      createLayerId,
      derived: derivedSurfaceCache,
      diagnostics,
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      exportBaked: (layerId, includeDisabled) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled }),
      getAdjustedSurface,
      getDocument: () => mirror.getDocument(),
      getMaskPattern: getMaskPatternTile,
      getReducerDocument,
      hasExportableContent: hasExportableLayerContent,
      history,
      installPrepared: (prepared) => installGeneratedPaintCache(prepared),
      isCacheReady: isLayerCacheReadyForOp,
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit),
      layers: layerCache,
      preparePixels: prepareGeneratedPaintCache,
      rasterize: (layerId) => rasterizeLayerPixelsForStructural(layerId),
    },
    commitGeneratedImageResult,
    newRasterLayer: {
      backend,
      capturePermit: () => captureDocumentEditPermit(),
      createLayerId,
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      history,
      installPrepared: (prepared) => installGeneratedPaintCache(prepared),
      isGestureActive: () => pipeline.isGestureActive(),
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit),
      layers: layerCache,
      preparePixels: prepareGeneratedPaintCache,
      selection: editingController.selection,
    },
    mask: {
      applyImagePatch,
      canEdit: () => canEditDocument(),
      deleteDerived: deleteDerivedSurfaces,
      discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
      dispatch: (action) => dispatchCanvasMutation(action),
      endBurst: () => endNudgeBurst(),
      getDocument: () => mirror.getDocument(),
      history,
      isCacheReady: isLayerCacheReadyForOp,
      isGestureActive: () => pipeline.isGestureActive(),
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
      restoreCache: restoreLayerCache,
    },
    merge: {
      backend,
      canEdit: () => canEditDocument(),
      ctx: mutationContext,
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId),
      hasExportableContent: hasExportableLayerContent,
      isCacheReady: isLayerCacheReadyForOp,
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
    },
    rasterize: {
      backend,
      canEdit: () => canEditDocument(),
      dispatch: (action) => dispatchCanvasMutation(action),
      endBurst: () => endNudgeBurst(),
      getDocument: () => mirror.getDocument(),
      history,
      isGestureActive: () => pipeline.isGestureActive(),
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
      rasterizeDeps: (document) => rasterizeDeps(document),
    },
    structural: structuralController,
    thumbnail: {
      backend,
      getActiveProjectId: () => projectId,
      getCheckerboard: getCheckerboardTile,
      getDocument: () => mirror.getDocument(),
      getEntry: (layerId) => layerCache.get(layerId),
      getMaskPattern: getMaskPatternTile,
      isDisposed: () => disposed,
      isSupportedSource: isSupportedExportSource,
      pin: (layerId) => rasterController.memory.pin(layerId, lifecycleGeneration),
      projectId,
      rasterize: rasterizeLayerForThumbnail,
      reportError: (layerId, error) => {
        try {
          reportError('Layer thumbnail rasterization failed', layerId, error);
        } catch {
          // Diagnostics must not turn a handled thumbnail failure into a rejection.
        }
      },
      reserve: (bytes) => {
        syncMemoryBaselines();
        return rasterController.memory.reserve(bytes, { generation: lifecycleGeneration, purpose: 'thumbnail' });
      },
      setStatus: (layerId, status) => {
        if (status) {
          stores.thumbnailStatus.set(layerId, status);
        } else {
          stores.thumbnailStatus.delete(layerId);
        }
      },
    },
  });
  const exportCapability: CanvasEngineInternalExportCapability = {
    captureLayerExportGuard: captureCurrentLayerExportGuard,
    captureRasterSnapshot,
    exportBakedLayerBlob,
    exportBakedLayerPixels,
    exportLayerPixels: rasterizeLayerPixels,
    exportRasterComposite,
    exportRasterLayersToPsd,
    extractMaskedArea,
    getCompositeExecutorDeps,
    hasExportableLayerContent,
    isLayerExportGuardCurrent,
  };
  const documentCapability: CanvasDocumentCapability = {
    captureSnapshot: captureDocumentSnapshot,
    getDocument: () => mirror.getDocument(),
  };
  const selectionCapability: CanvasEngineSelectionCapability = {
    deselect,
    eraseSelection,
    fillSelection,
    getSelectionBounds: () => selection.bounds(),
    exportSelectionBlob,
    getSelectionMaskRect: () => selection.mask()?.rect ?? null,
    invertSelection,
    liftSelectionToLayer,
    pasteImage,
    replaceSelectionFromImage,
    selectAll,
  };
  const toolsCapability: CanvasEngineToolCapability = {
    ...interactionController.tools,
    canTargetLayerFromContextMenu,
    handleEscapePriority,
    onStrokeCommitted,
    requestColorSample,
    setInteractionLocked,
  };
  const layersCapability: CanvasEngineLayerCapability = {
    ...layerController.layers,
    applyTransform,
    booleanMergeRasterLayers,
    cancelTextEdit,
    cancelTransform,
    clearMask,
    commitLayerConversion,
    commitLayerCopy,
    commitMaskImageResult,
    commitOpenTextSession,
    commitRasterFilterResult,
    commitStagedImage,
    commitTextEdit,
    copyLayerToRaster,
    cropLayerToBbox,
    mergeLayerDown,
    mergeVisibleRasterLayers,
    nudgeSelectedLayer,
    openTextCreate,
    openTextEdit,
    rasterizeLayer,
    setTextEditContentReader,
    updateTextEditStyle,
    updateTransformSession,
  };
  const previewCapability: CanvasEnginePreviewCapability = {
    ...layerController.previews,
    setGuardedFilterPreview,
    setStagedPreview,
  };
  const diagnosticsCapability: CanvasDiagnosticsCapability = {
    clearCaches,
    getDiagnostics: diagnostics.snapshot,
    logDebugInfo,
  };

  const applicationHost: CanvasApplicationHost = {
    captureGuard: captureCurrentLayerExportGuard,
    clearFilterPreview,
    clearSamPreview,
    commitFilter: (options) => commitRasterFilterResult(options, documentEditOwner),
    commitGenerated: (options) => commitGeneratedImageResult(options, documentEditOwner),
    commitMask: (options) => commitMaskImageResult(options, documentEditOwner),
    decodeSelectObjectPreview,
    encodeSurface: (surface) => backend.encodeSurface(surface, 'image/png'),
    exportBakedLayerBlob: (layerId) => exportBakedLayerBlob(layerId, { includeDisabled: true }),
    exportLayerPixels: rasterizeLayerPixelsForStructural,
    getCompositeExecutorDeps,
    getDocument: () => mirror.getDocument(),
    isGuardCurrent: isLayerExportGuardCurrent,
    isInteractionLocked: () => interactionLocked,
    isSamToolActive: () => interactionController.getActiveToolId() === 'sam',
    prepareSelectObjectStart,
    publishFilterPreview: (layerId, imageName, rect, guard, filterType) =>
      setGuardedFilterPreview(layerId, { filterType, imageName, rect }, guard),
    publishSamPreview: (preview) => {
      const isolationChanged = renderController.previews.getSam()?.isolated !== preview.isolated;
      renderController.previews.setSam(preview);
      scheduler.invalidate(preview.isolated || isolationChanged ? { all: true } : { overlay: true });
      return undefined;
    },
    replaceSelection: (guard, image, rect, signal) =>
      replaceSelectionFromImage(guard, image, rect, signal, documentEditOwner),
    replaceTemporaryRestoreTool: () => pipeline.replaceTemporaryRestoreTool('sam', 'view'),
    selectLayer: (layerId) => {
      if (mirror.getDocument()?.selectedLayerId !== layerId) {
        dispatchCanvasMutation({ id: layerId, type: 'setCanvasSelectedLayer' });
      }
    },
    setSamInputHandler: (handler) => {
      samInputHandler = handler;
    },
    setEscapeHandler: (handler) => {
      applicationEscapeHandler = handler;
    },
    setSamInteraction: (state) => {
      stores.samInteraction.set(state);
      scheduler.invalidate({ overlay: true });
    },
    setSamTool: () => setTool('sam'),
    setViewTool: () => setTool('view'),
    subscribeToolChanges: (listener) => {
      toolChangeListeners.add(listener);
      return () => toolChangeListeners.delete(listener);
    },
  };

  const engine: CanvasEngineImplementation = {
    diagnostics: diagnosticsCapability,
    document: documentCapability,
    edits: editingController.edits,
    exports: exportCapability,
    history: historyCapability,
    interaction,
    lifecycle,
    layers: layersCapability,
    projectId,
    previews: previewCapability,
    selection: selectionCapability,
    stores,
    surface,
    tools: toolsCapability,
    viewport: viewportCapability,
  };
  return { applicationHost, engine };
};
