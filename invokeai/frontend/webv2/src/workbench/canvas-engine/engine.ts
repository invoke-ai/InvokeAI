import type {
  CanvasEditRefusal,
  CanvasHistoryReplayStatus,
  CanvasHistoryCapability,
  CanvasDiagnosticsCapability,
  CanvasEngine,
  CanvasEngineExportCapability,
  CanvasFontCapability,
  CanvasFontReplacementResult,
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
  CopyLayerToRasterResult,
  CropLayerResult,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  ExtractMaskedAreaResult,
  MaskEditResult,
  MergeDownResult,
  MergeVisibleResult,
  NewRasterLayerResult,
  PsdExportResult,
  RasterizeLayerResult,
  StructuralCommitOptions,
  StructuralCommitResult,
} from '@workbench/canvas-engine/capabilities';
import type {
  CanvasCompositeExecutorDeps,
  CaptureRasterSnapshotResult,
  RasterReadLease,
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
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { ColorSampleProviders, ColorSampler } from '@workbench/canvas-engine/render/colorSample';
import type { CanvasFontRuntime, CanvasTextSource, FontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { OverlayCursor } from '@workbench/canvas-engine/render/overlayRenderer';
import type { RenderScheduler } from '@workbench/canvas-engine/render/scheduler';
import type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, RenderFlags, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { PixelEditController } from '@workbench/canvas-engine/controllers/controlPixelController';
import { EditingController } from '@workbench/canvas-engine/controllers/editingController';
import { FilterResultController } from '@workbench/canvas-engine/controllers/filterResultController';
import { GeneratedResultController } from '@workbench/canvas-engine/controllers/generatedResultController';
import { HistoryController } from '@workbench/canvas-engine/controllers/historyController';
import { InteractionController } from '@workbench/canvas-engine/controllers/interactionController';
import { LayerController } from '@workbench/canvas-engine/controllers/layerController';
import {
  type CapturedLayerCache,
  type DuplicateLayerRasterPlan,
  LayerMutationController,
} from '@workbench/canvas-engine/controllers/layerMutationController';
import { MaskResultController } from '@workbench/canvas-engine/controllers/maskResultController';
import {
  createCanvasMutationContext,
  type CanvasMutationContext,
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
  compileDocumentLeaves,
  createDocumentModel,
  type CanvasDocumentModel,
} from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentIndex, getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/document/documentIndex';
import {
  createEngineStores,
  type EngineStores,
  type ScalarStore,
  type TextStylePatch,
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
import { createColorSampler } from '@workbench/canvas-engine/render/colorSample';
import {
  compositeDocument,
  createCheckerboardTile,
  type CompositeOptions,
} from '@workbench/canvas-engine/render/compositor';
import { createFontLoader, domFontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import { hasLayerDisplayEffect } from '@workbench/canvas-engine/render/layerDisplayEffect';
import { createMaskPatternTile } from '@workbench/canvas-engine/render/maskFill';
import { colorLoupePixels, renderOverlay } from '@workbench/canvas-engine/render/overlayRenderer';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createDomRasterBackend, type RasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import { rasterizeSource, type ImageResolver, type RasterizeDeps } from '@workbench/canvas-engine/render/rasterizers';
import { fitThumbnailSize, getLayerThumbnailDisplayKey } from '@workbench/canvas-engine/render/thumbnail';
import { documentDeltaToLocal, liftSelectedPixels } from '@workbench/canvas-engine/selection/floatingSelection';
import { ANTS_STEP_PX, createAntsAnimator, type AntsAnimator } from '@workbench/canvas-engine/selection/marchingAnts';
import { traceMaskOutlinePath } from '@workbench/canvas-engine/selection/maskOutline';
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
import { FULL_DAMAGE } from '@workbench/canvas-engine/types';
import { createViewport, type Viewport } from '@workbench/canvas-engine/viewport';

import type { ImagePatchApply } from './history/imagePatch';
import type { CanvasProjectMutation } from './mutationContracts';
import type { StrokeCommittedEvent, Tool, ToolContext } from './tools/tool';

import { createBitmapStore, type BitmapStore } from './document/bitmapStore';
import { createDocumentMirror, type DocumentMirror } from './document/documentMirror';
import { decideLayerChange } from './document/layerChangeDecision';
import { getSourceBounds, getSourceContentRect, isRenderableLayer, renderableSourceOf } from './document/sources';
import { collectCanvasFontReferences, replaceCanvasFontReferences } from './fontReferences';
import { createLayerExportGuards, isSupportedExportSource } from './layerExportGuards';
import { createLayerRasterizer } from './layerRasterizer';
import { createPreviewPublisher } from './previewPublisher';
import { createRasterSnapshotCapture } from './rasterSnapshotCapture';
import { createCompositeFrame } from './render/compositeFrame';
import { floatingSelectionFrame } from './render/floatingSelectionFrame';
import { createOverlayFrame } from './render/overlayFrame';
import { createSelectObjectBridge } from './selectObjectBridge';
import { createStrokeEdits } from './strokeCommit';
import { createViewTool } from './tools/viewTool';

/**
 * PSD export returns exported, nothing (no raster content), too-large (union exceeds PSD limits), or not-ready (a
 * contributor is decoding; surface feedback).
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
  /** The raw failure; notifications show its message and diagnostics keep its stack. */
  context: { error: unknown; layerId?: string; label?: string };
  message:
    | 'Layer thumbnail rasterization failed'
    | 'Bitmap persistence failed'
    | 'Bitmap persistence suspended'
    | 'Structural edit was refused'
    | 'Structural edit could not be reverted'
    | 'Structural edit could not be mirrored'
    | 'Structural history replay was refused'
    | 'Structural history replay could not be mirrored'
    | 'History replay failed';
  namespace: 'canvas';
  projectId: string;
}

/** Options for {@link createCanvasEngine}. */
export interface CanvasEngineOptions {
  projectId: string;
  mutationPort: CanvasProjectMutationPort;
  /**
   * Uploads engine bitmaps durably because document layer refs must survive garbage collection. Networking remains
   * outside the core.
   */
  uploadImage(blob: Blob): Promise<{ height: number; imageName: string; width: number }>;
  /**
   * Uploads unreferenced composites as transient intermediates so each invocation does not leave a permanent
   * image.
   */
  uploadIntermediateImage(blob: Blob): Promise<{ height: number; imageName: string; width: number }>;
  /** Supplies the currently selected model base for core-created control layer contracts. */
  getMainModelBase?: () => string | null;
  /** Supplies the default control model key for core-created control layer contracts. */
  getDefaultControlModel?: (base: string | null) => string | null;
  /** Supplies the Layers panel's transient multi-selection for grouped moves. */
  getSelectedLayerIds?: () => readonly string[];
  /** Publishes engine history's transient multi-selection changes back to the Layers panel. */
  setSelectedLayerIds?: (primaryId: string | null, selectedIds: readonly string[]) => void;
  /** Reports structured engine failures without exposing the global workbench dispatcher. */
  reportError(report: CanvasEngineErrorReport): void;
  /** Raster surface/bitmap factory. Defaults to the DOM backend. */
  backend?: RasterBackend;
  /** Resolves persisted image assets to blobs for decoding. */
  imageResolver: ImageResolver;
  bitmapStore?: BitmapStore;
  /**
   * Font readiness seam for rerasterizing loaded text. Defaults to `document.fonts`, or a no-op without the DOM;
   * injectable for tests.
   */
  fonts?: FontLoadApi | CanvasFontRuntime | null;
  /** Enables deterministic raster/render counters. Disabled by default. */
  enableDiagnostics?: boolean;
}

export interface CanvasEngineSelectionCapability extends CanvasSelectionCapability {}

/**
 * Extends {@link CanvasEngineExportCapability} with Canvas-internal raster snapshots, pixel exports and executor
 * dependencies. Inheritance keeps public requirements synchronized while internal additions stay out of `api.ts`.
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
      context: { error, layerId },
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
    imageResolver,
    // A raster-image pixel transaction replaces the live cache with pixels that already include adjustments while
    // the contract keeps them until commit; applying them again would double them and snap back on pointer-up.
    isAdjustmentBaked: (layerId) => pixelEditController?.isOpenFor([layerId]) === true,
    isLayerHeld: (layerId) => isLayerHeldBySession(layerId) || bitmapStore.hasPendingWork(layerId),
    onVersionChange: (layerId) => editingController?.invalidateLayer(layerId),
  });
  const layerCache = rasterController.layers;
  const stores = createEngineStores();
  const publishLayerThumbnailVersion = (layerId: string, version: number): void => {
    stores.thumbnailVersion.set(layerId, version);
    stores.layerPixelEpoch.set(stores.layerPixelEpoch.get() + 1);
  };
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
    colorPair: stores.colorPair,
    documentEditingLocked: stores.documentEditingLocked,
    eraserOptions: stores.eraserOptions,
    gradientOptions: stores.gradientOptions,
    hasFloatingSelection: stores.hasFloatingSelection,
    hasSelection: stores.hasSelection,
    historyEpoch: stores.historyEpoch,
    invertBrushSizeScroll: stores.invertBrushSizeScroll,
    lassoOptions: stores.lassoOptions,
    marqueeOptions: stores.marqueeOptions,
    layerPixelEpoch: stores.layerPixelEpoch,
    documentEpoch: stores.documentEpoch,
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
  // Undefined uses browser font readiness; explicit null disables it. Loaded fonts trigger text rerasterization.
  const fontLoader = createFontLoader(opts.fonts === undefined ? domFontLoadApi() : opts.fonts);
  let syncActiveFontSources: () => void = () => undefined;
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

  // Render-time transform previews leave the mirror untouched; move overrides position, transform overrides the
  // full matrix.
  const transformOverrides = new Map<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  >();
  let pixelEditController: PixelEditController | null = null;

  const cancelLayerRasterization = (layerId: string): void => rasterController.cancelRasterization(layerId);
  const cancelAllLayerRasterizations = (): void => rasterController.cancelAllRasterization();

  let disposed = false;
  let lifecycleState: 'active' | 'cooling' | 'cool' | 'disposed' = 'active';
  let lifecycleGeneration = 0;
  let cooldownPromise: Promise<'cooled' | 'dirty'> | null = null;

  // The brush/eraser cursor ring, drawn on the overlay (set by the active tool).
  let overlayCursor: OverlayCursor | null = null;
  /** Whether the color picker wants its loupe; it sits at the pointer and is sampled when the overlay frame draws. */
  let colorLoupeShown = false;
  let colorLoupeSampler: ColorSampler | null = null;

  // Lazy checker tile reused until checker colors change, when the subscription clears it.
  let checkerboardTile: RasterSurface | null = null;
  const getCheckerboardTile = (): RasterSurface => {
    checkerboardTile ??= createCheckerboardTile(backend, stores.checkerColors.get());
    return checkerboardTile;
  };

  // Lazy mask tiles keyed by style/color; solid fills cache null.
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

  // Adjusted raster surfaces rebuild on cache-version or adjustment changes, not each frame.
  const derivedSurfaceCache = rasterController.derived;
  const deleteDerivedSurfaces = (layerId: string): void => rasterController.deleteDerivedSurfaces(layerId);
  const getAdjustedSurface = (layer: CanvasLayerContract, entry: LayerCacheEntry): RasterSurface | null =>
    rasterController.getAdjustedSurface(layer, entry);

  const groupSurfaces = rasterController.groups;
  const getGroupSurface: NonNullable<CompositeOptions['groupSurface']> = (scope, members, matrices, content) =>
    groupSurfaces.get(scope, members, matrices, content);
  /** Picking and the picker's loupe see the same adjusted, derived and grouped pixels as the display. */
  const colorSampleProviders: ColorSampleProviders = {
    adjustedSurface: getAdjustedSurface,
    derivedSurfaces: derivedSurfaceCache,
    groupSurface: getGroupSurface,
  };

  // Completed-stroke subscribers (persistence P2.2, history P2.3).
  const strokeListeners = new Set<(event: StrokeCommittedEvent) => void>();
  const toolChangeListeners = new Set<(change: { from: string; to: string; temporary: boolean }) => void>();
  let samInputHandler: ((input: SamVisualInput) => void) | null = null;
  let applicationEscapeHandler: ((gestureWasActive: boolean) => boolean) | null = null;

  /**
   * Current layer source shared by persistence guards and self-echo checks. The closure is called only after
   * `mirror` initialization.
   */
  const getLayerSourceById = (layerId: string): CanvasLayerSourceContract | null => {
    const doc = mirror.getDocument();
    const layer = getDocumentLayer(doc, layerId);
    // Synthetic paint sources let masks share persistence source guards and self-echo detection.
    return layer ? renderableSourceOf(layer) : null;
  };

  const getAuthoritativeLayerSourceById = (layerId: string): CanvasLayerSourceContract | null => {
    const layer = getDocumentLayer(mutationPort.getCanvasState()?.document, layerId);
    return layer ? renderableSourceOf(layer) : null;
  };

  /**
   * Swap persisted refs/offsets via paint-source actions for raster/control or mask-config actions preserving
   * fill. Self-echo tracking covers both, avoiding redundant rasterization.
   */
  const dispatchLayerBitmap = (layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }): boolean => {
    const doc = mirror.getDocument();
    const layer = getDocumentLayer(doc, layerId);
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
   * Clear empty paint/mask bitmaps through type-specific actions, restoring the same source state as a fresh empty
   * layer.
   */
  const clearLayerBitmap = (layerId: string): boolean => {
    const doc = mirror.getDocument();
    const layer = getDocumentLayer(doc, layerId);
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

  /** Sessions that own or frame a layer's live pixels, which nothing else can reconstruct while they are open. */
  function isLayerHeldBySession(layerId: string): boolean {
    return (
      stores.documentEditingLayerId.get() === layerId ||
      stores.transformSession.get()?.layerId === layerId ||
      stores.textEditSession.get()?.layerId === layerId ||
      floatingSelection.get()?.layerId === layerId ||
      pixelEditController?.isOpenFor([layerId]) === true
    );
  }

  /** Defer trim while a gesture, session or rasterization depends on the cache bounds. */
  const isLayerBusyForTrim = (layerId: string): boolean => {
    if (pipeline.isGestureActive() || isLayerHeldBySession(layerId)) {
      return true;
    }
    const layer = getDocumentLayer(mirror.getDocument(), layerId);
    return !!layer && isCurrentRasterizationJob(layer);
  };

  const bitmapStore: BitmapStore =
    opts.bitmapStore ??
    createBitmapStore({
      dispatch: (action) => mutationPort.dispatch(action, 'system'),
      clearBitmap: (layerId) => clearLayerBitmap(layerId),
      dispatchBitmap: (layerId, bitmap, offset) => dispatchLayerBitmap(layerId, bitmap, offset),
      encodeSurface: (surface) => backend.encodeSurface(surface),
      trimLayerPixels: (layerId) => {
        const result = trimPaintCacheToAlpha(
          {
            isLayerBusy: isLayerBusyForTrim,
            isLayerPinned: (candidate) => rasterController.memory.isPinned(candidate),
            layers: layerCache,
          },
          layerId
        );
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
        const entry = layerCache.peek(layerId);
        if (!entry) {
          return null;
        }
        if (entry.rect.width <= 0 || entry.rect.height <= 0) {
          return 'empty';
        }
        return { offset: { x: entry.rect.x, y: entry.rect.y }, surface: entry.surface };
      },
      onError: (error, layerId, info) =>
        reportError(
          info.willRetry ? 'Bitmap persistence failed' : 'Bitmap persistence suspended',
          layerId,
          // Lead circuit-opening feedback with persistence stopping until a fresh stroke, rather than repeating
          // the underlying error.
          info.willRetry ? error : new Error('Canvas changes are no longer uploading. A new stroke will retry.')
        ),
      uploadImage: (blob) => opts.uploadImage(blob),
    });
  const persistenceController = new PersistenceController(bitmapStore);

  const historyController = new HistoryController({
    // A live float holds pixels no history entry knows about, and a preview holds unrecorded document values:
    // replaying over either would land on state the entry never saw. Put both back first; neither is itself
    // undoable until it commits.
    beforeReplay: () => {
      floatingSelection.cancel();
      structuralController.endPreview();
    },
    canEdit: () => canEditDocument(),
    canRedoStore: stores.canRedo,
    canUndoStore: stores.canUndo,
    isGestureActive: () => pipeline.isGestureActive(),
    reportFailure: (label, error) =>
      opts.reportError({
        area: 'canvas-engine',
        context: { error, label },
        message: 'History replay failed',
        namespace: 'canvas',
        projectId,
      }),
  });
  const history = historyController.history;
  const unsubscribeHistoryEpoch = history.subscribe(() => stores.historyEpoch.set(stores.historyEpoch.get() + 1));
  const dispatchCanvasMutation = (action: CanvasProjectMutation, origin: 'system' | 'user' = 'user'): boolean =>
    mutationPort.dispatch(action, origin);
  // Direct pixel writes do not replace the reducer canvas object. Snapshot
  // freshness therefore also binds to this engine-local direct-pixel epoch.
  let directPixelEpoch = 0;
  const resolveSelectedLayerIds = (document: CanvasDocumentContractV3): readonly string[] => {
    const primaryId = document.selectedLayerId;
    if (!primaryId) {
      return [];
    }
    const requested = new Set(opts.getSelectedLayerIds?.() ?? [primaryId]);
    const reconciled = getDocumentLeaves(document)
      .filter((layer) => requested.has(layer.id))
      .map((layer) => layer.id);
    return requested.has(primaryId) && reconciled.length === requested.size ? reconciled : [primaryId];
  };
  const cancelOpenPixelEdit = (): void => {
    pixelEditController?.cancel();
  };

  let structuralController: StructuralLayerController;

  /**
   * Undo/redo restores cache pixels and marks dirty. A patch lands only on trustworthy pixels: an evicted or stale
   * cache is rasterized first under a read lease, and a layer that is gone moves the step as a no-op. An in-flight
   * newer upload self-echoes without replacing restored pixels; the serialized follow-up flush persists them.
   */
  const applyImagePatch: ImagePatchApply = async (layerId, rect, pixels) => {
    if (!getDocumentLayer(mirror.getDocument(), layerId)) {
      return;
    }
    const prepared = await rasterizeLayerPixels(layerId, { includeDisabled: true });
    if (prepared.status === 'missing') {
      return;
    }
    const resident = layerCache.get(layerId);
    if (prepared.status === 'empty' && resident?.stale) {
      // The source holds no pixels now, so stale ones must not survive around the patch.
      layerCache.delete(layerId);
    } else if (prepared.status !== 'ok' && prepared.status !== 'empty' && (!resident || resident.stale)) {
      throw new Error(`Layer pixels could not be restored (${prepared.status}).`);
    }
    try {
      // Grow to the layer-local patch rect before writing; undo/redo may reach beyond a cache trimmed since capture.
      const entry = layerCache.growToRect(layerId, rect);
      entry.surface.ctx.putImageData(pixels, rect.x - entry.rect.x, rect.y - entry.rect.y);
      entry.stale = false;
    } finally {
      if (prepared.status === 'ok') {
        prepared.release();
      }
    }
    notifyLayerPainted(layerId);
    bitmapStore.markLayerDirty(layerId);
  };

  /**
   * Replaces the whole cache with pixels at a layer-local rect. Transform undo/redo needs exact extent replacement
   * to remove stale pixels outside smaller bounds; shield from rasterization and mark dirty.
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
      directPixelEpoch += 1;
      publishLayerThumbnailVersion(layerId, entry.version);
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

  // Engine-owned selection masks clip strokes and drive fill/erase. Marching ants redraw only the overlay while
  // selection exists and the engine is attached.

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

  const documentEditOwner = Symbol('canvas-operation-document-edit-owner');
  // Later-defined engine values are passed as thunks: the context never
  // invokes them during construction.
  const mutationContext = createCanvasMutationContext({
    commitEdit: (intent) => mutationPort.commitEdit(intent),
    createLayerId,
    projectId,
    dispatch: (action, origin) => dispatchCanvasMutation(action, origin),
    editOwner: documentEditOwner,
    editingLocked: stores.documentEditingLocked,
    getDocument: () => mirror.getDocument(),
    getReducerDocument: () => mutationPort.getCanvasState()?.document ?? null,
    history,
    installPrepared: (prepared, persist) => installGeneratedPaintCache(prepared, persist),
    isGestureActive: () => pipeline.isGestureActive(),
    isGuardCurrent: (guard) => isLayerExportGuardCurrent(guard),
    preparePixels: (layerId, rect, pixels) => prepareGeneratedPaintCache(layerId, rect, pixels),
    refreshMirror: () => mirror.refresh(),
    report: (error, label) =>
      reportError(
        error.outcome === 'reverted-unmirrored'
          ? 'Structural edit could not be mirrored'
          : 'Structural edit could not be reverted',
        label,
        error
      ),
    reserveRaster: (bytes) => rasterController.memory.reserveOperation(bytes, { purpose: 'layer-operation' }),
    subscribeReducer: (listener) => mutationPort.subscribe(listener),
  });
  const editRefusalListeners = new Set<(refusal: CanvasEditRefusal) => void>();
  const reportEditRefusal = (refusal: CanvasEditRefusal): void => {
    for (const listener of editRefusalListeners) {
      try {
        listener(refusal);
      } catch {
        // A faulty notice must not affect the refused edit's cleanup.
      }
    }
  };
  /** Live strokes are admitted before their first pixel; auto-created layers compose into the same undo step. */
  const strokeEdits = createStrokeEdits({
    applyImagePatch,
    commitPaintEdit: () => mutationPort.commitEdit({ kind: 'paint' }),
    ctx: mutationContext,
    layerCache,
    markLayerDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
    notifyLayerPainted,
    reportRefusal: reportEditRefusal,
    strokeListeners,
  });
  const captureInsertionAnchor: CanvasMutationContext['captureInsertionAnchor'] = (stack, aboveId) =>
    mutationContext.captureInsertionAnchor(stack, aboveId);
  const editingController = new EditingController({
    floatingSelection: {
      applyImagePatch,
      backend,
      ctx: mutationContext,
      getDocument: () => mirror.getDocument(),
      invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
      onChange: () => stores.hasFloatingSelection.set(floatingSelection.has()),
      reportRefusal: reportEditRefusal,
      suspendPersistence: (layerId) => bitmapStore.suspendLayer(layerId),
    },
    getDocument: () => mirror.getDocument(),
    history,
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
      beginPixelEdit: (layerId) => beginPixelEdit(layerId),
      canEdit: () => canEditDocument(),
      deleteDerived: deleteDerivedSurfaces,
      getDocument: () => mirror.getDocument(),
      ctx: mutationContext,
      getFillColor: () => stores.brushOptions.get().color,
      invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
      isRasterCacheReady: (layer, document) => isLayerCacheReadyForOp(layer, document),
      isGestureActive: () => pipeline.isGestureActive(),
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
      reportRefusal: reportEditRefusal,
      requestRasterization: (layerId) => scheduleLayerRasterization([layerId]),
    },
    selectionImage: {
      concurrency: mutationContext,
      decodeImage: (image, options) => rasterController.decodeImage(image, options),
      getDocument: () => mirror.getDocument(),
      isGuardCurrent: (guard) => isLayerExportGuardCurrent(guard),
    },
    text: {
      canEdit: () => canEditDocument(),
      captureInsertionAnchor,
      commitStructural: (label, forward, inverse) => commitToolStructural(label, forward, inverse),
      createLayerId,
      getDocument: () => mirror.getDocument(),
      invalidate: (payload) => scheduler.invalidate(payload),
      isGestureActive: () => pipeline.isGestureActive(),
      colors: stores.colorPair,
      options: stores.textOptions,
      session: stores.textEditSession,
    },
    transform: {
      backend,
      canEdit: () => canEditDocument(),
      commitStructural: (label, forward, inverse) => commitStructural(label, forward, inverse),
      ctx: mutationContext,
      getCache: (layerId) => layerCache.get(layerId) ?? null,
      getDocument: () => mirror.getDocument(),
      invalidate: (payload) => scheduler.invalidate(payload),
      isGestureActive: () => pipeline.isGestureActive(),
      reportRefusal: reportEditRefusal,
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

  /** Resolve layer-local float placement and the matching document-space ants transform for the frame. */

  const nowMs = (): number =>
    typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
  const reducedMotionQuery =
    typeof globalThis.matchMedia === 'function' ? globalThis.matchMedia('(prefers-reduced-motion: reduce)') : null;
  // Reduced motion stills the pulse only; the 5fps ants march on, matching the
  // selection ants' pre-existing behavior.
  const samPulseActive = (): boolean =>
    renderController.previews.getSam() !== null && reducedMotionQuery?.matches === false;
  const antsAnimator: AntsAnimator = createAntsAnimator({
    cancelFrame: (handle) => globalThis.cancelAnimationFrame(handle),
    now: nowMs,
    onFrame: () => {
      // The SAM pulse is the one per-frame consumer; ants keep the 200ms step.
      if (samPulseActive()) {
        scheduler.invalidate({ overlay: true });
      }
    },
    onStep: () => {
      antsPhase += ANTS_STEP_PX;
      // Overlay-only: an ants tick never recomposites the document.
      scheduler.invalidate({ overlay: true });
    },
    requestFrame: (callback) => globalThis.requestAnimationFrame(callback),
  });

  // Assumes data dims === rect dims (the decode bridge validates this), so the
  // pixel-grid trace offset by rect origin is exact document space.
  const traceSamOutline = (preview: { data: RasterSurface; rect: Rect }): Path2D | null => {
    try {
      const { data, rect } = preview;
      const pixels = data.ctx.getImageData(0, 0, data.width, data.height);
      const outline = traceMaskOutlinePath(pixels, { x: rect.x, y: rect.y });
      return outline ? createPath2DImpl(outline) : null;
    } catch {
      // getImageData can throw (tainted/OOM); the preview then shows without ants.
      return null;
    }
  };

  /** Runs the ants loop only while a selection exists AND render targets are bound. */
  function updateAntsAnimation(): void {
    const animating = selection.hasSelection() || renderController.previews.getSam() !== null;
    if (!disposed && animating && renderController.getInputElement()) {
      antsAnimator.start();
    } else {
      antsAnimator.stop();
    }
  }

  /**
   * One-shot color claims stash samples during press/drag and settle on release, after structural edits are
   * allowed. Escape, tool change and teardown resolve null.
   */
  let pendingColorSample: {
    previousToolId: ToolId;
    resolve: (hex: string | null) => void;
    sampledHex: string | null;
  } | null = null;
  // Where unclaimed eyedropper samples land while a workbench is attached.
  let colorSampleRouter: ((hex: string) => boolean) | null = null;

  const toolContext: ToolContext = {
    applyTransform: () => applyTransform(),
    backend,
    captureInsertionAnchor,
    beginPixelEdit: (layerId) => beginPixelEdit(layerId, { gesture: true }),
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
      const layer = getDocumentLayer(mirror.getDocument(), layerId);
      return layer ? documentDeltaToLocal(layerMatrix(layer.transform), delta) : delta;
    },
    getFloatingSelection: () => floatingSelection.get(),
    isPointInSelection: (point) => selection.containsPoint(point),
    liftFloatingSelection: (layerId) => floatingSelection.lift(layerId),
    setFloatingTransform: (transform) => floatingSelection.setTransform(transform),
    createLayerId,
    createPath2D: createPath2DImpl,
    dispatch: (action) => dispatchCanvasMutation(action),
    beginStrokeEdit: (initialBytes) => strokeEdits.begin(initialBytes),
    getDocument: () => mirror.getDocument(),
    getSelectedLayerIds: () => {
      const document = mirror.getDocument();
      return document ? resolveSelectedLayerIds(document) : [];
    },
    getSelectionMask: () => selection.mask(),
    getStrokeClipRect: () => {
      // Capture bbox clipping at gesture start so moving the frame cannot alter the active stroke.
      const doc = mirror.getDocument();
      return stores.clipToBbox.get() && doc ? { ...doc.bbox } : null;
    },
    invalidate: (payload) => scheduler.invalidate(payload),
    layers: layerCache,
    notifyLayerPainted,
    // Detached engines produce no frames; render immediately rather than hold work until the next attach.
    scheduleFrame: (task) => {
      if (scheduler.isPaused) {
        task();
        return () => undefined;
      }
      return scheduler.beforeNextFrame(task);
    },
    requestLayerRasterization: (layerId) => scheduleLayerRasterization([layerId]),
    getSamInteraction: () => stores.samInteraction.get(),
    openTextCreate: (docPoint) => openTextCreate(docPoint),
    openTextEdit: (layerId) => openTextEdit(layerId),
    sampleProviders: colorSampleProviders,
    resolveColorSample: (hex) => {
      if (pendingColorSample) {
        pendingColorSample.sampledHex = hex;
        return true;
      }
      return colorSampleRouter?.(hex) ?? false;
    },
    commitColorSample: () => {
      if (pendingColorSample !== null && pendingColorSample.sampledHex !== null) {
        settleColorSample(pendingColorSample.sampledHex, true);
      }
    },
    discardColorSample: () => {
      if (pendingColorSample !== null) {
        pendingColorSample.sampledHex = null;
      }
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
    showColorLoupe: (shown) => {
      colorLoupeShown = shown;
      scheduler.invalidate({ overlay: true });
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
   * A pointer that left the canvas takes its hover chrome along, unless a gesture still owns it; the loupe follows
   * the pipeline's hover point, which leaves with it.
   */
  const onCanvasPointerLeave = (): void => {
    pipeline.onPointerLeave();
    if (!pipeline.isGestureActive()) {
      overlayCursor = null;
      scheduler.invalidate({ overlay: true });
    }
  };
  const resetInput = (): void => {
    pipeline.reset();
    overlayCursor = null;
  };

  /** Size changes without pointer events update the cursor radius at its last center and invalidate the overlay. */
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

  const rasterizeDeps = (doc: CanvasDocumentContractV3, signal?: AbortSignal): RasterizeDeps => ({
    backend,
    bitmapPool: rasterController.bitmaps,
    documentSize: { height: doc.height, width: doc.width },
    resolver: imageResolver,
    resolveFontFamily: fontLoader.resolveFamily,
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
    reportError,
    thumbnails: {
      setStatus: (layerId, status) => stores.thumbnailStatus.set(layerId, status),
      setVersion: publishLayerThumbnailVersion,
    },
  });

  structuralController = new StructuralLayerController({
    ctx: mutationContext,
    getSelectedLayerIds: resolveSelectedLayerIds,
    report: (message, label, error) => reportError(message, label, error),
  });
  const commitStructural = (
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation,
    options?: StructuralCommitOptions
  ): StructuralCommitResult => structuralController.commit(label, forward, inverse, options);
  const replaceAllFontReferences = (
    from: Parameters<CanvasFontCapability['replaceAllReferences']>[0],
    target: Parameters<CanvasFontCapability['replaceAllReferences']>[1]
  ): CanvasFontReplacementResult => {
    const before = mirror.getDocument();
    if (!before) {
      return { status: 'not-ready' };
    }
    const expectedRevision = mutationContext.getEditRevision();
    const summary = replaceCanvasFontReferences(before, from, target);
    if (summary.replacedCount === 0) {
      return { ...summary, status: 'unchanged' };
    }
    const result = commitStructural(
      'Replace missing font',
      { document: summary.document, type: 'replaceCanvasFontReferences' },
      { document: before, type: 'replaceCanvasFontReferences' },
      { expectedRevision }
    );
    return result.status === 'committed' ? { ...summary, status: 'committed' } : result;
  };
  const fonts: CanvasFontCapability = {
    collectReferences: collectCanvasFontReferences,
    ensurePreview: fontLoader.ensurePreview,
    replaceAllReferences: replaceAllFontReferences,
    resolveFamily: fontLoader.resolveFamily,
    subscribe: fontLoader.subscribe,
    waitForReady: fontLoader.waitForReady,
  };
  const nudgeSelectedLayer = (dx: number, dy: number): StructuralCommitResult => structuralController.nudge(dx, dy);
  // Tools commit at pointer-up with nowhere to show a refusal; contention is expected, anything else is logged.
  const commitToolStructural = (
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation
  ): StructuralCommitResult => {
    const result = commitStructural(label, forward, inverse);
    if (result.status !== 'committed' && result.status !== 'busy') {
      reportError('Structural edit was refused', label, result.status);
    }
    return result;
  };
  const canEditDocument = (owner?: symbol): boolean => mutationContext.canEdit(owner);

  const rasterExportController = new RasterExportController({
    backend,
    captureGuard: captureLayerExportGuard,
    getDocument: () => mirror.getDocument(),
    getOrStartRasterization: getOrStartLayerRasterization,
    isGuardCurrent: isLayerExportGuardCurrent,
    isRasterizing: isCurrentRasterizationJob,
    isSupportedSource: isSupportedExportSource,
    layers: layerCache,
    invalidateLayerCache,
    pin: (layerId) => rasterController.memory.pin(layerId),
    reserve: (bytes) =>
      rasterController.memory.reserve(bytes, { generation: lifecycleGeneration, purpose: 'raster-export' }),
    waitForFont: fontLoader.waitForReady,
  });
  const rasterizeLayerPixels = rasterExportController.rasterize.bind(rasterExportController);
  const prepareLayerRasterCache = async (layerId: string) => {
    const result = await rasterizeLayerPixels(layerId, { includeDisabled: true });
    if (result.status !== 'ok') {
      return { status: result.status === 'over-budget' ? ('over-budget' as const) : ('not-ready' as const) };
    }
    const layer = result.guard.layer;
    result.release();
    return { layer, status: 'ready' as const };
  };
  const scheduleLayerRasterization = (layerIds: readonly string[]): void => {
    void (async () => {
      for (const layerId of layerIds) {
        try {
          await prepareLayerRasterCache(layerId);
        } catch {
          // Each target can retry independently through its next paint/fill
          // gesture; one failed source must not starve the rest of the batch.
        }
      }
    })();
  };
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
    document: CanvasDocumentContractV3
  ): Promise<'published' | 'stale' | 'error'> => {
    const result = await getOrStartLayerRasterization(layer, document);
    return result === 'aborted' ? 'stale' : result;
  };

  const cropLayerToBbox = (layerId: string): Promise<CropLayerResult> => layerController.crop.crop(layerId);

  const copyLayerToRaster = (layerId: string): Promise<CopyLayerToRasterResult> =>
    layerController.copy.copyToRaster(layerId);

  /**
   * Leases rasterized layer pixels for a composite export. Missing, ineligible or unsupported layers throw so
   * exports cannot silently omit selected contributors.
   */
  const requireLayerSurfaceForExport = async (layerId: string): Promise<RasterReadLease> => {
    const result = await rasterizeLayerPixels(layerId);
    if (result.status === 'ok') {
      return result;
    }
    if (result.status === 'over-budget') {
      throw new RasterCompositeOverBudgetError();
    }
    throw new Error(`Cannot rasterize layer ${layerId} for export: ${result.status}.`);
  };

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
    const dpr = viewport.getDpr();
    const view = viewport.viewMatrix(dpr);

    if (!doc) {
      clearSurface(screen);
      clearSurface(overlay);
      return;
    }

    // Only pixel, ordering and viewport changes need recomposition. Overlay-only hover retains the screen frame,
    // avoiding layer upscaling whose fill cost grows with zoom.
    const samPreview = renderController.previews.getSam();
    // Resolve float placement once so composite pixels and selection ants agree.
    const floatRender = floatingSelectionFrame(floatingSelection.get(), doc);
    if (flags.all || flags.view || flags.layers.size > 0) {
      compositeFrame.draw(screen, doc, view, floatRender, samPreview, flags.damage);
    }

    // Redraw the overlay every frame in CSS pixels, scaled by the same device-pixel ratio as the composite.
    const cssSize = viewport.getViewportSize();
    const viewportSize = cssSize.width > 0 ? cssSize : { height: overlay.height / dpr, width: overlay.width / dpr };
    renderOverlay(
      overlay,
      overlayFrame.describe(doc, { dpr, view: viewport.viewMatrix(1), viewportSize }, floatRender, samPreview)
    );
  };

  const renderController = new RenderController({
    applyCursor: (value) => applyCursorToInput(value),
    clearPreview: () => clearStagedPreview(),
    // Stable handlers: detach removes exactly the listeners attach added.
    getInputHandlers: () => ({ ...pipeline, onPointerLeave: onCanvasPointerLeave, onWheel, reset: resetInput }),
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
    derivedSurfaceCache,
    diagnostics,
    getAdjustedSurface,
    getGroupSurface,
    getCheckerboardTile,
    getMaskPatternTile,
    layerCache,
    previews: renderController.previews,
    raster: rasterController,
    rasterizeLayer: (layer, doc) => void getOrStartLayerRasterization(layer, doc),
    stores,
    transformOverrides,
    viewport,
  });

  const overlayFrame = createOverlayFrame({
    getActiveToolId: () => interactionController.getActiveToolId(),
    getAntsPhase: () => antsPhase,
    getSamPulseTime: () => (samPulseActive() ? nowMs() : null),
    getFloatingSelection: () => floatingSelection.get(),
    getColorLoupe: (doc) => {
      const screenPoint = colorLoupeShown ? pipeline.hoverPoint() : null;
      if (!screenPoint) {
        return null;
      }
      const point = viewport.screenToDocument(screenPoint);
      colorLoupeSampler ??= createColorSampler(backend);
      const size = colorLoupePixels(viewport.getZoom());
      const area = colorLoupeSampler.sampleArea(doc, layerCache, point, size, colorSampleProviders);
      return area ? { checker: getCheckerboardTile(), color: area.center, pixels: area.pixels, point } : null;
    },
    getOverlayCursor: () => overlayCursor,
    selection,
    stores,
    transformOverrides,
  });
  // Stay paused until attached: invalidations accumulate but never request a
  // (DOM) frame, keeping the engine node-safe before it has render targets.
  scheduler.pause();

  // ---- Staged generation and filter previews ------------------------------

  const previewPublisher = createPreviewPublisher({
    decodeBlob: (blob, dimensions) => rasterController.decodeBlob(blob, dimensions),
    getDocument: () => mirror.getDocument(),
    invalidateAll: () => scheduler.invalidate({ all: true }),
    invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
    isGuardCurrent: (guard) => isLayerExportGuardCurrent(guard),
    previews: renderController.previews,
    resolveImage: imageResolver,
  });
  const {
    clearAllFilterPreviews,
    clearFilterPreview,
    clearStagedPreview,
    clearStagedPreviewCache,
    preloadStagedPreview,
    setGuardedFilterPreview,
    setStagedPreview,
  } = previewPublisher;

  // ---- Document mirror ----------------------------------------------------

  const mirror: DocumentMirror = createDocumentMirror(mutationPort, {
    onDocumentChanged: () => stores.documentEpoch.set(stores.documentEpoch.get() + 1),
    // Bbox changes need only overlay redraw unless a bbox-relative staged preview must move in the composite.
    // Explicitly placed candidates stay fixed.
    onBboxChanged: () => {
      const staged = renderController.previews.getStaged();
      scheduler.invalidate(staged && !staged.placement ? { all: true } : { overlay: true });
    },
    onDocumentReplaced: () => {
      const cleanup = createCleanupAccumulator();
      cleanup.run(() => groupSurfaces.clear());
      cleanup.run(() => editingController.invalidateDocument());
      cleanup.run(() => pipeline.cancelActiveGesture());
      cleanup.run(cancelOpenPixelEdit);
      cleanup.run(() => rasterController.invalidateDocument());
      cleanup.run(() => stores.thumbnailStatus.clear());
      // Whole-document swaps invalidate pixel history. Cancel the pointer gesture first to discard stale tool
      // anchors and clear active-gesture state; also clear any lingering bbox preview.
      cleanup.run(() => stores.bboxPreview.set(null));
      cleanup.run(() => history.clear());
      cleanup.run(() => stores.transformSession.set(null));
      cleanup.run(() => transformOverrides.clear());
      // A text-edit session likewise belongs to the outgoing document; drop it.
      cleanup.run(() => stores.textEditSession.set(null));
      cleanup.run(clearStagedPreview);
      // Per-layer control-filter previews likewise belong to the outgoing
      // document — a swap can reuse a layer id with different content, so
      // pruning only "missing" ids isn't enough; drop them all.
      cleanup.run(clearAllFilterPreviews);
      // Drop document-scoped selection/lasso state and cancel outgoing floats; committing would target a replaced
      // layer.
      cleanup.run(() => floatingSelection.cancel());
      cleanup.run(() => editingController.discardSelection());
      cleanup.run(() => stores.lassoPreview.set(null));
      cleanup.run(() => stores.marqueePreview.set(null));
      const doc = mirror.getDocument();
      const present = new Set(doc ? getDocumentLeaves(doc).map((layer) => layer.id) : []);
      rasterController.clearThumbnailKeys();
      for (const layer of getDocumentLeaves(doc)) {
        rasterController.setThumbnailKey(layer.id, getLayerThumbnailDisplayKey(layer));
      }
      for (const layerId of layerCache.layerIds()) {
        if (!present.has(layerId)) {
          cleanup.run(() => dropLayer(layerId));
        }
      }
      // Invalidate every incoming cache on wholesale replacement: reused ids may still hold outgoing pixels,
      // regardless of reference diffs.
      for (const layerId of present) {
        cleanup.run(() => invalidateLayerCache(layerId));
      }
      // Discard old-document self-echo and pending persistence state before ids are reused.
      cleanup.run(() => bitmapStore.reset());
      cleanup.run(() => scheduler.invalidate({ all: true }));
      cleanup.throwIfFailed();
    },
    onLayerOrderChanged: () => {
      const doc = mirror.getDocument();
      groupSurfaces.prune(doc ? new Set(getDocumentIndex(doc).byId.keys()) : new Set());
      scheduler.invalidate({ all: true });
    },
    onLayersRecomposite: (ids) => {
      scheduler.invalidate({ layers: ids });
    },
    onLayersChanged: (ids, sourceChangedIds, restructured) => {
      const cleanup = createCleanupAccumulator();
      // A group deleted with its leaves reports here, not onLayerOrderChanged; value edits remove no group.
      if (restructured) {
        const doc = mirror.getDocument();
        cleanup.run(() => groupSurfaces.prune(doc ? new Set(getDocumentIndex(doc).byId.keys()) : new Set()));
      }
      const floatLayerId = floatingSelection.get()?.layerId;
      if (floatLayerId && ids.includes(floatLayerId)) {
        // The float's layer was replaced or removed. Drop the float rather than
        // baking pixels into a layer whose content just changed underneath them;
        // `cancel` is a no-op restore when the layer is already gone.
        cleanup.run(() => floatingSelection.cancel());
      }
      if (pixelEditController?.isOpenFor(ids)) {
        cleanup.run(() => pipeline.cancelActiveGesture());
        cleanup.run(cancelOpenPixelEdit);
      }
      const doc = mirror.getDocument();
      // `sourceChangedIds` is a subset of `ids`, so each changed layer invalidates once.
      for (const id of ids) {
        cleanup.run(() => editingController.invalidateLayer(id));
      }
      const sourceChanged = new Set(sourceChangedIds);
      // Remove transform sessions and preview overrides when their layer disappears, even during temporary tool
      // switches.
      const session = stores.transformSession.get();
      const textSession = stores.textEditSession.get();
      for (const id of ids) {
        const decision = decideLayerChange({
          currentThumbnailKey: rasterController.getThumbnailKey(id),
          currentThumbnailVersion: stores.thumbnailVersion.get(id),
          hasTextEditSession: textSession?.layerId === id,
          hasTransformSession: session?.layerId === id,
          isSelfEcho: () => bitmapStore.isSelfEcho(id, getLayerSourceById(id)),
          layer: getDocumentLayer(doc, id) ?? undefined,
          sourceChanged: sourceChanged.has(id),
        });
        if (decision.kind === 'removed') {
          rasterController.deleteThumbnailKey(id);
          cleanup.run(() => dropLayer(id));
          // Removed preview layers must drop decoded pixels and advance decode tokens so late results or restored
          // ids cannot revive stale previews.
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
            publishLayerThumbnailVersion(id, decision.thumbnailDisplay.version);
          }
          continue;
        }
        rasterController.setThumbnailKey(id, decision.thumbnailKey);
        if (decision.invalidateCache) {
          cleanup.run(() => invalidateLayerCache(id));
        }
      }
      cleanup.run(() => scheduler.invalidate({ layers: ids }));
      cleanup.throwIfFailed();
    },
    /**
     * Panel selection is authoritative. Reconcile transient sessions and selection chrome without dispatch;
     * selection-only edits reuse stacks/bbox and trigger no other callback.
     */
    onSelectionChanged: (selectedLayerId) => {
      const cleanup = createCleanupAccumulator();
      // A float belongs to exactly one layer; selecting another banks it rather
      // than leaving pixels in flight over a layer the user is no longer on.
      cleanup.run(() => floatingSelection.commit());
      const session = stores.transformSession.get();
      if (session && session.layerId !== selectedLayerId) {
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

  let fontSourceStacks: CanvasDocumentContractV3['stacks'] | null = null;
  let documentFontSources: CanvasTextSource[] = [];
  let draftFontSource: CanvasTextSource | undefined;
  let defaultsFontOptions: TextToolOptions | undefined;
  let defaultsFontSource: CanvasTextSource | undefined;
  syncActiveFontSources = () => {
    const document = mutationPort.getCanvasState()?.document;
    const stacks = document?.stacks ?? null;
    const draft = document ? stores.textEditSession.get()?.source : undefined;
    // Text defaults also affect pane previews and the next created layer.
    const options = document ? stores.textOptions.get() : undefined;
    if (stacks === fontSourceStacks && draft === draftFontSource && options === defaultsFontOptions) {
      return;
    }
    if (stacks !== fontSourceStacks) {
      fontSourceStacks = stacks;
      documentFontSources = document
        ? getDocumentLeaves(document).flatMap((layer) =>
            (layer.type === 'raster' || layer.type === 'control') && layer.source.type === 'text' ? [layer.source] : []
          )
        : [];
    }
    if (options !== defaultsFontOptions) {
      defaultsFontOptions = options;
      defaultsFontSource = options ? { ...options, color: '#000000', content: '', type: 'text' } : undefined;
    }
    draftFontSource = draft;
    fontLoader.setActiveSources([
      ...documentFontSources,
      ...(draft ? [draft] : []),
      ...(defaultsFontSource ? [defaultsFontSource] : []),
    ]);
  };
  syncActiveFontSources();
  const unsubscribeTextFontSources = stores.textEditSession.subscribe(syncActiveFontSources);
  const unsubscribeTextDefaultsFontSources = stores.textOptions.subscribe(syncActiveFontSources);

  for (const layer of getDocumentLeaves(mirror.getDocument())) {
    rasterController.setThumbnailKey(layer.id, getLayerThumbnailDisplayKey(layer));
  }

  // A guarded filter preview belongs to one continuous active-project epoch.
  // Switching away invalidates published and in-flight work, so returning to
  // this project cannot resurrect it.
  let projectWasPresent = mutationPort.getCanvasState() !== null;
  const unsubscribeProjectPreviewLifecycle = mutationPort.subscribe(() => {
    syncActiveFontSources();
    const projectIsPresent = mutationPort.getCanvasState() !== null;
    if (projectWasPresent && !projectIsPresent) {
      const cleanup = createCleanupAccumulator();
      cleanup.run(() => editingController.invalidateProject());
      cleanup.run(() => pipeline.cancelActiveGesture());
      cleanup.run(cancelOpenPixelEdit);
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

  // During synchronous resize repaint, suppress viewport frame scheduling to avoid a second identical composite
  // next animation frame.
  let suppressViewportInvalidate = false;

  const unsubscribeViewport = viewport.subscribe(() => {
    stores.zoom.set(viewport.getZoom());
    if (!suppressViewportInvalidate) {
      scheduler.invalidate({ view: true });
    }
  });

  // Only a diameter change moves the ring; other option edits (color, opacity) leave the overlay untouched.
  const subscribeSizeChange = (store: ScalarStore<{ size: number }>, toolId: ToolId): (() => void) => {
    let size = store.get().size;
    return store.subscribe(() => {
      const next = store.get().size;
      if (next !== size) {
        size = next;
        if (interactionController.getActiveToolId() === toolId) {
          refreshBrushCursorRadius();
        }
      }
    });
  };
  const unsubscribeBrushOptions = subscribeSizeChange(stores.brushOptions, 'brush');
  const unsubscribeEraserOptions = subscribeSizeChange(stores.eraserOptions, 'eraser');
  const unsubscribeCheckerboard = stores.checkerboard.subscribe(() => scheduler.invalidate({ all: true }));
  const unsubscribeCheckerColors = stores.checkerColors.subscribe(() => {
    checkerboardTile = null;
    scheduler.invalidate({ all: true });
  });
  const unsubscribeShowGrid = stores.showGrid.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeBboxGrid = stores.bboxGrid.subscribe(() => {
    if (stores.showGrid.get()) {
      scheduler.invalidate({ overlay: true });
    }
  });
  // Bbox frame, shade and guides need only overlay redraw. Snap preference has no render effect.
  const unsubscribeShowBbox = stores.showBbox.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeBboxOverlay = stores.bboxOverlay.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeRuleOfThirds = stores.ruleOfThirds.subscribe(() => scheduler.invalidate({ overlay: true }));

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
    cancelGesture: () => pipeline.endForToolSwitch(),
    getTool: (toolId) => tools.get(toolId),
    getToolContext: () => toolContext,
    invalidateOverlay: () => scheduler.invalidate({ overlay: true }),
    isLocked: () => interactionLocked,
    publishActiveTool: (toolId) => stores.activeTool.set(toolId),
    stepBrushSize: stepActiveBrushSize,
    updateCursor,
  });
  /**
   * Settle a pending sample, optionally restoring the previous tool. Tool-switch callers skip restoration so their
   * requested switch wins.
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

  /** Samples the document composite and resolves hex on release, after the gesture, or null on cancellation. */
  const requestColorSample = (): Promise<string | null> => {
    // A second request supersedes the first; the earlier caller gets a cancel.
    settleColorSample(null, false);
    // End any held temporary tool first, so the request records the tool the user was really on.
    pipeline.endForToolSwitch();

    return new Promise<string | null>((resolve) => {
      pendingColorSample = { previousToolId: interactionController.getActiveToolId(), resolve, sampledHex: null };
      interactionController.setTool('colorPicker');
    });
  };

  /**
   * After gesture cancellation, Escape cancels defocused text editing, then transform, then selection. Focused
   * portals handle Escape themselves. If a gesture consumed Escape, preserve committed selection. Shared with
   * pipeline wiring and node tests.
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
      // Escape restores lifted pixels while retaining selection; Enter, deselect and tool switches commit the
      // float.
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
    getKeyboardRoot: () => renderController.getKeyboardRoot(),
    getToolContext: () => toolContext,
    handleEscape: handleEscapePriority,
    hasTool: (id) => tools.has(id),
    isReplaying: () => history.isReplaying(),
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

  // Lifecycle flushes reduce paint-loss exposure but cannot reliably block unload; invoke/export await the actual
  // barrier. Blur also resets held temporary tools.
  const kickPendingFlush = (): void => {
    // Lifecycle events cannot await this best-effort flush. Real persistence
    // failures are already reported through BitmapStore.onError; consume the
    // rejection so pagehide/visibilitychange never creates an unhandled promise.
    void persistenceController.flush().catch(() => undefined);
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
    resetInput();
    scheduler.invalidate({ overlay: true });
  };

  const clearSamPreview = (): void => {
    const previous = renderController.previews.clearSam();
    updateAntsAnimation();
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

  const attach = (
    screenCanvas: HTMLCanvasElement,
    overlayCanvas: HTMLCanvasElement,
    keyboardRoot?: HTMLElement
  ): void => renderController.attach(screenCanvas, overlayCanvas, keyboardRoot);
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
    clearStagedPreview();
    clearStagedPreviewCache();
    detach();
    cancelAllLayerRasterizations();
    cooldownPromise = persistenceController.flush().then(
      () => {
        if (disposed || lifecycleState !== 'cooling' || lifecycleGeneration !== generation) {
          return 'cooled';
        }
        rasterController.releaseReconstructible();
        renderController.previews.clearFilters();
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
    // Suppress viewport invalidation because the same-task render already repaints this resize.
    suppressViewportInvalidate = true;
    viewport.setViewportSize(cssWidth, cssHeight, dpr);
    suppressViewportInvalidate = false;
    // The viewport owns the clamped ratio so the backing size and every view matrix agree.
    const backingDpr = viewport.getDpr();
    const backingWidth = Math.round(cssWidth * backingDpr);
    const backingHeight = Math.round(cssHeight * backingDpr);
    renderController.resize(backingWidth, backingHeight);
    // Resize clears canvas pixels; force full recomposition in the same task to prevent a blank browser frame.
    // Detached rendering is a no-op.
    render({ all: true, damage: FULL_DAMAGE, layers: new Set<string>(), overlay: true, view: true });
  };

  let hasEverFitToView = false;

  const fitToView = (): void => {
    const doc = mirror.getDocument();
    if (!doc) {
      return;
    }
    // Fit content union bbox, using the bbox as the empty-canvas anchor and including content beyond it.
    let bounds: Rect = { ...doc.bbox };
    for (const leaf of compileDocumentLeaves(doc)) {
      if (leaf.contributionEnabled && isRenderableLayer(leaf.layer)) {
        bounds = union(bounds, getSourceBounds(leaf.layer, doc));
      }
    }
    viewport.fitToView(bounds, viewport.getViewportSize());
    hasEverFitToView = true;
  };

  /**
   * Fit only once per engine. Kept-alive widgets reattach after layout switches, and the project engine preserves
   * their existing zoom/pan.
   */
  const fitToViewOnFirstShow = (): void => {
    if (hasEverFitToView) {
      return;
    }

    fitToView();
  };

  const isLayerCacheReadyForOp = (layer: CanvasLayerContract, doc: CanvasDocumentContractV3): boolean => {
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

    // After prepared cache installation and document dispatch, observer, scheduler and persistence hooks only
    // notify; their failures cannot veto the completed transaction.
    const notifyBestEffort = (notify: () => void): void => {
      try {
        notify();
      } catch {
        // The document and cache are already converged. A later render or dirty
        // mark can retry ancillary work without reporting a false failed commit.
      }
    };
    notifyBestEffort(() => deleteDerivedSurfaces(layerId));
    notifyBestEffort(() => publishLayerThumbnailVersion(layerId, target.version));
    if (renderController.previews.hasFilter(layerId)) {
      notifyBestEffort(() => clearFilterPreview(layerId));
    }
    notifyBestEffort(() => scheduler.invalidate({ layers: [layerId] }));
    if (persist) {
      notifyBestEffort(() => bitmapStore.markLayerDirty(layerId));
    }
  };

  const getMainModelBase = (): string | null => {
    return opts.getMainModelBase?.() ?? null;
  };
  const getDefaultControlModel = (base: string | null): string | null => {
    return opts.getDefaultControlModel?.(base) ?? null;
  };

  /** Conversion reducers clone contracts, so their publication postcondition compares by value. */
  const documentHasLayerContract = (
    document: CanvasDocumentContractV3 | null,
    expected: CanvasLayerContract
  ): boolean => {
    const current = getDocumentLayer(document, expected.id);
    return current !== undefined && areJsonValuesStructurallyEqual(current, expected);
  };

  pixelEditController = new PixelEditController({
    applyImagePatch,
    backend,
    bitmapStore,
    canEdit: () => canEditDocument(),
    ctx: mutationContext,
    deleteDerived: deleteDerivedSurfaces,
    getActiveProjectId: () => projectId,
    getAdjustedSurface,
    getDocument: () => mirror.getDocument(),
    getTransformSession: () => stores.transformSession.get(),
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
    reportRefusal: reportEditRefusal,
    setTransformOverride: (layerId, transform) => {
      if (transform) {
        transformOverrides.set(layerId, transform);
      } else {
        transformOverrides.delete(layerId);
      }
    },
  });
  const beginPixelEdit = pixelEditController.begin.bind(pixelEditController);

  const captureLayerCache = (
    layer: CanvasLayerContract,
    doc: CanvasDocumentContractV3,
    admit: (rect: Rect) => boolean = () => true
  ): CapturedLayerCache => {
    const entry = layerCache.get(layer.id);
    if (!entry || isEmpty(entry.rect)) {
      return null;
    }
    if (isCurrentRasterizationJob(layer) || (entry.stale && !isEmpty(getSourceContentRect(layer, doc)))) {
      return 'not-ready';
    }
    if (!admit(entry.rect)) {
      return 'over-budget';
    }
    const pixels = backend.createSurface(entry.rect.width, entry.rect.height, { willReadFrequently: true });
    pixels.ctx.drawImage(entry.surface.canvas, 0, 0);
    return { pixels, rect: { ...entry.rect } };
  };

  const getDuplicateRasterPlan = (
    layer: CanvasLayerContract,
    doc: CanvasDocumentContractV3
  ): DuplicateLayerRasterPlan => {
    const source = renderableSourceOf(layer);
    if (!source) {
      return { type: 'empty' };
    }
    const entry = layerCache.get(layer.id);
    if (!entry || isEmpty(entry.rect)) {
      // Persistence clearing is failure-tolerant: the live cache may already be
      // empty while the contract still names the old bitmap. The cache is the
      // pixel authority in that state, so the duplicate must explicitly start
      // empty rather than resurrecting the durable image.
      if (source.type === 'paint' && bitmapStore.hasPendingClear(layer.id)) {
        return { type: 'empty' };
      }
      if (isEmpty(getSourceContentRect(layer, doc))) {
        return { type: 'empty' };
      }
      if (source.type === 'paint' && bitmapStore.hasPendingWork(layer.id)) {
        return { type: 'not-ready' };
      }
      if (layer.isEnabled) {
        return { type: 'not-ready' };
      }
      return { type: 'reference' };
    }
    if (isCurrentRasterizationJob(layer) || (entry.stale && !isEmpty(getSourceContentRect(layer, doc)))) {
      return { type: 'not-ready' };
    }
    const captureBytes = entry.rect.width * entry.rect.height * 4;
    const createsDerivedSurface =
      layer.type === 'regional_guidance' || layer.type === 'inpaint_mask' || hasLayerDisplayEffect(layer);
    const retainForHistory =
      source.type === 'paint' && (source.bitmap === null || bitmapStore.hasPendingWork(layer.id));
    const derivedBytes = createsDerivedSurface ? captureBytes : 0;
    return {
      captureBytes,
      initialReserveBytes: captureBytes * (retainForHistory ? 2 : 1) + derivedBytes,
      replayReserveBytes: captureBytes + derivedBytes,
      retainForHistory,
      type: 'capture',
    };
  };

  const layerNeedsPixelPersistence = (layer: CanvasLayerContract): boolean =>
    renderableSourceOf(layer)?.type === 'paint';

  const layerMutationController = new LayerMutationController({
    captureCache: captureLayerCache,
    ctx: mutationContext,
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    getDuplicateRasterPlan,
    getSelectedLayerIds: resolveSelectedLayerIds,
    hasPendingPixelWork: (layerId) => bitmapStore.hasPendingWork(layerId),
    needsPixelPersistence: layerNeedsPixelPersistence,
    prepareDuplicateRasterSource: prepareLayerRasterCache,
    pinDuplicateRasterSources: (layerIds) => {
      const leases = layerIds.map((layerId) => rasterController.memory.pin(layerId));
      return {
        release: () => {
          for (const lease of leases) {
            lease.release();
          }
        },
      };
    },
    publishSelectedLayerIds: (primaryId, selectedIds) => opts.setSelectedLayerIds?.(primaryId, selectedIds),
    scheduleDuplicateRasterization: scheduleLayerRasterization,
    sameContract: documentHasLayerContract,
    trackDetached: (bytes) => rasterController.memory.trackDetached(bytes),
  });
  const commitLayerCopy = layerMutationController.copy.bind(layerMutationController);
  const commitLayerConversion = layerMutationController.convert.bind(layerMutationController);
  const duplicateLayers = layerMutationController.duplicate.bind(layerMutationController);

  const replaceSelectionFromImage = editingController.selectionImage.replace.bind(editingController.selectionImage);

  const maskResultController = new MaskResultController({ ctx: mutationContext });
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
    createEventId,
    ctx: mutationContext,
    getCanvasState: () => mutationPort.getCanvasState(),
    now: () => new Date().toISOString(),
  });
  const commitStagedImage = stagedResultController.commit.bind(stagedResultController);

  const booleanMergeRasterLayers = (
    upperLayerId: string,
    operation: BooleanRasterOperation
  ): Promise<BooleanRasterResult> => layerController.booleanMerge.merge(upperLayerId, operation);

  const extractMaskedArea = (maskLayerId: string): Promise<ExtractMaskedAreaResult> =>
    layerController.extractMaskedArea.extract(maskLayerId);

  const mergeLayerDown = (upperLayerId: string): MergeDownResult => layerController.merge.mergeDown(upperLayerId);
  const mergeSelectedRasterLayers = (layerIds: readonly string[]): Promise<MergeVisibleResult> =>
    layerController.merge.mergeSelected(layerIds);
  const mergeVisibleRasterLayers = (): Promise<MergeVisibleResult> => layerController.merge.mergeVisible();

  const { captureDocumentSnapshot, captureRasterSnapshot, isDocumentSnapshotCurrent, releaseActiveSnapshots } =
    createRasterSnapshotCapture({
      createSurface: (width, height) => backend.createSurface(width, height),
      getCanvasState: () => mutationPort.getCanvasState(),
      getDirectPixelEpoch: () => directPixelEpoch,
      getDocumentGeneration: () => rasterController.getDocumentGeneration(),
      getLifecycleGeneration: () => lifecycleGeneration,
      isDisposed: () => disposed,
      isGuardCurrent: isLayerExportGuardCurrent,
      memory: rasterController.memory,
      rasterizeLayerPixels,
    });

  const psdExportController = new PsdExportController({
    backend,
    captureDocumentSnapshot,
    captureRasterSnapshot,
    getAvailableBytes: () => rasterController.memory.getAvailableBytes(),
    isDocumentSnapshotCurrent,
    reserve: (bytes) =>
      rasterController.memory.reserve(bytes, { generation: lifecycleGeneration, purpose: 'psd-export' }),
  });
  const exportRasterLayersToPsd = (fileName: string): Promise<PsdExportResult> => psdExportController.export(fileName);

  const rasterizeLayer = (layerId: string): RasterizeLayerResult => layerController.rasterize.rasterize(layerId);

  // Transform sessions span gestures. Apply records one undoable parameter edit for image layers or pixel bake for
  // paint; Cancel drops previews. Tools and public UI share the session API.

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

  // Text sessions expose a portal with DOM-owned typing until commit. Creation/add or source update records one
  // undo entry; unchanged edits and empty creation cancel. Live style changes remain in session state.

  const setTextEditContentReader = (reader: (() => string) | null): void =>
    editingController.text.setContentReader(reader);
  const openTextCreate = (point: Vec2): void => editingController.text.openCreate(point);
  // The pane keeps styling the layer after the portal's blur commits the
  // session, so opening an edit also makes that layer the document selection.
  const openTextEdit = (layerId: string): void => {
    editingController.text.openEdit(layerId);
    if (stores.textEditSession.get()?.layerId === layerId && mirror.getDocument()?.selectedLayerId !== layerId) {
      dispatchCanvasMutation({ id: layerId, type: 'setCanvasSelectedLayer' });
    }
  };
  const updateTextEditStyle = (patch: TextStylePatch): void => editingController.text.updateStyle(patch);
  const cancelTextEdit = (): void => editingController.text.cancel();
  const commitTextEdit = (content: string, styleChanges?: TextStylePatch): StructuralCommitResult | null =>
    editingController.text.commit(content, styleChanges);
  const commitOpenTextSession = (): boolean => editingController.text.commitOpen();

  // ---- Selection public API -----------------------------------------------

  /**
   * Select-all and invert use content union bbox as their bounded domain, matching fit-to-view and anchoring empty
   * canvases.
   */
  // Commit live floats before selection operations so they see displayed pixels; Escape instead abandons the
  // float.
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

  /** Copy reuses the float's nonmutating masked-copy path so clipboard and lift select identical pixels. */
  const exportSelectionBlob = (): Promise<Blob | null> => {
    floatingSelection.commit();
    const doc = mirror.getDocument();
    const layer = getDocumentLayer(doc, doc?.selectedLayerId);
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

  const clearMask = (layerId: string): MaskEditResult => layerController.mask.clear(layerId);
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
    cleanup.run(cancelOpenPixelEdit);
    cleanup.run(() => pixelEditController?.dispose());
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
    cleanup.run(() => stores.textEditSession.set(null));
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
    cleanup.run(unsubscribeTextFontSources);
    cleanup.run(unsubscribeTextDefaultsFontSources);
    cleanup.run(() => mutationContext.dispose());
    cleanup.run(unsubscribeHistoryEpoch);
    cleanup.run(() => historyController.dispose());
    cleanup.run(() => persistenceController.dispose());
    cleanup.run(() => mirror.dispose());
    cleanup.run(() => previewPublisher.dispose());
    cleanup.run(() => renderController.dispose());
    cleanup.run(() => rasterController.dispose());
    cleanup.run(() => fontLoader.dispose());
    cleanup.run(() => stores.thumbnailStatus.clear());
    cleanup.run(() => strokeListeners.clear());
    cleanup.run(() => editRefusalListeners.clear());
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

  const onEditRefused = (listener: (refusal: CanvasEditRefusal) => void): (() => void) => {
    editRefusalListeners.add(listener);
    return () => {
      editRefusalListeners.delete(listener);
    };
  };

  const clearCaches = async (): Promise<void> => {
    // Flush paint before invalidating caches; otherwise rerasterizing the older persisted source would erase
    // unuploaded strokes.
    await persistenceController.flush();
    const doc = mirror.getDocument();
    for (const layer of getDocumentLeaves(doc)) {
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
      document: doc ? { height: doc.height, layers: getDocumentLeaves(doc).length, width: doc.width } : null,
      hasSelection: selection.hasSelection(),
      projectId,
      selectedLayerId: doc?.selectedLayerId ?? null,
      zoom: viewport.getZoom(),
    });
  };

  /**
   * Menus target the selected layer without hit testing. Suppress them during gestures, transforms and text
   * editing, when right-click belongs to the interaction.
   */
  const canTargetLayerFromContextMenu = (): boolean =>
    !pipeline.isGestureActive() &&
    !stores.transformSession.get() &&
    !stores.textEditSession.get() &&
    mirror.getDocument() !== null;

  const replayHistory = async (direction: 'undo' | 'redo'): Promise<CanvasHistoryReplayStatus> => {
    const result = await (direction === 'undo' ? historyController.undo() : historyController.redo());
    return result.status;
  };
  const undo = (): Promise<CanvasHistoryReplayStatus> => replayHistory('undo');
  const redo = (): Promise<CanvasHistoryReplayStatus> => replayHistory('redo');
  const setBboxGrid = (size: number): void => stores.bboxGrid.set(size > 0 ? size : 1);
  const getViewport = (): Viewport => viewport;
  const getCompositeExecutorDeps = (): CanvasCompositeExecutorDeps => ({
    backend,
    reserve: (bytes) => rasterController.memory.reserveOperation(bytes, { purpose: 'invocation-composite' }),
    // Unreferenced generation inputs upload as reclaimable intermediates.
    uploadImage: (blob) => opts.uploadIntermediateImage(blob),
  });
  const exportRasterComposite = (request: RasterCompositeExportRequest) =>
    exportRasterCompositeWithDeps(request, {
      backend,
      captureSnapshot: (): RasterCompositeExportSnapshot => ({
        directPixelEpoch: directPixelEpoch,
        document: mirror.getDocument(),
        documentGeneration: rasterController.getDocumentGeneration(),
        lifecycleGeneration,
      }),
      adjustedSurface: (layerId, surface, adjustments) =>
        rasterController.getAdjustedCacheSurface(layerId, surface, adjustments),
      getLayerSurface: requireLayerSurfaceForExport,
      isSnapshotCurrent: (snapshot) =>
        !disposed &&
        mutationPort.getCanvasState() !== null &&
        snapshot.directPixelEpoch === directPixelEpoch &&
        snapshot.document === mirror.getDocument() &&
        snapshot.documentGeneration === rasterController.getDocumentGeneration() &&
        snapshot.lifecycleGeneration === lifecycleGeneration,
      pin: (layerIds) => {
        const leases = layerIds.map((layerId) => rasterController.memory.pin(layerId));
        return {
          release: () => {
            for (const lease of leases) {
              lease.release();
            }
          },
        };
      },
      reserve: (bytes) => rasterController.memory.reserveOperation(bytes, { purpose: 'background-snapshot' }),
    });
  const surface: CanvasSurfaceCapability = { attach, detach, resize };
  const viewportCapability: CanvasViewportCapability = { fitToView, fitToViewOnFirstShow, getViewport, setBboxGrid };
  const stepHistoryBy = async (offset: number): Promise<CanvasHistoryReplayStatus> => {
    let status: CanvasHistoryReplayStatus = 'empty';
    for (let step = 0; step < Math.abs(Math.trunc(offset)); step += 1) {
      status = await (offset < 0 ? undo() : redo());
      if (status !== 'applied') {
        break;
      }
    }
    return status;
  };
  const historyCapability: CanvasHistoryCapability = {
    clearHistory,
    getEntries: () => history.entries(),
    getHeldAssetRefs: () => history.heldAssetRefs(),
    redo,
    stepBy: stepHistoryBy,
    undo,
  };
  const lifecycle: CanvasLifecycleCapability = {
    activate,
    beginCooldown,
    dispose,
    flushPendingUploads: (options) => persistenceController.flush(options),
    getLifecycleState: () => lifecycleState,
  };
  const layerController = new LayerController({
    booleanMerge: {
      backend,
      ctx: mutationContext,
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId),
      isCacheReady: isLayerCacheReadyForOp,
      isGuardCurrent: isLayerExportGuardCurrent,
    },
    crop: {
      backend,
      captureCache: captureLayerCache,
      ctx: mutationContext,
      discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled: true }),
      isGuardCurrent: isLayerExportGuardCurrent,
      isSupportedSource: isSupportedExportSource,
    },
    copy: {
      backend,
      ctx: mutationContext,
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled: true }),
      isGuardCurrent: isLayerExportGuardCurrent,
    },
    extractMaskedArea: {
      backend,
      ctx: mutationContext,
      derived: derivedSurfaceCache,
      diagnostics,
      exportBaked: (layerId, includeDisabled) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled }),
      getAdjustedSurface,
      getGroupSurface,
      getMaskPattern: getMaskPatternTile,
      hasExportableContent: hasExportableLayerContent,
      isCacheReady: isLayerCacheReadyForOp,
      isGuardCurrent: isLayerExportGuardCurrent,
      layers: layerCache,
      rasterize: (layerId) => rasterizeLayerPixelsForStructural(layerId),
    },
    commitGeneratedImageResult,
    newRasterLayer: {
      backend,
      ctx: mutationContext,
      layers: layerCache,
      selection: editingController.selection,
    },
    mask: {
      applyImagePatch,
      ctx: mutationContext,
      deleteDerived: deleteDerivedSurfaces,
      discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
      isCacheReady: isLayerCacheReadyForOp,
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
      restoreCache: restoreLayerCache,
    },
    merge: {
      backend,
      ctx: mutationContext,
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId),
      hasExportableContent: hasExportableLayerContent,
      isCacheReady: isLayerCacheReadyForOp,
      layers: layerCache,
      needsPixelPersistence: layerNeedsPixelPersistence,
      publishSelectedLayerIds: (primaryId, selectedIds) => opts.setSelectedLayerIds?.(primaryId, selectedIds),
    },
    rasterize: {
      backend,
      ctx: mutationContext,
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
      pin: (layerId) => rasterController.memory.pin(layerId),
      projectId,
      rasterize: rasterizeLayerForThumbnail,
      reportError: (layerId, error) => {
        try {
          reportError('Layer thumbnail rasterization failed', layerId, error);
        } catch {
          // Diagnostics must not turn a handled thumbnail failure into a rejection.
        }
      },
      reserve: (bytes) =>
        rasterController.memory.reserve(bytes, { generation: lifecycleGeneration, purpose: 'thumbnail' }),
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
  let documentModel: CanvasDocumentModel | null = null;
  const currentDocumentModel = (): CanvasDocumentModel | null => {
    const document = mutationPort.getCanvasState()?.document ?? null;
    if (!document) {
      return null;
    }
    if (documentModel?.document !== document) {
      documentModel = createDocumentModel(document, {
        editRevision: mutationContext.getEditRevision(),
        projectId,
      });
    }
    return documentModel;
  };
  const documentCapability: CanvasDocumentCapability = {
    captureInsertionAnchor,
    captureRestoreAnchor: (layerId) => mutationContext.captureRestoreAnchor(layerId),
    captureSnapshot: captureDocumentSnapshot,
    getDocument: () => mirror.getDocument(),
    getEditRevision: () => mutationContext.getEditRevision(),
    model: currentDocumentModel,
    replaceDocument: (document) => dispatchCanvasMutation({ document, type: 'replaceCanvasDocument' }, 'user'),
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
    onEditRefused,
    onStrokeCommitted,
    requestColorSample,
    setColorSampleRouter: (router) => {
      colorSampleRouter = router;
      return () => {
        // Compare-and-clear: a later installer must not be evicted by an earlier one's cleanup.
        if (colorSampleRouter === router) {
          colorSampleRouter = null;
        }
      };
    },
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
    duplicateLayers,
    mergeLayerDown,
    mergeSelectedRasterLayers,
    mergeVisibleRasterLayers,
    nudgeSelectedLayer,
    openTextCreate,
    openTextEdit,
    rasterizeLayer,
    setTextEditContentReader,
    updateTextEditStyle,
    updateTransformSession,
  };
  let overviewRequestedDoc: CanvasDocumentContractV3 | null = null;
  const overviewRequestedLayerIds = new Set<string>();
  const drawDocumentOverview = (target: HTMLCanvasElement, maxSizePx: number): Rect | null => {
    const doc = mirror.getDocument();
    const ctx = target.getContext('2d');
    if (!doc || !ctx || doc.width <= 0 || doc.height <= 0) {
      return null;
    }
    // Overview requests missing offscreen caches and repaints after publication. Attempt each layer once per
    // document to avoid repeated failures; never pin against memory-budget eviction.
    if (overviewRequestedDoc !== doc) {
      overviewRequestedDoc = doc;
      overviewRequestedLayerIds.clear();
    }
    const missing: string[] = [];
    for (const leaf of getDocumentLeaves(doc)) {
      if (!leaf.isEnabled || !renderableSourceOf(leaf) || overviewRequestedLayerIds.has(leaf.id)) {
        continue;
      }
      const entry = layerCache.peek(leaf.id);
      if (!entry || entry.stale) {
        missing.push(leaf.id);
        overviewRequestedLayerIds.add(leaf.id);
      }
    }
    if (missing.length > 0) {
      scheduleLayerRasterization(missing);
    }
    const size = fitThumbnailSize(doc.width, doc.height, maxSizePx);
    target.width = size.width;
    target.height = size.height;
    const scale = size.width / doc.width;
    // Only `ctx`/`width`/`height` are read by the compositor; the element is not resized through the surface seam.
    const surfaceLike = { canvas: target, ctx, height: size.height, width: size.width } as unknown as RasterSurface;
    compositeDocument(
      surfaceLike,
      doc,
      layerCache,
      { a: scale, b: 0, c: 0, d: scale, e: 0, f: 0 },
      {
        adjustedSurface: getAdjustedSurface,
        backend,
        checkerboardTile: null,
        derivedSurfaces: derivedSurfaceCache,
        groupSurface: getGroupSurface,
        imageSmoothing: true,
        maskPatternTile: getMaskPatternTile,
        regionOverlays: true,
      }
    );
    return { height: doc.height, width: doc.width, x: 0, y: 0 };
  };
  const previewCapability: CanvasEnginePreviewCapability = {
    ...layerController.previews,
    drawDocumentOverview,
    preloadStagedPreview,
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
      const previous = renderController.previews.getSam();
      const isolationChanged = previous?.isolated !== preview.isolated;
      // An isolation toggle republishes the same surface; keep its outline.
      const outline = previous?.data === preview.data ? (previous.outline ?? null) : traceSamOutline(preview);
      renderController.previews.setSam({ ...preview, outline });
      updateAntsAnimation();
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
    fonts,
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
