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
  CanvasSurfaceAttachOptions,
  CanvasSurfaceCapability,
  CanvasViewportCapability,
  BooleanRasterOperation,
  BooleanRasterResult,
  CopyLayerResult,
  CropAllLayersResult,
  CropLayerResult,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  ExtractMaskedAreaResult,
  MaskEditResult,
  MergeDownResult,
  MergeInpaintMasksResult,
  MergeVisibleResult,
  NewRasterLayerResult,
  PsdExportResult,
  RasterizeLayerResult,
  StructuralCommitOptions,
  StructuralCommitResult,
} from '@workbench/canvas-engine/capabilities';
import type {
  CanvasCompositeExecutorDeps,
  CaptureRasterSnapshotOptions,
  CaptureRasterSnapshotResult,
  RasterReadLease,
} from '@workbench/canvas-engine/rasterTransactions';

import type { TransformResampler } from './controllers/transformEditingController';
import type * as VectorRuntimeModule from './vectorRuntime';
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
import type { LayerCacheReadiness } from '@workbench/canvas-engine/controllers/editSteps';
import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { ColorSampleProviders, ColorSampler } from '@workbench/canvas-engine/render/colorSample';
import type { AdjustmentRefusal } from '@workbench/canvas-engine/render/exactAdjustments';
import type { CanvasFontRuntime, CanvasTextSource, FontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import type { LayerCacheEntry, LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { OverlayCursor } from '@workbench/canvas-engine/render/overlayRenderer';
import type { RenderScheduler } from '@workbench/canvas-engine/render/scheduler';
import type { SpatialAdjustmentExecutor } from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect, RenderFlags, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { VectorRuntime } from '@workbench/canvas-engine/vectorRuntime';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';

import { createItemId } from '@platform/browser/randomUuid';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { ColorSampleController } from '@workbench/canvas-engine/controllers/colorSampleController';
import { ControlDerivationController } from '@workbench/canvas-engine/controllers/controlDerivationController';
import { PixelEditController } from '@workbench/canvas-engine/controllers/controlPixelController';
import { ControlViewController } from '@workbench/canvas-engine/controllers/controlViewController';
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
import { PaintLineEndController } from '@workbench/canvas-engine/controllers/paintLineEndController';
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
import {
  getDocumentIndex,
  getDocumentLayer,
  getDocumentLeaves,
  getDocumentNode,
} from '@workbench/canvas-engine/document/documentIndex';
import {
  createEngineStores,
  type EngineStores,
  type LayerAdjustmentStatus,
  type ScalarStore,
  type TextStylePatch,
  type TextToolOptions,
} from '@workbench/canvas-engine/engineStores';
import {
  exportRasterComposite as exportRasterCompositeWithDeps,
  type RasterCompositeExportRequest,
  type RasterCompositeExportSnapshot,
} from '@workbench/canvas-engine/exportRasterComposite';
import { createPointerPipeline, type PointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createWheelInput } from '@workbench/canvas-engine/input/wheel';
import { isEmpty, union } from '@workbench/canvas-engine/math/rect';
import { hasSpatialAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { createColorSampler } from '@workbench/canvas-engine/render/colorSample';
import {
  compositeDocument,
  createCheckerboardTile,
  shouldSmoothAtZoom,
  type CompositeOptions,
} from '@workbench/canvas-engine/render/compositor';
import { createFontLoader, domFontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import { hasLayerDisplayEffect } from '@workbench/canvas-engine/render/layerDisplayEffect';
import { createMaskPatternTile } from '@workbench/canvas-engine/render/maskFill';
import { colorLoupePixels, renderOverlay } from '@workbench/canvas-engine/render/overlayRenderer';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createDomRasterBackend, type RasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import {
  RasterCompositeNotReadyError,
  RasterCompositeOverBudgetError,
  RasterCompositeUnreadableError,
} from '@workbench/canvas-engine/render/rasterComposite';
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
import type { LayerUnreadable } from './layerExportGuards';
import type { CanvasProjectMutation } from './mutationContracts';
import type { StrokeCommittedEvent, Tool, ToolContext } from './tools/tool';

import { createBitmapStore, type BitmapStore } from './document/bitmapStore';
import { createDocumentMirror, type DocumentMirror } from './document/documentMirror';
import { decideLayerChange, nextThumbnailDisplayToken } from './document/layerChangeDecision';
import { getSourceBounds, getSourceContentRect, isRenderableLayer, renderableSourceOf } from './document/sources';
import { collectCanvasFontReferences, replaceCanvasFontReferences } from './fontReferences';
import { createLayerExportGuards, isSupportedExportSource, layerUnreadable } from './layerExportGuards';
import { createLayerRasterizer } from './layerRasterizer';
import { createPreviewPublisher } from './previewPublisher';
import { createRasterSnapshotCapture } from './rasterSnapshotCapture';
import { createCompositeFrame, isInStagingHiddenStack } from './render/compositeFrame';
import { floatingSelectionFrame } from './render/floatingSelectionFrame';
import { createOverlayFrame } from './render/overlayFrame';
import { createSelectObjectBridge } from './selectObjectBridge';
import { createStrokeEdits } from './strokeCommit';
import { createViewTool } from './tools/viewTool';

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
  context: { error: unknown; layerId?: string; label?: string };
  message:
    | 'Vector tools could not be loaded'
    | 'PSD export failed'
    | 'Layer thumbnail rasterization failed'
    | 'Bitmap persistence failed'
    | 'Bitmap persistence suspended'
    | 'Structural edit was refused'
    | 'Structural edit could not be reverted'
    | 'Structural edit could not be mirrored'
    | 'Structural history replay was refused'
    | 'Structural history replay could not be mirrored'
    | 'History replay failed'
    | 'Transform resampling failed';
  namespace: 'canvas';
  projectId: string;
}

export interface CanvasEngineOptions {
  resampleTransform?: TransformResampler;
  projectId: string;
  loadVectorRuntime?: () => Promise<typeof VectorRuntimeModule>;
  /** Undo steps the project's previous engine was disposed with; History reports them until it is next cleared. */
  droppedHistorySteps?: number;
  mutationPort: CanvasProjectMutationPort;
  /** Durable upload: document layer refs must survive garbage collection. */
  uploadImage(blob: Blob): Promise<{ height: number; imageName: string; width: number }>;
  /** Transient intermediate upload, so each invocation leaves no permanent image. */
  uploadIntermediateImage(blob: Blob): Promise<{ height: number; imageName: string; width: number }>;
  getMainModelBase?: () => string | null;
  getSelectedLayerIds?: () => readonly string[];
  setSelectedLayerIds?: (primaryId: string | null, selectedIds: readonly string[]) => void;
  reportError(report: CanvasEngineErrorReport): void;
  backend?: RasterBackend;
  imageResolver: ImageResolver;
  bitmapStore?: BitmapStore;
  /** Undefined uses `document.fonts`; null disables font readiness. */
  fonts?: FontLoadApi | CanvasFontRuntime | null;
  enableDiagnostics?: boolean;
  /** Spatial adjustment executor factory; defaults to a module worker. Node tests pass an inline one. */
  createSpatialExecutor?: () => SpatialAdjustmentExecutor;
}

export interface CanvasEngineSelectionCapability extends CanvasSelectionCapability {}

/** Adds Canvas-internal raster snapshots, pixel exports and executor deps that stay out of `api.ts`. */
export interface CanvasEngineInternalExportCapability extends CanvasEngineExportCapability {
  captureRasterSnapshot(
    documentSnapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options?: CaptureRasterSnapshotOptions
  ): Promise<CaptureRasterSnapshotResult>;
  exportBakedLayerPixels(
    layerId: string,
    options?: ExportBakedLayerPixelsOptions
  ): Promise<ExportBakedLayerPixelsResult>;
  exportLayerPixels(layerId: string, options?: ExportLayerPixelsOptions): Promise<ExportLayerPixelsResult>;
  getCompositeExecutorDeps(): CanvasCompositeExecutorDeps;
}

export interface CanvasEngineImplementation extends CanvasEngine {
  readonly exports: CanvasEngineInternalExportCapability;
  readonly stores: EngineStores;
}

export interface CanvasEngineCoreComposition {
  readonly engine: CanvasEngineImplementation;
  readonly applicationHost: CanvasApplicationHost;
}

const createLayerId = (): string => createItemId('layer');
const createEventId = (): string => `event-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const clearSurface = (surface: RasterSurface): void => {
  surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
  surface.ctx.clearRect(0, 0, surface.width, surface.height);
};

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
    // Raster pixel transactions already include adjustments; applying them again would double them.
    isAdjustmentBaked: (layerId) => pixelEditController?.isOpenFor([layerId]) === true,
    isLayerHeld: (layerId) => isLayerHeldBySession(layerId) || bitmapStore.hasPendingWork(layerId),
    onAdjustedSurfaceSettled: (layerId) => onAdjustedSurfaceSettled(layerId),
    onAdjustmentStatus: (ownerId, status) => onAdjustmentStatus(ownerId, status),
    onGroupSurfaceSettled: (groupId) => onGroupSurfaceSettled(groupId),
    onVersionChange: (layerId) => editingController?.invalidateLayer(layerId),
    createSpatialExecutor: opts.createSpatialExecutor,
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
    bboxPreview: stores.bboxPreview,
    brushOptions: stores.brushOptions,
    canRedo: stores.canRedo,
    canUndo: stores.canUndo,
    checkerboard: stores.checkerboard,
    checkerColors: stores.checkerColors,
    clipToBbox: stores.clipToBbox,
    colorPair: stores.colorPair,
    controlLayerView: stores.controlLayerView,
    documentEditingLocked: stores.documentEditingLocked,
    psdExportStatus: stores.psdExportStatus,
    eraserOptions: stores.eraserOptions,
    gradientOptions: stores.gradientOptions,
    hasFloatingSelection: stores.hasFloatingSelection,
    hasSelection: stores.hasSelection,
    historyEpoch: stores.historyEpoch,
    invertBrushSizeScroll: stores.invertBrushSizeScroll,
    lassoOptions: stores.lassoOptions,
    layerQuickSwitch: stores.layerQuickSwitch,
    marqueeOptions: stores.marqueeOptions,
    layerPixelEpoch: stores.layerPixelEpoch,
    documentEpoch: stores.documentEpoch,
    ruleOfThirds: stores.ruleOfThirds,
    scrollInput: stores.scrollInput,
    shapeOptions: stores.shapeOptions,
    showBbox: stores.showBbox,
    showGrid: stores.showGrid,
    snapToGrid: stores.snapToGrid,
    stagingHidesOverlays: stores.stagingHidesOverlays,
    textEditSession: stores.textEditSession,
    vectorEditSession: stores.vectorEditSession,
    vectorToolsStatus: stores.vectorToolsStatus,
    textOptions: stores.textOptions,
    trackpadDetected: stores.trackpadDetected,
    transformOptions: stores.transformOptions,
    transformSession: stores.transformSession,
    pendingTransformLayerId: stores.pendingTransformLayerId,
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
    getLayerAdjustmentStatus: (layerId) => stores.adjustmentStatus.get(layerId) ?? 'idle',
    subscribeLayerAdjustmentStatus: (layerId, listener) => stores.adjustmentStatus.subscribeKey(layerId, listener),
  };
  const fontLoader = createFontLoader(opts.fonts === undefined ? domFontLoadApi() : opts.fonts);
  let syncActiveFontSources: () => void = () => undefined;
  const tools = new Map<ToolId, Tool>([
    ['view', createViewTool()],
    ['brush', createBrushTool()],
    ['eraser', createEraserTool()],
    ['move', createMoveTool()],
    [
      'spline',
      {
        id: 'spline',
        usesAltKey: true,
        cursor: () => vectorRuntime?.editor.cursor() ?? 'crosshair',
        onActivate: () => vectorRuntime?.editor.onActivate(),
        onDeactivate: (ctx, options) => vectorRuntime?.editor.onDeactivate(ctx, options),
        onPointerDown: (ctx, input) => vectorRuntime?.editor.onPointerDown(ctx, input),
        onPointerMove: (ctx, input) => vectorRuntime?.editor.onPointerMove(ctx, input),
        onPointerUp: () => vectorRuntime?.editor.onPointerUp(),
        onPointerCancel: () => vectorRuntime?.editor.onPointerCancel(),
        onKeyCommand: (ctx, command) => vectorRuntime?.editor.onKeyCommand(ctx, command) ?? false,
      },
    ],
    ['transform', createTransformTool()],
    ['bbox', createBboxTool()],
    ['colorPicker', createColorPickerTool()],
    ['lasso', createLassoTool()],
    ['marquee', createMarqueeTool()],
    ['shape', createShapeTool({ getVectorTool: () => vectorRuntime?.shape ?? null })],
    ['gradient', createGradientTool()],
    ['text', createTextTool()],
    ['sam', createSamTool()],
  ]);

  // Render-time transform previews: move overrides position, transform the full matrix.
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
  let vectorWorkEpoch = 0;
  let cooldownPromise: Promise<'cooled' | 'dirty'> | null = null;

  let overlayCursor: OverlayCursor | null = null;
  let colorLoupeShown = false;
  let colorLoupeSampler: ColorSampler | null = null;

  let checkerboardTile: RasterSurface | null = null;
  const getCheckerboardTile = (): RasterSurface => {
    checkerboardTile ??= createCheckerboardTile(backend, stores.checkerColors.get());
    return checkerboardTile;
  };

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

  const derivedSurfaceCache = rasterController.derived;
  const deleteDerivedSurfaces = (layerId: string): void => rasterController.deleteDerivedSurfaces(layerId);
  const getAdjustedSurface = (layer: CanvasLayerContract, entry: LayerCacheEntry): RasterSurface | null =>
    rasterController.getAdjustedSurface(layer, entry);

  const groupSurfaces = rasterController.groups;
  const getGroupSurface: NonNullable<CompositeOptions['groupSurface']> = (scope, members, matrices, content) =>
    groupSurfaces.get(scope, members, matrices, content);
  /** Picking and the loupe sample the image underneath the mask overlays. */
  const colorSampleProviders: ColorSampleProviders = {
    adjustedSurface: getAdjustedSurface,
    derivedSurfaces: derivedSurfaceCache,
    groupSurface: getGroupSurface,
    maskPatterns: {
      screen: () => {
        const dpr = viewport.getDpr();
        return { dpr, smoothing: shouldSmoothAtZoom(viewport.getZoom()), view: viewport.viewMatrix(dpr) };
      },
      tile: getMaskPatternTile,
    },
    showOverlayStacks: { control: true, inpaint_mask: false, regional_guidance: false, vector: true },
  };

  const strokeListeners = new Set<(event: StrokeCommittedEvent) => void>();
  const toolChangeListeners = new Set<(change: { from: string; to: string; temporary: boolean }) => void>();
  let samInputHandler: ((input: SamVisualInput) => void) | null = null;
  let applicationEscapeHandler: ((gestureWasActive: boolean) => boolean) | null = null;

  const getLayerSourceById = (layerId: string): CanvasLayerSourceContract | null => {
    const doc = mirror.getDocument();
    const layer = getDocumentLayer(doc, layerId);
    return layer ? renderableSourceOf(layer) : null;
  };

  const getAuthoritativeLayerSourceById = (layerId: string): CanvasLayerSourceContract | null => {
    const layer = getDocumentLayer(mutationPort.getCanvasState()?.document, layerId);
    return layer ? renderableSourceOf(layer) : null;
  };

  /** Mask actions preserve `fill`; self-echo tracking avoids redundant rasterization. */
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
          // Derived surfaces are keyed on the old extent; drop them before the clear dispatch.
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
      onError: (error, layerId, info) => {
        // A retry may still land the pixels; only a failure no retry follows ends the wait.
        if (!info.willRetry) {
          editingController.transform.onUploadFailed(layerId);
        }
        reportError(
          info.willRetry ? 'Bitmap persistence failed' : 'Bitmap persistence suspended',
          layerId,
          info.willRetry ? error : new Error('Canvas changes are no longer uploading. A new stroke will retry.')
        );
      },
      uploadImage: (blob) => opts.uploadImage(blob),
    });
  const persistenceController = new PersistenceController(bitmapStore);

  const historyController = new HistoryController({
    droppedSteps: opts.droppedHistorySteps ?? 0,
    // A live float or preview holds unrecorded state; restore both before replaying history.
    beforeReplay: () => {
      paintLineEnds.clear();
      editingController.transform.onReplay();
      floatingSelection.cancel();
      structuralController.endPreview();
    },
    canEdit: () => canEditDocument(),
    canRedoStore: { set: (value) => stores.canRedo.set(value) },
    canUndoStore: { set: (value) => stores.canUndo.set(value) },
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
  let engineMutationDepth = 0;
  const dispatchCanvasMutation = (action: CanvasProjectMutation, origin: 'system' | 'user' = 'user'): boolean => {
    // Engine edits are not explicit layer navigation, even when user-originated.
    engineMutationDepth += 1;
    try {
      return mutationPort.dispatch(action, origin);
    } finally {
      engineMutationDepth -= 1;
    }
  };
  // Direct pixel writes keep the reducer canvas object, so snapshot freshness also binds to this epoch.
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

  /** Patches land only on trustworthy pixels: stale caches rasterize first; a missing layer makes the step a no-op. */
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

  /** Exact extent replacement, so transform undo removes stale pixels outside smaller bounds. */
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
    controlView.clear((id) => id === layerId);
    scheduler.invalidate({ layers: [layerId] });
  };

  const invalidateAncestorGroups = (ownerId: string): boolean => {
    const document = mirror.getDocument();
    const ancestors = document ? (getDocumentIndex(document).byId.get(ownerId)?.path ?? []) : [];
    if (ancestors.length === 0) {
      return false;
    }
    groupSurfaces.invalidate(ancestors);
    return true;
  };

  /** A spatial result changed the display without a cache publication; the layer's own adjusted surface stays. */
  const onAdjustedSurfaceSettled = (layerId: string): void => {
    if (disposed) {
      return;
    }
    stores.thumbnailVersion.set(layerId, nextThumbnailDisplayToken(stores.thumbnailVersion.get(layerId)));
    stores.layerPixelEpoch.set(stores.layerPixelEpoch.get() + 1);
    invalidateAncestorGroups(layerId);
    scheduler.invalidate({ layers: [layerId] });
  };

  /** Never redraws the landed group's own surface, or every landing would start its next job. */
  const onGroupSurfaceSettled = (groupId: string): void => {
    if (disposed) {
      return;
    }
    stores.layerPixelEpoch.set(stores.layerPixelEpoch.get() + 1);
    invalidateAncestorGroups(groupId);
    scheduler.invalidate({ layers: [groupId] });
  };

  /** A job that ended without landing still redraws holding groups, which wait for it. */
  const onAdjustmentStatus = (ownerId: string, status: LayerAdjustmentStatus | null): void => {
    const wasComputing = stores.adjustmentStatus.get(ownerId) === 'computing';
    if (status) {
      stores.adjustmentStatus.set(ownerId, status);
    } else {
      stores.adjustmentStatus.delete(ownerId);
    }
    if (wasComputing && status !== 'computing' && !disposed && invalidateAncestorGroups(ownerId)) {
      scheduler.invalidate({ layers: [ownerId] });
    }
  };

  /** Drops spatial state of owners that no longer draw a spatial stack, clearing their status too. */
  const retainSpatialSurfaces = (): void => {
    const document = mirror.getDocument();
    const byId = document ? getDocumentIndex(document).byId : null;
    rasterController.adjustments.retainSpatial((layerId) => {
      const node = byId?.get(layerId)?.node;
      return node?.type === 'raster' && hasSpatialAdjustments(node.adjustments);
    });
    groupSurfaces.retainSpatial((groupId) => {
      const entry = byId?.get(groupId);
      return (
        entry?.stack === 'raster' &&
        entry.node.type === 'group' &&
        entry.node.isEnabled &&
        entry.ancestorsEnabled &&
        hasSpatialAdjustments(entry.node.adjustments)
      );
    });
  };

  const invalidateLayerCache = (layerId: string): void => {
    cancelLayerRasterization(layerId);
    layerCache.invalidate(layerId);
    stores.thumbnailStatus.delete(layerId);
    if (renderController.previews.hasFilter(layerId)) {
      clearFilterPreview(layerId);
    }
    controlView.clear((id) => id === layerId);
  };

  let antsPhase = 0;

  const onSelectionChanged = (): void => {
    // Selection is already applied; observers are best-effort so a fault cannot report a false failure.
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
  // Later-defined engine values are passed as thunks: the context never invokes them during construction.
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
    endStructuralPreview: () => structuralController.endPreview(),
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
      getTransformSessionLayerId: () => stores.transformSession.get()?.layerId ?? null,
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
      getOptions: stores.transformOptions.get,
      resample: opts.resampleTransform,
      runSpatial: (request) => rasterController.spatial.run(request),
      reportFailure: (layerId, error) => reportError('Transform resampling failed', layerId, error),
      canEdit: () => canEditDocument(),
      commitStructural: (label, forward, inverse) => commitStructural(label, forward, inverse),
      ctx: mutationContext,
      enterTool: () => setTool('transform'),
      getCache: (layerId) => layerCache.get(layerId) ?? null,
      getDocument: () => mirror.getDocument(),
      getUserToolId: () => pipeline.heldRestoreToolId() ?? interactionController.getActiveToolId(),
      hasFloatingSelection: () => floatingSelection.has(),
      getFloatingSelection: () => floatingSelection.get(),
      setFloatingTransform: (transform) => floatingSelection.setTransform(transform),
      invalidate: (payload) => scheduler.invalidate(payload),
      isGestureActive: () => pipeline.isGestureActive(),
      peekCacheRect: (layerId) => layerCache.peek(layerId)?.rect,
      pending: stores.pendingTransformLayerId,
      reportRefusal: reportEditRefusal,
      rebaseLineEnd: (layerId, from) => paintLineEnds.rebase(layerId, from),
      select: (layerId) => dispatchCanvasMutation({ id: layerId, type: 'setCanvasSelectedLayer' }),
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

  const nowMs = (): number =>
    typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
  const reducedMotionQuery =
    typeof globalThis.matchMedia === 'function' ? globalThis.matchMedia('(prefers-reduced-motion: reduce)') : null;
  // Reduced motion stills the pulse only; ants keep marching.
  const samPulseActive = (): boolean =>
    renderController.previews.getSam() !== null && reducedMotionQuery?.matches === false;
  const antsAnimator: AntsAnimator = createAntsAnimator({
    cancelFrame: (handle) => globalThis.cancelAnimationFrame(handle),
    now: nowMs,
    onFrame: () => {
      if (samPulseActive()) {
        scheduler.invalidate({ overlay: true });
      }
    },
    onStep: () => {
      antsPhase += ANTS_STEP_PX;
      scheduler.invalidate({ overlay: true });
    },
    requestFrame: (callback) => globalThis.requestAnimationFrame(callback),
  });

  // Assumes data dims equal rect dims (the decode bridge validates this).
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

  function updateAntsAnimation(): void {
    const animating = selection.hasSelection() || renderController.previews.getSam() !== null;
    if (!disposed && animating && renderController.getInputElement()) {
      antsAnimator.start();
    } else {
      antsAnimator.stop();
    }
  }

  const colorSamples = new ColorSampleController({
    endHeldTool: () => pipeline.endForToolSwitch(),
    getActiveToolId: () => interactionController.getActiveToolId(),
    setTool: (toolId) => interactionController.setTool(toolId),
  });
  const paintLineEnds = new PaintLineEndController({
    getLayer: (layerId) => getDocumentLayer(mirror.getDocument(), layerId),
  });

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
      // Land the float first so the committed op applies to the visible document.
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
    beginSelectedLayerEdit: () => beginSelectedLayerEdit(),
    reportEditRefusal,
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
    paintLineEnds,
    sampleProviders: colorSampleProviders,
    resolveColorSample: (hex, phase) => colorSamples.route(hex, phase),
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

  const applyCursorToInput = (cursor: string): void => {
    const style = renderController.getInputElement()?.style;
    if (style) {
      style.cursor = cursor;
    }
  };

  const updateCursor = (): void => {
    const cursor = activeTool()?.cursor?.(toolContext) ?? 'default';
    stores.cursor.set(cursor);
    applyCursorToInput(cursor);
  };

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

  const {
    dispose: disposeLayerRasterizer,
    getOrStartLayerRasterization,
    getFailure: getRasterizationFailure,
    isBackingOff: isRasterizationBackingOff,
  } = createLayerRasterizer({
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

  let vectorTraceAbort: AbortController | null = null;
  let vectorRuntime: VectorRuntime | null = null;
  let vectorPreparation: Promise<boolean> | null = null;
  const prepareVectors = (): Promise<boolean> => {
    if (disposed) {
      return Promise.resolve(false);
    }
    if (vectorRuntime) {
      return Promise.resolve(true);
    }
    if (vectorPreparation) {
      return vectorPreparation;
    }
    vectorPreparation = Promise.resolve()
      .then(() => (opts.loadVectorRuntime ? opts.loadVectorRuntime() : import('./vectorRuntime')))
      .then(({ createVectorRuntime }) => {
        if (disposed) {
          return false;
        }
        vectorRuntime = createVectorRuntime({
          captureInsertionAnchor: (layerId) => captureInsertionAnchor('vector', layerId),
          session: {
            get: () => stores.vectorEditSession.get(),
            set: (value) => {
              stores.vectorEditSession.set(value);
              stores.canUndo.set(history.canUndo());
              stores.canRedo.set(history.canRedo());
            },
          },
          getDocument: () => mirror.getDocument(),
          canEdit: () => canEditDocument(),
          isGestureActive: () => pipeline.isGestureActive(),
          getToolId: () => interactionController.getActiveToolId(),
          commit: (label, forward, inverse) => commitToolStructural(label, forward, inverse),
          invalidate: () => scheduler.invalidate({ overlay: true }),
          createId: createLayerId,
          reportRefusal: reportEditRefusal,
          viewport,
        });
        if (lifecycleState === 'active') {
          if (interactionController.getActiveToolId() === 'spline') {
            vectorRuntime?.editor.onActivate();
          } else {
            vectorRuntime?.editor.refreshAvailability();
          }
        }
        stores.vectorToolsStatus.set('ready');
        scheduler.invalidate({ overlay: true });
        return true;
      })
      .catch((error: unknown) => {
        if (!disposed) {
          stores.vectorToolsStatus.set('failed');
          opts.reportError({
            area: 'canvas-engine',
            namespace: 'canvas',
            projectId,
            message: 'Vector tools could not be loaded',
            context: { error },
          });
        }
        return false;
      })
      .finally(() => {
        vectorPreparation = null;
      });
    stores.vectorToolsStatus.set('loading');
    return vectorPreparation;
  };

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
    peekAdjustedCacheSurface: (layerId, surface, adjustments) =>
      rasterController.peekAdjustedCacheSurface(layerId, surface, adjustments),
    pin: (layerId) => rasterController.memory.pin(layerId),
    reserve: (bytes) =>
      rasterController.memory.reserve(bytes, { generation: lifecycleGeneration, purpose: 'raster-export' }),
    runSpatial: (request) => rasterController.spatial.run(request),
    waitForFont: fontLoader.waitForReady,
  });
  const rasterizeLayerPixels = rasterExportController.rasterize.bind(rasterExportController);
  const prepareLayerRasterCache = async (layerId: string) => {
    const result = await rasterizeLayerPixels(layerId, { includeDisabled: true });
    if (result.status === 'unreadable') {
      return result;
    }
    if (result.status !== 'ok') {
      return { status: result.status === 'over-budget' ? result.status : ('not-ready' as const) };
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
          // One failed source must not starve the rest of the batch.
        }
      }
    })();
  };
  const exportBakedLayerPixels = rasterExportController.baked.bind(rasterExportController);
  const exportBakedLayerBlob = rasterExportController.blob.bind(rasterExportController);
  type StructuralExportLayerPixelsResult =
    | Extract<ExportLayerPixelsResult, { status: 'ok' }>
    | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' }
    | LayerUnreadable
    | AdjustmentRefusal;
  const normalizeStructuralExport = async (
    result: Promise<ExportLayerPixelsResult>
  ): Promise<StructuralExportLayerPixelsResult> => {
    const resolved = await result;
    if (resolved.status === 'ok' || resolved.status === 'adjustment-refused' || resolved.status === 'unreadable') {
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
    return result === 'aborted' ? 'stale' : result === 'unavailable' ? 'error' : result;
  };

  const cropLayerToBbox = (layerId: string): Promise<CropLayerResult> => layerController.crop.crop(layerId);

  const copyLayerToRaster = (layerId: string): Promise<CopyLayerResult> => layerController.copy.copy(layerId, 'raster');
  const copyLayerToControl = (layerId: string): Promise<CopyLayerResult> =>
    layerController.copy.copy(layerId, 'control');

  /** Throws for missing or ineligible layers so exports cannot silently omit contributors. */
  const requireLayerSurfaceForExport = async (layerId: string): Promise<RasterReadLease> => {
    const result = await rasterizeLayerPixels(layerId);
    if (result.status === 'ok') {
      return result;
    }
    if (result.status === 'over-budget') {
      throw new RasterCompositeOverBudgetError();
    }
    if (result.status === 'unreadable') {
      throw new RasterCompositeUnreadableError(result);
    }
    if (result.status === 'not-ready') {
      throw new RasterCompositeNotReadyError(layerId);
    }
    throw new Error(`Cannot rasterize layer ${layerId} for export: ${result.status}.`);
  };

  const dropLayer = (layerId: string): void => {
    // Cancel persistence first: a late upload must never target a recreated layer with the same id.
    try {
      bitmapStore.discardLayer(layerId);
    } catch {
      // Keep authoritative removal cleanup observer-safe for injected stores.
    }
    rasterController.dropLayer(layerId);
    stores.thumbnailVersion.delete(layerId);
    stores.thumbnailStatus.delete(layerId);
  };

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
      renderController.releaseMaskScratch();
      clearSurface(screen);
      clearSurface(overlay);
      return;
    }

    // Overlay-only hover keeps the screen frame; upscaling layers costs more with zoom.
    const samPreview = renderController.previews.getSam();
    // Resolve float placement once so composite pixels and selection ants agree.
    const floatRender = floatingSelectionFrame(floatingSelection.get(), doc);
    if (flags.all || flags.view || flags.layers.size > 0) {
      compositeFrame.draw(screen, doc, view, floatRender, samPreview, flags.damage);
    }

    const cssSize = viewport.getViewportSize();
    const viewportSize = cssSize.width > 0 ? cssSize : { height: overlay.height / dpr, width: overlay.width / dpr };
    renderOverlay(
      overlay,
      overlayFrame.describe(doc, { dpr, view: viewport.viewMatrix(1), viewportSize }, floatRender, samPreview),
      vectorRuntime?.render
    );
  };

  const renderController = new RenderController({
    applyCursor: (value) => applyCursorToInput(value),
    backend,
    rasterMemory: rasterController.memory,
    getInputHandlers: () => ({ ...pipeline, ...wheelInput, onPointerLeave: onCanvasPointerLeave, reset: resetInput }),
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

  let gestureEditsStagingHiddenStack = false;
  let sessionEditsStagingHiddenStack = false;
  const setStagingHiddenStackEdits = (gesture: boolean, session: boolean): void => {
    const revealed = gestureEditsStagingHiddenStack || sessionEditsStagingHiddenStack;
    gestureEditsStagingHiddenStack = gesture;
    sessionEditsStagingHiddenStack = session;
    if ((gesture || session) !== revealed && renderController.previews.hidesStagingOverlays()) {
      scheduler.invalidate({ all: true });
    }
  };
  const setGestureEditsStagingHiddenStack = (edits: boolean): void =>
    setStagingHiddenStackEdits(edits, sessionEditsStagingHiddenStack);
  const syncSessionEditsStagingHiddenStack = (): void => {
    const layerId = stores.transformSession.get()?.layerId ?? floatingSelection.get()?.layerId ?? null;
    // With nothing held, no document is read: the mirror reports changes before construction completes.
    const document = layerId === null ? null : mirror.getDocument();
    setStagingHiddenStackEdits(gestureEditsStagingHiddenStack, !!document && isInStagingHiddenStack(document, layerId));
  };
  const unsubscribeTransformReveal = stores.transformSession.subscribe(syncSessionEditsStagingHiddenStack);
  const unsubscribeFloatReveal = stores.hasFloatingSelection.subscribe(syncSessionEditsStagingHiddenStack);
  const beginSelectedLayerEdit = (): void => {
    const document = mirror.getDocument();
    if (pipeline.isGestureActive() && document && isInStagingHiddenStack(document, document.selectedLayerId)) {
      setGestureEditsStagingHiddenStack(true);
    }
  };
  const onGestureChange = (active: boolean): void => {
    if (!active) {
      setGestureEditsStagingHiddenStack(false);
    } else if (activeTool()?.editsSelectedLayer?.(toolContext)) {
      beginSelectedLayerEdit();
    }
  };

  const compositeFrame = createCompositeFrame({
    backend,
    derivedSurfaceCache,
    diagnostics,
    getAdjustedSurface,
    getGroupSurface: (scope, members, matrices, content) =>
      groupSurfaces.get(scope, members, matrices, content, { frame: true }),
    getCheckerboardTile,
    getMaskPatternTile,
    getControlView: () => controlView.get(),
    isEditingStagingHiddenStack: () => gestureEditsStagingHiddenStack || sessionEditsStagingHiddenStack,
    layerCache,
    previews: renderController.previews,
    maskScratch: renderController,
    raster: rasterController,
    // A failed source waits out its retry backoff instead of failing every frame.
    rasterizeLayer: (layer, doc) => {
      if (!isRasterizationBackingOff(layer)) {
        void getOrStartLayerRasterization(layer, doc);
      }
    },
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
      const area = colorLoupeSampler.sampleArea(doc, layerCache, point, size, colorSampleProviders, screenPoint);
      return area ? { checker: getCheckerboardTile(), color: area.center, pixels: area.pixels, point } : null;
    },
    getOverlayCursor: () => overlayCursor,
    getStagedRegions: () => renderController.previews.getStagedRegions(),
    selection,
    stores,
    transformOverrides,
  });
  // Paused until attached, so the engine stays node-safe before it has render targets.
  scheduler.pause();

  const previewPublisher = createPreviewPublisher({
    decodeBlob: (blob) => rasterController.decodeBlob(blob),
    getDocument: () => mirror.getDocument(),
    invalidateAll: () => scheduler.invalidate({ all: true }),
    invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
    isGuardCurrent: (guard) => isLayerExportGuardCurrent(guard),
    previews: renderController.previews,
    publishHidesStagingOverlays: (hidden) => stores.stagingHidesOverlays.set(hidden),
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

  const controlView = new ControlViewController({
    getDocument: () => mirror.getDocument(),
    invalidateLayer: (layerId) => scheduler.invalidate({ layers: [layerId] }),
    isLocked: () => stores.documentEditingLocked.get(),
    publish: (view) => stores.controlLayerView.set(view),
    rasterize: (source, document, signal) => rasterizeSource(source, rasterizeDeps(document, signal)),
    reserve: (bytes) => rasterController.memory.reserveOperation(bytes, { purpose: 'layer-operation' }),
    trackDetached: (bytes) => rasterController.memory.trackDetached(bytes),
  });

  /** Last non-null selection, kept across deselect/reselect. */
  let lastSelectedLayerId: string | null = null;
  let toolsReady = false;
  const mirror: DocumentMirror = createDocumentMirror(mutationPort, {
    onDocumentChanged: () => {
      vectorTraceAbort?.abort();
      editingController.transform.onDocumentChanged();
      syncSessionEditsStagingHiddenStack();
      stores.documentEpoch.set(stores.documentEpoch.get() + 1);
      // Skipped until the interaction controller exists; the cursor reads it.
      if (toolsReady) {
        vectorRuntime?.editor.onDocumentChanged();
        tools.get('shape')?.onDocumentChanged?.(toolContext);
        updateCursor();
      }
    },
    // Staged previews sit at the rect pinned at submission, so a bbox change redraws only the overlay.
    onBboxChanged: () => scheduler.invalidate({ overlay: true }),
    onDocumentReplaced: () => {
      const cleanup = createCleanupAccumulator();
      cleanup.run(() => renderController.releaseMaskScratch());
      cleanup.run(() => vectorRuntime?.editor.discard());
      cleanup.run(() => groupSurfaces.clear());
      cleanup.run(() => editingController.invalidateDocument());
      cleanup.run(() => pipeline.cancelActiveGesture());
      cleanup.run(cancelOpenPixelEdit);
      cleanup.run(() => rasterController.invalidateDocument());
      cleanup.run(() => stores.thumbnailStatus.clear());
      // Whole-document swaps invalidate pixel history; cancel the gesture first to drop stale anchors.
      cleanup.run(() => stores.bboxPreview.set(null));
      cleanup.run(() => history.clear());
      paintLineEnds.clear();
      cleanup.run(() => editingController.transform.onDocumentReplaced());
      cleanup.run(() => transformOverrides.clear());
      cleanup.run(() => stores.textEditSession.set(null));
      cleanup.run(clearStagedPreview);
      // A swap can reuse a layer id with different content, so drop every filter preview.
      cleanup.run(clearAllFilterPreviews);
      cleanup.run(() => controlView.clear());
      // Committing an outgoing float would target a replaced layer.
      cleanup.run(() => floatingSelection.cancel());
      cleanup.run(() => structuralController.dropPreview());
      cleanup.run(() => editingController.discardSelection());
      cleanup.run(() => stores.lassoPreview.set(null));
      cleanup.run(() => tools.get('shape')?.onPointerCancel?.(toolContext));
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
      // Reused ids may still hold outgoing pixels, so invalidate every incoming cache.
      for (const layerId of present) {
        cleanup.run(() => invalidateLayerCache(layerId));
      }
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
      retainSpatialSurfaces();
      scheduler.invalidate({ layers: ids });
    },
    onLayersChanged: (ids, sourceChangedIds, restructured) => {
      const cleanup = createCleanupAccumulator();
      // A group deleted with its leaves reports here, not onLayerOrderChanged; value edits remove no group.
      if (restructured) {
        const doc = mirror.getDocument();
        cleanup.run(() => groupSurfaces.prune(doc ? new Set(getDocumentIndex(doc).byId.keys()) : new Set()));
      }
      cleanup.run(retainSpatialSurfaces);
      const floatLayerId = floatingSelection.get()?.layerId;
      if (floatLayerId && ids.includes(floatLayerId)) {
        // Drop the float instead of baking pixels into a layer whose content changed underneath.
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
      const session = stores.transformSession.get();
      const textSession = stores.textEditSession.get();
      for (const id of ids) {
        paintLineEnds.onLayerChanged(id);
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
          // Advance decode tokens so late results cannot revive stale previews.
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
      cleanup.run(() => controlView.reconcile());
      cleanup.run(() => scheduler.invalidate({ layers: ids }));
      cleanup.throwIfFailed();
    },
    /** Panel selection is authoritative; reconcile sessions and chrome without dispatch. */
    onSelectionChanged: (selectedLayerId) => {
      const cleanup = createCleanupAccumulator();
      if (selectedLayerId !== null && selectedLayerId !== lastSelectedLayerId) {
        // Keep the surviving previous choice instead of a layer the deletion or merge removed.
        if (
          engineMutationDepth === 0 &&
          lastSelectedLayerId !== null &&
          getDocumentNode(mirror.getDocument(), lastSelectedLayerId)
        ) {
          stores.layerQuickSwitch.set({ ...stores.layerQuickSwitch.get(), previousLayerId: lastSelectedLayerId });
        }
        // The next explicit switch returns to the layer actually left, including fallbacks.
        lastSelectedLayerId = selectedLayerId;
      }
      cleanup.run(() => floatingSelection.commit());
      cleanup.run(() => controlView.clear((id) => id !== selectedLayerId));
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
  });
  lastSelectedLayerId = mirror.getDocument()?.selectedLayerId ?? null;

  let fontSourceStacks: CanvasDocumentContractV3['stacks'] | null = null;
  let documentFontSources: CanvasTextSource[] = [];
  let draftFontSource: CanvasTextSource | undefined;
  let defaultsFontOptions: TextToolOptions | undefined;
  let defaultsFontSource: CanvasTextSource | undefined;
  syncActiveFontSources = () => {
    const document = mutationPort.getCanvasState()?.document;
    const stacks = document?.stacks ?? null;
    const draft = document ? stores.textEditSession.get()?.source : undefined;
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

  // Leaving the project invalidates guarded filter previews, so returning cannot resurrect them.
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
      cleanup.run(() => controlView.clear());
      projectWasPresent = false;
      cleanup.throwIfFailed();
      return;
    }
    projectWasPresent = projectIsPresent;
  });

  // Suppressed during synchronous resize repaint to avoid a second identical composite.
  let suppressViewportInvalidate = false;

  const unsubscribeViewport = viewport.subscribe(() => {
    stores.zoom.set(viewport.getZoom());
    vectorRuntime?.editor.refreshAvailability();
    if (!suppressViewportInvalidate) {
      scheduler.invalidate({ view: true });
    }
  });

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
  const unsubscribeControlViewLock = stores.documentEditingLocked.subscribe(() => {
    if (stores.documentEditingLocked.get()) {
      controlView.clear();
    }
  });
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
  const unsubscribeShowBbox = stores.showBbox.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeBboxOverlay = stores.bboxOverlay.subscribe(() => scheduler.invalidate({ overlay: true }));
  const unsubscribeRuleOfThirds = stores.ruleOfThirds.subscribe(() => scheduler.invalidate({ overlay: true }));

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
      // Real switches bank a float; a temporary modifier hold (space to pan) must not.
      if (!switchOptions?.temporary) {
        floatingSelection.commit();
        editingController.transform.onToolSwitch(to);
      }
      for (const listener of toolChangeListeners) {
        listener({ from, temporary: switchOptions?.temporary === true, to });
      }
    },
    cancelGesture: () => pipeline.endForToolSwitch(),
    getTool: (toolId) => tools.get(toolId),
    getToolContext: () => toolContext,
    invalidateOverlay: () => scheduler.invalidate({ overlay: true }),
    publishActiveTool: (toolId) => stores.activeTool.set(toolId),
    stepBrushSize: stepActiveBrushSize,
    updateCursor,
  });
  toolsReady = true;
  const unsubscribeShapeOptions = stores.shapeOptions.subscribe(() => {
    if (interactionController.getActiveToolId() === 'shape') {
      updateCursor();
    }
  });
  let toolRequest = 0;
  const setTool = (toolId: ToolId, options?: { temporary?: boolean }): void => {
    const request = ++toolRequest;
    if (toolId === 'spline' && !vectorRuntime) {
      const generation = vectorWorkEpoch;
      void prepareVectors().then((ready) => {
        if (
          ready &&
          !disposed &&
          request === toolRequest &&
          vectorWorkEpoch === generation &&
          lifecycleState === 'active'
        ) {
          setTool(toolId, options);
        }
      });
      return;
    }
    colorSamples.onToolChange(toolId);
    interactionController.setTool(toolId, options);
  };

  /** Escape order: text edit, transform, float, tool session, operation, then selection. */
  const handleEscapePriority = ({ gestureWasActive }: { gestureWasActive: boolean }): void => {
    if (vectorTraceAbort) {
      vectorTraceAbort.abort();
      return;
    }
    if (tools.get('shape')?.hasSession?.()) {
      tools.get('shape')?.onKeyCommand?.(toolContext, 'cancel');
      return;
    }
    if (vectorRuntime?.editor.hasSession()) {
      vectorRuntime?.editor.onKeyCommand(toolContext, 'cancel');
      return;
    }
    if (colorSamples.onEscape()) {
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
    if (editingController.transform.onEscape()) {
      return;
    }
    if (floatingSelection.has()) {
      floatingSelection.cancel();
      return;
    }
    // The tool the user is really on: a Space-held pan mid-polygon keeps the polygon's session on the held tool.
    const sessionTool = tools.get(pipeline.heldRestoreToolId() ?? interactionController.getActiveToolId());
    if (sessionTool?.onKeyCommand?.(toolContext, 'cancel')) {
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
    getKeyOwnerResolver: () => renderController.getKeyOwnerResolver(),
    getToolContext: () => toolContext,
    handleEscape: handleEscapePriority,
    hasFocus: () => typeof document === 'undefined' || document.hasFocus(),
    hasTool: (id) => tools.has(id),
    isReplaying: () => history.isReplaying(),
    maybeCommitModalSession: () => commitOpenTextSession(),
    onGestureChange,
    setTool: (id, opts) => setTool(id, opts),
    updateCursor,
    viewport,
  });

  // A view change under a still pointer moves the document beneath it; hover-derived tool state must follow.
  const unsubscribeHoverRefresh = viewport.subscribe(() => pipeline.refreshHover());

  const wheelInput = createWheelInput({
    getActiveTool: activeTool,
    getControlKeyState: () => pipeline.controlKeyState(),
    getInputElement: () => renderController.getInputElement(),
    getInvertBrushSizeScroll: () => stores.invertBrushSizeScroll.get(),
    getScrollInput: () => stores.scrollInput.get(),
    invalidate: (payload) => scheduler.invalidate(payload),
    isGestureActive: () => pipeline.isGestureActive(),
    onWheelWithoutControl: () => pipeline.onWheelWithoutControl(),
    stepActiveBrushSize,
    trackpadDetected: stores.trackpadDetected,
    viewport,
  });

  // Lifecycle flushes are best-effort; invoke and export await the real barrier.
  const kickPendingFlush = (): void => {
    // Failures already report through BitmapStore.onError; never leave an unhandled rejection.
    void persistenceController.flush().catch(() => undefined);
  };
  const onPageHide = (): void => {
    kickPendingFlush();
  };
  const onVisibilityChange = (): void => {
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
      kickPendingFlush();
      // Key releases while hidden never arrive; held keys (Space, Alt, Control) must not outlive the page.
      onWindowBlur();
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

  const attach = (
    screenCanvas: HTMLCanvasElement,
    overlayCanvas: HTMLCanvasElement,
    keyboardRoot?: HTMLElement,
    options?: CanvasSurfaceAttachOptions
  ): void => renderController.attach(screenCanvas, overlayCanvas, keyboardRoot, options);
  const detach = (): void => renderController.detach();

  const activate = (): void => {
    if (disposed) {
      return;
    }
    rasterController.memory.releaseGeneration(lifecycleGeneration);
    lifecycleGeneration += 1;
    lifecycleState = 'active';
    editingController.activate();
    vectorRuntime?.editor.onDocumentChanged();
    cooldownPromise = null;
  };

  const beginCooldown = (): Promise<'cooled' | 'dirty'> => {
    vectorWorkEpoch += 1;
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
    vectorTraceAbort?.abort();
    vectorRuntime?.editor.discard();
    editingController.cooldown();
    clearStagedPreview();
    clearStagedPreviewCache();
    controlView.clear();
    detach();
    cancelAllLayerRasterizations();
    rasterController.cancelSpatialAdjustments();
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
          // Clear the completed attempt so a zero-reference registry entry can retry persistence.
          cooldownPromise = null;
        }
        return 'dirty';
      }
    );
    return cooldownPromise;
  };

  const resize = (cssWidth: number, cssHeight: number, dpr: number): void => {
    suppressViewportInvalidate = true;
    viewport.setViewportSize(cssWidth, cssHeight, dpr);
    suppressViewportInvalidate = false;
    // The viewport owns the clamped ratio so the backing size and every view matrix agree.
    const backingDpr = viewport.getDpr();
    const backingWidth = Math.round(cssWidth * backingDpr);
    const backingHeight = Math.round(cssHeight * backingDpr);
    renderController.resize(backingWidth, backingHeight);
    // Resize clears canvas pixels; recomposite in the same task to avoid a blank frame.
    render({ all: true, damage: FULL_DAMAGE, layers: new Set<string>(), overlay: true, view: true });
  };

  let hasEverFitToView = false;

  /** The bbox joined with every enabled renderable layer's bounds; shared by fit-to-view and the overview. */
  const getDocumentExtent = (doc: CanvasDocumentContractV3): Rect => {
    let extent: Rect = { ...doc.bbox };
    for (const leaf of compileDocumentLeaves(doc)) {
      if (leaf.contributionEnabled && isRenderableLayer(leaf.layer)) {
        extent = union(extent, getSourceBounds(leaf.layer, doc));
      }
    }
    return extent;
  };

  const fitToView = (): void => {
    const doc = mirror.getDocument();
    if (!doc) {
      return;
    }
    viewport.fitToView(getDocumentExtent(doc), viewport.getViewportSize());
    hasEverFitToView = true;
  };

  /** Fit once per engine; kept-alive widgets reattach and keep their zoom and pan. */
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
  /** An `unavailable` image retries at once on request; an `unreadable` one only after its backoff. */
  const layerCacheReadinessForOp = (layer: CanvasLayerContract, doc: CanvasDocumentContractV3): LayerCacheReadiness => {
    if (isLayerCacheReadyForOp(layer, doc)) {
      return 'ready';
    }
    const failure = isCurrentRasterizationJob(layer) ? null : getRasterizationFailure(layer);
    if (!failure) {
      return 'not-ready';
    }
    if (failure === 'unavailable' || !isRasterizationBackingOff(layer)) {
      scheduleLayerRasterization([layer.id]);
    }
    return failure === 'unreadable' ? layerUnreadable(doc, layer.id) : 'not-ready';
  };

  const prepareGeneratedPaintCache = (layerId: string, rect: Rect, pixels: RasterSurface) =>
    layerCache.prepareReplacement(layerId, rect, pixels);

  const installGeneratedPaintCache = (
    prepared: ReturnType<LayerCacheStore['prepareReplacement']>,
    persist = true
  ): void => {
    const { layerId } = prepared;
    const target = layerCache.installReplacement(prepared);

    // After the commit, hooks notify best-effort; their failures cannot veto the transaction.
    const notifyBestEffort = (notify: () => void): void => {
      try {
        notify();
      } catch {
        // Document and cache already converged; later work can retry.
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
    getAdjustmentStatus: (layerId) => stores.adjustmentStatus.get(layerId),
    peekAdjustedSurface: (layer, entry) => rasterController.peekAdjustedSurface(layer, entry),
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
  const pixelEdits = pixelEditController;
  const beginPixelEdit: PixelEditController['begin'] = (layerId, options) => {
    const transaction = pixelEdits.begin(layerId, options);
    if (transaction) {
      controlView.clear((id) => id === layerId);
    }
    return transaction;
  };

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
      // A pending clear makes the cache the pixel authority; the duplicate starts empty, not from the old bitmap.
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
    backend,
    captureCache: captureLayerCache,
    ctx: mutationContext,
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled: true }),
    getMainModelBase,
    getDuplicateRasterPlan,
    getSelectedLayerIds: resolveSelectedLayerIds,
    hasPendingPixelWork: (layerId) => bitmapStore.hasPendingWork(layerId),
    isGuardCurrent: isLayerExportGuardCurrent,
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
  const convertLayerToControl = layerMutationController.convertToControl.bind(layerMutationController);
  const duplicateLayers = layerMutationController.duplicate.bind(layerMutationController);

  const replaceSelectionFromImage = editingController.selectionImage.replace.bind(editingController.selectionImage);

  const maskResultController = new MaskResultController({ ctx: mutationContext });
  const commitMaskImageResult = maskResultController.commit.bind(maskResultController);

  const filterResultController = new FilterResultController({
    captureCache: captureLayerCache,
    ctx: mutationContext,
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    getMainModelBase,
    needsPixelPersistence: layerNeedsPixelPersistence,
  });
  const commitRasterFilterResult = filterResultController.commit.bind(filterResultController);

  const controlDerivationController = new ControlDerivationController({
    captureCache: captureLayerCache,
    ctx: mutationContext,
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    getDocumentRevision: () => mutationPort.getCanvasState()?.documentRevision ?? null,
    hasPendingPixelWork: (layerId) => bitmapStore.hasPendingWork(layerId),
    isLayerSessionHeld: isLayerHeldBySession,
  });

  const generatedResultController = new GeneratedResultController({
    captureCache: captureLayerCache,
    clearPreview: clearFilterPreview,
    ctx: mutationContext,
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    getMainModelBase,
    needsPixelPersistence: layerNeedsPixelPersistence,
  });
  const commitGeneratedImageResult = generatedResultController.commit.bind(generatedResultController);

  const stagedResultController = new StagedResultController({
    createEventId,
    ctx: mutationContext,
    getCanvasState: () => mutationPort.getCanvasState(),
    now: () => new Date().toISOString(),
    settleEditSessions: () => {
      if (floatingSelection.has()) {
        floatingSelection.commit();
      }
      if (stores.transformSession.get()) {
        editingController.transform.apply();
      }
      return !floatingSelection.has() && stores.transformSession.get() === null;
    },
  });
  const commitStagedImage = stagedResultController.commit.bind(stagedResultController);

  const booleanMergeRasterLayers = (
    upperLayerId: string,
    operation: BooleanRasterOperation
  ): Promise<BooleanRasterResult> => layerController.booleanMerge.merge(upperLayerId, operation);

  const extractMaskedArea = (maskLayerId: string): Promise<ExtractMaskedAreaResult> =>
    layerController.extractMaskedArea.extract(maskLayerId);

  const mergeLayerDown = (upperLayerId: string): Promise<MergeDownResult> =>
    layerController.merge.mergeDown(upperLayerId);
  const mergeSelectedRasterLayers = (layerIds: readonly string[]): Promise<MergeVisibleResult> =>
    layerController.merge.mergeSelected(layerIds);
  const mergeVisibleRasterLayers = (): Promise<MergeVisibleResult> => layerController.merge.mergeVisible();
  const mergeInpaintMasks = (layerIds: readonly string[]): Promise<MergeInpaintMasksResult> =>
    layerController.merge.mergeInpaintMasks(layerIds);
  const cropAllLayersToBbox = (): Promise<CropAllLayersResult> => layerController.crop.cropAll();

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
    onStatus: (status) => stores.psdExportStatus.set(status),
    captureDocumentSnapshot,
    captureRasterSnapshot,
    getAvailableBytes: () => rasterController.memory.getAvailableBytes(),
    isDocumentSnapshotCurrent,
    reserve: (bytes) => rasterController.memory.reserveOperation(bytes, { purpose: 'psd-export' }),
    runSpatial: (request) => rasterController.spatial.run(request),
  });
  const exportRasterLayersToPsd = async (fileName: string): Promise<PsdExportResult> => {
    try {
      return await psdExportController.export(fileName);
    } catch (error) {
      opts.reportError({
        namespace: 'canvas',
        area: 'canvas-engine',
        projectId,
        message: 'PSD export failed',
        context: { error },
      });
      throw error;
    }
  };

  const rasterizeLayer = (layerId: string): Promise<RasterizeLayerResult> =>
    layerController.rasterize.rasterize(layerId);

  const beginTransformSession = (layerId: string): void => editingController.transform.begin(layerId);
  const updateTransformSession = (transform: LayerTransform): void => editingController.transform.update(transform);
  // A framed float is the session; only without one do these reach the layer transform session.
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

  const setTextEditContentReader = (reader: (() => string) | null): void =>
    editingController.text.setContentReader(reader);
  const openTextCreate = (point: Vec2): void => editingController.text.openCreate(point);
  // The pane styles the layer after blur commits, so opening an edit also selects it.
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

  // Commit live floats first so selection ops see displayed pixels.

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
  const admitSelectionErase = (): boolean => editingController.selectionPixels.admitErase();

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
    vectorWorkEpoch += 1;
    paintLineEnds.dispose();
    disposeLayerRasterizer();
    releaseActiveSnapshots();
    rasterController.memory.releaseGeneration(lifecycleGeneration);
    lifecycleGeneration += 1;
    lifecycleState = 'disposed';
    const cleanup = createCleanupAccumulator();
    cleanup.run(() => pipeline.cancelActiveGesture());
    cleanup.run(cancelOpenPixelEdit);
    cleanup.run(() => pixelEditController?.dispose());
    cleanup.run(() => filterResultController.dispose());
    cleanup.run(() => controlDerivationController.dispose());
    cleanup.run(() => generatedResultController.dispose());
    cleanup.run(() => stagedResultController.dispose());
    cleanup.run(() => editingController.dispose());
    cleanup.run(() => vectorTraceAbort?.abort());
    cleanup.run(() => vectorRuntime?.editor.discard());
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
    cleanup.run(() => colorSamples.dispose());
    cleanup.run(() => activeTool()?.onDeactivate?.(toolContext));
    cleanup.run(unsubscribeViewport);
    cleanup.run(unsubscribeHoverRefresh);
    cleanup.run(unsubscribeBrushOptions);
    cleanup.run(unsubscribeEraserOptions);
    cleanup.run(unsubscribeCheckerboard);
    cleanup.run(unsubscribeTransformReveal);
    cleanup.run(unsubscribeFloatReveal);
    cleanup.run(unsubscribeControlViewLock);
    cleanup.run(unsubscribeCheckerColors);
    cleanup.run(unsubscribeShowGrid);
    cleanup.run(unsubscribeBboxGrid);
    cleanup.run(unsubscribeShowBbox);
    cleanup.run(unsubscribeBboxOverlay);
    cleanup.run(unsubscribeRuleOfThirds);
    cleanup.run(unsubscribeShapeOptions);
    cleanup.run(unsubscribeProjectPreviewLifecycle);
    cleanup.run(unsubscribeTextFontSources);
    cleanup.run(unsubscribeTextDefaultsFontSources);
    cleanup.run(() => mutationContext.dispose());
    cleanup.run(unsubscribeHistoryEpoch);
    cleanup.run(() => historyController.dispose());
    cleanup.run(() => persistenceController.dispose());
    cleanup.run(() => mirror.dispose());
    cleanup.run(() => previewPublisher.dispose());
    cleanup.run(() => controlView.dispose());
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
    // Flush first, or rerasterizing the older persisted source erases unuploaded strokes.
    await persistenceController.flush();
    const doc = mirror.getDocument();
    for (const layer of getDocumentLeaves(doc)) {
      invalidateLayerCache(layer.id);
      deleteDerivedSurfaces(layer.id);
    }
    checkerboardTile = null;
    maskPatternTiles.clear();
    renderController.releaseMaskScratch();
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

  /** Right-click belongs to gestures, transforms and text editing, so menus are suppressed then. */
  const canTargetLayerFromContextMenu = (): boolean =>
    !pipeline.isGestureActive() &&
    !vectorRuntime?.editor.hasSession() &&
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
    runSpatial: (request) => rasterController.spatial.run(request),
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
      // Never builds the display's copy, and reuses only an exact one: a miss adjusts what the export reserved.
      adjustedSurface: (layerId, surface, adjustments) =>
        rasterController.peekAdjustedCacheSurface(layerId, surface, adjustments),
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
      runSpatial: (request) => rasterController.spatial.run(request),
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
      cacheReadiness: layerCacheReadinessForOp,
      isGuardCurrent: isLayerExportGuardCurrent,
    },
    crop: {
      backend,
      captureCache: captureLayerCache,
      ctx: mutationContext,
      discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
      exportBaked: (layerId, options) =>
        exportBakedLayerPixelsForStructural(layerId, { ...options, includeDisabled: true }),
      // Held pixels belong to an open edit; crop all must not wait for that edit to end.
      flushPendingPixels: () => persistenceController.flush({ waitForHeldPixels: false }),
      isGuardCurrent: isLayerExportGuardCurrent,
      isSupportedSource: isSupportedExportSource,
    },
    copy: {
      backend,
      commitLayerCopy,
      ctx: mutationContext,
      exportBaked: (layerId) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled: true }),
      getMainModelBase,
      isGuardCurrent: isLayerExportGuardCurrent,
    },
    extractMaskedArea: {
      backend,
      ctx: mutationContext,
      exportBaked: (layerId, includeDisabled) => exportBakedLayerPixelsForStructural(layerId, { includeDisabled }),
      hasExportableContent: hasExportableLayerContent,
      cacheReadiness: layerCacheReadinessForOp,
      isGuardCurrent: isLayerExportGuardCurrent,
      layers: layerCache,
      peekAdjustedCacheSurface: (layerId, surface, adjustments) =>
        rasterController.peekAdjustedCacheSurface(layerId, surface, adjustments),
      rasterize: (layerId) => rasterizeLayerPixelsForStructural(layerId),
      runSpatial: (request) => rasterController.spatial.run(request),
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
      cacheReadiness: layerCacheReadinessForOp,
      layers: layerCache,
      needsPixelPersistence: layerNeedsPixelPersistence,
      peekAdjustedSurface: (layer, entry) => rasterController.peekAdjustedSurface(layer, entry),
      publishSelectedLayerIds: (primaryId, selectedIds) => opts.setSelectedLayerIds?.(primaryId, selectedIds),
      runSpatial: (request) => rasterController.spatial.run(request),
    },
    rasterize: {
      backend,
      ctx: mutationContext,
      rasterizeDeps: (document) => rasterizeDeps(document),
      runSpatial: (request) => rasterController.spatial.run(request),
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
      peekDisplayedSurface: (layer, entry) => rasterController.peekDisplayedAdjustedSurface(layer, entry),
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
    cancelPsdExport: () => psdExportController.cancel(),
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
    restoreSnapshot: (snapshotId, keepCurrentAs) =>
      dispatchCanvasMutation(
        { snapshotId, type: 'restoreCanvasSnapshot', ...(keepCurrentAs ? { keepCurrentAs } : {}) },
        'user'
      ),
  };
  const selectionCapability: CanvasEngineSelectionCapability = {
    deselect,
    admitSelectionErase,
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
    setTool,
    canTargetLayerFromContextMenu,
    handleEscapePriority,
    onEditRefused,
    onStrokeCommitted,
    requestColorSample: () => colorSamples.requestSample(),
    setColorSampleRouter: (router) => colorSamples.setRouter(router),
  };
  const runVectorTrace = async (layerId: string, mode: 'trace' | 'fill' | 'appearance', pathId?: string) => {
    vectorTraceAbort?.abort();
    const controller = new AbortController();
    vectorTraceAbort = controller;
    const document = mirror.getDocument();
    try {
      if (!document) {
        return { status: 'not-ready' as const };
      }
      if (!(await prepareVectors()) || !vectorRuntime) {
        return { status: 'not-ready' as const };
      }
      return await vectorRuntime.trace({
        document,
        layerId,
        pathId,
        appearance: mode === 'appearance',
        fillOnly: mode === 'fill',
        ctx: mutationContext,
        backend,
        id: createLayerId(),
        anchor: captureInsertionAnchor('raster', null),
        signal: controller.signal,
        isCurrent: () => !disposed && mirror.getDocument() === document,
      });
    } finally {
      if (vectorTraceAbort === controller) {
        vectorTraceAbort = null;
      }
    }
  };
  const layersCapability: CanvasEngineLayerCapability = {
    ...layerController.layers,
    traceVectorLayer: (layerId, pathId) => runVectorTrace(layerId, 'trace', pathId),
    fillVectorPath: (layerId, pathId) => runVectorTrace(layerId, 'fill', pathId),
    traceFillVectorLayer: (layerId) => runVectorTrace(layerId, 'appearance'),
    cancelVectorTrace: () => vectorTraceAbort?.abort(),
    vectorEdit: {
      prepare: prepareVectors,
      createLayer: async (input) => {
        const document = mirror.getDocument();
        const generation = vectorWorkEpoch;
        if (!document) {
          return null;
        }
        if (
          !(await prepareVectors()) ||
          !vectorRuntime ||
          disposed ||
          lifecycleState !== 'active' ||
          vectorWorkEpoch !== generation ||
          mirror.getDocument() !== document
        ) {
          return null;
        }
        return vectorRuntime.createLayer({
          ...input,
          name:
            input.name || vectorRuntime.nextLayerName(getDocumentIndex(document).nodes.map((entry) => entry.node.name)),
        });
      },
      setPointType: (type) => vectorRuntime?.editor.setPointType(type),
      targetContextMenu: (point) => vectorRuntime?.editor.targetContextMenu(point) ?? false,
      deletePath: () => vectorRuntime?.editor.deletePath(),
      nudge: (dx, dy) => vectorRuntime?.editor.nudge(dx, dy),
      smooth: (selectedOnly) => vectorRuntime?.editor.smooth(selectedOnly),
      deleteSelection: () => vectorRuntime?.editor.deleteSelection(),
      copyPath: () => vectorRuntime?.editor.copyPath(),
      extractPath: () => vectorRuntime?.editor.extractPath(),
      transformPath: () => vectorRuntime?.editor.transformPath(),
      split: () => vectorRuntime?.editor.split(),
      join: () => vectorRuntime?.editor.join(),
      canSplit: () => vectorRuntime?.editor.canSplit() ?? false,
      canJoin: () => vectorRuntime?.editor.canJoin() ?? false,
    },
    applyTransform,
    beginTransform: (layerId) => editingController.transform.request(layerId),
    flipLayer: (layerId, direction, label) => editingController.transform.flip(layerId, direction, label),
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
    convertLayerToControl,
    copyLayerToControl,
    copyLayerToRaster,
    cropAllLayersToBbox,
    cropLayerToBbox,
    duplicateLayers,
    mergeInpaintMasks,
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
    if (!doc || !ctx) {
      return null;
    }
    const extent = getDocumentExtent(doc);
    if (isEmpty(extent)) {
      return null;
    }
    // Request each missing cache once per document; never pin against budget eviction.
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
    const size = fitThumbnailSize(extent.width, extent.height, maxSizePx);
    target.width = size.width;
    target.height = size.height;
    const scale = size.width / extent.width;
    // Only `ctx`/`width`/`height` are read by the compositor; the element is not resized through the surface seam.
    const surfaceLike = { canvas: target, ctx, height: size.height, width: size.width } as unknown as RasterSurface;
    compositeDocument(
      surfaceLike,
      doc,
      layerCache,
      { a: scale, b: 0, c: 0, d: scale, e: -extent.x * scale, f: -extent.y * scale },
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
    return extent;
  };
  const previewCapability: CanvasEnginePreviewCapability = {
    ...layerController.previews,
    drawDocumentOverview,
    preloadStagedPreview,
    setControlLayerView: (layerId, view) => controlView.set(layerId, view),
    setGuardedFilterPreview,
    setStagedPreview,
    setStagedRegions: (regions) => {
      if (renderController.previews.setStagedRegions(regions)) {
        scheduler.invalidate({ overlay: true });
      }
    },
  };
  const diagnosticsCapability: CanvasDiagnosticsCapability = {
    clearCaches,
    getDiagnostics: diagnostics.snapshot,
    logDebugInfo,
  };

  const applicationHost: CanvasApplicationHost = {
    captureControlDetectionGuard: (layerId) => controlDerivationController.captureControlDetectionGuard(layerId),
    captureGuard: captureCurrentLayerExportGuard,
    clearFilterPreview,
    clearSamPreview,
    commitControlLayerPatch: (options) => controlDerivationController.commitControlLayerPatch(options),
    commitControlMap: (options) => controlDerivationController.commitControlMap(options),
    commitFilter: (options) => commitRasterFilterResult(options, documentEditOwner),
    commitGenerated: (options) => commitGeneratedImageResult(options, documentEditOwner),
    commitMask: (options) => commitMaskImageResult(options, documentEditOwner),
    decodeSelectObjectPreview,
    encodeSurface: (surface) => backend.encodeSurface(surface, 'image/png'),
    exportBakedLayerBlob: (layerId) => exportBakedLayerBlob(layerId, { includeDisabled: true }),
    exportLayerPixels: rasterizeLayerPixelsForStructural,
    flushLayerUploads: (layerId, options) => persistenceController.flushLayer(layerId, options),
    getCompositeExecutorDeps,
    getDocument: () => mirror.getDocument(),
    isGuardCurrent: isLayerExportGuardCurrent,
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
