/**
 * `CanvasEngine`: the per-project imperative heart of the canvas.
 *
 * One engine instance exists per project (managed by `engineRegistry.ts`) and
 * shared by the canvas and layers widgets. It owns interaction, pixels, and
 * rendering; the reducer owns the document. The document flows in one way
 * through the {@link createDocumentMirror | document mirror}; the engine never
 * dispatches from pointer input except for the single gesture-start
 * `addCanvasLayer` a paint tool emits when auto-creating a layer.
 *
 * Responsibilities:
 * - Hold the {@link Viewport} (pan/zoom) and the transient {@link EngineStores}
 *   React subscribes to.
 * - `attach`/`detach` bind two stacked canvases (composited document + overlay)
 *   as render targets, wire pointer/wheel/key listeners, and drive the
 *   scheduler; `resize` sizes the backing stores to `css × min(dpr, 2)`.
 * - Route normalized pointer/wheel input to the active {@link Tool} via the
 *   pointer pipeline and wheel handler (middle-mouse pan, space/alt temp tools,
 *   plain-wheel zoom, ctrl+wheel brush-size step), and relay committed strokes
 *   to `onStrokeCommitted` subscribers.
 * - Orchestrate rasterization: ensure a raster cache per visible layer, compose
 *   the document, and draw the overlay each scheduled frame.
 *
 * DOM APIs are confined to `attach`/`detach`/`resize` and the DOM raster
 * backend, so importing this module stays node-safe. Zero React imports.
 */

import type {
  CanvasHistoryCapability,
  CanvasEditCapability,
  CanvasCompositeExecutorDeps,
  CanvasDocumentCapability,
  CanvasExportCapability,
  CanvasLifecycleCapability,
  CanvasLayerCapability,
  CanvasPreviewCapability,
  CanvasSelectionCapability,
  CanvasSurfaceCapability,
  CanvasToolCapability,
  CanvasViewportCapability,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  LayerExportGuard,
} from '@workbench/canvas-engine/api';
export type {
  CommitGeneratedImageOptions,
  CommitGeneratedImageResult,
  ExportBakedLayerBlobResult,
  ExportBakedLayerPixelsOptions,
  ExportLayerPixelsOptions,
  GeneratedImageTarget,
  LayerExportGuard,
  LayerThumbnailRequestResult,
  ReplaceSelectionFromImageResult,
} from '@workbench/canvas-engine/api';
export type {
  CommitMaskImageResult,
  CommitMaskImageResultOptions,
  MaskImageResultTarget,
} from '@workbench/canvas-engine/controllers/maskResultController';
export type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
} from '@workbench/canvas-engine/controllers/filterResultController';
import type { CanvasApplicationHost, SelectObjectStartContext } from '@workbench/canvas-engine/applicationHost';
import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { FontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import type { OverlayCursor } from '@workbench/canvas-engine/render/overlayRenderer';
import type { RenderScheduler } from '@workbench/canvas-engine/render/scheduler';
import type { SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { Rect, RenderFlags, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type {
  CanvasImageRef,
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  WorkbenchState,
} from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { ControlPixelController } from '@workbench/canvas-engine/controllers/controlPixelController';
import { EditingController } from '@workbench/canvas-engine/controllers/editingController';
import {
  FilterResultController,
  type CommitRasterFilterOptions,
  type CommitRasterFilterResult,
} from '@workbench/canvas-engine/controllers/filterResultController';
import { GeneratedResultController } from '@workbench/canvas-engine/controllers/generatedResultController';
import { HistoryController } from '@workbench/canvas-engine/controllers/historyController';
import { InteractionController } from '@workbench/canvas-engine/controllers/interactionController';
import { LayerController } from '@workbench/canvas-engine/controllers/layerController';
import { LayerMutationController } from '@workbench/canvas-engine/controllers/layerMutationController';
import {
  MaskResultController,
  type CommitMaskImageResult,
  type CommitMaskImageResultOptions,
} from '@workbench/canvas-engine/controllers/maskResultController';
import { PersistenceController } from '@workbench/canvas-engine/controllers/persistenceController';
import { PsdExportController } from '@workbench/canvas-engine/controllers/psdExportController';
import { RasterController, type RasterizationJob } from '@workbench/canvas-engine/controllers/rasterController';
import { RasterExportController } from '@workbench/canvas-engine/controllers/rasterExportController';
import { RenderController } from '@workbench/canvas-engine/controllers/renderController';
import { StructuralLayerController } from '@workbench/canvas-engine/controllers/structuralLayerController';
import { createCanvasDiagnostics, type CanvasDiagnosticsSnapshot } from '@workbench/canvas-engine/diagnostics';
import { createEngineStores, type EngineStores, type TextToolOptions } from '@workbench/canvas-engine/engineStores';
import { LayerFilterOutputDimensionError } from '@workbench/canvas-engine/filterError';
import { createPointerPipeline, type PointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createWheelHandler } from '@workbench/canvas-engine/input/wheel';
import { fromTRS } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import {
  compositeDocument,
  createCheckerboardTile,
  shouldSmoothAtZoom,
} from '@workbench/canvas-engine/render/compositor';
import { createFontLoader, domFontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import {
  DEFAULT_CACHE_BUDGET_BYTES,
  type LayerCacheEntry,
  type LayerCacheStore,
} from '@workbench/canvas-engine/render/layerCache';
import { createMaskPatternTile } from '@workbench/canvas-engine/render/maskFill';
import { renderOverlay } from '@workbench/canvas-engine/render/overlayRenderer';
import { createDomRasterBackend, type RasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import { rasterizeSource, type ImageResolver, type RasterizeDeps } from '@workbench/canvas-engine/render/rasterizers';
import { textFontString } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';
import { enforceSurfaceBudget } from '@workbench/canvas-engine/render/surfaceBudget';
import { getLayerThumbnailDisplayKey } from '@workbench/canvas-engine/render/thumbnail';
import { ANTS_STEP_PX, createAntsAnimator, type AntsAnimator } from '@workbench/canvas-engine/selection/marchingAnts';
import { createBboxTool } from '@workbench/canvas-engine/tools/bboxTool';
import { createBrushTool } from '@workbench/canvas-engine/tools/brushTool';
import { createColorPickerTool } from '@workbench/canvas-engine/tools/colorPickerTool';
import { createEraserTool } from '@workbench/canvas-engine/tools/eraserTool';
import { createGradientTool } from '@workbench/canvas-engine/tools/gradientTool';
import { createLassoTool } from '@workbench/canvas-engine/tools/lassoTool';
import { hittableLayerRect, layerOutlineCorners, topLayerAt } from '@workbench/canvas-engine/tools/moveHitTest';
import { createMoveTool } from '@workbench/canvas-engine/tools/moveTool';
import { stepBrushSize } from '@workbench/canvas-engine/tools/paintConstants';
import { createSamTool } from '@workbench/canvas-engine/tools/samTool';
import { createShapeTool } from '@workbench/canvas-engine/tools/shapeTool';
import { createTextTool } from '@workbench/canvas-engine/tools/textTool';
import { createTransformTool } from '@workbench/canvas-engine/tools/transformTool';
import { type LayerTransform, transformOverlayGeometry } from '@workbench/canvas-engine/transform/transformMath';
import { createViewport, MAX_DPR, type Viewport } from '@workbench/canvas-engine/viewport';

import type { HistoryEntry } from './history/history';
import type { StrokeCommittedEvent, Tool, ToolContext } from './tools/tool';

import { createBitmapStore, type BitmapStore } from './document/bitmapStore';
import { createDocumentMirror, type DocumentMirror } from './document/documentMirror';
import { getSourceBounds, getSourceContentRect, isRenderableLayer, renderableSourceOf } from './document/sources';
import { createImagePatchEntry, type ImagePatchApply } from './history/imagePatch';
import { createViewTool } from './tools/viewTool';

/**
 * The input to {@link CanvasEnginePreviewCapability.setStagedPreview}: either a persisted image
 * (decoded via the engine's `imageResolver`, sized to the decoded pixels — the
 * final staged candidate, optionally with candidate-specific placement) or an
 * inline data-URL with explicit document-space dimensions (a live
 * denoise-progress frame, scaled to fill those dims at the current bbox).
 */
export interface StagedPreviewPlacement extends Rect {
  opacity: number;
}

export type StagedPreviewInput =
  | { imageName: string; placement?: StagedPreviewPlacement }
  | { dataUrl: string; width: number; height: number };

export interface FilterPreviewInput {
  imageName: string;
  rect: Rect;
  filterType?: string;
}

/**
 * Result of {@link CanvasEngineLayerCapability.mergeVisibleRasterLayers}: `'merged'` when a new
 * composite layer was inserted, `'not-ready'` when a contributor could not be
 * rasterized consistently, `'busy'` when another edit owns the document, and
 * `'nothing'` when fewer than two visible rasters have content.
 */
export type MergeVisibleResult = 'merged' | 'not-ready' | 'busy' | 'nothing';

export type BooleanRasterOperation = 'intersect' | 'cutout' | 'cutaway' | 'exclude';
export type BooleanRasterResult = 'merged' | 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty';

export type ExtractMaskedAreaResult =
  | { status: 'extracted'; layerId: string }
  | { status: 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty' };

export type CropLayerResult =
  | { status: 'cropped' }
  | { status: 'missing' | 'locked' | 'unsupported' | 'empty' | 'not-ready' | 'busy' }
  | { status: 'failed'; message: string };

/**
 * Result of {@link CanvasEngineExportCapability.exportRasterLayersToPsd}: `'exported'` on
 * success, `'nothing'` when there are no raster layers with content, `'too-large'`
 * when the union bounds exceed the PSD dimension cap, and `'not-ready'` when a
 * participant's cache is still decoding (nothing exported — surface feedback).
 */
export type PsdExportResult = 'exported' | 'nothing' | 'too-large' | 'not-ready';

/** Opaque snapshot identity carried through async layer operations. */
export type ExportLayerPixelsResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' };

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

/** Structural equality for JSON-safe canvas contracts (including synthetic mask paint sources). */
const isDeeplyEqual = (left: unknown, right: unknown): boolean => {
  if (Object.is(left, right)) {
    return true;
  }
  if (typeof left !== 'object' || left === null || typeof right !== 'object' || right === null) {
    return false;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => isDeeplyEqual(value, right[index]))
    );
  }
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftRecord);
  const rightKeys = Object.keys(rightRecord);
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key) =>
        Object.prototype.hasOwnProperty.call(rightRecord, key) && isDeeplyEqual(leftRecord[key], rightRecord[key])
    )
  );
};

/** The minimal workbench store the engine depends on. */
export interface EngineStore {
  getState(): WorkbenchState;
  subscribe(listener: () => void): () => void;
  dispatch(action: WorkbenchAction): void;
}

/** Options for {@link createCanvasEngine}. */
export interface CanvasEngineOptions {
  projectId: string;
  store: EngineStore;
  /** Persists encoded engine-owned bitmaps. Application networking stays outside the core. */
  uploadImage(blob: Blob): Promise<{ height: number; imageName: string; width: number }>;
  /** Supplies the currently selected model base for core-created control layer contracts. */
  getMainModelBase?: () => string | null;
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

export interface CanvasEngineToolCapability extends CanvasToolCapability {
  contextMenuLayerIdAt(screenPoint: Vec2): string | null;
  handleEscapePriority(options: { gestureWasActive: boolean }): void;
  onStrokeCommitted(listener: (event: StrokeCommittedEvent) => void): () => void;
  setInteractionLocked(locked: boolean): void;
}

export interface CanvasEngineLayerCapability extends CanvasLayerCapability {
  applyTransform(): void;
  booleanMergeRasterLayers(upperLayerId: string, operation: BooleanRasterOperation): Promise<BooleanRasterResult>;
  cancelTextEdit(): void;
  cancelTransform(): void;
  clearMask(layerId: string): boolean;
  commitLayerConversion(label: string, expectedLiveLayer: CanvasLayerContract, after: CanvasLayerContract): boolean;
  commitLayerCopy(label: string, sourceLayerId: string, layer: CanvasLayerContract, index: number): boolean;
  commitMaskImageResult(options: CommitMaskImageResultOptions): Promise<CommitMaskImageResult>;
  commitOpenTextSession(): boolean;
  commitRasterFilterResult(options: CommitRasterFilterOptions): Promise<CommitRasterFilterResult>;
  commitTextEdit(content: string, styleChanges?: Partial<TextToolOptions>): void;
  copyLayerToRaster(layerId: string): Promise<string | null>;
  cropLayerToBbox(layerId: string): Promise<CropLayerResult>;
  mergeLayerDown(upperLayerId: string): boolean;
  mergeVisibleRasterLayers(): Promise<MergeVisibleResult>;
  nudgeSelectedLayer(dx: number, dy: number): void;
  openTextCreate(docPoint: Vec2): void;
  openTextEdit(layerId: string): void;
  rasterizeLayer(layerId: string): boolean;
  setTextEditContentReader(reader: (() => string) | null): void;
  updateTextEditStyle(patch: Partial<TextToolOptions>): void;
  updateTransformSession(transform: LayerTransform): void;
}

export interface CanvasEngineExportCapability extends CanvasExportCapability {
  exportBakedLayerPixels(
    layerId: string,
    options?: ExportBakedLayerPixelsOptions
  ): Promise<ExportBakedLayerPixelsResult>;
  exportLayerPixels(layerId: string, options?: ExportLayerPixelsOptions): Promise<ExportLayerPixelsResult>;
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
  extractMaskedArea(maskLayerId: string): Promise<ExtractMaskedAreaResult>;
}

export interface CanvasEngineSelectionCapability extends CanvasSelectionCapability {}

export interface CanvasEnginePreviewCapability extends CanvasPreviewCapability {
  setGuardedFilterPreview(
    layerId: string,
    input: FilterPreviewInput,
    guard: LayerExportGuard
  ): Promise<'shown' | 'missing' | 'stale'>;
  setStagedPreview(input: StagedPreviewInput | null): void;
}

export interface CanvasDiagnosticsCapability {
  clearCaches(): Promise<void>;
  getDiagnostics(): Readonly<CanvasDiagnosticsSnapshot>;
  logDebugInfo(): void;
}

/** The public engine handle. */
export interface CanvasEngine {
  readonly projectId: string;
  readonly surface: CanvasSurfaceCapability;
  readonly viewport: CanvasViewportCapability;
  readonly tools: CanvasEngineToolCapability;
  readonly history: CanvasHistoryCapability;
  readonly lifecycle: CanvasLifecycleCapability;
  readonly layers: CanvasEngineLayerCapability;
  readonly previews: CanvasEnginePreviewCapability;
  readonly selection: CanvasEngineSelectionCapability;
  readonly edits: CanvasEditCapability;
  readonly document: CanvasDocumentCapability;
  readonly exports: CanvasEngineExportCapability;
  readonly diagnostics: CanvasDiagnosticsCapability;
  /** The transient stores React subscribes to. */
  readonly stores: EngineStores;
}

export interface CanvasEngineCoreComposition {
  readonly engine: CanvasEngine;
  readonly applicationHost: CanvasApplicationHost;
}

/**
 * Decodes a `data:` URL to a `Blob` without a DOM (`atob`/`Blob`/`Uint8Array`
 * are all node-safe), so the staged-progress decode path runs in vitest through
 * the injected raster backend just like the imageName path runs through the
 * injected resolver.
 */
const dataUrlToBlob = (dataUrl: string): Blob => {
  const commaIndex = dataUrl.indexOf(',');
  const header = commaIndex >= 0 ? dataUrl.slice(0, commaIndex) : '';
  const data = commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
  const mime = /^data:([^;,]+)/i.exec(header)?.[1] ?? 'application/octet-stream';
  if (/;base64/i.test(header)) {
    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type: mime });
  }
  return new Blob([decodeURIComponent(data)], { type: mime });
};

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

const clearSurface = (surface: RasterSurface): void => {
  surface.ctx.setTransform(1, 0, 0, 1, 0, 0);
  surface.ctx.clearRect(0, 0, surface.width, surface.height);
};

/** Creates a per-project canvas engine. */
export const createCanvasEngine = (opts: CanvasEngineOptions): CanvasEngineCoreComposition => {
  const { imageResolver, projectId, store } = opts;
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

  /**
   * Applies a persisted bitmap ref + offset to a layer's document contract — the
   * bitmap store's single swap-on-success dispatch. Raster/control layers take a
   * `paint` source (`updateCanvasLayerSource`); mask layers take their `mask`
   * bitmap + offset (`updateCanvasLayerConfig`, preserving the fill). The self-echo
   * `lastApplied` name the store records covers both, so a mask flush round-tripping
   * back through the mirror is skipped for re-rasterization exactly like a paint one.
   */
  const dispatchLayerBitmap = (layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }): void => {
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!layer) {
      return;
    }
    if (layer.type === 'raster' || layer.type === 'control') {
      store.dispatch({ id: layerId, source: { bitmap, offset, type: 'paint' }, type: 'updateCanvasLayerSource' });
    } else if (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance') {
      store.dispatch({
        config: { layerType: layer.type, mask: { bitmap, offset } },
        id: layerId,
        type: 'updateCanvasLayerConfig',
      });
    }
  };

  // Paint persistence: debounced PNG encode → SHA-256 dedupe → upload → a single
  // swap-on-success `updateCanvasLayerSource` (paint) / `updateCanvasLayerConfig`
  // (mask). Wired to committed strokes below.
  const bitmapStore: BitmapStore =
    opts.bitmapStore ??
    createBitmapStore({
      dispatch: (action) => store.dispatch(action),
      dispatchBitmap: (layerId, bitmap, offset) => dispatchLayerBitmap(layerId, bitmap, offset),
      encodeSurface: (surface) => backend.encodeSurface(surface),
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
  let controlPixelController: ControlPixelController | null = null;
  const cancelOpenControlPixelEdit = (): void => {
    controlPixelController?.cancel();
  };

  const structuralController = new StructuralLayerController({
    canEdit: () => canEditDocument(),
    dispatch: (action) => store.dispatch(action),
    getDocument: () => mirror.getDocument(),
    history,
    isGestureActive: () => pipeline.isGestureActive(),
  });
  const endNudgeBurst = (): void => structuralController.endBurst();
  const commitStructural = (label: string, forward: WorkbenchAction, inverse: WorkbenchAction): void =>
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

  const createPath2DImpl: CreatePath2D = (d) => new Path2D(d);

  /** Bumps a layer's cache version after a direct paint (pixels stay fresh) and recomposites. */
  const notifyLayerPainted = (layerId: string): void => {
    const entry = layerCache.publishPixels(layerId);
    if (entry) {
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
  const createComposedPaintEntry = (
    created: { layer: CanvasLayerContract; index: number },
    event: StrokeCommittedEvent,
    label: string
  ): HistoryEntry => {
    const { afterImageData, dirtyRect, layerId } = event;
    const rect: Rect = { height: dirtyRect.height, width: dirtyRect.width, x: dirtyRect.x, y: dirtyRect.y };
    const bytes = event.beforeImageData.data.byteLength + afterImageData.data.byteLength + 256;
    return {
      bytes,
      label,
      redo: () => {
        store.dispatch({ index: created.index, layer: created.layer, type: 'addCanvasLayer' });
        // Re-create an EMPTY cache marked fresh so the async rasterize pass can't
        // clobber the restored stroke; `applyImagePatch` grows it to the stroke's
        // content bounds and writes the `after` pixels.
        const entry = layerCache.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 });
        entry.stale = false;
        applyImagePatch(layerId, rect, afterImageData);
      },
      undo: () => {
        store.dispatch({ ids: [layerId], type: 'removeCanvasLayers' });
      },
    };
  };

  const commitOrdinaryStroke = (event: StrokeCommittedEvent): void => {
    // A fresh stroke ends any open nudge-coalescing burst.
    endNudgeBurst();
    notifyLayerPainted(event.layerId);
    // Persistence first: mark the layer dirty so a debounced upload fires even
    // when no external subscriber is attached.
    bitmapStore.markLayerDirty(event.layerId);
    // Record the edit on the engine-owned history. Guarded against re-entrancy:
    // an undo/redo replay routes pixels through `applyImagePatch`, not a fresh
    // stroke, so this never fires during apply — the guard is belt-and-braces.
    if (!history.isApplying()) {
      const label = event.tool === 'eraser' ? 'Eraser stroke' : 'Brush stroke';
      history.push(
        event.createdLayer
          ? createComposedPaintEntry(event.createdLayer, event, label)
          : createImagePatchEntry({
              after: event.afterImageData,
              apply: applyImagePatch,
              before: event.beforeImageData,
              label,
              layerId: event.layerId,
              rect: event.dirtyRect,
            })
      );
    }
    for (const listener of strokeListeners) {
      listener(event);
    }
  };

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
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
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
      dispatch: (action) => store.dispatch(action),
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

  const toolContext: ToolContext = {
    applyTransform: () => applyTransform(),
    backend,
    beginControlPixelEdit: (layerId) => beginControlPixelEdit(layerId),
    beginTransformSession: (layerId) => beginTransformSession(layerId),
    cancelTextEdit: () => cancelTextEdit(),
    cancelTransform: () => cancelTransform(),
    commitSelection: (commit) => selection.commit(commit),
    commitStructural: (label, forward, inverse) => commitStructural(label, forward, inverse),
    createLayerId,
    createPath2D: createPath2DImpl,
    dispatch: (action) => store.dispatch(action),
    emitStrokeCommitted: (event) => commitOrdinaryStroke(event),
    getDocument: () => mirror.getDocument(),
    getSelectionMask: () => selection.mask(),
    invalidate: (payload) => scheduler.invalidate(payload),
    layers: layerCache,
    notifyLayerPainted,
    getSamInteraction: () => stores.samInteraction.get(),
    openTextCreate: (docPoint) => openTextCreate(docPoint),
    openTextEdit: (layerId) => openTextEdit(layerId),
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
    documentSize: { height: doc.height, width: doc.width },
    resolver: imageResolver,
    signal,
    store: layerCache,
  });

  const isSupportedExportSource = (source: CanvasLayerSourceContract): boolean => {
    if (source.type === 'shape') {
      return source.kind !== 'polygon';
    }
    return true;
  };

  const isCurrentRasterizationJob = (layer: CanvasLayerContract): boolean => {
    const job = rasterController.getRasterizationJob(layer.id);
    const source = renderableSourceOf(layer);
    return (
      !!job &&
      !!source &&
      job.version === layerCache.version(layer.id) &&
      job.documentGeneration === rasterController.getDocumentGeneration() &&
      isDeeplyEqual(job.source, source)
    );
  };

  /**
   * Starts (or joins) an isolated rasterization. Pixels land in a scratch
   * surface and are copied into the live cache only while this exact job still
   * describes the current layer source, cache version, and document.
   */
  const getOrStartLayerRasterization = (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV2
  ): Promise<'published' | 'stale' | 'error'> => {
    if (disposed || store.getState().activeProjectId !== projectId) {
      return Promise.resolve('stale');
    }
    const liveSource = renderableSourceOf(layer);
    if (!liveSource || !isSupportedExportSource(liveSource)) {
      return Promise.resolve('stale');
    }

    const contentRect = getSourceContentRect(layer, document);
    const entry = layerCache.getOrCreateRect(layer.id, contentRect);
    const version = entry.version;
    const documentGeneration = rasterController.getDocumentGeneration();
    const source = structuredClone(liveSource);
    const existing = rasterController.getRasterizationJob(layer.id);
    if (
      existing &&
      existing.version === version &&
      existing.documentGeneration === documentGeneration &&
      isDeeplyEqual(existing.source, source)
    ) {
      return existing.promise;
    }
    cancelLayerRasterization(layer.id);

    if (source.type === 'text') {
      fontLoader.ensure(textFontString(source), () => {
        const currentDocument = mirror.getDocument();
        const currentLayer = currentDocument?.layers.find((candidate) => candidate.id === layer.id);
        if (
          disposed ||
          store.getState().activeProjectId !== projectId ||
          !currentLayer ||
          rasterController.getDocumentGeneration() !== documentGeneration ||
          layerCache.version(layer.id) !== version ||
          !isDeeplyEqual(renderableSourceOf(currentLayer), source)
        ) {
          return;
        }
        invalidateLayerCache(layer.id);
        scheduler.invalidate({ layers: [layer.id] });
      });
    }

    const scratch = backend.createSurface(contentRect.width, contentRect.height);
    const controller = new AbortController();
    let settleJob!: (result: 'published' | 'stale' | 'error') => void;
    const promise = new Promise<'published' | 'stale' | 'error'>((resolve) => {
      settleJob = resolve;
    });
    const job: RasterizationJob = {
      controller,
      documentGeneration,
      promise,
      source,
      version,
    };
    rasterController.installRasterizationJob(layer.id, job);
    let published = false;
    void (async () => {
      try {
        const result = await rasterizeSource(source, rasterizeDeps(document, controller.signal), scratch);
        const currentDocument = mirror.getDocument();
        const currentLayer = currentDocument?.layers.find((candidate) => candidate.id === layer.id);
        const currentEntry = layerCache.get(layer.id);
        if (
          disposed ||
          store.getState().activeProjectId !== projectId ||
          rasterController.getRasterizationJob(layer.id) !== job ||
          rasterController.getDocumentGeneration() !== documentGeneration ||
          !currentLayer ||
          !currentEntry ||
          currentEntry.version !== version ||
          !isDeeplyEqual(renderableSourceOf(currentLayer), source)
        ) {
          return 'stale';
        }

        if (currentEntry.surface.width !== result.rect.width || currentEntry.surface.height !== result.rect.height) {
          currentEntry.surface.resize(result.rect.width, result.rect.height);
        }
        const ctx = currentEntry.surface.ctx;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, result.rect.width, result.rect.height);
        if (!isEmpty(result.rect)) {
          ctx.drawImage(result.surface.canvas, 0, 0);
        }
        currentEntry.rect = { ...result.rect };
        const publishedEntry = layerCache.publishPixels(layer.id);
        if (!publishedEntry) {
          return 'stale';
        }
        trackPublishedLayerImage(currentLayer);
        published = true;
        stores.thumbnailVersion.set(layer.id, publishedEntry.version);
        stores.thumbnailStatus.set(layer.id, 'ready');
        scheduler.invalidate({ layers: [layer.id] });
        return 'published';
      } catch (error) {
        const currentDocument = mirror.getDocument();
        const currentLayer = currentDocument?.layers.find((candidate) => candidate.id === layer.id);
        if (
          disposed ||
          store.getState().activeProjectId !== projectId ||
          rasterController.getRasterizationJob(layer.id) !== job ||
          rasterController.getDocumentGeneration() !== documentGeneration ||
          !currentLayer ||
          layerCache.version(layer.id) !== version ||
          !isDeeplyEqual(renderableSourceOf(currentLayer), source)
        ) {
          return 'stale';
        }
        stores.thumbnailStatus.set(layer.id, 'error');
        try {
          store.dispatch({
            area: 'canvas-engine',
            context: { error: error instanceof Error ? error.message : String(error), layerId: layer.id },
            message: 'Layer thumbnail rasterization failed',
            namespace: 'canvas',
            projectId,
            type: 'recordError',
          });
        } catch {
          // Diagnostics must not turn a handled thumbnail failure into a rejection.
        }
        return 'error';
      } finally {
        rasterController.finishRasterizationJob(layer.id, job);
        const imageName = sourceImageName(source);
        if (!published && imageName) {
          releaseBitmapIfUnreferenced(imageName);
        }
      }
    })().then(settleJob, () => settleJob('stale'));
    return promise;
  };

  const captureLayerExportGuard = (layer: CanvasLayerContract, entry: LayerCacheEntry): LayerExportGuard => ({
    cacheVersion: entry.version,
    documentGeneration: rasterController.getDocumentGeneration(),
    layer,
    layerId: layer.id,
    projectId,
  });

  const isLayerExportGuardCurrent = (guard: LayerExportGuard): boolean => {
    if (
      disposed ||
      guard.projectId !== projectId ||
      store.getState().activeProjectId !== projectId ||
      guard.documentGeneration !== rasterController.getDocumentGeneration()
    ) {
      return false;
    }
    const document = mirror.getDocument();
    const liveLayer = document?.layers.find((candidate) => candidate.id === guard.layerId);
    const entry = layerCache.get(guard.layerId);
    return !!entry && liveLayer === guard.layer && entry.version === guard.cacheVersion;
  };

  const captureCurrentLayerExportGuard = (layerId: string): LayerExportGuard | null => {
    const document = mirror.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    const source = layer ? renderableSourceOf(layer) : null;
    const entry = layerCache.get(layerId);
    if (
      !layer ||
      !source ||
      !isSupportedExportSource(source) ||
      !entry ||
      entry.stale ||
      isCurrentRasterizationJob(layer) ||
      isEmpty(entry.rect)
    ) {
      return null;
    }
    const guard = captureLayerExportGuard(layer, entry);
    return isLayerExportGuardCurrent(guard) ? guard : null;
  };

  const documentEditOwner = Symbol('canvas-operation-document-edit-owner');
  type DocumentEditPermit = { epoch: number; owner?: symbol };
  let documentEditEpoch = 0;
  let documentEditingLocked = false;
  const syncDocumentEditingLock = (): void => {
    const nextLocked = stores.documentEditingLocked.get();
    if (nextLocked !== documentEditingLocked) {
      documentEditingLocked = nextLocked;
      documentEditEpoch += 1;
    }
  };
  const unsubscribeDocumentEditingLock = stores.documentEditingLocked.subscribe(syncDocumentEditingLock);
  const canEditDocument = (owner?: symbol): boolean =>
    owner === documentEditOwner || !stores.documentEditingLocked.get();
  const captureDocumentEditPermit = (owner?: symbol): DocumentEditPermit | null =>
    canEditDocument(owner) ? { epoch: documentEditEpoch, owner } : null;
  const isDocumentEditPermitCurrent = (permit: DocumentEditPermit): boolean =>
    permit.owner === documentEditOwner || (!stores.documentEditingLocked.get() && permit.epoch === documentEditEpoch);

  const ensureLayerCaches = (doc: CanvasDocumentContractV2): void => {
    for (const layer of doc.layers) {
      // The layer's rasterizable source: a raster/control `source`, or a mask
      // layer's alpha bitmap viewed as a paint source (colorized at composite).
      const source = renderableSourceOf(layer);
      if (!layer.isEnabled || !source) {
        continue;
      }
      if (!isRenderableLayer(layer)) {
        continue;
      }

      // Ensure a cache entry exists WITHOUT resizing an existing one: a paint
      // layer's live cache may have grown past its persisted content rect (fresh
      // unflushed strokes), so we must never resize it down to the contract size
      // here — that would destroy those pixels. The rasterizer (below, guarded by
      // `stale`) owns sizing the surface + placing its content rect.
      const entry = layerCache.getOrCreateRect(layer.id, getSourceContentRect(layer, doc));
      if (!entry.stale) {
        continue;
      }

      void getOrStartLayerRasterization(layer, doc);
    }
  };

  const hasExportableLayerContent = (layerId: string): boolean => {
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!doc || !layer) {
      return false;
    }
    const source = renderableSourceOf(layer);
    if (!source || !isSupportedExportSource(source)) {
      return false;
    }
    if (!isEmpty(getSourceContentRect(layer, doc))) {
      return true;
    }
    // Only paint-backed layers (including masks through renderableSourceOf) can
    // have real pixels beyond their persisted source rect. The live cache must
    // describe the current source revision and must not be mid-rasterization.
    if (source.type !== 'paint') {
      return false;
    }
    const entry = layerCache.get(layerId);
    return !!entry && !entry.stale && !isCurrentRasterizationJob(layer) && !isEmpty(entry.rect);
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
  });
  const rasterizeLayerPixels = rasterExportController.rasterize.bind(rasterExportController);
  const exportBakedLayerPixels = rasterExportController.baked.bind(rasterExportController);
  const exportBakedLayerBlob = rasterExportController.blob.bind(rasterExportController);

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
  const getLayerSurfaceForExport = async (layerId: string): Promise<{ surface: RasterSurface; rect: Rect }> => {
    const result = await rasterizeLayerPixels(layerId);
    if (result.status === 'ok') {
      return { rect: result.rect, surface: result.surface };
    }
    throw new Error(`Cannot rasterize layer ${layerId} for generation: ${result.status}.`);
  };

  const releaseBitmapIfUnreferenced = (imageName: string): void =>
    rasterController.releaseBitmapIfUnreferenced(imageName);
  const trackPublishedLayerImage = (layer: CanvasLayerContract): void =>
    rasterController.trackPublishedLayerImage(layer);

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

  // ---- Staged generation preview ------------------------------------------

  /** Drops any staged preview and bumps the token so an in-flight decode is discarded. */
  const clearStagedPreview = (): void => {
    if (renderController.previews.clearStaged()) {
      scheduler.invalidate({ all: true });
    }
  };

  /** Decodes a staged-preview input to a surface (imageName via resolver, dataUrl via the backend seam). */
  const decodeStagedPreview = async (
    input: StagedPreviewInput
  ): Promise<{ surface: RasterSurface; width: number; height: number; placement?: StagedPreviewPlacement }> => {
    if ('imageName' in input) {
      const blob = await imageResolver(input.imageName);
      const decoded = await rasterController.decodeBlob(blob);
      return {
        height: decoded.decodedHeight,
        placement: input.placement ? { ...input.placement } : undefined,
        surface: decoded.surface,
        width: decoded.decodedWidth,
      };
    }
    const { dataUrl, height, width } = input;
    const decoded = await rasterController.decodeBlob(dataUrlToBlob(dataUrl), { height, scale: true, width });
    return { height, surface: decoded.surface, width };
  };

  const setStagedPreview = (input: StagedPreviewInput | null): void => {
    if (input === null) {
      clearStagedPreview();
      return;
    }
    const token = renderController.previews.nextStagedToken();
    decodeStagedPreview(input)
      .then((decoded) => {
        // A newer set/clear superseded this decode while it was in flight.
        if (renderController.previews.publishStaged(token, decoded)) {
          scheduler.invalidate({ all: true });
        }
      })
      .catch(() => {
        // A transient decode failure leaves any prior preview untouched rather
        // than blanking the canvas; the next selection re-drives a decode.
      });
  };

  /**
   * Drops a layer's filter-preview state and bumps its token so an in-flight
   * decode for it is discarded — even if the id is later reused (e.g. an undo
   * that restores a deleted layer must not resurrect a stale decode result
   * that resolves afterward). The token is bumped, never reset/deleted, so a
   * later guarded preview for the same id can never collide with a
   * still-in-flight decode's captured token.
   */
  const clearFilterPreview = (layerId: string): void => {
    if (renderController.previews.clearFilter(layerId)) {
      scheduler.invalidate({ layers: [layerId] });
    }
  };

  /**
   * Drops every layer's filter-preview state. Used on a wholesale document
   * replace: none of the outgoing document's previews describe the incoming
   * document, even if a layer id happens to be reused.
   */
  const clearAllFilterPreviews = (): void => {
    for (const id of renderController.previews.filterLayerIds()) {
      clearFilterPreview(id);
    }
  };

  const publishFilterPreview = async (
    layerId: string,
    input: FilterPreviewInput,
    validate: () => 'shown' | 'missing' | 'stale',
    guard: LayerExportGuard
  ): Promise<'shown' | 'missing' | 'stale'> => {
    const nextToken = renderController.previews.beginGuardedFilter(layerId);
    const dropGuardedRequest = (): void => {
      renderController.previews.finishGuardedFilter(layerId, nextToken);
    };
    const beforeDecode = validate();
    if (beforeDecode !== 'shown') {
      dropGuardedRequest();
      return beforeDecode;
    }
    try {
      const decoded = await decodeStagedPreview({ imageName: input.imageName });
      if (input.filterType && (decoded.width !== input.rect.width || decoded.height !== input.rect.height)) {
        throw new LayerFilterOutputDimensionError(
          input.filterType,
          { height: decoded.height, width: decoded.width },
          input.rect
        );
      }
      const beforePublish = validate();
      if (beforePublish !== 'shown') {
        dropGuardedRequest();
        return beforePublish;
      }
      // A newer set/clear for THIS layer superseded the decode in flight.
      if (!renderController.previews.isFilterTokenCurrent(layerId, nextToken)) {
        dropGuardedRequest();
        return 'stale';
      }
      renderController.previews.publishFilter(layerId, nextToken, {
        guard,
        rect: { ...input.rect },
        surface: decoded.surface,
      });
      scheduler.invalidate({ layers: [layerId] });
      return 'shown';
    } catch (error) {
      // Transient decode failure leaves any prior preview untouched.
      dropGuardedRequest();
      if (error instanceof LayerFilterOutputDimensionError) {
        throw error;
      }
      return 'stale';
    }
  };

  const setGuardedFilterPreview = (
    layerId: string,
    input: FilterPreviewInput,
    guard: LayerExportGuard
  ): Promise<'shown' | 'missing' | 'stale'> => {
    const validate = (): 'shown' | 'missing' | 'stale' => {
      const liveLayer = mirror.getDocument()?.layers.find((candidate) => candidate.id === layerId);
      if (!liveLayer) {
        return 'missing';
      }
      if (layerId !== guard.layerId || !isLayerExportGuardCurrent(guard)) {
        return 'stale';
      }
      return 'shown';
    };
    return publishFilterPreview(layerId, input, validate, guard);
  };

  // ---- Render loop --------------------------------------------------------

  /**
   * The selected-layer bounds outline for the move overlay, or `null`. Only the
   * move tool draws it; a layer mid-drag (with a live override) is preferred over
   * the committed selection so the marquee tracks the preview.
   */
  const moveOutlineCorners = (doc: CanvasDocumentContractV2): readonly { x: number; y: number }[] | null => {
    if (interactionController.getActiveToolId() !== 'move') {
      return null;
    }
    const overridden = doc.layers.find((layer) => transformOverrides.has(layer.id));
    const target = overridden ?? doc.layers.find((layer) => layer.id === doc.selectedLayerId);
    if (!target) {
      return null;
    }
    return layerOutlineCorners(target, doc, transformOverrides.get(target.id) ?? null);
  };

  /** The transform-tool frame (rotated bounds + handles + rotation nub), or `null`. */
  const transformFrameOverlay = (
    doc: CanvasDocumentContractV2
  ): {
    corners: { x: number; y: number }[];
    handles: { x: number; y: number }[];
    center: { x: number; y: number };
    rotationAnchor: { x: number; y: number };
  } | null => {
    if (interactionController.getActiveToolId() !== 'transform') {
      return null;
    }
    const session = stores.transformSession.get();
    if (!session) {
      return null;
    }
    const layer = doc.layers.find((candidate) => candidate.id === session.layerId);
    // The layer's LOCAL content rect (off-origin aware): the frame must wrap the
    // pixels where the compositor draws them, not an assumed origin-anchored box.
    const rect = layer ? hittableLayerRect(layer, doc) : null;
    if (!rect) {
      return null;
    }
    return transformOverlayGeometry(session.transform, rect);
  };

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
    const needsComposite = flags.all || flags.view || flags.layers.size > 0;
    const stagedPreview = renderController.previews.getStaged();
    const samPreview = renderController.previews.getSam();
    const filterPreviews = renderController.previews.filterSnapshot();
    if (needsComposite) {
      ensureLayerCaches(doc);
      const stagedPlacement = stagedPreview?.placement;
      const isolatedGuard = samPreview?.isolated ? samPreview.guard : null;
      const isolatedIds = isolatedGuard ? new Set([isolatedGuard.layerId]) : null;
      const compositeDoc = isolatedIds
        ? { ...doc, layers: doc.layers.filter((layer) => isolatedIds.has(layer.id)) }
        : doc;
      compositeDocument(screen, compositeDoc, layerCache, view, {
        // Memoized adjusted surfaces for raster layers with brightness/contrast/
        // saturation/curves (not recomputed per frame — see adjustedSurfaceCache).
        adjustedSurface: getAdjustedSurface,
        derivedSurfaces: derivedSurfaceCache,
        diagnostics,
        // The raster backend + mask fill tile resolver drive the mask colorize
        // path (alpha stencil → source-in fill colour/pattern, above all layers).
        backend,
        maskPatternTile: getMaskPatternTile,
        // Non-destructive control-filter previews (drawn in place of the layer's
        // committed pixels). Only allocated when a preview is actually active.
        layerPreviews:
          !isolatedGuard && filterPreviews.size > 0
            ? new Map(Array.from(filterPreviews, ([id, preview]) => [id, preview]))
            : null,
        clipRect: isolatedGuard && samPreview ? samPreview.rect : null,
        // Feed the cached checkerboard tile only while the toggle is ON; passing
        // `null` renders transparent documents without a checkerboard.
        checkerboardTile: stores.checkerboard.get() ? getCheckerboardTile() : null,
        // Crisp + cheap when zoomed in (nearest-neighbor up-scale), smooth when
        // shrinking (bilinear down-scale). See `shouldSmoothAtZoom`.
        imageSmoothing: shouldSmoothAtZoom(viewport.getZoom()),
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

      const visibleIds = doc.layers
        .filter((layer) => isolatedIds?.has(layer.id) || isRenderableLayer(layer))
        .map((layer) => layer.id);
      const { evictedBaseLayerIds: evicted } = enforceSurfaceBudget(
        layerCache,
        derivedSurfaceCache,
        visibleIds,
        DEFAULT_CACHE_BUDGET_BYTES,
        diagnostics
      );
      // Prune version-keyed dependents for every evicted id (mirrors dropLayer):
      // the evicted layer's surface is gone, so its adjusted-surface memo and
      // thumbnail state must not linger keyed to a version the re-rasterized entry
      // will exceed (the cache floor keeps versions monotonic across the recreate).
      for (const id of evicted) {
        deleteDerivedSurfaces(id);
        stores.thumbnailVersion.delete(id);
        stores.thumbnailStatus.delete(id);
      }
    }

    // The overlay is cheap (a handful of screen-space strokes, independent of
    // zoom and document size) and shares the `view` transform with the composite,
    // so redraw it whenever any frame runs — including overlay-only frames.
    // While the bbox tool drags, the transient preview stands in for the
    // committed frame so the overlay (rect + handles) tracks the gesture.
    const bboxPreview = stores.bboxPreview.get();
    const samSession = stores.samInteraction.get();
    renderOverlay(overlay, {
      bbox: bboxPreview ?? doc.bbox,
      bboxHandles: interactionController.getActiveToolId() === 'bbox',
      cursor: overlayCursor,
      // The grid spans the whole viewport at the bbox snap size when the setting
      // is on (the document rect no longer bounds it).
      gridSize: stores.bboxGrid.get(),
      layerOutline: moveOutlineCorners(doc),
      // The in-progress lasso path (transient) and the committed selection's
      // animated marching ants. Both are overlay chrome; ants advance via the
      // ants animator's overlay-only ticks.
      gradientPreview: stores.gradientPreview.get(),
      lassoPreview: stores.lassoPreview.get(),
      marchingAnts: selection.hasSelection() ? { paths: selection.antsPaths(), phase: antsPhase } : null,
      samInput: samSession?.input.type === 'visual' ? samSession.input : null,
      samPreview: samPreview ? { opacity: 0.45, rect: samPreview.rect, surface: samPreview.data } : null,
      bboxOverlay: stores.bboxOverlay.get(),
      ruleOfThirds: stores.ruleOfThirds.get(),
      shapePreview: stores.shapePreview.get(),
      // The passive bbox frame follows the setting, but always renders while the
      // bbox tool is active so its handles have a frame to attach to (editable).
      showBbox: stores.showBbox.get() || interactionController.getActiveToolId() === 'bbox',
      showGrid: stores.showGrid.get(),
      transformFrame: transformFrameOverlay(doc),
      view,
    });
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
  // Stay paused until attached: invalidations accumulate but never request a
  // (DOM) frame, keeping the engine node-safe before it has render targets.
  scheduler.pause();

  // ---- Document mirror ----------------------------------------------------

  const mirror: DocumentMirror = createDocumentMirror(store, projectId, {
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
      cleanup.run(() => selection.clear());
      cleanup.run(() => stores.lassoPreview.set(null));
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
      const present = new Set(doc ? doc.layers.map((layer) => layer.id) : []);
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
        const layer = doc?.layers.find((candidate) => candidate.id === id);
        if (!present.has(id)) {
          rasterController.deleteThumbnailKey(id);
          cleanup.run(() => dropLayer(id));
          const previousImageName = previousImageNames.get(id);
          if (previousImageName) {
            cleanup.run(() => releaseBitmapIfUnreferenced(previousImageName));
          }
          // A control-filter preview (session + decoded surface) belongs to a
          // specific layer; a layer removed out from under an in-flight or
          // already-decoded preview (delete via the layers panel, or an undo
          // that removes it) must have its preview dropped and its decode
          // token bumped, or a late-resolving decode — or a later undo that
          // restores this same id — would repopulate a stale preview.
          cleanup.run(() => clearFilterPreview(id));
          if (session && session.layerId === id) {
            cleanup.run(cancelTransform);
          }
          // An edit-mode text session whose layer was deleted out from under it
          // (layers panel, or undo of the add) is torn down the same way.
          if (textSession && textSession.layerId === id) {
            cleanup.run(cancelTextEdit);
          }
          continue;
        }
        const preview = renderController.previews.getFilter(id);
        if (preview && !isLayerExportGuardCurrent(preview.guard)) {
          cleanup.run(() => clearFilterPreview(id));
        }
        if (!sourceChanged.has(id)) {
          if (layer) {
            const displayKey = getLayerThumbnailDisplayKey(layer);
            if (rasterController.getThumbnailKey(id) !== displayKey) {
              rasterController.setThumbnailKey(id, displayKey);
              const currentVersion = stores.thumbnailVersion.get(id);
              // Cache versions are positive. Negative display tokens cannot
              // collide with the next cache publication and suppress its redraw.
              stores.thumbnailVersion.set(
                id,
                currentVersion !== undefined && currentVersion < 0 ? currentVersion - 1 : -1
              );
            }
          }
          // Prop/transform-only change (opacity, blend, lock, visibility,
          // rename, nudge): the layer object was replaced but its SOURCE
          // reference is unchanged, so the rasterized pixels are still valid.
          // Invalidating here would be wasteful for an image layer and
          // *destructive* for an unflushed paint layer — a `bitmap: null` paint
          // source rasterizes to a CLEARED surface, wiping strokes that live
          // only in the cache until their debounced upload lands. The compositor
          // applies transform/opacity/blend at draw time, so the scheduled
          // recomposite below is all a prop change needs.
          continue;
        }
        // The layer's source genuinely changed (image swap, or a paint-bitmap
        // swap from undo/import). Self-echo guard: skip when it's the exact
        // paint-bitmap ref the bitmap store just applied — the cache already
        // holds those pixels, so re-rasterizing would needlessly re-fetch and
        // could flicker. Any other swap invalidates → re-rasterizes.
        if (layer) {
          rasterController.setThumbnailKey(id, getLayerThumbnailDisplayKey(layer));
        }
        cleanup.run(() => rasterController.untrackLayerImage(id));
        const previousImageName = previousImageNames.get(id);
        if (previousImageName) {
          cleanup.run(() => releaseBitmapIfUnreferenced(previousImageName));
        }
        const source = getLayerSourceById(id);
        if (bitmapStore.isSelfEcho(id, source)) {
          continue;
        }
        cleanup.run(() => invalidateLayerCache(id));
      }
      cleanup.run(() => scheduler.invalidate({ layers: ids }));
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
  let lastActiveProjectId = store.getState().activeProjectId;
  const unsubscribeProjectPreviewLifecycle = store.subscribe(() => {
    const activeProjectId = store.getState().activeProjectId;
    if (lastActiveProjectId === projectId && activeProjectId !== projectId) {
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
      lastActiveProjectId = activeProjectId;
      cleanup.throwIfFailed();
      return;
    }
    lastActiveProjectId = activeProjectId;
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
  const setTool = (toolId: ToolId, options?: { temporary?: boolean }): void =>
    interactionController.setTool(toolId, options);

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
    if (stores.textEditSession.get()) {
      cancelTextEdit();
      return;
    }
    if (stores.transformSession.get()) {
      cancelTransform();
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

  const decodeSelectObjectPreview = async (
    result: { image: CanvasImageRef; rect: Rect },
    signal: AbortSignal
  ): Promise<RasterSurface> => {
    const validateDecoded = (width: number, height: number): void => {
      const valid =
        Number.isInteger(width) &&
        width > 0 &&
        Number.isInteger(height) &&
        height > 0 &&
        width === result.image.width &&
        height === result.image.height &&
        width === result.rect.width &&
        height === result.rect.height;
      if (!valid) {
        throw Object.assign(
          new Error(
            `Decoded Select Object preview dimensions ${String(width)}x${String(height)} do not match SAM output ${result.image.width}x${result.image.height} and preview rect ${result.rect.width}x${result.rect.height}.`
          ),
          { samErrorCode: 'output-dimension' as const }
        );
      }
    };
    const decoded = await rasterController.decodeImage(result.image, {
      scaleToImage: false,
      signal,
      validateDecoded,
    });
    if (decoded.status !== 'ok') {
      throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
    }
    const surface = decoded.surface;
    surface.ctx.globalCompositeOperation = 'source-in';
    surface.ctx.fillStyle = '#38bdf8';
    surface.ctx.fillRect(0, 0, surface.width, surface.height);
    surface.ctx.globalCompositeOperation = 'source-over';
    if (signal.aborted) {
      throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
    }
    return surface;
  };

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
          lifecycleState = 'cool';
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
    render({ all: true, layers: new Set<string>(), overlay: true, view: true });
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

  const getReducerDocument = (): CanvasDocumentContractV2 | null =>
    store.getState().projects.find((project) => project.id === projectId)?.canvas.document ?? null;
  const getMainModelBase = (): string | null => {
    return opts.getMainModelBase?.() ?? null;
  };

  const dispatchPreparedMutation = (
    action: WorkbenchAction,
    isApplied: () => boolean,
    isMirrored: () => boolean
  ): void => {
    if (store.getState().activeProjectId !== projectId) {
      throw new Error('Canvas project is not active');
    }
    try {
      store.dispatch(action);
    } catch (error) {
      // Store subscribers run after the reducer has accepted an action. A
      // faulty observer must not strand an applied document mutation before
      // its matching engine state and history are published. Preserve real
      // reducer/dispatch failures by swallowing only when the exact intended
      // postcondition is visible in the authoritative reducer state.
      if (!isApplied()) {
        throw error;
      }
      // Notification may have been interrupted before DocumentMirror's
      // subscriber ran. Reconcile it synchronously from authoritative state
      // before publishing follow-up state or history.
      try {
        mirror.refresh();
      } catch (refreshError) {
        if (!isMirrored()) {
          throw refreshError;
        }
      }
      if (!isMirrored()) {
        throw error;
      }
      return;
    }

    // A reducer may reject a guarded transaction by returning the unchanged
    // state without throwing. Do not install its prepared cache or consume a
    // failure-atomic history entry unless the authoritative postcondition
    // actually landed.
    if (!isApplied()) {
      throw new Error('Canvas document mutation was rejected');
    }
    if (!isMirrored()) {
      try {
        mirror.refresh();
      } catch (refreshError) {
        if (!isMirrored()) {
          throw refreshError;
        }
      }
      if (!isMirrored()) {
        throw new Error('Canvas document mutation was not mirrored');
      }
    }
  };

  /** Conversion reducers clone contracts, so their publication postcondition compares by value. */
  const documentHasLayerContract = (
    document: CanvasDocumentContractV2 | null,
    expected: CanvasLayerContract
  ): boolean => {
    const current = document?.layers.find((candidate) => candidate.id === expected.id);
    return current !== undefined && isDeeplyEqual(current, expected);
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
    getActiveProjectId: () => store.getState().activeProjectId,
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
    capturePermit: (owner) => captureDocumentEditPermit(owner),
    createLayerId,
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    dispatchPrepared: dispatchPreparedMutation,
    endBurst: () => endNudgeBurst(),
    getDocument: () => mirror.getDocument(),
    getMainModelBase,
    getReducerDocument,
    history,
    installPrepared: installGeneratedPaintCache,
    isGestureActive: () => pipeline.isGestureActive(),
    isGuardCurrent: isLayerExportGuardCurrent,
    isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
    needsPixelPersistence: layerNeedsPixelPersistence,
    preparePixels: prepareGeneratedPaintCache,
  });
  const commitRasterFilterResult = filterResultController.commit.bind(filterResultController);

  const generatedResultController = new GeneratedResultController({
    captureCache: captureLayerCache,
    capturePermit: (owner) => captureDocumentEditPermit(owner),
    clearPreview: clearFilterPreview,
    createLayerId,
    decodeImage: (image, options) => rasterController.decodeImage(image, options),
    discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
    dispatchPrepared: dispatchPreparedMutation,
    endBurst: () => endNudgeBurst(),
    getDocument: () => mirror.getDocument(),
    getMainModelBase,
    getReducerDocument,
    history,
    installPrepared: installGeneratedPaintCache,
    isGestureActive: () => pipeline.isGestureActive(),
    isGuardCurrent: isLayerExportGuardCurrent,
    isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
    needsPixelPersistence: layerNeedsPixelPersistence,
    preparePixels: prepareGeneratedPaintCache,
  });
  const commitGeneratedImageResult = generatedResultController.commit.bind(generatedResultController);

  const booleanMergeRasterLayers = (
    upperLayerId: string,
    operation: BooleanRasterOperation
  ): Promise<BooleanRasterResult> => layerController.booleanMerge.merge(upperLayerId, operation);

  const extractMaskedArea = (maskLayerId: string): Promise<ExtractMaskedAreaResult> =>
    layerController.extractMaskedArea.extract(maskLayerId);

  const mergeLayerDown = (upperLayerId: string): boolean => layerController.merge.mergeDown(upperLayerId);
  const mergeVisibleRasterLayers = (): Promise<MergeVisibleResult> => layerController.merge.mergeVisible();

  /**
   * Rasterizes a layer to its cache for PSD export and returns the surface + its
   * content rect. Like {@link getLayerSurfaceForExport} but does NOT gate on
   * `isEnabled` (hidden layers are exported too). Reads the live cache when it is
   * not stale (capturing unflushed paint strokes), else rasterizes synchronously
   * from the source.
   */
  const getLayerSurfaceForPsd = async (layerId: string): Promise<{ surface: RasterSurface; rect: Rect }> => {
    const result = await rasterizeLayerPixels(layerId, { includeDisabled: true });
    if (result.status === 'ok') {
      return { rect: result.rect, surface: result.surface };
    }
    throw new Error(`Cannot rasterize layer ${layerId} for PSD export: ${result.status}.`);
  };

  const psdExportController = new PsdExportController({
    backend,
    getDocument: () => mirror.getDocument(),
    getLayerSurface: getLayerSurfaceForPsd,
    isRasterizationCurrent: isCurrentRasterizationJob,
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
  const cancelTransform = (): void => editingController.transform.cancel();
  const applyTransform = (): void => editingController.transform.apply();

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
  const selectAll = (): void => editingController.selectAll();
  const deselect = (): void => editingController.deselect();
  const invertSelection = (): void => editingController.invertSelection();

  const fillSelection = (): void => editingController.selectionPixels.run('fill');
  const eraseSelection = (): void => editingController.selectionPixels.run('erase');

  const clearMask = (layerId: string): boolean => layerController.mask.clear(layerId);
  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    lifecycleGeneration += 1;
    lifecycleState = 'disposed';
    const cleanup = createCleanupAccumulator();
    cleanup.run(() => pipeline.cancelActiveGesture());
    cleanup.run(cancelOpenControlPixelEdit);
    cleanup.run(() => controlPixelController?.dispose());
    cleanup.run(() => filterResultController.dispose());
    cleanup.run(() => generatedResultController.dispose());
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
    cleanup.run(unsubscribeDocumentEditingLock);
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

  const contextMenuLayerIdAt = (screenPoint: Vec2): string | null => {
    // Never open the menu over an in-progress edit: a live paint/drag gesture, or
    // an open transform / text-edit session. Right-click during those belongs to
    // the interaction, not to picking a layer. Mirrors the mid-gesture guards on
    // merge/nudge/undo above.
    if (pipeline.isGestureActive() || stores.transformSession.get() || stores.textEditSession.get()) {
      return null;
    }
    const doc = mirror.getDocument();
    if (!doc) {
      return null;
    }
    // Same screen→document conversion and group-rank-consistent, live-cache-aware
    // hit-test the move tool uses for click-selection, so right-clicking a layer
    // targets exactly the layer a left-click there would select.
    const documentPoint = viewport.screenToDocument(screenPoint);
    const hit = topLayerAt(
      doc,
      documentPoint,
      (layer) => layer.isEnabled,
      (layerId) => layerCache.get(layerId)?.rect
    );
    return hit?.id ?? null;
  };

  const undo = (): void => historyController.undo();
  const redo = (): void => historyController.redo();
  const setBboxGrid = (size: number): void => stores.bboxGrid.set(size > 0 ? size : 1);
  const getViewport = (): Viewport => viewport;
  const getCompositeExecutorDeps = (): CanvasCompositeExecutorDeps => ({
    backend,
    getLayerSurface: getLayerSurfaceForExport,
    uploadImage: (blob) => opts.uploadImage(blob),
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
      exportBaked: (layerId) => exportBakedLayerPixels(layerId),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      history,
      installPrepared: (prepared) =>
        installGeneratedPaintCache(prepared as ReturnType<LayerCacheStore['prepareReplacement']>),
      isCacheReady: isLayerCacheReadyForOp,
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
      preparePixels: prepareGeneratedPaintCache,
    },
    crop: {
      backend,
      captureCache: captureLayerCache,
      capturePermit: () => captureDocumentEditPermit(),
      discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      exportBaked: (layerId) => exportBakedLayerPixels(layerId, { includeDisabled: true }),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      history,
      installPrepared: (prepared) =>
        installGeneratedPaintCache(prepared as ReturnType<LayerCacheStore['prepareReplacement']>),
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
      isSupportedSource: isSupportedExportSource,
      preparePixels: prepareGeneratedPaintCache,
    },
    copy: {
      capturePermit: () => captureDocumentEditPermit(),
      createLayerId,
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      exportBaked: (layerId) => exportBakedLayerPixels(layerId, { includeDisabled: true }),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      history,
      installPrepared: (prepared) =>
        installGeneratedPaintCache(prepared as ReturnType<LayerCacheStore['prepareReplacement']>),
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
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
      exportBaked: (layerId, includeDisabled) => exportBakedLayerPixels(layerId, { includeDisabled }),
      getAdjustedSurface,
      getDocument: () => mirror.getDocument(),
      getMaskPattern: getMaskPatternTile,
      getReducerDocument,
      hasExportableContent: hasExportableLayerContent,
      history,
      installPrepared: (prepared) =>
        installGeneratedPaintCache(prepared as ReturnType<LayerCacheStore['prepareReplacement']>),
      isCacheReady: isLayerCacheReadyForOp,
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
      layers: layerCache,
      preparePixels: prepareGeneratedPaintCache,
      rasterize: (layerId) => rasterizeLayerPixels(layerId),
    },
    commitGeneratedImageResult,
    mask: {
      applyImagePatch,
      canEdit: () => canEditDocument(),
      deleteDerived: deleteDerivedSurfaces,
      discardPersisted: (layerId) => bitmapStore.discardLayer(layerId),
      dispatch: (action) => store.dispatch(action),
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
      capturePermit: () => captureDocumentEditPermit(),
      createLayerId,
      dispatch: (action) => store.dispatch(action),
      dispatchPrepared: dispatchPreparedMutation,
      endBurst: () => endNudgeBurst(),
      exportBaked: (layerId) => exportBakedLayerPixels(layerId),
      getDocument: () => mirror.getDocument(),
      getReducerDocument,
      hasExportableContent: hasExportableLayerContent,
      history,
      installPrepared: (prepared) =>
        installGeneratedPaintCache(prepared as ReturnType<LayerCacheStore['prepareReplacement']>),
      isCacheReady: isLayerCacheReadyForOp,
      isGestureActive: () => pipeline.isGestureActive(),
      isGuardCurrent: isLayerExportGuardCurrent,
      isPermitCurrent: (permit) => isDocumentEditPermitCurrent(permit as DocumentEditPermit),
      layers: layerCache,
      markDirty: (layerId) => bitmapStore.markLayerDirty(layerId),
      notifyPainted: notifyLayerPainted,
      preparePixels: prepareGeneratedPaintCache,
    },
    rasterize: {
      backend,
      canEdit: () => canEditDocument(),
      dispatch: (action) => store.dispatch(action),
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
      getActiveProjectId: () => store.getState().activeProjectId,
      getCheckerboard: getCheckerboardTile,
      getDocument: () => mirror.getDocument(),
      getEntry: (layerId) => layerCache.get(layerId),
      getMaskPattern: getMaskPatternTile,
      isDisposed: () => disposed,
      isSupportedSource: isSupportedExportSource,
      projectId,
      rasterize: getOrStartLayerRasterization,
      reportError: (layerId, error) => {
        try {
          store.dispatch({
            area: 'canvas-engine',
            context: { error: error instanceof Error ? error.message : String(error), layerId },
            message: 'Layer thumbnail rasterization failed',
            namespace: 'canvas',
            projectId,
            type: 'recordError',
          });
        } catch {
          // Diagnostics must not turn a handled thumbnail failure into a rejection.
        }
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
  const exportCapability: CanvasEngineExportCapability = {
    captureLayerExportGuard: captureCurrentLayerExportGuard,
    exportBakedLayerBlob,
    exportBakedLayerPixels,
    exportLayerPixels: rasterizeLayerPixels,
    exportRasterLayersToPsd,
    extractMaskedArea,
    getCompositeExecutorDeps,
    hasExportableLayerContent,
    isLayerExportGuardCurrent,
  };
  const documentCapability: CanvasDocumentCapability = { getDocument: () => mirror.getDocument() };
  const selectionCapability: CanvasEngineSelectionCapability = {
    deselect,
    eraseSelection,
    fillSelection,
    getSelectionBounds: () => selection.bounds(),
    getSelectionMaskRect: () => selection.mask()?.rect ?? null,
    invertSelection,
    replaceSelectionFromImage,
    selectAll,
  };
  const toolsCapability: CanvasEngineToolCapability = {
    ...interactionController.tools,
    contextMenuLayerIdAt,
    handleEscapePriority,
    onStrokeCommitted,
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

  const prepareSelectObjectStart = (layerId: string): SelectObjectStartContext => {
    const document = mirror.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer) {
      return { status: 'missing' };
    }
    if (layer.type !== 'raster' && layer.type !== 'control') {
      return { status: 'unsupported' };
    }
    if (!layer.isEnabled) {
      return { status: 'disabled' };
    }
    if (layer.isLocked) {
      return { status: 'locked' };
    }
    const guard = captureCurrentLayerExportGuard(layer.id);
    const entry = layerCache.get(layer.id);
    if (!guard || !entry) {
      return { status: 'not-ready' };
    }
    const sourceRect = roundOut(
      transformBounds(
        fromTRS(
          { x: layer.transform.x, y: layer.transform.y },
          layer.transform.rotation,
          layer.transform.scaleX,
          layer.transform.scaleY
        ),
        entry.rect
      )
    );
    if (isEmpty(sourceRect)) {
      return { status: 'not-ready' };
    }
    return { guard, layerId, layerName: layer.name, layerType: layer.type, sourceRect, status: 'ready' };
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
    exportLayerPixels: rasterizeLayerPixels,
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
        store.dispatch({ id: layerId, type: 'setCanvasSelectedLayer' });
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

  const engine: CanvasEngine = {
    diagnostics: diagnosticsCapability,
    document: documentCapability,
    edits: editingController.edits,
    exports: exportCapability,
    history: historyCapability,
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
