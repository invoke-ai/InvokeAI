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

import type { ExecuteCompositePlanDeps } from '@workbench/canvas-engine/export/compositeForGeneration';
import type { PsdExportLayerInput } from '@workbench/canvas-engine/export/psdExport';
import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { FontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import type { OverlayCursor } from '@workbench/canvas-engine/render/overlayRenderer';
import type { RenderScheduler } from '@workbench/canvas-engine/render/scheduler';
import type { Rect, RenderFlags, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { SamInput, SamModel } from '@workbench/generation/canvas/samGraph';
import type {
  CanvasImageRef,
  CanvasDocumentContractV2,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
  WorkbenchState,
} from '@workbench/types';
import type {
  FilterCommitTarget,
  FilterOperationSession,
  LayerFilterSettings,
} from '@workbench/widgets/layers/filterOperationSession';
import type {
  SelectObjectSession,
  SelectObjectSessionPreview,
  SelectObjectSessionProcessResult,
} from '@workbench/widgets/layers/selectObjectSession';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { socketHub } from '@workbench/backend/socketHub';
import { runUtilityGraph, type UtilityGraphResult } from '@workbench/canvas-engine/backend/utilityQueue';
import {
  createEngineStores,
  type EngineStores,
  type SamPointLabel,
  type TextToolOptions,
} from '@workbench/canvas-engine/engineStores';
import { executePsdExport, planPsdExport } from '@workbench/canvas-engine/export/psdExport';
import { createPointerPipeline, type PointerPipeline } from '@workbench/canvas-engine/input/pointerPipeline';
import { createWheelHandler } from '@workbench/canvas-engine/input/wheel';
import { fromTRS, invert as invertMatrix } from '@workbench/canvas-engine/math/mat2d';
import { intersect, isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { createAdjustedSurfaceCache } from '@workbench/canvas-engine/render/adjustedSurfaceCache';
import { applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import {
  blendToComposite,
  compositeDocument,
  createCheckerboardTile,
  shouldSmoothAtZoom,
} from '@workbench/canvas-engine/render/compositor';
import { renderControlTransparency } from '@workbench/canvas-engine/render/controlTransparency';
import { createFontLoader, domFontLoadApi } from '@workbench/canvas-engine/render/fontLoader';
import {
  createLayerCacheStore,
  type LayerCacheEntry,
  type LayerCacheStore,
} from '@workbench/canvas-engine/render/layerCache';
import { colorizeMask, createMaskPatternTile } from '@workbench/canvas-engine/render/maskFill';
import { renderOverlay } from '@workbench/canvas-engine/render/overlayRenderer';
import { createDomRasterBackend, type RasterBackend, type RasterSurface } from '@workbench/canvas-engine/render/raster';
import { rasterizeSource, type ImageResolver, type RasterizeDeps } from '@workbench/canvas-engine/render/rasterizers';
import { textFontString, type TextSource } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';
import { createRenderScheduler } from '@workbench/canvas-engine/render/scheduler';
import { fitThumbnailSize, getLayerThumbnailDisplayKey } from '@workbench/canvas-engine/render/thumbnail';
import { ANTS_STEP_PX, createAntsAnimator, type AntsAnimator } from '@workbench/canvas-engine/selection/marchingAnts';
import { eraseMaskedRegion, fillMaskedRegion } from '@workbench/canvas-engine/selection/selectionOps';
import { createSelectionState, type SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import { createBboxTool } from '@workbench/canvas-engine/tools/bboxTool';
import { createBrushTool } from '@workbench/canvas-engine/tools/brushTool';
import { createColorPickerTool } from '@workbench/canvas-engine/tools/colorPickerTool';
import { createEraserTool } from '@workbench/canvas-engine/tools/eraserTool';
import { createGradientTool } from '@workbench/canvas-engine/tools/gradientTool';
import { createLassoTool } from '@workbench/canvas-engine/tools/lassoTool';
import {
  hittableLayerRect,
  hittableLayerSize,
  layerOutlineCorners,
  topLayerAt,
} from '@workbench/canvas-engine/tools/moveHitTest';
import { createMoveTool } from '@workbench/canvas-engine/tools/moveTool';
import { stepBrushSize } from '@workbench/canvas-engine/tools/paintConstants';
import { createSamTool } from '@workbench/canvas-engine/tools/samTool';
import { createShapeTool } from '@workbench/canvas-engine/tools/shapeTool';
import { createTextTool } from '@workbench/canvas-engine/tools/textTool';
import { createTransformTool } from '@workbench/canvas-engine/tools/transformTool';
import {
  bakeMatrix,
  type LayerTransform,
  transformOverlayGeometry,
} from '@workbench/canvas-engine/transform/transformMath';
import { createViewport, MAX_DPR, type Viewport } from '@workbench/canvas-engine/viewport';
import {
  buildFilterDefaults,
  DEFAULT_CONTROL_FILTER_TYPE,
  getFilterDefinition,
} from '@workbench/generation/canvas/filterGraphs';
import { createFilterOperationSession } from '@workbench/widgets/layers/filterOperationSession';
import { runLayerFilter } from '@workbench/widgets/layers/layerFilterRunner';
import {
  createInpaintMaskFromImage,
  createControlLayer,
  createRegionalGuidanceFromImage,
  DEFAULT_INPAINT_MASK_FILL,
  nextInpaintMaskName,
  nextControlLayerName,
  nextRegionalGuidanceFillColor,
  nextRegionalGuidanceName,
} from '@workbench/widgets/layers/layerOps';
import { getSelectedModelBase } from '@workbench/widgets/layers/selectedModel';
import { createSelectObjectSession } from '@workbench/widgets/layers/selectObjectSession';

import type { StrokeCommittedEvent, Tool, ToolContext } from './tools/tool';

import { uploadCanvasImage } from './backend/canvasImages';
import { createCanvasOperationController, type CanvasOperationController } from './canvasOperationController';
import { createBitmapStore, type BitmapStore } from './document/bitmapStore';
import { createDocumentMirror, type DocumentMirror } from './document/documentMirror';
import { mergeDownMatrix } from './document/mergeDown';
import { planMergeVisibleRuns, planNextMergeVisibleStep } from './document/mergeVisible';
import {
  getSourceBounds,
  getSourceContentRect,
  isMaskLayer,
  isMergeableRasterLayer,
  isRenderableLayer,
  renderableSourceOf,
} from './document/sources';
import { createDocumentPatchEntry } from './history/documentPatch';
import { createHistory, type History, type HistoryEntry } from './history/history';
import { createImagePatchEntry, type ImagePatchApply } from './history/imagePatch';
import { createViewTool } from './tools/viewTool';

/**
 * The input to {@link CanvasEngine.setStagedPreview}: either a persisted image
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
}

/**
 * Result of {@link CanvasEngine.mergeVisibleRasterLayers}: `'merged'` when the
 * fold ran, `'not-ready'` when a participant's cache is still decoding/stale (the
 * whole op was refused — surface feedback), `'nothing'` when there was nothing to
 * merge.
 */
export type MergeVisibleResult = 'merged' | 'not-ready' | 'nothing';

export type BooleanRasterOperation = 'intersect' | 'cutout' | 'cutaway' | 'exclude';
export type BooleanRasterResult = 'merged' | 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty';

export type ExtractMaskedAreaResult =
  | { status: 'extracted'; layerId: string }
  | { status: 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty' };

export type CropLayerResult =
  | { status: 'cropped' }
  | { status: 'missing' | 'locked' | 'unsupported' | 'empty' | 'not-ready' | 'busy' }
  | { status: 'failed'; message: string };

export type CommitRasterFilterResult =
  | { status: 'committed'; layerId: string }
  | { status: 'missing' | 'locked' | 'stale' | 'unsupported' | 'busy' | 'aborted' }
  | { status: 'failed'; message: string };

export interface CommitRasterFilterOptions {
  guard: LayerExportGuard;
  image: CanvasImageRef;
  rect: Rect;
  mode: 'replace' | 'copy';
  filter?: LayerFilterSettings;
  target?: FilterCommitTarget;
  signal?: AbortSignal;
}

export type GeneratedImageTarget = 'replace' | 'copy-raster' | 'copy-control';

export type CommitGeneratedImageResult =
  | { status: 'committed'; layerId: string }
  | { status: 'missing' | 'locked' | 'stale' | 'unsupported' | 'busy' | 'aborted' }
  | { status: 'failed'; message: string };

export interface CommitGeneratedImageOptions {
  guard: LayerExportGuard;
  image: CanvasImageRef;
  origin: Vec2;
  target: GeneratedImageTarget;
  signal?: AbortSignal;
}

export type ReplaceSelectionFromImageResult =
  | { status: 'selected' }
  | { status: 'aborted' | 'missing' | 'locked' | 'stale' | 'unsupported' | 'busy' }
  | { status: 'failed'; message: string };

export type MaskImageResultTarget = 'inpaint_mask' | 'regional_guidance';

export interface CommitMaskImageResultOptions {
  guard: LayerExportGuard;
  image: CanvasImageRef;
  rect: Rect;
  target: MaskImageResultTarget;
  signal?: AbortSignal;
}

export type CommitMaskImageResult =
  | { status: 'committed'; layerId: string }
  | { status: 'aborted' | 'missing' | 'locked' | 'stale' | 'unsupported' | 'busy' };

/**
 * Result of {@link CanvasEngine.exportRasterLayersToPsd}: `'exported'` on
 * success, `'nothing'` when there are no raster layers with content, `'too-large'`
 * when the union bounds exceed the PSD dimension cap, and `'not-ready'` when a
 * participant's cache is still decoding (nothing exported — surface feedback).
 */
export type PsdExportResult = 'exported' | 'nothing' | 'too-large' | 'not-ready';

export interface ExportLayerPixelsOptions {
  /** Export disabled/hidden layers too. Used by PSD/save-layer actions. */
  includeDisabled?: boolean;
  /** Defaults false. When true, bakes raster adjustments into a scratch surface. */
  applyAdjustments?: boolean;
}

/** Opaque snapshot identity carried through async layer operations. */
export interface LayerExportGuard {
  readonly projectId: string;
  readonly layerId: string;
  readonly layer: CanvasLayerContract;
  readonly cacheVersion: number;
  readonly documentGeneration: number;
}

export type ExportLayerPixelsResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' };

export type ExportBakedLayerPixelsOptions = Omit<ExportLayerPixelsOptions, 'applyAdjustments'> & {
  /** Defaults true. When false, leaves raster adjustments non-destructive. */
  applyAdjustments?: boolean;
};

export type ExportBakedLayerPixelsResult = ExportLayerPixelsResult;

export type ExportBakedLayerBlobResult =
  | { status: 'ok'; blob: Blob; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' };

interface RasterizationJob {
  controller: AbortController;
  version: number;
  documentGeneration: number;
  source: CanvasLayerSourceContract;
  promise: Promise<'published' | 'stale' | 'error'>;
}

export type LayerThumbnailRequestResult = 'ready' | 'stale' | 'error' | 'missing' | 'unsupported';

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
  /** Raster surface/bitmap factory. Defaults to the DOM backend. */
  backend?: RasterBackend;
  /** Resolves persisted image assets to blobs for decoding. */
  imageResolver: ImageResolver;
  /** Injectable Select Object queue/upload seams; production defaults use the utility queue. */
  selectObjectDeps?: {
    uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
    runGraph(options: {
      graph: Parameters<typeof runUtilityGraph>[0]['graph'];
      outputNodeId?: string;
      signal?: AbortSignal;
    }): Promise<Pick<UtilityGraphResult, 'imageName' | 'origin'>>;
  };
  /** Injectable filter queue/upload seams; production defaults use intermediate canvas images and the utility queue. */
  filterDeps?: {
    uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
    runGraph(options: {
      graph: Parameters<typeof runUtilityGraph>[0]['graph'];
      outputNodeId?: string;
      signal?: AbortSignal;
    }): Promise<UtilityGraphResult>;
  };
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
}

export type StartSelectObjectSessionResult = 'started' | 'missing' | 'disabled' | 'unsupported' | 'not-ready';
export type StartFilterOperationResult = 'started' | 'missing' | 'disabled' | 'locked' | 'unsupported' | 'not-ready';

export type SelectObjectSaveTarget = 'raster' | 'control' | MaskImageResultTarget;
export type SaveSelectObjectSessionResult = CommitGeneratedImageResult | CommitMaskImageResult;

export interface SelectObjectSessionUpdate {
  input?: SamInput;
  pointLabel?: SamPointLabel;
  model?: SamModel;
  invert?: boolean;
  applyPolygonRefinement?: boolean;
  autoProcess?: boolean;
  isolatedPreview?: boolean;
}

interface SelectObjectCommitOwner {
  controller: AbortController;
  guard: LayerExportGuard;
  inputHash: string;
  preview: SelectObjectSessionPreview<RasterSurface>;
  previewId: number;
  session: SelectObjectSession<RasterSurface>;
  token: number;
}

/** The public engine handle. */
export interface CanvasEngine {
  readonly projectId: string;
  /** The single active guarded canvas operation and its React-free external-store state. */
  readonly canvasOperations: CanvasOperationController;
  /** The pan/zoom viewport. */
  getViewport(): Viewport;
  /** The transient stores React subscribes to. */
  readonly stores: EngineStores;
  /** Binds the two stacked canvases as render targets and starts the render loop. */
  attach(screenCanvas: HTMLCanvasElement, overlayCanvas: HTMLCanvasElement): void;
  /** Unbinds the canvases, removes listeners, and pauses the render loop. */
  detach(): void;
  /** Sizes the backing stores to `css × min(dpr, 2)` and updates the viewport. */
  resize(cssWidth: number, cssHeight: number, dpr: number): void;
  /** Activates a tool (deactivating the previous one). */
  setTool(toolId: ToolId, opts?: { temporary?: boolean }): void;
  /** Selects the requested raster/control layer, starts Select Object, and activates the SAM tool. */
  startSelectObject(layerId: string): StartSelectObjectSessionResult;
  /** Starts one engine-owned guarded filter operation for a raster or control layer. */
  startFilterOperation(layerId: string, recommendedFilterType?: string | null): StartFilterOperationResult;
  updateFilterOperation(draft: LayerFilterSettings): void;
  processFilterOperation(): Promise<void>;
  resetFilterOperation(settings: Record<string, unknown>): void;
  commitFilterOperation(
    target: FilterCommitTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<void>;
  cancelFilterOperation(): void;
  updateSelectObjectSession(changes: SelectObjectSessionUpdate): void;
  processSelectObjectSession(): Promise<SelectObjectSessionProcessResult>;
  applySelectObjectSession(): Promise<ReplaceSelectionFromImageResult>;
  saveSelectObjectSession(
    target: SelectObjectSaveTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<SaveSelectObjectSessionResult>;
  resetSelectObjectSession(): void;
  cancelSelectObjectSession(): void;
  disposeSelectObjectSession(): void;
  /** Locks interaction to the view tool; used while staged canvas results await a decision. */
  setInteractionLocked(locked: boolean): void;
  /**
   * Steps the active brush/eraser diameter by one notch (`+1` grows, `-1`
   * shrinks) — the same helper the ctrl+wheel path drives, so the `[`/`]`
   * hotkeys and ctrl+wheel stay in lockstep. A no-op when the active tool is
   * neither brush nor eraser.
   */
  stepBrushSize(direction: 1 | -1): void;
  /**
   * Subscribes to completed brush/eraser strokes. Downstream tasks wire
   * persistence (P2.2) and undo/redo history (P2.3) to this; the engine itself
   * does not upload or record history. Returns an unsubscribe function.
   */
  onStrokeCommitted(listener: (event: StrokeCommittedEvent) => void): () => void;
  /**
   * Flushes any pending paint-bitmap uploads immediately and resolves once they
   * settle. A barrier for invoke/export/autosave: the persisted document is
   * guaranteed to reference the latest painted pixels after this resolves.
   */
  flushPendingUploads(): Promise<void>;
  /**
   * The side-effecting dependencies `executeCompositePlan` needs, wired to this
   * engine: its raster `backend`, a rasterize-or-throw `getLayerSurface`, and an
   * intermediate-image `uploadImage`. The caller supplies the dedupe cache (it
   * is per-engine, not per-invoke). `getLayerSurface` never returns a blank
   * surface — a missing/unrasterizable layer throws, so generation can't
   * silently omit a layer.
   */
  getCompositeExecutorDeps(): Omit<ExecuteCompositePlanDeps, 'dedupe'>;
  /** Centers and zooms to fit the document content. */
  fitToView(): void;
  /**
   * Sets the grid size (document px) the bbox tool snaps positions and sizes to.
   * The engine can't know the active model, so React feeds this from generate
   * settings (see `gridSizeForModelBase`); defaults to 8.
   */
  setBboxGrid(size: number): void;
  /**
   * Reverts the most recent canvas edit via the engine-owned history (paint
   * pixels or structural document patch). A no-op when there is nothing to undo.
   * Canvas-scoped: it never touches project-level (reducer) undo.
   */
  undo(): void;
  /** Re-applies the most recently undone canvas edit. A no-op when there is nothing to redo. */
  redo(): void;
  /**
   * Draws a layer's cached pixels, scaled to fit a `maxSize` box (aspect ratio
   * preserved, never upscaled), onto `target`, sizing `target`'s backing store
   * to the fitted dimensions. Returns `false` (drawing nothing) when the layer
   * has no live raster cache — the caller then falls back to a placeholder. The
   * layers panel redraws when a layer's `stores.thumbnailVersion` bumps.
   */
  drawLayerThumbnail(layerId: string, target: HTMLCanvasElement, maxSize: number): boolean;
  /** Ensures thumbnail pixels exist, independent of canvas attachment or layer visibility. */
  requestLayerThumbnail(layerId: string): Promise<LayerThumbnailRequestResult>;
  /**
   * Composites the layer identified by `upperLayerId` down INTO the layer
   * directly below it, baking the upper layer's pixels (with its opacity/blend) into a new
   * content-sized (union-extent) paint cache in the below layer's local space, then dispatches
   * `mergeCanvasLayersDown` so the reducer collapses the two layers. The new
   * pixels are persisted through the normal bitmap-store path. Returns `false`
   * (a no-op) when there is no layer below, either layer is not a live raster
   * cache, or the below transform is non-invertible. Not recorded on the
   * engine's undo history (the inverse would need the discarded pixels).
   */
  mergeLayerDown(upperLayerId: string): boolean;
  /**
   * Applies a Canvas2D boolean operation to an adjacent raster pair, inserts the
   * paint-backed result above them, and disables (but preserves) both sources.
   * The complete change is one engine-history entry.
   */
  booleanMergeRasterLayers(upperLayerId: string, operation: BooleanRasterOperation): Promise<BooleanRasterResult>;
  /** Extracts the enabled raster-layer composite through one inpaint mask. */
  extractMaskedArea(maskLayerId: string): Promise<ExtractMaskedAreaResult>;
  /**
   * Folds ALL visible mergeable raster layers together (the raster group-header
   * "merge visible" action; legacy `mergeVisibleOfType`). Plans runs via the pure
   * `planMergeVisibleRuns` — interleaved non-raster layers and hidden rasters do
   * not block a merge (the compositor draws by group rank, so they never render
   * between rasters); a rendering non-participant raster (visible locked /
   * parametric) splits the fold into independent runs. Pre-flights the ENTIRE
   * fold — every participant's cache must be ready and every consecutive pair's
   * merge matrix computable — BEFORE the first pixel is touched, so the op never
   * leaves a half-merged stack: returns `'not-ready'` (doing nothing) when any
   * participant is still decoding/stale, `'nothing'` when there is nothing to
   * merge (or mid-gesture), else performs the fold (reorder + `mergeLayerDown`
   * per step) and returns `'merged'`. Not undoable, like `mergeLayerDown`.
   */
  mergeVisibleRasterLayers(): MergeVisibleResult;
  /**
   * Exports ALL raster layers (with content) to a Photoshop `.psd` file and
   * triggers a browser download. Read-only: no dispatches, no history entries,
   * no document mutation. Hidden layers are exported with `hidden: true`; each
   * layer's non-destructive adjustments are baked so the PSD matches the canvas.
   * The `ag-psd` library is loaded lazily (its own chunk). Pre-flights readiness:
   * returns `'not-ready'` (exporting nothing) when a participant's cache is still
   * decoding, `'nothing'` when no raster layer has content, `'too-large'` when the
   * union bounds exceed the PSD dimension cap, else `'exported'`. `fileName` is
   * the download name (extension appended if absent).
   */
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
  /**
   * True when a layer has pixels that Save/Clipboard can export. Hidden layers
   * are eligible; unsupported polygon sources, missing layers, and empty or
   * stale unpersisted paint/mask caches are not.
   */
  hasExportableLayerContent(layerId: string): boolean;
  /**
   * Ensures one layer's pixels are rasterized in the shared layer cache and returns
   * the cache surface plus its layer-local content rect. Read-only: no dispatches,
   * no history entry, no document mutation. Joins an equivalent in-flight
   * rasterization and returns `not-ready` only when that work becomes stale or fails.
   */
  exportLayerPixels(layerId: string, options?: ExportLayerPixelsOptions): Promise<ExportLayerPixelsResult>;
  /**
   * Exports a layer as a new world-space surface: its cache is drawn through the
   * layer transform into the returned document-space rect, and raster adjustments
   * are baked by default. Opacity/blend are intentionally not baked.
   */
  exportBakedLayerPixels(
    layerId: string,
    options?: ExportBakedLayerPixelsOptions
  ): Promise<ExportBakedLayerPixelsResult>;
  /** Encodes {@link exportBakedLayerPixels} as a PNG blob for save/copy/upload actions. */
  exportBakedLayerBlob(layerId: string, options?: ExportBakedLayerPixelsOptions): Promise<ExportBakedLayerBlobResult>;
  /** Captures the exact current layer/cache/document/project snapshot when its cache is ready. */
  captureLayerExportGuard(layerId: string): LayerExportGuard | null;
  /** True only while the exact exported layer/cache/document/project snapshot remains live. */
  isLayerExportGuardCurrent(guard: LayerExportGuard): boolean;
  /**
   * Destructively crops a layer to its overlap with the current bbox, baking its
   * transform/adjustments into content-sized paint/mask pixels. The full layer
   * contract and cache snapshot are restored by one undoable history entry.
   */
  cropLayerToBbox(layerId: string): Promise<CropLayerResult>;
  /** Commits a durable raster-filter result only while its export guard remains current. */
  commitRasterFilterResult(options: CommitRasterFilterOptions): Promise<CommitRasterFilterResult>;
  /** Replaces or copies a layer with a durable generated image while its export guard remains current. */
  commitGeneratedImageResult(options: CommitGeneratedImageOptions): Promise<CommitGeneratedImageResult>;
  /** Adds a durable SAM image as a new mask layer while its source export guard remains current. */
  commitMaskImageResult(options: CommitMaskImageResultOptions): Promise<CommitMaskImageResult>;
  /** Creates a new raster paint layer above `layerId` from that layer's baked pixels. */
  copyLayerToRaster(layerId: string): Promise<string | null>;
  /**
   * Rasterizes a parametric (shape/gradient/text) layer to a paint layer: bakes its
   * current appearance into a content-sized paint bitmap (persisted through the
   * normal dirty path) and converts the source via `convertCanvasLayer`. UNDOABLE
   * via a composed entry whose inverse re-converts to the original parametric
   * source (regenerated from params, so no pixel snapshot is kept). Returns
   * `false` (a no-op) mid-gesture or when the layer is missing, not a raster
   * layer, locked, or not a rasterizable parametric source.
   */
  rasterizeLayer(layerId: string): boolean;
  /**
   * Performs a UI-initiated structural document edit under the engine-owned
   * history: dispatches `forward` immediately and records a reversible entry
   * whose undo dispatches `inverse` and whose redo re-dispatches `forward`. Use
   * for panel ops (add/remove/duplicate/reorder/rename/opacity/blend/eye/lock)
   * so they share the canvas undo stack with paint edits.
   */
  commitStructural(label: string, forward: WorkbenchAction, inverse: WorkbenchAction): void;
  /** Applies a guarded live structural preview without adding a history entry. */
  applyStructuralPreview(action: WorkbenchAction): boolean;
  /** Adds a contract-built layer while preserving the source layer's live cache pixels. */
  commitLayerCopy(label: string, sourceLayerId: string, layer: CanvasLayerContract, index: number): boolean;
  /** Converts the expected immutable live layer while preserving its cache pixels through undo/redo. */
  commitLayerConversion(label: string, expectedLiveLayer: CanvasLayerContract, after: CanvasLayerContract): boolean;
  /**
   * Updates the active transform session's live transform (a numeric options-bar
   * edit). Refreshes the preview; no dispatch until Apply. A no-op with no session.
   */
  updateTransformSession(transform: LayerTransform): void;
  /**
   * Commits the active transform session as ONE undoable entry — a param edit for
   * image layers, a pixel bake for paint layers — then clears the session. A no-op
   * with no session or an unchanged transform. Bound to `enter` and the options
   * bar's Apply button.
   */
  applyTransform(): void;
  /**
   * Cancels the active transform session (drops the preview, no dispatch). Bound
   * to `esc`, tool switch, and the options bar's Cancel button.
   */
  cancelTransform(): void;
  /**
   * Opens a CREATE-mode text-editing session at `docPoint` (no layer yet), seeded
   * from the current text options. The text tool calls this on an empty click.
   */
  openTextCreate(docPoint: Vec2): void;
  /**
   * Opens an EDIT-mode text-editing session on an existing text layer (captures
   * its committed source for the undo inverse). The text tool calls this on a hit.
   */
  openTextEdit(layerId: string): void;
  /**
   * Updates the active text session's live style (font/size/weight/lineHeight/
   * align/color). No dispatch — the portal restyles and the change folds into the
   * single commit. The text options bar drives this while a session is open.
   */
  updateTextEditStyle(patch: Partial<TextToolOptions>): void;
  /**
   * Commits the active text session as ONE undoable edit: create → `addCanvasLayer`
   * with the final `content` (inverse removes); edit → `updateCanvasLayerSource`
   * (exact inverse). A no-change edit or an empty create dispatches nothing (cancel
   * semantics). `content` comes from the React contenteditable (the engine never
   * sees per-keystroke input); `styleChanges` folds in any final style override.
   * Bound to blur / click-outside / `mod+enter`.
   */
  commitTextEdit(content: string, styleChanges?: Partial<TextToolOptions>): void;
  /**
   * Registers (or clears with `null`) a getter the React text portal provides so
   * the engine can read the live contenteditable content at commit time without
   * per-keystroke traffic. Set in the portal's ref callback; cleared on unmount.
   */
  setTextEditContentReader(reader: (() => string) | null): void;
  /**
   * Commits an open text-edit session using the live content from the registered
   * reader (falling back to the session content when none is set). Returns `true`
   * when a session was open and committed. The pointer pipeline calls this on a
   * canvas pointerdown so clicking away commits the session ("click elsewhere to
   * commit"); it runs before any gesture starts so the commit is never swallowed.
   */
  commitOpenTextSession(): boolean;
  /**
   * Runs the Escape priority ladder (text session → transform session → deselect)
   * after the pointer pipeline has cancelled any in-flight gesture. The pipeline
   * wires this to the window keydown; exposed for tests. `gestureWasActive`
   * suppresses deselect when a drag just consumed the Escape.
   */
  handleEscapePriority(opts: { gestureWasActive: boolean }): void;
  /** Cancels the active text-editing session (drops it, no dispatch). Bound to `esc`. */
  cancelTextEdit(): void;
  /** Selects the whole document (transient selection; `mod+a`). */
  selectAll(): void;
  /** Clears the selection (`mod+d` / Escape when idle). */
  deselect(): void;
  /** Replaces the transient selection from a guarded alpha-bearing image result. */
  replaceSelectionFromImage(
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal
  ): Promise<ReplaceSelectionFromImageResult>;
  /** Inverts the selection within the document (`mod+shift+i`). */
  invertSelection(): void;
  /** Clears an inpaint/regional mask as one undoable edit, preserving its non-pixel settings. */
  clearMask(layerId: string): boolean;
  /**
   * Inverts a mask layer's alpha in place over `content ∪ bbox`, as one undoable
   * edit persisted through the normal dirty path. A no-op (returns `false`)
   * mid-gesture, or when the layer is missing, not a mask, or locked/hidden.
   */
  invertMask(layerId: string): boolean;
  /**
   * Fills the selection region on the selected paint layer with the current brush
   * color, as one undoable edit. A no-op mid-gesture, with no selection, or with a
   * locked/hidden/non-paint selected layer.
   */
  fillSelection(): void;
  /** Erases the selection region on the selected paint layer, as one undoable edit. */
  eraseSelection(): void;
  /**
   * Nudges the selected layer by `(dx, dy)` document pixels as a single undoable
   * structural edit (bounds/lock logic stays engine-side). A no-op with no
   * selection or a locked/hidden selected layer. Rapid same-layer nudges coalesce
   * into one history entry. Widget arrow-key hotkeys drive this.
   */
  nudgeSelectedLayer(dx: number, dy: number): void;
  /**
   * Shows a staged generation preview above committed layers, or clears it with
   * `null`. A final `imageName` candidate may provide its own placement; legacy
   * candidates and live `dataUrl` progress frames follow the CURRENT bbox.
   *
   * The input is decoded to a surface off the main path — an `imageName` through
   * the engine's `imageResolver`, or a `dataUrl` with explicit dimensions.
   * Decoding is async and latest-call-wins: a stale decode resolving after a
   * newer `setStagedPreview`/clear is discarded (version-guarded like the
   * rasterize path), so rapid candidate cycling never flashes an older result.
   * Explicit placement matches candidate acceptance; otherwise the preview's
   * top-left tracks the bbox at render time.
   */
  setStagedPreview(input: StagedPreviewInput | null): void;
  /** Shows a filter preview only while the exact exported layer snapshot remains current. */
  setGuardedFilterPreview(
    layerId: string,
    input: FilterPreviewInput,
    guard: LayerExportGuard
  ): Promise<'shown' | 'missing' | 'stale'>;
  /**
   * The top-most VISIBLE layer id under a SCREEN-space point (relative to the
   * input element's top-left), for a right-click context menu on the canvas
   * surface — or `null` when nothing is hit OR an interaction is mid-flight (a
   * live paint/drag gesture, or an open transform / text-edit session), in which
   * case the menu must not open over an in-progress edit. Group-rank consistent
   * and live-cache aware: it shares {@link topLayerAt} with the move tool, so the
   * layer this returns is exactly the one the move tool would auto-select there.
   */
  contextMenuLayerIdAt(screenPoint: Vec2): string | null;
  /** The current mirrored document, or `null`. */
  getDocument(): CanvasDocumentContractV2 | null;
  /**
   * Debug action: drops every layer's raster/adjusted cache and the
   * checkerboard/mask pattern tiles, then forces a full recomposite so everything
   * re-rasterizes from source. Recovers from a suspected stale cache; not undoable
   * (it touches no document state). Flushes pending bitmap uploads first so an
   * un-flushed stroke inside the debounce window is not lost.
   */
  clearCaches(): Promise<void>;
  /** Debug action: resets the engine-owned canvas undo/redo history. Not undoable. */
  clearHistory(): void;
  /** Debug action: logs a summary of the engine's live state (document, tool, view, caches) to the console. */
  logDebugInfo(): void;
  /** Tears everything down: listeners, subscriptions, caches, scheduler. */
  dispose(): void;
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

/** Wraps an existing DOM canvas as a {@link RasterSurface} render target. */
const wrapCanvasSurface = (canvas: HTMLCanvasElement): RasterSurface => {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to acquire a 2D context from the canvas element');
  }
  return {
    canvas,
    ctx,
    get height() {
      return canvas.height;
    },
    resize(width: number, height: number) {
      canvas.width = width;
      canvas.height = height;
    },
    get width() {
      return canvas.width;
    },
  };
};

/** Creates a per-project canvas engine. */
export const createCanvasEngine = (opts: CanvasEngineOptions): CanvasEngine => {
  const { imageResolver, projectId, store } = opts;
  const backend = opts.backend ?? createDomRasterBackend();

  const viewport = createViewport();
  let canvasOperations: CanvasOperationController;
  const layerCache: LayerCacheStore = createLayerCacheStore(backend, {
    onVersionChange: (layerId) => canvasOperations?.invalidateSource(projectId, layerId),
  });
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
  let activeToolId: ToolId = 'view';
  let interactionLocked = false;

  // Transient per-layer transform overrides driving the move/transform drag
  // preview (compositor + overlay read at render time; the mirror stays untouched).
  // The move tool sets only x/y; the transform tool sets the full transform.
  const transformOverrides = new Map<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  >();

  /** Layer id → decoded image name, so removed layers can release their bitmaps. */
  const trackedImageNames = new Map<string, string>();
  /** Every live layer's source image, including layers that have never rasterized. */
  const mirroredLayerImageNames = new Map<string, string>();
  /** Display-only thumbnail inputs mirrored separately from source-cache identity. */
  const thumbnailDisplayKeys = new Map<string, string>();
  /** The newest isolated rasterization job for each layer id. */
  const layerRasterizationJobs = new Map<string, RasterizationJob>();
  /** All physically running jobs, including canceled jobs superseded in the per-layer map. */
  const activeLayerRasterizationJobs = new Set<RasterizationJob>();
  const cancelLayerRasterization = (layerId: string): void => {
    const job = layerRasterizationJobs.get(layerId);
    if (!job) {
      return;
    }
    layerRasterizationJobs.delete(layerId);
    job.controller.abort();
  };
  const cancelAllLayerRasterizations = (): void => {
    const jobs = [...layerRasterizationJobs.values()];
    layerRasterizationJobs.clear();
    for (const job of jobs) {
      job.controller.abort();
    }
  };
  /** Invalidates jobs and export guards across wholesale document replacement. */
  let rasterDocumentGeneration = 0;

  let screenSurface: RasterSurface | null = null;
  let overlaySurface: RasterSurface | null = null;
  let inputEl: HTMLCanvasElement | null = null;
  let disposed = false;

  let selectObjectSession: SelectObjectSession<RasterSurface> | null = null;
  let selectObjectUnsubscribe: (() => void) | null = null;
  let selectObjectGuard: LayerExportGuard | null = null;
  let selectObjectSourceRect: Rect | null = null;
  let selectObjectPointLabel: SamPointLabel = 'include';
  let selectObjectCommitOwner: SelectObjectCommitOwner | null = null;
  let selectObjectCommitToken = 0;
  let samPreview: SelectObjectSessionPreview<RasterSurface> | null = null;
  let filterSession: FilterOperationSession | null = null;
  let filterUnsubscribe: (() => void) | null = null;
  let filterControllerUnsubscribe: (() => void) | null = null;

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
  const adjustedSurfaceCache = createAdjustedSurfaceCache(backend);
  const getAdjustedSurface = (layer: CanvasLayerContract, entry: LayerCacheEntry): RasterSurface | null =>
    layer.type === 'raster' ? adjustedSurfaceCache.get(layer.id, entry, layer.adjustments) : null;

  // The decoded staged-generation preview drawn over the current bbox, or null.
  // `stagedPreviewToken` version-guards the async decode: every set/clear bumps
  // it, and a decode whose token no longer matches is dropped so a slow decode
  // can't clobber a newer selection (mirrors the rasterize race guard).
  let stagedPreview: {
    surface: RasterSurface;
    width: number;
    height: number;
    placement?: StagedPreviewPlacement;
  } | null = null;
  let stagedPreviewToken = 0;
  // Completed-stroke subscribers (persistence P2.2, history P2.3).
  const strokeListeners = new Set<(event: StrokeCommittedEvent) => void>();

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
      uploadImage: (blob) => uploadCanvasImage(blob),
    });

  // Engine-owned canvas history (paint pixel patches + structural patches).
  // Project-level undo deliberately no longer covers the canvas (Phase 0).
  const history: History = createHistory();

  // Nudge coalescing: a rapid burst of same-layer arrow nudges collapses into a
  // single history entry whose inverse restores the burst's ORIGINAL position.
  // Any other edit (stroke/structural/undo/redo/replace) ends the burst.
  const NUDGE_COALESCE_MS = 500;
  let nudgeBurst: { layerId: string; origin: { x: number; y: number }; expiresAt: number } | null = null;
  const endNudgeBurst = (): void => {
    nudgeBurst = null;
  };

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
    adjustedSurfaceCache.delete(layerId);
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
    if (filterPreviews.has(layerId)) {
      clearFilterPreview(layerId);
    }
    scheduler.invalidate({ layers: [layerId] });
  };

  /** Invalidates cached pixels and drops only previews tied to that exact cache version. */
  const invalidateLayerCache = (layerId: string): void => {
    cancelLayerRasterization(layerId);
    layerCache.invalidate(layerId);
    stores.thumbnailStatus.delete(layerId);
    if (filterPreviews.has(layerId)) {
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

  const selection: SelectionState = createSelectionState({
    backend,
    createPath2D: createPath2DImpl,
    getDocumentSize: () => {
      const doc = mirror.getDocument();
      return doc ? { height: doc.height, width: doc.width } : null;
    },
    onChange: () => onSelectionChanged(),
  });

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
    if (!disposed && selection.hasSelection() && inputEl) {
      antsAnimator.start();
    } else {
      antsAnimator.stop();
    }
  }

  const toolContext: ToolContext = {
    applyTransform: () => applyTransform(),
    backend,
    beginTransformSession: (layerId) => beginTransformSession(layerId),
    cancelTextEdit: () => cancelTextEdit(),
    cancelTransform: () => cancelTransform(),
    commitSelection: (commit) => selection.commit(commit),
    commitStructural: (label, forward, inverse) => commitStructural(label, forward, inverse),
    createLayerId,
    createPath2D: createPath2DImpl,
    dispatch: (action) => store.dispatch(action),
    emitStrokeCommitted: (event) => {
      // A fresh stroke ends any open nudge-coalescing burst.
      endNudgeBurst();
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
            ? // The gesture auto-created its paint layer: compose layer creation +
              // stroke into ONE entry, so an undo removes the now-empty layer AND the
              // stroke (skipping the pixel restore — the layer's cache is gone), and a
              // redo re-adds the layer then re-applies the stroke pixels.
              createComposedPaintEntry(event.createdLayer, event, label)
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
    },
    getDocument: () => mirror.getDocument(),
    getSelectionMask: () => selection.mask(),
    invalidate: (payload) => scheduler.invalidate(payload),
    layers: layerCache,
    notifyLayerPainted,
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
    updateSamInput: (input) => updateSelectObjectSession({ input }),
    updateTransformSession: (transform) => updateTransformSession(transform),
    viewport,
  };

  const activeTool = (): Tool | undefined => tools.get(activeToolId);

  /** Applies a CSS cursor to the input element, guarded for node stubs without `style`. */
  const applyCursorToInput = (cursor: string): void => {
    const style = inputEl?.style;
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
    if (activeToolId === 'brush') {
      size = stores.brushOptions.get().size;
    } else if (activeToolId === 'eraser') {
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
    const job = layerRasterizationJobs.get(layer.id);
    const source = renderableSourceOf(layer);
    return (
      !!job &&
      !!source &&
      job.version === layerCache.version(layer.id) &&
      job.documentGeneration === rasterDocumentGeneration &&
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
    const documentGeneration = rasterDocumentGeneration;
    const source = structuredClone(liveSource);
    const existing = layerRasterizationJobs.get(layer.id);
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
          rasterDocumentGeneration !== documentGeneration ||
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
    layerRasterizationJobs.set(layer.id, job);
    activeLayerRasterizationJobs.add(job);
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
          layerRasterizationJobs.get(layer.id) !== job ||
          rasterDocumentGeneration !== documentGeneration ||
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
          layerRasterizationJobs.get(layer.id) !== job ||
          rasterDocumentGeneration !== documentGeneration ||
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
        if (layerRasterizationJobs.get(layer.id) === job) {
          layerRasterizationJobs.delete(layer.id);
        }
        activeLayerRasterizationJobs.delete(job);
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
    documentGeneration: rasterDocumentGeneration,
    layer,
    layerId: layer.id,
    projectId,
  });

  const applyLocalRasterAdjustments = (
    result: Extract<ExportLayerPixelsResult, { status: 'ok' }>,
    shouldApply: boolean
  ): Extract<ExportLayerPixelsResult, { status: 'ok' }> => {
    const layer = result.guard.layer;
    if (!shouldApply || layer.type !== 'raster' || !layer.adjustments) {
      return result;
    }
    const surface = backend.createSurface(result.rect.width, result.rect.height);
    const ctx = surface.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, result.rect.width, result.rect.height);
    ctx.drawImage(result.surface.canvas, 0, 0);
    const imageData = ctx.getImageData(0, 0, result.rect.width, result.rect.height);
    applyAdjustments(imageData, layer.adjustments);
    ctx.putImageData(imageData, 0, 0);
    return { ...result, surface };
  };

  const isLayerExportGuardCurrent = (guard: LayerExportGuard): boolean => {
    if (
      disposed ||
      guard.projectId !== projectId ||
      store.getState().activeProjectId !== projectId ||
      guard.documentGeneration !== rasterDocumentGeneration
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

  canvasOperations = createCanvasOperationController({ isGuardCurrent: isLayerExportGuardCurrent });
  const documentEditOwner = Symbol('canvas-operation-document-edit-owner');
  type DocumentEditPermit = { epoch: number; owner?: symbol };
  let documentEditEpoch = 0;
  let documentEditingLocked = false;
  const syncDocumentEditingLock = (): void => {
    const nextLocked = canvasOperations.getSnapshot().status === 'active';
    if (nextLocked !== documentEditingLocked) {
      documentEditingLocked = nextLocked;
      documentEditEpoch += 1;
    }
    stores.documentEditingLocked.set(nextLocked);
  };
  const unsubscribeDocumentEditingLock = canvasOperations.subscribe(syncDocumentEditingLock);
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

  const rasterizeLayerPixels = async (
    layerId: string,
    options: ExportLayerPixelsOptions = {}
  ): Promise<ExportLayerPixelsResult> => {
    const doc = mirror.getDocument();
    if (!doc) {
      return { status: 'missing' };
    }
    const layer = doc.layers.find((candidate) => candidate.id === layerId);
    const source = layer ? renderableSourceOf(layer) : null;
    if (!layer || !source) {
      return { status: 'missing' };
    }
    if (!options.includeDisabled && !layer.isEnabled) {
      return { status: 'disabled' };
    }
    if (!isSupportedExportSource(source)) {
      return { status: 'unsupported' };
    }
    const liveEntry = layerCache.get(layerId);
    if (liveEntry && !liveEntry.stale && !isCurrentRasterizationJob(layer) && !isEmpty(liveEntry.rect)) {
      return applyLocalRasterAdjustments(
        {
          guard: captureLayerExportGuard(layer, liveEntry),
          rect: liveEntry.rect,
          status: 'ok',
          surface: liveEntry.surface,
        },
        options.applyAdjustments === true
      );
    }
    const contentRect = getSourceContentRect(layer, doc);
    if (isEmpty(contentRect)) {
      return { status: 'empty' };
    }
    const rasterized = await getOrStartLayerRasterization(layer, doc);
    if (rasterized !== 'published') {
      return { status: 'not-ready' };
    }
    const currentDocument = mirror.getDocument();
    const currentLayer = currentDocument?.layers.find((candidate) => candidate.id === layerId);
    const entry = layerCache.get(layerId);
    if (!currentLayer || !entry || entry.stale) {
      return { status: 'not-ready' };
    }
    const currentSource = renderableSourceOf(currentLayer);
    if (!currentSource) {
      return { status: 'missing' };
    }
    if (!options.includeDisabled && !currentLayer.isEnabled) {
      return { status: 'disabled' };
    }
    if (!isSupportedExportSource(currentSource)) {
      return { status: 'unsupported' };
    }
    if (isEmpty(entry.rect)) {
      return { status: 'empty' };
    }
    return applyLocalRasterAdjustments(
      {
        guard: captureLayerExportGuard(currentLayer, entry),
        rect: entry.rect,
        status: 'ok',
        surface: entry.surface,
      },
      options.applyAdjustments === true
    );
  };

  const exportBakedLayerPixels = async (
    layerId: string,
    options: ExportBakedLayerPixelsOptions = {}
  ): Promise<ExportBakedLayerPixelsResult> => {
    const raw = await rasterizeLayerPixels(layerId, { ...options, applyAdjustments: false });
    if (raw.status !== 'ok') {
      return raw;
    }
    const layer = raw.guard.layer;

    const matrix = fromTRS(
      { x: layer.transform.x, y: layer.transform.y },
      layer.transform.rotation,
      layer.transform.scaleX,
      layer.transform.scaleY
    );
    const worldRect = roundOut(transformBounds(matrix, raw.rect));
    if (isEmpty(worldRect)) {
      return { status: 'empty' };
    }

    const surface = backend.createSurface(worldRect.width, worldRect.height);
    const ctx = surface.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, worldRect.width, worldRect.height);
    ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - worldRect.x, matrix.f - worldRect.y);
    ctx.drawImage(raw.surface.canvas, raw.rect.x, raw.rect.y);

    if (options.applyAdjustments !== false && layer.type === 'raster' && layer.adjustments) {
      const imageData = ctx.getImageData(0, 0, worldRect.width, worldRect.height);
      applyAdjustments(imageData, layer.adjustments);
      ctx.putImageData(imageData, 0, 0);
    }

    return { guard: raw.guard, rect: worldRect, status: 'ok', surface };
  };

  const exportBakedLayerBlob = async (
    layerId: string,
    options: ExportBakedLayerPixelsOptions = {}
  ): Promise<ExportBakedLayerBlobResult> => {
    const result = await exportBakedLayerPixels(layerId, options);
    if (result.status !== 'ok') {
      return result;
    }
    const blob = await backend.encodeSurface(result.surface, 'image/png');
    if (!isLayerExportGuardCurrent(result.guard)) {
      return { status: 'not-ready' };
    }
    return { blob, guard: result.guard, rect: result.rect, status: 'ok' };
  };

  const cropLayerToBbox = async (layerId: string): Promise<CropLayerResult> => {
    const permit = captureDocumentEditPermit();
    if (!permit || pipeline.isGestureActive()) {
      return { status: 'busy' };
    }
    endNudgeBurst();
    const document = mirror.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer) {
      return { status: 'missing' };
    }
    if (layer.isLocked) {
      return { status: 'locked' };
    }
    const source = renderableSourceOf(layer);
    if (!source || !isSupportedExportSource(source)) {
      return { status: 'unsupported' };
    }

    try {
      const exported = await exportBakedLayerPixels(layerId, { includeDisabled: true });
      if (!isDocumentEditPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      if (exported.status !== 'ok') {
        switch (exported.status) {
          case 'missing':
          case 'unsupported':
          case 'empty':
          case 'not-ready':
            return { status: exported.status };
          case 'disabled':
            return { status: 'not-ready' };
        }
      }

      const liveDocument = mirror.getDocument();
      const liveLayer = liveDocument?.layers.find((candidate) => candidate.id === layerId);
      if (!liveDocument || !liveLayer) {
        return { status: 'missing' };
      }
      if (!isDocumentEditPermitCurrent(permit) || pipeline.isGestureActive()) {
        return { status: 'busy' };
      }
      if (liveLayer.isLocked) {
        return { status: 'locked' };
      }
      if (!isLayerExportGuardCurrent(exported.guard)) {
        return { status: 'not-ready' };
      }
      const liveSource = renderableSourceOf(liveLayer);
      if (!liveSource || !isSupportedExportSource(liveSource)) {
        return { status: 'unsupported' };
      }

      const overlap = intersect(exported.rect, roundOut(liveDocument.bbox));
      if (!overlap) {
        return { status: 'empty' };
      }
      const cropRect = roundOut(overlap);
      if (isEmpty(cropRect)) {
        return { status: 'empty' };
      }

      const beforePixels = captureLayerCache(liveLayer, liveDocument);
      if (!beforePixels || beforePixels === 'not-ready') {
        return { status: 'not-ready' };
      }
      const before = structuredClone(liveLayer);
      const cropped = backend.createSurface(cropRect.width, cropRect.height);
      const cctx = cropped.ctx;
      cctx.setTransform(1, 0, 0, 1, 0, 0);
      cctx.clearRect(0, 0, cropRect.width, cropRect.height);
      cctx.drawImage(exported.surface.canvas, exported.rect.x - cropRect.x, exported.rect.y - cropRect.y);

      const identity = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
      const paintSource = { bitmap: null, offset: { x: cropRect.x, y: cropRect.y }, type: 'paint' } as const;
      let after: CanvasLayerContract;
      if (before.type === 'raster') {
        const { adjustments: _adjustments, ...rest } = before;
        after = { ...rest, source: paintSource, transform: identity };
      } else if (before.type === 'control') {
        const { filter: _filter, ...rest } = before;
        after = { ...rest, source: paintSource, transform: identity };
      } else {
        after = {
          ...before,
          mask: { ...before.mask, bitmap: null, offset: { x: cropRect.x, y: cropRect.y } },
          transform: identity,
        };
      }

      const publishSnapshot = (
        contract: CanvasLayerContract,
        prepared: ReturnType<LayerCacheStore['prepareReplacement']>
      ): void => {
        dispatchPreparedMutation(
          { layer: contract, layerId, type: 'replaceCanvasLayer' },
          () => getReducerDocument()?.layers.find((candidate) => candidate.id === layerId) === contract,
          () => mirror.getDocument()?.layers.find((candidate) => candidate.id === layerId) === contract
        );
        // A crop creates a new live-paint incarnation for the same layer id.
        // Invalidate any pending or in-flight upload from the outgoing pixels
        // before publishing and dirtying the replacement cache.
        try {
          bitmapStore.discardLayer(layerId);
        } catch {
          // Persistence bookkeeping is ancillary after reducer acceptance.
        }
        installGeneratedPaintCache(prepared);
      };
      const applySnapshot = (contract: CanvasLayerContract, snapshot: { pixels: RasterSurface; rect: Rect }): void => {
        const prepared = prepareGeneratedPaintCache(layerId, snapshot.rect, snapshot.pixels);
        publishSnapshot(contract, prepared);
      };
      const afterPixels = { pixels: cropped, rect: cropRect };
      const preparedAfter = prepareGeneratedPaintCache(layerId, afterPixels.rect, afterPixels.pixels);
      if (!isDocumentEditPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      publishSnapshot(after, preparedAfter);
      history.push({
        bytes:
          beforePixels.rect.width * beforePixels.rect.height * 4 +
          afterPixels.rect.width * afterPixels.rect.height * 4 +
          256,
        label: 'Crop layer to bbox',
        redo: () => applySnapshot(after, afterPixels),
        replayFailureAtomic: true,
        undo: () => applySnapshot(before, beforePixels),
      });
      return { status: 'cropped' };
    } catch (error) {
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  };

  const copyLayerToRaster = async (layerId: string): Promise<string | null> => {
    const permit = captureDocumentEditPermit();
    if (!permit) {
      return null;
    }
    if (pipeline.isGestureActive()) {
      return null;
    }
    endNudgeBurst();
    const doc = mirror.getDocument();
    const sourceLayer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!doc || !sourceLayer) {
      return null;
    }
    const baked = await exportBakedLayerPixels(layerId, { includeDisabled: true });
    if (!isDocumentEditPermitCurrent(permit)) {
      return null;
    }
    if (baked.status !== 'ok') {
      return null;
    }
    if (
      !isDocumentEditPermitCurrent(permit) ||
      pipeline.isGestureActive() ||
      !isLayerExportGuardCurrent(baked.guard) ||
      baked.guard.layer !== sourceLayer
    ) {
      return null;
    }
    const liveDocument = mirror.getDocument();
    const liveSourceLayer = liveDocument?.layers.find((layer) => layer.id === layerId);
    const sourceIndex = liveDocument?.layers.findIndex((layer) => layer.id === layerId) ?? -1;
    if (!liveDocument || liveSourceLayer !== sourceLayer || sourceIndex < 0) {
      return null;
    }

    const newId = createLayerId();
    const layer: CanvasLayerContract = {
      blendMode: 'normal',
      id: newId,
      isEnabled: true,
      isLocked: false,
      name: `${sourceLayer.name} copy`,
      opacity: 1,
      source: { bitmap: null, offset: { x: baked.rect.x, y: baked.rect.y }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const selectedLayerId = liveDocument.selectedLayerId;
    const apply = (): void => {
      const prepared = prepareGeneratedPaintCache(newId, baked.rect, baked.surface);
      dispatchPreparedMutation(
        {
          add: { index: sourceIndex, layer },
          enabledUpdates: [],
          selectedLayerId: newId,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          getReducerDocument()?.selectedLayerId === newId &&
          getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () =>
          mirror.getDocument()?.selectedLayerId === newId &&
          mirror.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
      installGeneratedPaintCache(prepared);
    };
    if (!isDocumentEditPermitCurrent(permit)) {
      return null;
    }
    apply();
    history.push({
      bytes: baked.rect.width * baked.rect.height * 4 + 256,
      label: 'Copy layer to raster',
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        dispatchPreparedMutation(
          {
            enabledUpdates: [],
            removeIds: [newId],
            selectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          () =>
            getReducerDocument()?.selectedLayerId === selectedLayerId &&
            getReducerDocument()?.layers.some((candidate) => candidate.id === newId) === false,
          () =>
            mirror.getDocument()?.selectedLayerId === selectedLayerId &&
            mirror.getDocument()?.layers.some((candidate) => candidate.id === newId) === false
        ),
    });
    return newId;
  };

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

  const releaseBitmapIfUnreferenced = (imageName: string): void => {
    const doc = mirror.getDocument();
    const usedInDoc = doc ? doc.layers.some((layer) => layerImageName(layer) === imageName) : false;
    if (usedInDoc) {
      return;
    }
    for (const tracked of trackedImageNames.values()) {
      if (tracked === imageName) {
        return;
      }
    }
    for (const job of activeLayerRasterizationJobs) {
      if (sourceImageName(job.source) === imageName) {
        return;
      }
    }
    layerCache.deleteBitmap(imageName);
  };

  const untrackLayerImage = (layerId: string): void => {
    const imageName = trackedImageNames.get(layerId);
    if (!imageName) {
      return;
    }
    trackedImageNames.delete(layerId);
    releaseBitmapIfUnreferenced(imageName);
  };

  const trackPublishedLayerImage = (layer: CanvasLayerContract): void => {
    const previous = trackedImageNames.get(layer.id);
    const current = layerImageName(layer);
    if (current) {
      trackedImageNames.set(layer.id, current);
    } else {
      trackedImageNames.delete(layer.id);
    }
    if (previous && previous !== current) {
      releaseBitmapIfUnreferenced(previous);
    }
  };

  const dropLayer = (layerId: string): void => {
    cancelLayerRasterization(layerId);
    // Generation-cancel persistence before the id can be restored by undo/redo.
    // A late upload from the removed incarnation must never target a recreated
    // paint layer with the same id.
    try {
      bitmapStore.discardLayer(layerId);
    } catch {
      // Keep authoritative removal cleanup observer-safe for injected stores.
    }
    layerCache.delete(layerId);
    adjustedSurfaceCache.delete(layerId);
    stores.thumbnailVersion.delete(layerId);
    stores.thumbnailStatus.delete(layerId);
    untrackLayerImage(layerId);
  };

  // ---- Staged generation preview ------------------------------------------

  /** Drops any staged preview and bumps the token so an in-flight decode is discarded. */
  const clearStagedPreview = (): void => {
    stagedPreviewToken += 1;
    if (stagedPreview) {
      stagedPreview = null;
      scheduler.invalidate({ all: true });
    }
  };

  /** Decodes a staged-preview input to a surface (imageName via resolver, dataUrl via the backend seam). */
  const decodeStagedPreview = async (
    input: StagedPreviewInput
  ): Promise<{ surface: RasterSurface; width: number; height: number; placement?: StagedPreviewPlacement }> => {
    if ('imageName' in input) {
      const blob = await imageResolver(input.imageName);
      const bitmap = await backend.createImageBitmap(blob);
      const width = bitmap.width;
      const height = bitmap.height;
      try {
        const surface = backend.createSurface(width, height);
        surface.ctx.clearRect(0, 0, width, height);
        surface.ctx.drawImage(bitmap, 0, 0);
        return {
          height,
          placement: input.placement ? { ...input.placement } : undefined,
          surface,
          width,
        };
      } finally {
        bitmap.close();
      }
    }
    const { dataUrl, height, width } = input;
    const bitmap = await backend.createImageBitmap(dataUrlToBlob(dataUrl));
    try {
      const surface = backend.createSurface(width, height);
      surface.ctx.clearRect(0, 0, width, height);
      surface.ctx.drawImage(bitmap, 0, 0, width, height);
      return { height, surface, width };
    } finally {
      bitmap.close();
    }
  };

  const setStagedPreview = (input: StagedPreviewInput | null): void => {
    if (input === null) {
      clearStagedPreview();
      return;
    }
    stagedPreviewToken += 1;
    const token = stagedPreviewToken;
    decodeStagedPreview(input)
      .then((decoded) => {
        // A newer set/clear superseded this decode while it was in flight.
        if (token !== stagedPreviewToken) {
          return;
        }
        stagedPreview = decoded;
        scheduler.invalidate({ all: true });
      })
      .catch(() => {
        // A transient decode failure leaves any prior preview untouched rather
        // than blanking the canvas; the next selection re-drives a decode.
      });
  };

  // Per-layer guarded filter previews: layerId → decoded filtered surface, plus a
  // per-layer decode token so stale work never overwrites a newer request.
  const filterPreviews = new Map<string, { surface: RasterSurface; rect: Rect; guard: LayerExportGuard }>();
  const filterPreviewTokens = new Map<string, number>();
  /** Current guarded request/publication token per layer. */
  const guardedFilterPreviewTokens = new Map<string, number>();

  /**
   * Drops a layer's filter-preview state and bumps its token so an in-flight
   * decode for it is discarded — even if the id is later reused (e.g. an undo
   * that restores a deleted layer must not resurrect a stale decode result
   * that resolves afterward). The token is bumped, never reset/deleted, so a
   * later guarded preview for the same id can never collide with a
   * still-in-flight decode's captured token.
   */
  const clearFilterPreview = (layerId: string): void => {
    filterPreviewTokens.set(layerId, (filterPreviewTokens.get(layerId) ?? 0) + 1);
    guardedFilterPreviewTokens.delete(layerId);
    if (filterPreviews.delete(layerId)) {
      scheduler.invalidate({ layers: [layerId] });
    }
  };

  /**
   * Drops every layer's filter-preview state. Used on a wholesale document
   * replace: none of the outgoing document's previews describe the incoming
   * document, even if a layer id happens to be reused.
   */
  const clearAllFilterPreviews = (): void => {
    const ids = new Set<string>([...filterPreviews.keys(), ...filterPreviewTokens.keys()]);
    for (const id of ids) {
      clearFilterPreview(id);
    }
  };

  const publishFilterPreview = async (
    layerId: string,
    input: FilterPreviewInput,
    validate: () => 'shown' | 'missing' | 'stale',
    guard: LayerExportGuard
  ): Promise<'shown' | 'missing' | 'stale'> => {
    const nextToken = (filterPreviewTokens.get(layerId) ?? 0) + 1;
    filterPreviewTokens.set(layerId, nextToken);
    guardedFilterPreviewTokens.set(layerId, nextToken);
    const dropGuardedRequest = (): void => {
      if (guardedFilterPreviewTokens.get(layerId) === nextToken) {
        guardedFilterPreviewTokens.delete(layerId);
      }
    };
    const beforeDecode = validate();
    if (beforeDecode !== 'shown') {
      dropGuardedRequest();
      return beforeDecode;
    }
    try {
      const decoded = await decodeStagedPreview({ imageName: input.imageName });
      const beforePublish = validate();
      if (beforePublish !== 'shown') {
        dropGuardedRequest();
        return beforePublish;
      }
      // A newer set/clear for THIS layer superseded the decode in flight.
      if (filterPreviewTokens.get(layerId) !== nextToken) {
        dropGuardedRequest();
        return 'stale';
      }
      filterPreviews.set(layerId, { guard, rect: { ...input.rect }, surface: decoded.surface });
      scheduler.invalidate({ layers: [layerId] });
      return 'shown';
    } catch {
      // Transient decode failure leaves any prior preview untouched.
      dropGuardedRequest();
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
    if (activeToolId !== 'move') {
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
    if (activeToolId !== 'transform') {
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
    const screen = screenSurface;
    const overlay = overlaySurface;
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
    if (needsComposite) {
      ensureLayerCaches(doc);
      const stagedPlacement = stagedPreview?.placement;
      const isolatedLayerId = samPreview?.isolated ? samPreview.guard.layerId : null;
      compositeDocument(screen, doc, layerCache, view, {
        // Memoized adjusted surfaces for raster layers with brightness/contrast/
        // saturation/curves (not recomputed per frame — see adjustedSurfaceCache).
        adjustedSurface: getAdjustedSurface,
        // The raster backend + mask fill tile resolver drive the mask colorize
        // path (alpha stencil → source-in fill colour/pattern, above all layers).
        backend,
        maskPatternTile: getMaskPatternTile,
        // Non-destructive control-filter previews (drawn in place of the layer's
        // committed pixels). Only allocated when a preview is actually active.
        layerPreviews:
          filterPreviews.size > 0 ? new Map(Array.from(filterPreviews, ([id, preview]) => [id, preview])) : null,
        onlyLayerId: isolatedLayerId,
        // Feed the cached checkerboard tile only while the toggle is ON; passing
        // `null` renders transparent documents without a checkerboard.
        checkerboardTile: stores.checkerboard.get() ? getCheckerboardTile() : null,
        // Crisp + cheap when zoomed in (nearest-neighbor up-scale), smooth when
        // shrinking (bilinear down-scale). See `shouldSmoothAtZoom`.
        imageSmoothing: shouldSmoothAtZoom(viewport.getZoom()),
        // Candidate-specific placement wins for final images. Progress frames
        // and legacy image inputs continue to follow the CURRENT bbox origin.
        stagedPreview: stagedPreview
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
        transformOverrides: transformOverrides.size > 0 ? transformOverrides : null,
      });

      const visibleIds = doc.layers
        .filter((layer) => layer.id === isolatedLayerId || isRenderableLayer(layer))
        .map((layer) => layer.id);
      const evicted = layerCache.evictHidden(visibleIds);
      // Prune version-keyed dependents for every evicted id (mirrors dropLayer):
      // the evicted layer's surface is gone, so its adjusted-surface memo and
      // thumbnail state must not linger keyed to a version the re-rasterized entry
      // will exceed (the cache floor keeps versions monotonic across the recreate).
      for (const id of evicted) {
        adjustedSurfaceCache.delete(id);
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
    const samSession = stores.samSession.get();
    renderOverlay(overlay, {
      bbox: bboxPreview ?? doc.bbox,
      bboxHandles: activeToolId === 'bbox',
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
      showBbox: stores.showBbox.get() || activeToolId === 'bbox',
      showGrid: stores.showGrid.get(),
      transformFrame: transformFrameOverlay(doc),
      view,
    });
  };

  const scheduler: RenderScheduler = createRenderScheduler({ render });
  // Stay paused until attached: invalidations accumulate but never request a
  // (DOM) frame, keeping the engine node-safe before it has render targets.
  scheduler.pause();

  // ---- Document mirror ----------------------------------------------------

  const mirror: DocumentMirror = createDocumentMirror(store, projectId, {
    // The bbox rectangle/handles are overlay chrome, so a bbox move is normally
    // overlay-only (no recomposite). The one exception: a legacy/progress staged
    // preview is drawn in the COMPOSITE at the current bbox origin, so it must
    // recomposite to follow the bbox. Explicitly placed candidates do not.
    onBboxChanged: () =>
      scheduler.invalidate(stagedPreview && !stagedPreview.placement ? { all: true } : { overlay: true }),
    onDocumentReplaced: () => {
      canvasOperations.invalidateDocument(projectId);
      cancelSelectObjectSession();
      const previousImageNames = [...mirroredLayerImageNames.values()];
      rasterDocumentGeneration += 1;
      cancelAllLayerRasterizations();
      stores.thumbnailStatus.clear();
      // A wholesale document swap — project switch, dims/background change, or a
      // snapshot restore that changes dims — invalidates the pixel history: its
      // entries reference layers/pixels that no longer describe the live document.
      //
      // Cancel any in-flight tool gesture FIRST: a swap mid-drag leaves stale tool
      // state (a bbox `startBbox`, a move drag anchor) whose pointer-up would
      // otherwise commit against the replaced document. Routing through the
      // pipeline clears `gestureActive` and runs the tool's `onPointerCancel`, so
      // the tool drops its own transient state.
      pipeline.cancelActiveGesture();
      // Defensive: a non-bbox active tool won't have cleared a lingering preview.
      stores.bboxPreview.set(null);
      history.clear();
      endNudgeBurst();
      // A transform session (which outlives individual gestures) belongs to the
      // outgoing document; tear it down alongside its preview override.
      stores.transformSession.set(null);
      transformOverrides.clear();
      // A text-edit session likewise belongs to the outgoing document; drop it.
      stores.textEditSession.set(null);
      // A staged preview belongs to the outgoing document's bbox/candidates; a
      // wholesale swap (project switch, snapshot restore) invalidates it.
      clearStagedPreview();
      // Per-layer control-filter previews likewise belong to the outgoing
      // document — a swap can reuse a layer id with different content, so
      // pruning only "missing" ids isn't enough; drop them all.
      clearAllFilterPreviews();
      // The selection is document-scoped interaction state: a swap drops it (and
      // any in-progress lasso preview), stopping the ants loop via onChange.
      selection.clear();
      stores.lassoPreview.set(null);
      const doc = mirror.getDocument();
      const present = new Set(doc ? doc.layers.map((layer) => layer.id) : []);
      mirroredLayerImageNames.clear();
      thumbnailDisplayKeys.clear();
      for (const layer of doc?.layers ?? []) {
        thumbnailDisplayKeys.set(layer.id, getLayerThumbnailDisplayKey(layer));
        const imageName = layerImageName(layer);
        if (imageName) {
          mirroredLayerImageNames.set(layer.id, imageName);
        }
      }
      // Snapshot ids first: dropLayer mutates trackedImageNames during iteration.
      const trackedIds = Array.from(trackedImageNames.keys());
      for (const layerId of trackedIds) {
        if (!present.has(layerId)) {
          dropLayer(layerId);
        } else {
          untrackLayerImage(layerId);
        }
      }
      // A wholesale replacement can reuse a layer id with a DIFFERENT source, so
      // a surviving cache entry may hold pixels from the outgoing document.
      // Invalidate EVERY id in the incoming document — not just ids whose
      // reference happened to change — to force a re-rasterize from the new
      // source; a diff can't be trusted across a full swap.
      for (const layerId of present) {
        invalidateLayerCache(layerId);
      }
      for (const imageName of previousImageNames) {
        releaseBitmapIfUnreferenced(imageName);
      }
      // Persistence bookkeeping (the self-echo `lastApplied` map and pending
      // debounced flushes) described the OLD document. Drop it so a reused layer
      // id can't have its next legit persistence dispatch suppressed as a stale
      // self-echo.
      bitmapStore.reset();
      scheduler.invalidate({ all: true });
    },
    onLayerOrderChanged: () => scheduler.invalidate({ all: true }),
    onLayersChanged: (ids, sourceChangedIds) => {
      for (const id of sourceChangedIds) {
        canvasOperations.invalidateSource(projectId, id);
      }
      for (const id of ids) {
        canvasOperations.invalidateLayer(projectId, id);
      }
      if (selectObjectGuard && !isLayerExportGuardCurrent(selectObjectGuard)) {
        cancelSelectObjectSession();
      }
      const doc = mirror.getDocument();
      const present = new Set(doc ? doc.layers.map((layer) => layer.id) : []);
      const sourceChanged = new Set(sourceChangedIds);
      const previousImageNames = new Map(ids.map((id) => [id, mirroredLayerImageNames.get(id)]));
      for (const id of ids) {
        const layer = doc?.layers.find((candidate) => candidate.id === id);
        const imageName = layer ? layerImageName(layer) : null;
        if (imageName) {
          mirroredLayerImageNames.set(id, imageName);
        } else {
          mirroredLayerImageNames.delete(id);
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
          thumbnailDisplayKeys.delete(id);
          dropLayer(id);
          const previousImageName = previousImageNames.get(id);
          if (previousImageName) {
            releaseBitmapIfUnreferenced(previousImageName);
          }
          // A control-filter preview (session + decoded surface) belongs to a
          // specific layer; a layer removed out from under an in-flight or
          // already-decoded preview (delete via the layers panel, or an undo
          // that removes it) must have its preview dropped and its decode
          // token bumped, or a late-resolving decode — or a later undo that
          // restores this same id — would repopulate a stale preview.
          clearFilterPreview(id);
          if (session && session.layerId === id) {
            cancelTransform();
          }
          // An edit-mode text session whose layer was deleted out from under it
          // (layers panel, or undo of the add) is torn down the same way.
          if (textSession && textSession.layerId === id) {
            cancelTextEdit();
          }
          continue;
        }
        const preview = filterPreviews.get(id);
        if (preview && !isLayerExportGuardCurrent(preview.guard)) {
          clearFilterPreview(id);
        }
        if (!sourceChanged.has(id)) {
          if (layer) {
            const displayKey = getLayerThumbnailDisplayKey(layer);
            if (thumbnailDisplayKeys.get(id) !== displayKey) {
              thumbnailDisplayKeys.set(id, displayKey);
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
          thumbnailDisplayKeys.set(id, getLayerThumbnailDisplayKey(layer));
        }
        untrackLayerImage(id);
        const previousImageName = previousImageNames.get(id);
        if (previousImageName) {
          releaseBitmapIfUnreferenced(previousImageName);
        }
        const source = getLayerSourceById(id);
        if (bitmapStore.isSelfEcho(id, source)) {
          continue;
        }
        invalidateLayerCache(id);
      }
      scheduler.invalidate({ layers: ids });
    },
    onStagingChanged: () => scheduler.invalidate({ overlay: true }),
  });

  for (const layer of mirror.getDocument()?.layers ?? []) {
    thumbnailDisplayKeys.set(layer.id, getLayerThumbnailDisplayKey(layer));
    const imageName = layerImageName(layer);
    if (imageName) {
      mirroredLayerImageNames.set(layer.id, imageName);
    }
  }

  // A guarded filter preview belongs to one continuous active-project epoch.
  // Switching away invalidates published and in-flight work, so returning to
  // this project cannot resurrect it.
  let lastActiveProjectId = store.getState().activeProjectId;
  const unsubscribeProjectPreviewLifecycle = store.subscribe(() => {
    const activeProjectId = store.getState().activeProjectId;
    if (lastActiveProjectId === projectId && activeProjectId !== projectId) {
      canvasOperations.invalidateProject(projectId);
      cancelSelectObjectSession();
      rasterDocumentGeneration += 1;
      cancelAllLayerRasterizations();
      stores.thumbnailStatus.clear();
      const ids = new Set<string>(guardedFilterPreviewTokens.keys());
      for (const layerId of filterPreviews.keys()) {
        ids.add(layerId);
      }
      for (const layerId of ids) {
        clearFilterPreview(layerId);
      }
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

  // ---- History → stores ---------------------------------------------------

  const syncHistoryStores = (): void => {
    // History has already moved its entry before notifying. External-store
    // observers are ancillary UI updates and must not make an applied engine
    // transaction report failure or prevent the sibling flag from syncing.
    try {
      stores.canUndo.set(history.canUndo());
    } catch {
      // The scalar store commits its value before notifying subscribers.
    }
    try {
      stores.canRedo.set(history.canRedo());
    } catch {
      // Keep undo/redo notifications isolated from each other.
    }
  };
  const unsubscribeHistory = history.subscribe(syncHistoryStores);

  // ---- Pointer / wheel / key input ---------------------------------------
  //
  // Normalization, capture, coalescing, temp-tool holds, and gesture cancel live
  // in the pointer pipeline; wheel routing (zoom vs brush-size step) lives in the
  // wheel handler. The engine just supplies seams and wires the DOM listeners.

  /** Steps the active brush/eraser diameter by one notch (ctrl+wheel or the `[`/`]` hotkeys). */
  const stepActiveBrushSize = (direction: 1 | -1): void => {
    if (activeToolId === 'brush') {
      const opts = stores.brushOptions.get();
      stores.brushOptions.set({ ...opts, size: stepBrushSize(opts.size, direction) });
    } else if (activeToolId === 'eraser') {
      const opts = stores.eraserOptions.get();
      stores.eraserOptions.set({ ...opts, size: stepBrushSize(opts.size, direction) });
    }
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
    if (stores.textEditSession.get()) {
      cancelTextEdit();
      return;
    }
    if (stores.transformSession.get()) {
      cancelTransform();
      return;
    }
    if (!gestureWasActive && filterSession) {
      cancelFilterOperation();
      return;
    }
    if (!gestureWasActive && selectObjectSession) {
      cancelSelectObjectSession();
      return;
    }
    if (!gestureWasActive && selection.hasSelection()) {
      selection.clear();
    }
  };

  const pipeline: PointerPipeline = createPointerPipeline({
    getActiveTool: activeTool,
    getActiveToolId: () => activeToolId,
    getInputElement: () => inputEl,
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
    getInputElement: () => inputEl,
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
    void bitmapStore.flushPendingUploads();
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
    const previous = samPreview;
    samPreview = null;
    if (previous) {
      scheduler.invalidate(previous.isolated ? { all: true } : { overlay: true });
    }
  };

  const syncSelectObjectStore = (): void => {
    const session = selectObjectSession;
    const sourceRect = selectObjectSourceRect;
    if (!session || !sourceRect) {
      stores.samSession.set(null);
      return;
    }
    const state = session.getSnapshot();
    stores.samSession.set({
      applyPolygonRefinement: state.applyPolygonRefinement,
      autoProcess: state.autoProcess,
      error: state.error,
      hasPreview: state.preview !== null,
      input:
        state.input.type === 'visual'
          ? {
              bbox: state.input.bbox ? { ...state.input.bbox } : null,
              excludePoints: state.input.excludePoints.map((point) => ({ ...point })),
              includePoints: state.input.includePoints.map((point) => ({ ...point })),
              type: 'visual',
            }
          : { prompt: state.input.prompt, type: 'prompt' },
      invert: state.invert,
      isolatedPreview: state.isolatedPreview,
      layerId: selectObjectGuard?.layerId ?? '',
      model: state.model,
      pointLabel: selectObjectPointLabel,
      sourceRect,
      status: selectObjectCommitOwner?.session === session ? 'committing' : state.status,
    });
    scheduler.invalidate({ overlay: true });
  };

  const revokeSelectObjectCommit = (): void => {
    selectObjectCommitToken += 1;
    const owner = selectObjectCommitOwner;
    selectObjectCommitOwner = null;
    owner?.controller.abort();
  };

  const invalidateSelectObjectCommit = (): void => {
    revokeSelectObjectCommit();
    syncSelectObjectStore();
  };

  const isSelectObjectCommitOwnerCurrent = (owner: SelectObjectCommitOwner): boolean => {
    const state = owner.session.getSnapshot();
    return (
      selectObjectCommitOwner === owner &&
      selectObjectCommitToken === owner.token &&
      selectObjectSession === owner.session &&
      selectObjectGuard === owner.guard &&
      state.preview?.previewId === owner.previewId &&
      state.preview.inputHash === owner.inputHash &&
      owner.preview.inputHash === owner.inputHash &&
      isLayerExportGuardCurrent(owner.guard) &&
      !owner.controller.signal.aborted
    );
  };

  const beginSelectObjectCommit = (
    session: SelectObjectSession<RasterSurface>,
    preview: SelectObjectSessionPreview<RasterSurface>
  ): SelectObjectCommitOwner | null => {
    if (selectObjectCommitOwner) {
      return null;
    }
    const owner: SelectObjectCommitOwner = {
      controller: new AbortController(),
      guard: preview.guard,
      inputHash: preview.inputHash,
      preview,
      previewId: preview.previewId,
      session,
      token: ++selectObjectCommitToken,
    };
    selectObjectCommitOwner = owner;
    syncSelectObjectStore();
    return owner;
  };

  const finishSelectObjectCommit = (owner: SelectObjectCommitOwner): boolean => {
    if (!isSelectObjectCommitOwnerCurrent(owner)) {
      return false;
    }
    selectObjectCommitOwner = null;
    syncSelectObjectStore();
    return true;
  };

  const decodeSelectObjectPreview = async (
    result: { image: CanvasImageRef; rect: Rect },
    signal: AbortSignal
  ): Promise<RasterSurface> => {
    const blob = await imageResolver(result.image.imageName, signal);
    if (signal.aborted) {
      throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
    }
    const bitmap = await backend.createImageBitmap(blob);
    try {
      if (signal.aborted) {
        throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
      }
      const surface = backend.createSurface(result.rect.width, result.rect.height);
      const ctx = surface.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, surface.width, surface.height);
      ctx.drawImage(bitmap, 0, 0, surface.width, surface.height);
      ctx.globalCompositeOperation = 'source-in';
      ctx.fillStyle = '#38bdf8';
      ctx.fillRect(0, 0, surface.width, surface.height);
      ctx.globalCompositeOperation = 'source-over';
      if (signal.aborted) {
        throw new DOMException('Select Object preview decode was aborted.', 'AbortError');
      }
      return surface;
    } finally {
      bitmap.close();
    }
  };

  const clearOwnedSelectObjectSession = (expectedOwner?: SelectObjectCommitOwner): boolean => {
    if (expectedOwner && !isSelectObjectCommitOwnerCurrent(expectedOwner)) {
      return false;
    }
    const session = selectObjectSession;
    if (expectedOwner && session !== expectedOwner.session) {
      return false;
    }
    pipeline.replaceTemporaryRestoreTool('sam', 'view');
    selectObjectSession = null;
    selectObjectUnsubscribe?.();
    selectObjectUnsubscribe = null;
    selectObjectGuard = null;
    selectObjectSourceRect = null;
    selectObjectPointLabel = 'include';
    revokeSelectObjectCommit();
    clearSamPreview();
    canvasOperations.cancel();
    session?.dispose();
    if (selectObjectSession === null) {
      stores.samSession.set(null);
    }
    scheduler.invalidate({ overlay: true });
    return true;
  };

  const startSelectObject = (layerId: string): StartSelectObjectSessionResult => {
    const doc = mirror.getDocument();
    if (!doc) {
      return 'missing';
    }
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!layer) {
      return 'missing';
    }
    if (layer.type !== 'raster' && layer.type !== 'control') {
      return 'unsupported';
    }
    if (!layer.isEnabled) {
      return 'disabled';
    }
    const entry = layerCache.get(layer.id);
    const guard = captureCurrentLayerExportGuard(layer.id);
    if (!entry || !guard) {
      return 'not-ready';
    }
    const matrix = fromTRS(
      { x: layer.transform.x, y: layer.transform.y },
      layer.transform.rotation,
      layer.transform.scaleX,
      layer.transform.scaleY
    );
    const sourceRect = roundOut(transformBounds(matrix, entry.rect));
    if (isEmpty(sourceRect)) {
      return 'not-ready';
    }

    if (doc.selectedLayerId !== layer.id) {
      store.dispatch({ id: layer.id, type: 'setCanvasSelectedLayer' });
    }

    clearOwnedFilterSession();
    clearOwnedSelectObjectSession();
    selectObjectGuard = guard;
    selectObjectSourceRect = sourceRect;
    const operation = canvasOperations.start({
      cleanupPreview: clearSamPreview,
      guard,
      identity: { kind: 'select-object', layerId: layer.id, projectId },
    });
    if (!operation) {
      selectObjectGuard = null;
      selectObjectSourceRect = null;
      return 'not-ready';
    }
    const queueDeps = opts.selectObjectDeps;
    selectObjectSession = createSelectObjectSession({
      deps: {
        captureGuard: () =>
          selectObjectGuard && isLayerExportGuardCurrent(selectObjectGuard) ? selectObjectGuard : null,
        cleanupPreview: clearSamPreview,
        controller: canvasOperations,
        decodePreview: decodeSelectObjectPreview,
        exportLayer: (layerId) => exportBakedLayerBlob(layerId, { includeDisabled: true }),
        isGuardCurrent: isLayerExportGuardCurrent,
        publishPreview: (preview) => {
          const isolationChanged = samPreview?.isolated !== preview.isolated;
          samPreview = preview;
          scheduler.invalidate(preview.isolated || isolationChanged ? { all: true } : { overlay: true });
          return undefined;
        },
        runGraph: (options) => queueDeps?.runGraph(options) ?? runUtilityGraph({ ...options, hub: socketHub }),
        uploadIntermediate: async (blob, signal) => {
          if (queueDeps) {
            return queueDeps.uploadIntermediate(blob, signal);
          }
          if (signal?.aborted) {
            throw new DOMException('Select Object upload was aborted.', 'AbortError');
          }
          const uploaded = await uploadCanvasImage(blob, { isIntermediate: true, signal });
          if (signal?.aborted) {
            throw new DOMException('Select Object upload was aborted.', 'AbortError');
          }
          return { imageName: uploaded.imageName };
        },
      },
      layerId: layer.id,
      projectId,
    });
    const installedSession = selectObjectSession;
    selectObjectUnsubscribe = installedSession.subscribe(() => {
      const owner = selectObjectCommitOwner;
      if (
        owner?.session === installedSession &&
        installedSession.getSnapshot().preview?.previewId !== owner.previewId
      ) {
        invalidateSelectObjectCommit();
      }
      syncSelectObjectStore();
    });
    selectObjectSession.update({
      input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
    });
    syncSelectObjectStore();
    setTool('sam');
    return 'started';
  };

  const updateSelectObjectSession = (changes: SelectObjectSessionUpdate): void => {
    const session = selectObjectSession;
    if (!session) {
      return;
    }
    const { pointLabel, ...sessionChanges } = changes;
    const state = session.getSnapshot();
    const processingChanged =
      ('input' in sessionChanges && sessionChanges.input !== state.input) ||
      ('model' in sessionChanges && sessionChanges.model !== state.model) ||
      ('invert' in sessionChanges && sessionChanges.invert !== state.invert) ||
      ('applyPolygonRefinement' in sessionChanges &&
        sessionChanges.applyPolygonRefinement !== state.applyPolygonRefinement);
    if (processingChanged) {
      invalidateSelectObjectCommit();
    }
    if (pointLabel) {
      selectObjectPointLabel = pointLabel;
    }
    session.update(sessionChanges);
    syncSelectObjectStore();
  };

  const processSelectObjectSession = (): Promise<SelectObjectSessionProcessResult> => {
    invalidateSelectObjectCommit();
    return selectObjectSession?.process() ?? Promise.resolve('stale');
  };

  const getCurrentSelectObjectPreview = (): SelectObjectSessionPreview<RasterSurface> | null => {
    const state = selectObjectSession?.getSnapshot();
    const preview = state?.preview;
    if (
      !state ||
      state.status === 'processing' ||
      state.status === 'scheduled' ||
      !preview ||
      preview.guard !== selectObjectGuard
    ) {
      return null;
    }
    return isLayerExportGuardCurrent(preview.guard) ? preview : null;
  };

  const applySelectObjectSession = async (): Promise<ReplaceSelectionFromImageResult> => {
    const session = selectObjectSession;
    const preview = getCurrentSelectObjectPreview();
    if (!session || !preview) {
      return { status: 'stale' };
    }
    const owner = beginSelectObjectCommit(session, preview);
    if (!owner) {
      return { status: 'busy' };
    }
    if (!isSelectObjectCommitOwnerCurrent(owner)) {
      invalidateSelectObjectCommit();
      return { status: 'stale' };
    }
    const result = await replaceSelectionFromImage(
      preview.guard,
      preview.image,
      preview.rect,
      owner.controller.signal,
      documentEditOwner
    );
    if (!isSelectObjectCommitOwnerCurrent(owner)) {
      return result;
    }
    if (result.status === 'selected') {
      if (!clearOwnedSelectObjectSession(owner)) {
        return result;
      }
      if (selectObjectSession === null && activeToolId === 'sam') {
        setTool('view');
      }
    } else {
      if (!finishSelectObjectCommit(owner)) {
        return result;
      }
      owner.session.reportError(
        result.status === 'failed' ? result.message : `Select Object apply is ${result.status}.`
      );
    }
    return result;
  };

  const saveSelectObjectSession = async (
    target: SelectObjectSaveTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<SaveSelectObjectSessionResult> => {
    const session = selectObjectSession;
    const preview = getCurrentSelectObjectPreview();
    if (!session || !preview) {
      return { status: 'stale' };
    }
    const owner = beginSelectObjectCommit(session, preview);
    if (!owner) {
      return { status: 'busy' };
    }
    if (!isSelectObjectCommitOwnerCurrent(owner)) {
      invalidateSelectObjectCommit();
      return { status: 'stale' };
    }
    try {
      await makeImageDurable(preview.image.imageName);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (finishSelectObjectCommit(owner)) {
        owner.session.reportError(message);
      }
      return { message, status: 'failed' };
    }
    if (!isSelectObjectCommitOwnerCurrent(owner)) {
      return { status: 'stale' };
    }
    let result: SaveSelectObjectSessionResult;
    try {
      result =
        target === 'raster' || target === 'control'
          ? await commitGeneratedImageResult(
              {
                guard: preview.guard,
                image: preview.image,
                origin: { x: preview.rect.x, y: preview.rect.y },
                signal: owner.controller.signal,
                target: target === 'raster' ? 'copy-raster' : 'copy-control',
              },
              documentEditOwner
            )
          : await commitMaskImageResult(
              {
                guard: preview.guard,
                image: preview.image,
                rect: preview.rect,
                signal: owner.controller.signal,
                target,
              },
              documentEditOwner
            );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (finishSelectObjectCommit(owner)) {
        owner.session.reportError(message);
      }
      return { message, status: 'failed' };
    }
    if (!isSelectObjectCommitOwnerCurrent(owner)) {
      return result;
    }
    if (result.status === 'committed') {
      if (!clearOwnedSelectObjectSession(owner)) {
        return result;
      }
      if (selectObjectSession === null && activeToolId === 'sam') {
        setTool('view');
      }
    } else {
      if (!finishSelectObjectCommit(owner)) {
        return result;
      }
      owner.session.reportError(
        result.status === 'failed' ? result.message : `Select Object save is ${result.status}.`
      );
    }
    return result;
  };

  const resetSelectObjectSession = (): void => {
    const session = selectObjectSession;
    const guard = selectObjectGuard;
    if (!session || !guard) {
      return;
    }
    invalidateSelectObjectCommit();
    selectObjectPointLabel = 'include';
    session.reset();
    session.update({ input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' } });
    canvasOperations.start({
      cleanupPreview: clearSamPreview,
      guard,
      identity: { kind: 'select-object', layerId: guard.layerId, projectId },
    });
    syncSelectObjectStore();
  };

  const cancelSelectObjectSession = (): void => {
    clearOwnedSelectObjectSession();
    if (selectObjectSession === null && activeToolId === 'sam') {
      setTool('view');
    }
  };

  let filterMakeDurable: (imageName: string) => Promise<void> = () =>
    Promise.reject(new Error('The filter result cannot be preserved.'));

  const syncFilterStore = (): void => {
    stores.filterSession.set(filterSession?.getSnapshot() ?? null);
  };

  const clearOwnedFilterSession = (): void => {
    const session = filterSession;
    filterSession = null;
    filterUnsubscribe?.();
    filterUnsubscribe = null;
    filterControllerUnsubscribe?.();
    filterControllerUnsubscribe = null;
    session?.dispose();
    stores.filterSession.set(null);
  };

  const startFilterOperation = (layerId: string, recommendedFilterType?: string | null): StartFilterOperationResult => {
    // Recommendations are opportunistic. Never replace an active manual filter,
    // preview, Select Object session, or other canvas operation.
    if (recommendedFilterType && canvasOperations.getSnapshot().status !== 'idle') {
      return 'not-ready';
    }
    const document = mirror.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer) {
      return 'missing';
    }
    if (layer.type !== 'raster' && layer.type !== 'control') {
      return 'unsupported';
    }
    if (recommendedFilterType && layer.filter) {
      return 'not-ready';
    }
    if (!layer.isEnabled) {
      return 'disabled';
    }
    if (layer.isLocked) {
      return 'locked';
    }
    const guard = captureCurrentLayerExportGuard(layer.id);
    if (!guard) {
      return 'not-ready';
    }
    const initialType = layer.filter?.type ?? recommendedFilterType ?? DEFAULT_CONTROL_FILTER_TYPE;
    const definition = getFilterDefinition(initialType);
    const initialFilter = layer.filter ? structuredClone(layer.filter) : null;
    const draft = initialFilter ?? {
      settings: definition ? buildFilterDefaults(definition) : {},
      type: definition?.type ?? DEFAULT_CONTROL_FILTER_TYPE,
    };

    clearOwnedSelectObjectSession();
    clearOwnedFilterSession();
    if (document.selectedLayerId !== layer.id) {
      store.dispatch({ id: layer.id, type: 'setCanvasSelectedLayer' });
    }
    const queueDeps = opts.filterDeps;
    filterSession = createFilterOperationSession({
      deps: {
        clearPreview: () => clearFilterPreview(layer.id),
        commit: ({ draft, guard, image, rect, signal, target }) =>
          commitRasterFilterResult(
            {
              filter: draft,
              guard,
              image,
              mode: target === 'apply' ? 'replace' : 'copy',
              rect,
              signal,
              target,
            },
            documentEditOwner
          ),
        controller: canvasOperations,
        exportPixels: () => rasterizeLayerPixels(layer.id, { applyAdjustments: true, includeDisabled: true }),
        isGuardCurrent: isLayerExportGuardCurrent,
        makeDurable: (imageName) => filterMakeDurable(imageName),
        publishPreview: (imageName, rect, previewGuard) =>
          setGuardedFilterPreview(layer.id, { imageName, rect }, previewGuard),
        runFilter: (options) =>
          runLayerFilter({
            ...options,
            deps: {
              encodeSurface: (surface) => backend.encodeSurface(surface, 'image/png'),
              runFilterGraph: async ({ graph, outputNodeId, signal }) => {
                const output = await (queueDeps?.runGraph({ graph, outputNodeId, signal }) ??
                  runUtilityGraph({ graph, hub: socketHub, outputNodeId, signal }));
                return { height: output.height, imageName: output.imageName, width: output.width };
              },
              uploadIntermediate: async (blob, signal) => {
                const uploaded = await (queueDeps?.uploadIntermediate(blob, signal) ??
                  uploadCanvasImage(blob, { isIntermediate: true, signal }));
                return { imageName: uploaded.imageName };
              },
            },
          }),
      },
      guard,
      initialDraft: draft,
      initialFilter,
      layerType: layer.type,
    });
    if (!filterSession) {
      return 'not-ready';
    }
    const installed = filterSession;
    filterUnsubscribe = installed.subscribe(syncFilterStore);
    filterControllerUnsubscribe = canvasOperations.subscribe(() => {
      if (filterSession === installed && canvasOperations.getSnapshot().status === 'idle') {
        clearOwnedFilterSession();
      }
    });
    syncFilterStore();
    setTool('view');
    return 'started';
  };

  const updateFilterOperation = (draft: LayerFilterSettings): void => filterSession?.updateDraft(draft);
  const processFilterOperation = (): Promise<void> => filterSession?.process() ?? Promise.resolve();
  const resetFilterOperation = (settings: Record<string, unknown>): void => filterSession?.reset(settings);
  const commitFilterOperation = async (
    target: FilterCommitTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<void> => {
    const session = filterSession;
    if (!session) {
      return;
    }
    filterMakeDurable = makeImageDurable;
    await session.commit(target);
    if (canvasOperations.getSnapshot().status === 'idle' && filterSession === session) {
      filterSession = null;
      filterUnsubscribe?.();
      filterUnsubscribe = null;
      stores.filterSession.set(null);
    }
  };

  const cancelFilterOperation = (): void => clearOwnedFilterSession();

  // ---- Public API ---------------------------------------------------------

  function setTool(toolId: ToolId, opts?: { temporary?: boolean }): void {
    if (interactionLocked && toolId !== 'view') {
      return;
    }
    if (toolId === activeToolId) {
      return;
    }
    if (activeToolId === 'sam' && toolId !== 'sam' && !opts?.temporary) {
      clearOwnedSelectObjectSession();
    }
    tools.get(activeToolId)?.onDeactivate?.(toolContext, opts);
    activeToolId = toolId;
    tools.get(toolId)?.onActivate?.(toolContext, opts);
    stores.activeTool.set(toolId);
    updateCursor();
    // The overlay draws tool-specific chrome (bbox handles, move outline) keyed
    // on the active tool, so a switch must repaint it even without a doc change.
    scheduler.invalidate({ overlay: true });
  }

  const setInteractionLocked = (locked: boolean): void => {
    if (interactionLocked === locked) {
      return;
    }
    interactionLocked = locked;
    if (locked) {
      pipeline.cancelActiveGesture();
      setTool('view');
    }
  };

  const attach = (screenCanvas: HTMLCanvasElement, overlayCanvas: HTMLCanvasElement): void => {
    if (disposed) {
      return;
    }
    if (inputEl) {
      detach();
    }
    screenSurface = wrapCanvasSurface(screenCanvas);
    overlaySurface = wrapCanvasSurface(overlayCanvas);
    // The overlay canvas sits on top, so it receives pointer/wheel input.
    inputEl = overlayCanvas;

    inputEl.addEventListener('pointerdown', pipeline.onPointerDown);
    inputEl.addEventListener('pointermove', pipeline.onPointerMove);
    inputEl.addEventListener('pointerup', pipeline.onPointerUp);
    inputEl.addEventListener('pointercancel', pipeline.onPointerCancel);
    inputEl.addEventListener('pointerenter', pipeline.onPointerEnter);
    inputEl.addEventListener('pointerleave', pipeline.onPointerLeave);
    inputEl.addEventListener('wheel', onWheel, { passive: false });
    if (typeof globalThis.addEventListener === 'function') {
      globalThis.addEventListener('keydown', pipeline.onKeyDown);
      globalThis.addEventListener('keyup', pipeline.onKeyUp);
      globalThis.addEventListener('pagehide', onPageHide);
      globalThis.addEventListener('blur', onWindowBlur);
    }
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibilityChange);
    }

    scheduler.resume();
    stores.viewportReady.set(true);
    // Reflect the active tool's cursor on the freshly-bound input element.
    updateCursor();
    scheduler.invalidate({ all: true });
    // Resume marching ants if a selection outlived a detach/attach cycle.
    updateAntsAnimation();
  };

  const detach = (): void => {
    if (inputEl) {
      inputEl.removeEventListener('pointerdown', pipeline.onPointerDown);
      inputEl.removeEventListener('pointermove', pipeline.onPointerMove);
      inputEl.removeEventListener('pointerup', pipeline.onPointerUp);
      inputEl.removeEventListener('pointercancel', pipeline.onPointerCancel);
      inputEl.removeEventListener('pointerenter', pipeline.onPointerEnter);
      inputEl.removeEventListener('pointerleave', pipeline.onPointerLeave);
      inputEl.removeEventListener('wheel', onWheel);
      // Restore the element's default cursor so a detached surface isn't stuck
      // with a stale tool cursor.
      applyCursorToInput('');
    }
    if (typeof globalThis.removeEventListener === 'function') {
      globalThis.removeEventListener('keydown', pipeline.onKeyDown);
      globalThis.removeEventListener('keyup', pipeline.onKeyUp);
      globalThis.removeEventListener('pagehide', onPageHide);
      globalThis.removeEventListener('blur', onWindowBlur);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    pipeline.reset();
    // Drop the staged preview surface so a detached engine holds no stale
    // decoded pixels; React re-drives it after a re-attach.
    clearStagedPreview();
    scheduler.pause();
    stores.viewportReady.set(false);
    screenSurface = null;
    overlaySurface = null;
    inputEl = null;
    // No render targets: pause marching ants (the selection itself is retained).
    updateAntsAnimation();
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
    screenSurface?.resize(backingWidth, backingHeight);
    overlaySurface?.resize(backingWidth, backingHeight);
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

  const drawLayerThumbnail = (layerId: string, target: HTMLCanvasElement, maxSize: number): boolean => {
    const entry = layerCache.get(layerId);
    const layer = mirror.getDocument()?.layers.find((candidate) => candidate.id === layerId);
    if (!entry || !entry.hasPublishedPixels || !layer) {
      return false;
    }
    const { height, width } = fitThumbnailSize(entry.surface.width, entry.surface.height, maxSize);
    if (width === 0 || height === 0) {
      return false;
    }
    const ctx = target.getContext('2d');
    if (!ctx) {
      return false;
    }
    target.width = width;
    target.height = height;
    ctx.clearRect(0, 0, width, height);

    // Downscale first. Every display effect below works only on this bounded
    // surface, never on the potentially multi-megapixel layer cache.
    const thumbnailSurface = backend.createSurface(width, height);
    const thumbnailCtx = thumbnailSurface.ctx;
    thumbnailCtx.setTransform(1, 0, 0, 1, 0, 0);
    thumbnailCtx.clearRect(0, 0, width, height);
    thumbnailCtx.globalAlpha = 1;
    thumbnailCtx.globalCompositeOperation = 'source-over';
    thumbnailCtx.drawImage(entry.surface.canvas, 0, 0, width, height);

    const checkerPattern = ctx.createPattern(getCheckerboardTile().canvas as CanvasImageSource, 'repeat');
    if (checkerPattern) {
      ctx.fillStyle = checkerPattern;
      ctx.fillRect(0, 0, width, height);
    }

    let displaySurface = thumbnailSurface;
    if (layer.type === 'raster' && !isIdentityAdjustments(layer.adjustments)) {
      const imageData = thumbnailCtx.getImageData(0, 0, width, height);
      applyAdjustments(imageData, layer.adjustments);
      thumbnailCtx.putImageData(imageData, 0, 0);
    } else if (layer.type === 'control' && layer.withTransparencyEffect) {
      displaySurface = renderControlTransparency(backend, thumbnailSurface, width, height);
    } else if (layer.type === 'inpaint_mask' || layer.type === 'regional_guidance') {
      const { fill } = layer.mask;
      displaySurface = colorizeMask(
        backend,
        thumbnailSurface,
        width,
        height,
        fill,
        getMaskPatternTile(fill.style, fill.color)
      );
    }
    ctx.globalAlpha = layer.opacity;
    ctx.drawImage(displaySurface.canvas as CanvasImageSource, 0, 0);
    return true;
  };

  const requestLayerThumbnail = async (layerId: string): Promise<LayerThumbnailRequestResult> => {
    if (disposed || store.getState().activeProjectId !== projectId) {
      stores.thumbnailStatus.delete(layerId);
      return disposed ? 'missing' : 'stale';
    }
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!doc || !layer) {
      stores.thumbnailStatus.delete(layerId);
      return 'missing';
    }
    const source = renderableSourceOf(layer);
    if (!source || !isSupportedExportSource(source)) {
      stores.thumbnailStatus.delete(layerId);
      return 'unsupported';
    }
    const entry = layerCache.get(layerId);
    if (entry?.hasPublishedPixels && !entry.stale) {
      stores.thumbnailStatus.set(layerId, 'ready');
      return 'ready';
    }

    stores.thumbnailStatus.set(layerId, 'loading');
    let result: Awaited<ReturnType<typeof getOrStartLayerRasterization>>;
    try {
      result = await getOrStartLayerRasterization(layer, doc);
    } catch (error) {
      stores.thumbnailStatus.set(layerId, 'error');
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
      return 'error';
    }
    if (result === 'published') {
      return 'ready';
    }
    return result;
  };

  /**
   * True when a layer's cache is safe to READ + mutate in place for a structural
   * op (invert mask, merge down). The hazard is a layer with PERSISTED pixels
   * (`mask.bitmap` / image / paint bitmap) that haven't been decoded into a ready
   * cache: the surface then holds blank/old pixels, so the op would bake garbage
   * AND the in-flight `rasterizeSource` completion would later redraw over the
   * op's result. Such a layer is safe only once its cache exists and is neither
   * stale (re-rasterize pending) nor in its current decode job.
   *
   * A layer with NO persisted content is always safe: its cache — whether a fresh
   * live paint (unflushed stroke) or a genuinely empty 0×0 stale entry — already
   * reflects the current pixels, with nothing to decode that could clobber it.
   */
  const isLayerCacheReadyForOp = (layer: CanvasLayerContract, doc: CanvasDocumentContractV2): boolean => {
    if (isEmpty(getSourceContentRect(layer, doc))) {
      return true;
    }
    const entry = layerCache.get(layer.id);
    return !!entry && !entry.stale && !isCurrentRasterizationJob(layer);
  };

  const booleanCompositeModes: Record<BooleanRasterOperation, GlobalCompositeOperation> = {
    cutaway: 'source-out',
    cutout: 'destination-in',
    exclude: 'xor',
    intersect: 'source-in',
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
    notifyBestEffort(() => adjustedSurfaceCache.delete(layerId));
    notifyBestEffort(() => stores.thumbnailVersion.set(layerId, target.version));
    if (filterPreviews.has(layerId)) {
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
    const project = store.getState().projects.find((candidate) => candidate.id === projectId);
    return project ? getSelectedModelBase(project) : null;
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

  const commitLayerCopy = (
    label: string,
    sourceLayerId: string,
    layer: CanvasLayerContract,
    index: number
  ): boolean => {
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return false;
    }
    endNudgeBurst();
    const doc = mirror.getDocument();
    const source = doc?.layers.find((candidate) => candidate.id === sourceLayerId);
    if (!doc || !source || doc.layers.some((candidate) => candidate.id === layer.id)) {
      return false;
    }
    const captured = captureLayerCache(source, doc);
    if (captured === 'not-ready') {
      return false;
    }
    const selectedLayerId = doc.selectedLayerId;
    const apply = (): void => {
      const prepared = captured ? prepareGeneratedPaintCache(layer.id, captured.rect, captured.pixels) : null;
      dispatchPreparedMutation(
        {
          add: { index, layer },
          enabledUpdates: [],
          selectedLayerId: layer.id,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          getReducerDocument()?.selectedLayerId === layer.id &&
          getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () =>
          mirror.getDocument()?.selectedLayerId === layer.id &&
          mirror.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );
      if (prepared) {
        installGeneratedPaintCache(prepared, layerNeedsPixelPersistence(layer));
      }
    };
    apply();
    history.push({
      bytes: captured ? captured.rect.width * captured.rect.height * 4 + 256 : 256,
      label,
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        dispatchPreparedMutation(
          {
            enabledUpdates: [],
            removeIds: [layer.id],
            selectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          () =>
            getReducerDocument()?.selectedLayerId === selectedLayerId &&
            getReducerDocument()?.layers.some((candidate) => candidate.id === layer.id) === false,
          () =>
            mirror.getDocument()?.selectedLayerId === selectedLayerId &&
            mirror.getDocument()?.layers.some((candidate) => candidate.id === layer.id) === false
        ),
    });
    return true;
  };

  const commitLayerConversion = (
    label: string,
    expectedLiveLayer: CanvasLayerContract,
    after: CanvasLayerContract
  ): boolean => {
    if (
      !canEditDocument() ||
      pipeline.isGestureActive() ||
      expectedLiveLayer.id !== after.id ||
      expectedLiveLayer.type === after.type
    ) {
      return false;
    }
    endNudgeBurst();
    const doc = mirror.getDocument();
    const current = doc?.layers.find((candidate) => candidate.id === expectedLiveLayer.id);
    if (
      !doc ||
      !current ||
      current !== expectedLiveLayer ||
      current.isLocked ||
      current.type !== expectedLiveLayer.type
    ) {
      return false;
    }
    const captured = captureLayerCache(current, doc);
    if (captured === 'not-ready') {
      return false;
    }
    const apply = (layer: CanvasLayerContract): void => {
      const prepared = captured ? prepareGeneratedPaintCache(layer.id, captured.rect, captured.pixels) : null;
      dispatchPreparedMutation(
        { id: layer.id, layer, targetType: layer.type, type: 'convertCanvasLayer' },
        () => documentHasLayerContract(getReducerDocument(), layer),
        () => documentHasLayerContract(mirror.getDocument(), layer)
      );
      // Conversion republishes a cache incarnation under the same layer id.
      // Cancel persistence from the outgoing contract before exposing it.
      try {
        bitmapStore.discardLayer(layer.id);
      } catch {
        // Persistence bookkeeping is ancillary after reducer acceptance.
      }
      if (prepared) {
        installGeneratedPaintCache(prepared, layerNeedsPixelPersistence(layer));
      }
    };
    const before = structuredClone(current);
    apply(after);
    history.push({
      bytes: captured ? captured.rect.width * captured.rect.height * 4 + 256 : 256,
      label,
      redo: () => apply(after),
      replayFailureAtomic: true,
      undo: () => apply(before),
    });
    return true;
  };

  const replaceSelectionFromImage = async (
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal,
    owner?: symbol
  ): Promise<ReplaceSelectionFromImageResult> => {
    const permit = captureDocumentEditPermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    try {
      const blob = await imageResolver(image.imageName, signal);
      if (!isDocumentEditPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      if (signal?.aborted) {
        return { status: 'aborted' };
      }
      const bitmap = await backend.createImageBitmap(blob);
      if (!isDocumentEditPermitCurrent(permit)) {
        bitmap.close();
        return { status: 'busy' };
      }
      if (signal?.aborted) {
        bitmap.close();
        return { status: 'aborted' };
      }
      let pixels: RasterSurface;
      try {
        // Allocation belongs to the decoded bitmap's lifetime too. If it throws,
        // the finally block must still release the browser bitmap.
        pixels = backend.createSurface(image.width, image.height);
        pixels.ctx.setTransform(1, 0, 0, 1, 0, 0);
        pixels.ctx.clearRect(0, 0, image.width, image.height);
        pixels.ctx.drawImage(bitmap, 0, 0, image.width, image.height);
      } finally {
        bitmap.close();
      }
      if (signal?.aborted) {
        return { status: 'aborted' };
      }

      const document = mirror.getDocument();
      const liveLayer = document?.layers.find((candidate) => candidate.id === guard.layerId);
      if (!document || !liveLayer) {
        return { status: 'missing' };
      }
      if (liveLayer.isLocked) {
        return { status: 'locked' };
      }
      if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
        return { status: 'unsupported' };
      }
      if (!isDocumentEditPermitCurrent(permit) || pipeline.isGestureActive()) {
        return { status: 'busy' };
      }
      if (!isLayerExportGuardCurrent(guard)) {
        return { status: 'stale' };
      }
      if (signal?.aborted) {
        return { status: 'aborted' };
      }

      if (!isDocumentEditPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      selection.replaceMask({ rect: { ...rect }, surface: pixels });
      return { status: 'selected' };
    } catch (error) {
      if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        return { status: 'aborted' };
      }
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  };

  const commitMaskImageResult = (
    options: CommitMaskImageResultOptions,
    owner?: symbol
  ): Promise<CommitMaskImageResult> => {
    if (!canEditDocument(owner)) {
      return Promise.resolve({ status: 'busy' });
    }
    if (options.signal?.aborted) {
      return Promise.resolve({ status: 'aborted' });
    }
    const document = mirror.getDocument();
    const liveLayer = document?.layers.find((candidate) => candidate.id === options.guard.layerId);
    if (!document || !liveLayer) {
      return Promise.resolve({ status: 'missing' });
    }
    if (liveLayer.isLocked) {
      return Promise.resolve({ status: 'locked' });
    }
    if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
      return Promise.resolve({ status: 'unsupported' });
    }
    if (pipeline.isGestureActive()) {
      return Promise.resolve({ status: 'busy' });
    }
    if (!isLayerExportGuardCurrent(options.guard)) {
      return Promise.resolve({ status: 'stale' });
    }
    if (options.signal?.aborted) {
      return Promise.resolve({ status: 'aborted' });
    }

    const sourceIndex = document.layers.findIndex((candidate) => candidate.id === liveLayer.id);
    if (sourceIndex < 0) {
      return Promise.resolve({ status: 'missing' });
    }
    const names = document.layers.map((layer) => layer.name);
    const layerId = createLayerId();
    const layer =
      options.target === 'inpaint_mask'
        ? createInpaintMaskFromImage({
            fill: DEFAULT_INPAINT_MASK_FILL,
            id: layerId,
            image: options.image,
            name: nextInpaintMaskName(names),
            rect: options.rect,
          })
        : createRegionalGuidanceFromImage({
            fill: {
              color: nextRegionalGuidanceFillColor(
                document.layers.filter((candidate) => candidate.type === 'regional_guidance').length
              ),
              style: 'solid',
            },
            id: layerId,
            image: options.image,
            name: nextRegionalGuidanceName(names),
            rect: options.rect,
          });
    const selectedLayerId = document.selectedLayerId;
    const apply = (): void =>
      dispatchPreparedMutation(
        { index: sourceIndex, layer, type: 'addCanvasLayer' },
        () => getReducerDocument()?.layers.some((candidate) => candidate === layer) === true,
        () => mirror.getDocument()?.layers.some((candidate) => candidate === layer) === true
      );

    endNudgeBurst();
    apply();
    history.push({
      bytes: 256,
      label: options.target === 'inpaint_mask' ? 'Create inpaint mask from object' : 'Create region from object',
      redo: apply,
      replayFailureAtomic: true,
      undo: () => {
        // Restore selection first. It is idempotent, so a pre-dispatch removal
        // failure leaves the history entry retryable without duplicating a
        // structural mutation or losing the caller's prior selection.
        dispatchPreparedMutation(
          { id: selectedLayerId, type: 'setCanvasSelectedLayer' },
          () => getReducerDocument()?.selectedLayerId === selectedLayerId,
          () => mirror.getDocument()?.selectedLayerId === selectedLayerId
        );
        dispatchPreparedMutation(
          { ids: [layerId], type: 'removeCanvasLayers' },
          () => getReducerDocument()?.layers.some((candidate) => candidate.id === layerId) === false,
          () => mirror.getDocument()?.layers.some((candidate) => candidate.id === layerId) === false
        );
      },
    });
    return Promise.resolve({ layerId, status: 'committed' });
  };

  const commitRasterFilterResult = async (
    options: CommitRasterFilterOptions,
    owner?: symbol
  ): Promise<CommitRasterFilterResult> => {
    const permit = captureDocumentEditPermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    try {
      const blob = await imageResolver(options.image.imageName, options.signal);
      if (!isDocumentEditPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }
      const bitmap = await backend.createImageBitmap(blob);
      if (!isDocumentEditPermitCurrent(permit)) {
        bitmap.close();
        return { status: 'busy' };
      }
      if (options.signal?.aborted) {
        bitmap.close();
        return { status: 'aborted' };
      }
      const pixels = backend.createSurface(options.image.width, options.image.height);
      try {
        pixels.ctx.clearRect(0, 0, options.image.width, options.image.height);
        pixels.ctx.drawImage(bitmap, 0, 0, options.image.width, options.image.height);
      } finally {
        bitmap.close();
      }

      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }

      const document = mirror.getDocument();
      const liveLayer = document?.layers.find((candidate) => candidate.id === options.guard.layerId);
      if (!document || !liveLayer) {
        return { status: 'missing' };
      }
      if (liveLayer.isLocked) {
        return { status: 'locked' };
      }
      if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
        return { status: 'unsupported' };
      }
      if (!isDocumentEditPermitCurrent(permit) || pipeline.isGestureActive()) {
        return { status: 'busy' };
      }
      if (!isLayerExportGuardCurrent(options.guard)) {
        return { status: 'stale' };
      }
      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }

      if (!isDocumentEditPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      endNudgeBurst();
      const image = structuredClone(options.image);
      const rect = { ...options.rect };
      const paintSource = { bitmap: image, offset: { x: rect.x, y: rect.y }, type: 'paint' } as const;
      if (options.mode === 'replace') {
        const beforePixels = captureLayerCache(liveLayer, document);
        if (!beforePixels || beforePixels === 'not-ready') {
          return { status: 'stale' };
        }
        const before = structuredClone(liveLayer);
        const after: CanvasLayerContract =
          liveLayer.type === 'raster'
            ? (() => {
                const { adjustments: _adjustments, ...base } = liveLayer;
                return structuredClone({ ...base, filter: options.filter, source: paintSource });
              })()
            : structuredClone({ ...liveLayer, filter: options.filter, source: paintSource });
        const afterPixels = { pixels, rect };
        const publishSnapshot = (
          contract: CanvasLayerContract,
          prepared: ReturnType<LayerCacheStore['prepareReplacement']>,
          options: { discardPersistence: boolean; persist: boolean }
        ): void => {
          dispatchPreparedMutation(
            { layer: contract, layerId: liveLayer.id, type: 'replaceCanvasLayer' },
            () => getReducerDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract,
            () => mirror.getDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract
          );
          if (options.discardPersistence) {
            // The reducer now points at a durable filtered bitmap. Invalidate
            // any pending timer or in-flight upload for the prior live paint
            // pixels before publishing the matching cache; otherwise that old
            // upload could later overwrite this durable contract.
            try {
              bitmapStore.discardLayer(liveLayer.id);
            } catch {
              // Persistence bookkeeping is an ancillary post-commit hook.
            }
          }
          installGeneratedPaintCache(prepared, options.persist);
        };
        const applySnapshot = (
          contract: CanvasLayerContract,
          snapshot: { pixels: RasterSurface; rect: Rect },
          options: { discardPersistence: boolean; persist: boolean }
        ): void => {
          const prepared = prepareGeneratedPaintCache(liveLayer.id, snapshot.rect, snapshot.pixels);
          publishSnapshot(contract, prepared, options);
        };
        // The durable filter result already backs `after.source`; clone its
        // cache before dispatch so allocation/draw failures cannot partially
        // mutate the document, and avoid a redundant persistence upload.
        const preparedAfter = prepareGeneratedPaintCache(liveLayer.id, afterPixels.rect, afterPixels.pixels);
        publishSnapshot(after, preparedAfter, { discardPersistence: true, persist: false });
        history.push({
          bytes:
            beforePixels.rect.width * beforePixels.rect.height * 4 +
            afterPixels.rect.width * afterPixels.rect.height * 4 +
            256,
          label: 'Replace layer with filter result',
          redo: () => applySnapshot(after, afterPixels, { discardPersistence: true, persist: false }),
          replayFailureAtomic: true,
          undo: () =>
            applySnapshot(before, beforePixels, {
              discardPersistence: false,
              persist: layerNeedsPixelPersistence(before),
            }),
        });
        return { layerId: liveLayer.id, status: 'committed' };
      }

      const sourceIndex = document.layers.findIndex((candidate) => candidate.id === liveLayer.id);
      if (sourceIndex < 0) {
        return { status: 'missing' };
      }
      const selectedLayerId = document.selectedLayerId;
      const layerId = createLayerId();
      let copy: CanvasLayerContract;
      if (options.target === 'control') {
        const base =
          liveLayer.type === 'control'
            ? structuredClone(liveLayer)
            : createControlLayer(`${liveLayer.name} filtered`, layerId, getMainModelBase());
        copy = {
          ...base,
          filter: options.filter,
          id: layerId,
          name: `${liveLayer.name} filtered`,
          source: paintSource,
          transform: structuredClone(liveLayer.transform),
        };
      } else if (options.target === 'raster' && liveLayer.type === 'control') {
        copy = {
          blendMode: liveLayer.blendMode,
          filter: options.filter,
          id: layerId,
          isEnabled: true,
          isLocked: false,
          name: `${liveLayer.name} filtered`,
          opacity: liveLayer.opacity,
          source: paintSource,
          transform: structuredClone(liveLayer.transform),
          type: 'raster',
        };
      } else {
        const { adjustments: _adjustments, ...base } = structuredClone(liveLayer as CanvasRasterLayerContractV2);
        copy = {
          ...base,
          filter: options.filter,
          id: layerId,
          name: `${liveLayer.name} filtered`,
          source: paintSource,
          type: 'raster',
        };
      }
      const apply = (): void => {
        // Prepare the independent live-cache clone before adding the layer. The
        // result image is already durable, so no dirty upload is needed.
        const prepared = prepareGeneratedPaintCache(layerId, rect, pixels);
        dispatchPreparedMutation(
          { index: sourceIndex, layer: copy, type: 'addCanvasLayer' },
          () => getReducerDocument()?.layers.some((candidate) => candidate === copy) === true,
          () => mirror.getDocument()?.layers.some((candidate) => candidate === copy) === true
        );
        installGeneratedPaintCache(prepared, false);
      };
      apply();
      history.push({
        bytes: rect.width * rect.height * 4 + 256,
        label: 'Copy layer filter result',
        redo: apply,
        replayFailureAtomic: true,
        undo: () => {
          // Restore selection first. It is idempotent, so if the subsequent
          // removal fails before reducer application the failure-atomic history
          // entry can retry without leaving the copy selected or duplicating a
          // structural mutation.
          dispatchPreparedMutation(
            { id: selectedLayerId, type: 'setCanvasSelectedLayer' },
            () => getReducerDocument()?.selectedLayerId === selectedLayerId,
            () => mirror.getDocument()?.selectedLayerId === selectedLayerId
          );
          dispatchPreparedMutation(
            { ids: [layerId], type: 'removeCanvasLayers' },
            () => getReducerDocument()?.layers.some((candidate) => candidate.id === layerId) === false,
            () => mirror.getDocument()?.layers.some((candidate) => candidate.id === layerId) === false
          );
        },
      });
      return { layerId, status: 'committed' };
    } catch (error) {
      if (options.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        return { status: 'aborted' };
      }
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  };

  const commitGeneratedImageResult = async (
    options: CommitGeneratedImageOptions,
    owner?: symbol
  ): Promise<CommitGeneratedImageResult> => {
    const permit = captureDocumentEditPermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }

    const validateGuard = ():
      | {
          document: CanvasDocumentContractV2;
          liveLayer: Extract<CanvasLayerContract, { type: 'raster' | 'control' }>;
        }
      | { result: CommitGeneratedImageResult } => {
      if (!isDocumentEditPermitCurrent(permit)) {
        return { result: { status: 'busy' } };
      }
      const document = mirror.getDocument();
      const liveLayer = document?.layers.find((candidate) => candidate.id === options.guard.layerId);
      if (!document || !liveLayer) {
        return { result: { status: 'missing' } };
      }
      if (liveLayer.isLocked) {
        return { result: { status: 'locked' } };
      }
      if (liveLayer.type !== 'raster' && liveLayer.type !== 'control') {
        return { result: { status: 'unsupported' } };
      }
      if (pipeline.isGestureActive()) {
        return { result: { status: 'busy' } };
      }
      if (!isLayerExportGuardCurrent(options.guard)) {
        return { result: { status: 'stale' } };
      }
      return { document, liveLayer };
    };

    try {
      const blob = await imageResolver(options.image.imageName, options.signal);
      if (!isDocumentEditPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }

      const bitmap = await backend.createImageBitmap(blob);
      if (!isDocumentEditPermitCurrent(permit)) {
        bitmap.close();
        return { status: 'busy' };
      }
      if (options.signal?.aborted) {
        bitmap.close();
        return { status: 'aborted' };
      }

      let generatedPixels: RasterSurface;
      try {
        generatedPixels = backend.createSurface(options.image.width, options.image.height);
        generatedPixels.ctx.setTransform(1, 0, 0, 1, 0, 0);
        generatedPixels.ctx.clearRect(0, 0, options.image.width, options.image.height);
        generatedPixels.ctx.drawImage(bitmap, 0, 0, options.image.width, options.image.height);
      } finally {
        bitmap.close();
      }
      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }

      const checked = validateGuard();
      if ('result' in checked) {
        return checked.result;
      }
      const { document, liveLayer } = checked;
      const image = structuredClone(options.image);
      const origin = { ...options.origin };
      const rect = { height: image.height, width: image.width, ...origin };
      const source = { bitmap: image, offset: origin, type: 'paint' } as const;
      const identityTransform: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

      const publishSnapshot = (
        contract: CanvasLayerContract,
        prepared: ReturnType<LayerCacheStore['prepareReplacement']>,
        publishOptions: { discardPersistence: boolean; persist: boolean }
      ): void => {
        dispatchPreparedMutation(
          { layer: contract, layerId: liveLayer.id, type: 'replaceCanvasLayer' },
          () => getReducerDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract,
          () => mirror.getDocument()?.layers.find((candidate) => candidate.id === liveLayer.id) === contract
        );
        if (publishOptions.discardPersistence) {
          try {
            bitmapStore.discardLayer(liveLayer.id);
          } catch {
            // Persistence generation cancellation is ancillary once the durable
            // contract has been accepted by the reducer.
          }
        }
        try {
          clearFilterPreview(liveLayer.id);
        } catch {
          // A transient preview is ancillary to the committed document/cache.
        }
        installGeneratedPaintCache(prepared, publishOptions.persist);
      };

      const applySnapshot = (
        contract: CanvasLayerContract,
        snapshot: { pixels: RasterSurface; rect: Rect },
        publishOptions: { discardPersistence: boolean; persist: boolean }
      ): void => {
        const prepared = prepareGeneratedPaintCache(liveLayer.id, snapshot.rect, snapshot.pixels);
        publishSnapshot(contract, prepared, publishOptions);
      };

      if (options.target === 'replace') {
        const beforePixels = captureLayerCache(liveLayer, document);
        if (!beforePixels || beforePixels === 'not-ready') {
          return { status: 'stale' };
        }
        if (options.signal?.aborted) {
          return { status: 'aborted' };
        }

        const before = structuredClone(liveLayer);
        let after: CanvasLayerContract;
        if (liveLayer.type === 'raster') {
          const { adjustments: _adjustments, ...base } = structuredClone(liveLayer);
          after = { ...base, source, transform: identityTransform };
        } else {
          after = { ...structuredClone(liveLayer), source, transform: identityTransform };
        }
        const afterPixels = { pixels: generatedPixels, rect };
        const preparedAfter = prepareGeneratedPaintCache(liveLayer.id, rect, generatedPixels);

        if (options.signal?.aborted) {
          return { status: 'aborted' };
        }
        const finalCheck = validateGuard();
        if ('result' in finalCheck) {
          return finalCheck.result;
        }

        endNudgeBurst();
        publishSnapshot(after, preparedAfter, { discardPersistence: true, persist: false });
        history.push({
          bytes:
            beforePixels.rect.width * beforePixels.rect.height * 4 +
            afterPixels.rect.width * afterPixels.rect.height * 4 +
            256,
          label: 'Replace layer with workflow result',
          redo: () => applySnapshot(after, afterPixels, { discardPersistence: true, persist: false }),
          replayFailureAtomic: true,
          undo: () =>
            applySnapshot(before, beforePixels, {
              discardPersistence: false,
              persist: layerNeedsPixelPersistence(before),
            }),
        });
        return { layerId: liveLayer.id, status: 'committed' };
      }

      const sourceIndex = document.layers.findIndex((candidate) => candidate.id === liveLayer.id);
      if (sourceIndex < 0) {
        return { status: 'missing' };
      }
      const layerId = createLayerId();
      const selectedLayerId = document.selectedLayerId;
      const copy: CanvasLayerContract =
        options.target === 'copy-control'
          ? {
              ...createControlLayer(
                nextControlLayerName(document.layers.map((layer) => layer.name)),
                layerId,
                getMainModelBase()
              ),
              source,
              transform: identityTransform,
            }
          : {
              blendMode: 'normal',
              id: layerId,
              isEnabled: true,
              isLocked: false,
              name: `${liveLayer.name} workflow result`,
              opacity: 1,
              source,
              transform: identityTransform,
              type: 'raster',
            };
      const publishCopy = (prepared: ReturnType<LayerCacheStore['prepareReplacement']>): void => {
        dispatchPreparedMutation(
          { index: sourceIndex, layer: copy, type: 'addCanvasLayer' },
          () => getReducerDocument()?.layers.some((candidate) => candidate === copy) === true,
          () => mirror.getDocument()?.layers.some((candidate) => candidate === copy) === true
        );
        installGeneratedPaintCache(prepared, false);
      };
      const applyCopy = (): void => {
        const prepared = prepareGeneratedPaintCache(layerId, rect, generatedPixels);
        publishCopy(prepared);
      };
      const preparedCopy = prepareGeneratedPaintCache(layerId, rect, generatedPixels);

      if (options.signal?.aborted) {
        return { status: 'aborted' };
      }
      const finalCheck = validateGuard();
      if ('result' in finalCheck) {
        return finalCheck.result;
      }

      endNudgeBurst();
      publishCopy(preparedCopy);
      history.push({
        bytes: rect.width * rect.height * 4 + 256,
        label:
          options.target === 'copy-control'
            ? 'Copy workflow result to control layer'
            : 'Copy workflow result to raster layer',
        redo: applyCopy,
        replayFailureAtomic: true,
        undo: () => {
          dispatchPreparedMutation(
            { id: selectedLayerId, type: 'setCanvasSelectedLayer' },
            () => getReducerDocument()?.selectedLayerId === selectedLayerId,
            () => mirror.getDocument()?.selectedLayerId === selectedLayerId
          );
          dispatchPreparedMutation(
            { ids: [layerId], type: 'removeCanvasLayers' },
            () => getReducerDocument()?.layers.some((candidate) => candidate.id === layerId) === false,
            () => mirror.getDocument()?.layers.some((candidate) => candidate.id === layerId) === false
          );
        },
      });
      return { layerId, status: 'committed' };
    } catch (error) {
      if (options.signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        return { status: 'aborted' };
      }
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  };

  const booleanMergeRasterLayers = async (
    upperLayerId: string,
    operation: BooleanRasterOperation
  ): Promise<BooleanRasterResult> => {
    const permit = captureDocumentEditPermit();
    if (!permit || pipeline.isGestureActive()) {
      return 'busy';
    }
    endNudgeBurst();
    const doc = mirror.getDocument();
    if (!doc) {
      return 'missing';
    }
    const upperIndex = doc.layers.findIndex((layer) => layer.id === upperLayerId);
    if (upperIndex < 0) {
      return 'missing';
    }
    const upper = doc.layers[upperIndex];
    const below = doc.layers[upperIndex + 1];
    if (!upper || !below) {
      return 'missing';
    }
    if (!isMergeableRasterLayer(upper) || !isMergeableRasterLayer(below)) {
      return 'unsupported';
    }
    if (!isLayerCacheReadyForOp(upper, doc) || !isLayerCacheReadyForOp(below, doc)) {
      return 'not-ready';
    }

    const [upperPixels, belowPixels] = await Promise.all([
      exportBakedLayerPixels(upper.id),
      exportBakedLayerPixels(below.id),
    ]);
    if (!isDocumentEditPermitCurrent(permit)) {
      return 'busy';
    }
    if (upperPixels.status !== 'ok' || belowPixels.status !== 'ok') {
      if (upperPixels.status === 'not-ready' || belowPixels.status === 'not-ready') {
        return 'not-ready';
      }
      if (
        upperPixels.status === 'disabled' ||
        upperPixels.status === 'unsupported' ||
        belowPixels.status === 'disabled' ||
        belowPixels.status === 'unsupported'
      ) {
        return 'unsupported';
      }
      return 'empty';
    }
    if (!isDocumentEditPermitCurrent(permit) || pipeline.isGestureActive()) {
      return 'busy';
    }
    if (
      upperPixels.guard.layer !== upper ||
      belowPixels.guard.layer !== below ||
      !isLayerExportGuardCurrent(upperPixels.guard) ||
      !isLayerExportGuardCurrent(belowPixels.guard)
    ) {
      return 'not-ready';
    }
    const liveDocument = mirror.getDocument();
    const liveUpperIndex = liveDocument?.layers.findIndex((layer) => layer.id === upperLayerId) ?? -1;
    const liveUpper = liveDocument?.layers[liveUpperIndex];
    const liveBelow = liveDocument?.layers[liveUpperIndex + 1];
    if (!liveDocument || liveUpper !== upper || liveBelow !== below) {
      return 'not-ready';
    }
    if (!isMergeableRasterLayer(liveUpper) || !isMergeableRasterLayer(liveBelow)) {
      return 'unsupported';
    }

    const resultRect = roundOut(union(upperPixels.rect, belowPixels.rect));
    if (isEmpty(resultRect)) {
      return 'empty';
    }
    const resultPixels = backend.createSurface(resultRect.width, resultRect.height);
    const ctx = resultPixels.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, resultRect.width, resultRect.height);
    ctx.globalAlpha = below.opacity;
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(belowPixels.surface.canvas, belowPixels.rect.x - resultRect.x, belowPixels.rect.y - resultRect.y);
    ctx.globalAlpha = upper.opacity;
    ctx.globalCompositeOperation = booleanCompositeModes[operation];
    ctx.drawImage(upperPixels.surface.canvas, upperPixels.rect.x - resultRect.x, upperPixels.rect.y - resultRect.y);

    const resultId = createLayerId();
    const resultLayer: CanvasLayerContract = {
      blendMode: 'normal',
      id: resultId,
      isEnabled: true,
      isLocked: false,
      name: `${upper.name} ${operation}`,
      opacity: 1,
      source: { bitmap: null, offset: { x: resultRect.x, y: resultRect.y }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const originalEnabled = [
      { id: upper.id, isEnabled: upper.isEnabled },
      { id: below.id, isEnabled: below.isEnabled },
    ];
    const disabled = originalEnabled.map(({ id }) => ({ id, isEnabled: false }));
    const selectedLayerId = liveDocument.selectedLayerId;
    const hasEnabledState = (
      document: CanvasDocumentContractV2 | null,
      updates: readonly { id: string; isEnabled: boolean }[]
    ): boolean =>
      updates.every(
        (update) => document?.layers.find((candidate) => candidate.id === update.id)?.isEnabled === update.isEnabled
      );
    const apply = (): void => {
      const prepared = prepareGeneratedPaintCache(resultId, resultRect, resultPixels);
      dispatchPreparedMutation(
        {
          add: { index: liveUpperIndex, layer: resultLayer },
          enabledUpdates: disabled,
          selectedLayerId: resultId,
          type: 'applyCanvasLayerStackMutation',
        },
        () => {
          const document = getReducerDocument();
          return (
            document?.selectedLayerId === resultId &&
            document.layers.some((candidate) => candidate === resultLayer) &&
            hasEnabledState(document, disabled)
          );
        },
        () => {
          const document = mirror.getDocument();
          return (
            document?.selectedLayerId === resultId &&
            document.layers.some((candidate) => candidate === resultLayer) &&
            hasEnabledState(document, disabled)
          );
        }
      );
      installGeneratedPaintCache(prepared);
    };

    if (!isDocumentEditPermitCurrent(permit)) {
      return 'busy';
    }
    apply();
    history.push({
      bytes: resultRect.width * resultRect.height * 4 + 256,
      label: `Boolean ${operation}`,
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        dispatchPreparedMutation(
          {
            enabledUpdates: originalEnabled,
            removeIds: [resultId],
            selectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          () => {
            const document = getReducerDocument();
            return (
              document?.selectedLayerId === selectedLayerId &&
              document.layers.some((candidate) => candidate.id === resultId) === false &&
              hasEnabledState(document, originalEnabled)
            );
          },
          () => {
            const document = mirror.getDocument();
            return (
              document?.selectedLayerId === selectedLayerId &&
              document.layers.some((candidate) => candidate.id === resultId) === false &&
              hasEnabledState(document, originalEnabled)
            );
          }
        ),
    });
    return 'merged';
  };

  const extractMaskedArea = async (maskLayerId: string): Promise<ExtractMaskedAreaResult> => {
    const permit = captureDocumentEditPermit();
    if (!permit || pipeline.isGestureActive()) {
      return { status: 'busy' };
    }
    endNudgeBurst();
    const doc = mirror.getDocument();
    if (!doc) {
      return { status: 'missing' };
    }
    const maskIndex = doc.layers.findIndex((layer) => layer.id === maskLayerId);
    if (maskIndex < 0) {
      return { status: 'missing' };
    }
    const maskLayer = doc.layers[maskIndex];
    if (!maskLayer || maskLayer.type !== 'inpaint_mask') {
      return { status: 'unsupported' };
    }
    if (maskLayer.isLocked) {
      return { status: 'unsupported' };
    }
    const liveMaskCache = layerCache.get(maskLayerId);
    if (isEmpty(getSourceContentRect(maskLayer, doc)) && (!liveMaskCache || isEmpty(liveMaskCache.rect))) {
      return { status: 'empty' };
    }
    const contributors = doc.layers.filter(
      (layer) => layer.isEnabled && layer.type === 'raster' && hasExportableLayerContent(layer.id)
    );
    if (contributors.length === 0) {
      return { status: 'empty' };
    }
    if (!isLayerCacheReadyForOp(maskLayer, doc) || contributors.some((layer) => !isLayerCacheReadyForOp(layer, doc))) {
      return { status: 'not-ready' };
    }

    const [maskPixels, contributorPixels] = await Promise.all([
      exportBakedLayerPixels(maskLayerId, { includeDisabled: true }),
      Promise.all(contributors.map((layer) => rasterizeLayerPixels(layer.id))),
    ]);
    if (!isDocumentEditPermitCurrent(permit)) {
      return { status: 'busy' };
    }
    if (maskPixels.status !== 'ok') {
      return { status: maskPixels.status === 'not-ready' ? 'not-ready' : 'empty' };
    }
    if (contributorPixels.some((pixels) => pixels.status !== 'ok')) {
      return {
        status: contributorPixels.some((pixels) => pixels.status === 'not-ready') ? 'not-ready' : 'empty',
      };
    }
    if (!isDocumentEditPermitCurrent(permit) || pipeline.isGestureActive()) {
      return { status: 'busy' };
    }
    if (maskPixels.guard.layer !== maskLayer || !isLayerExportGuardCurrent(maskPixels.guard)) {
      return { status: 'not-ready' };
    }
    const liveDocument = mirror.getDocument();
    const liveMaskIndex = liveDocument?.layers.findIndex((layer) => layer.id === maskLayerId) ?? -1;
    const currentMask = liveDocument?.layers[liveMaskIndex];
    if (!liveDocument || !currentMask) {
      return { status: 'missing' };
    }
    if (currentMask !== maskLayer) {
      return { status: currentMask.type === 'inpaint_mask' && currentMask.isLocked ? 'unsupported' : 'not-ready' };
    }
    const liveContributors = liveDocument.layers.filter(
      (layer) => layer.isEnabled && layer.type === 'raster' && hasExportableLayerContent(layer.id)
    );
    if (
      liveMaskIndex !== maskIndex ||
      liveContributors.length !== contributors.length ||
      liveContributors.some((layer, index) => layer !== contributors[index])
    ) {
      return { status: 'not-ready' };
    }
    for (let index = 0; index < contributorPixels.length; index += 1) {
      const pixels = contributorPixels[index];
      const contributor = contributors[index];
      if (
        !pixels ||
        pixels.status !== 'ok' ||
        !contributor ||
        pixels.guard.layer !== contributor ||
        !isLayerExportGuardCurrent(pixels.guard)
      ) {
        return { status: 'not-ready' };
      }
    }
    const resultRect = maskPixels.rect;
    if (isEmpty(resultRect)) {
      return { status: 'empty' };
    }

    const resultPixels = backend.createSurface(resultRect.width, resultRect.height);
    const compositeDoc: CanvasDocumentContractV2 = { ...doc, layers: contributors };
    compositeDocument(
      resultPixels,
      compositeDoc,
      layerCache,
      { a: 1, b: 0, c: 0, d: 1, e: -resultRect.x, f: -resultRect.y },
      {
        adjustedSurface: getAdjustedSurface,
        backend,
        maskPatternTile: getMaskPatternTile,
      }
    );
    const ctx = resultPixels.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(maskPixels.surface.canvas, 0, 0);

    const resultId = createLayerId();
    const resultLayer: CanvasLayerContract = {
      blendMode: 'normal',
      id: resultId,
      isEnabled: true,
      isLocked: false,
      name: `${maskLayer.name} extraction`,
      opacity: 1,
      source: { bitmap: null, offset: { x: resultRect.x, y: resultRect.y }, type: 'paint' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
    const selectedLayerId = liveDocument.selectedLayerId;
    const apply = (): void => {
      const prepared = prepareGeneratedPaintCache(resultId, resultRect, resultPixels);
      dispatchPreparedMutation(
        {
          add: { index: maskIndex, layer: resultLayer },
          enabledUpdates: [],
          selectedLayerId: resultId,
          type: 'applyCanvasLayerStackMutation',
        },
        () =>
          getReducerDocument()?.selectedLayerId === resultId &&
          getReducerDocument()?.layers.some((candidate) => candidate === resultLayer) === true,
        () =>
          mirror.getDocument()?.selectedLayerId === resultId &&
          mirror.getDocument()?.layers.some((candidate) => candidate === resultLayer) === true
      );
      installGeneratedPaintCache(prepared);
    };

    if (!isDocumentEditPermitCurrent(permit)) {
      return { status: 'busy' };
    }
    apply();
    history.push({
      bytes: resultRect.width * resultRect.height * 4 + 256,
      label: 'Extract masked area',
      redo: apply,
      replayFailureAtomic: true,
      undo: () =>
        dispatchPreparedMutation(
          {
            enabledUpdates: [],
            removeIds: [resultId],
            selectedLayerId,
            type: 'applyCanvasLayerStackMutation',
          },
          () =>
            getReducerDocument()?.selectedLayerId === selectedLayerId &&
            getReducerDocument()?.layers.some((candidate) => candidate.id === resultId) === false,
          () =>
            mirror.getDocument()?.selectedLayerId === selectedLayerId &&
            mirror.getDocument()?.layers.some((candidate) => candidate.id === resultId) === false
        ),
    });
    return { layerId: resultId, status: 'extracted' };
  };

  const mergeLayerDown = (upperLayerId: string): boolean => {
    // No-op mid-gesture (a mod+e mashed during a paint drag): merging writes
    // pixels and dispatches a structural collapse under the open stroke session.
    // Matches the commitStructural/nudge guards; this op is not undoable, so the
    // guard is doubly important. The pipeline reports gesture state.
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return false;
    }
    endNudgeBurst();
    const doc = mirror.getDocument();
    if (!doc) {
      return false;
    }
    const upperIndex = doc.layers.findIndex((layer) => layer.id === upperLayerId);
    if (upperIndex === -1) {
      return false;
    }
    const upper = doc.layers[upperIndex];
    const below = doc.layers[upperIndex + 1];
    // Both layers must be mergeable raster layers (enabled, unlocked, image/paint
    // sourced `raster` type) — the SAME predicate the layers panel's context menu
    // uses to enable/disable merge-down (`isMergeableRasterLayer`), so the hotkey
    // and the menu can never disagree. `isRenderableLayer` is deliberately NOT used
    // here: it is broader (masks, control layers, shapes, text, gradients are all
    // renderable) and gating on it let a mask on either side merge — the reducer's
    // merge unconditionally produces a `type: 'raster'` result, so merging a mask
    // blitted its stencil into the layer below and/or clobbered a mask below into
    // a raster layer, destroying its config. Not undoable, so this guard matters.
    if (!upper || !below || !isMergeableRasterLayer(upper) || !isMergeableRasterLayer(below)) {
      return false;
    }
    const upperCache = layerCache.get(upper.id);
    const belowCache = layerCache.get(below.id);
    if (!upperCache || !belowCache) {
      return false;
    }
    // Refuse while either cache is stale or mid-decode: merging would bake blank/
    // old pixels, and the in-flight rasterize completion would then redraw over
    // the merged result. (This op is not undoable, so a corrupt bake is permanent.)
    if (!isLayerCacheReadyForOp(upper, doc) || !isLayerCacheReadyForOp(below, doc)) {
      return false;
    }
    // Map the upper cache into the below layer's local space (the reducer keeps
    // the below transform verbatim, so we pre-warp the upper pixels to match).
    const matrix = mergeDownMatrix(below.transform, upper.transform);
    if (!matrix) {
      return false;
    }

    // Both layers are empty (content-sized 0×0 caches): the merge is trivially just
    // deleting the upper layer, the below staying empty. Skip every pixel step — a
    // 0×0 merged surface throws — and collapse the pair in the reducer with an empty
    // paint source. Merge-visible folds an all-empty run this way instead of stalling
    // on a silent no-op that leaves the stack half-merged (F4).
    if (isEmpty(belowCache.rect) && isEmpty(upperCache.rect)) {
      store.dispatch({
        source: { bitmap: null, offset: { x: 0, y: 0 }, type: 'paint' },
        type: 'mergeCanvasLayersDown',
        upperLayerId,
      });
      // The dispatch dropped the upper cache and invalidated the below one; the below
      // layer stays empty, so just clear its cache (a fresh empty one rebuilds on the
      // next composite) and persist the (empty) result like any other merge.
      layerCache.delete(below.id);
      notifyLayerPainted(below.id);
      bitmapStore.markLayerDirty(below.id);
      return true;
    }
    // The merged (below-local) content rect is the UNION of both layers' content
    // bounds mapped into the below layer's local space: the below cache's own rect
    // plus the upper cache's rect warped through `matrix`. Content-sized: no
    // document clipping, so nothing outside the old doc rect is lost.
    const upperInBelow = transformBounds(matrix, upperCache.rect);
    const mergedRect = roundOut(union(belowCache.rect, upperInBelow));

    // Composite below + upper into a fresh union-sized surface BEFORE dispatching:
    // the dispatch invalidates the below cache, which would otherwise destroy the
    // pixels we need as the base. Everything is drawn in merged-local space
    // (origin `mergedRect.origin`).
    const merged = backend.createSurface(mergedRect.width, mergedRect.height);
    const mctx = merged.ctx;
    mctx.setTransform(1, 0, 0, 1, 0, 0);
    mctx.clearRect(0, 0, mergedRect.width, mergedRect.height);
    // Below: its surface holds pixels for `belowCache.rect` in below-local; place
    // it at that origin minus the merged origin. Skip when the below layer is empty
    // (content-sizing means an empty paint cache is a 0×0 surface, and drawing a
    // zero-dimension canvas throws in browsers) — the merge is then just the warped
    // upper.
    if (!isEmpty(belowCache.rect)) {
      mctx.drawImage(belowCache.surface.canvas, belowCache.rect.x - mergedRect.x, belowCache.rect.y - mergedRect.y);
    }
    // Upper: warp upper-local → below-local via `matrix`, then shift into
    // merged-local by subtracting the merged origin. Its surface holds pixels for
    // `upperCache.rect` in upper-local, so draw at that local origin. Skip when the
    // upper layer is empty (0×0 surface — see above): the merge then degenerates to
    // deleting the upper layer, leaving the below pixels unchanged.
    if (!isEmpty(upperCache.rect)) {
      mctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - mergedRect.x, matrix.f - mergedRect.y);
      mctx.globalAlpha = upper.opacity;
      mctx.globalCompositeOperation = blendToComposite(upper.blendMode);
      mctx.drawImage(upperCache.surface.canvas, upperCache.rect.x, upperCache.rect.y);
      mctx.setTransform(1, 0, 0, 1, 0, 0);
      mctx.globalAlpha = 1;
      mctx.globalCompositeOperation = 'source-over';
    }

    // Collapse the two layers in the reducer (pixels stay in the cache; the
    // bitmap store persists them from there). Destructive: not undoable here.
    // The merged rect's origin rides along as the paint offset (matching the
    // rasterize bake) so the contract already records where the content sits;
    // the bitmap-store flush re-dispatches the same offset with the real ref.
    store.dispatch({
      source: { bitmap: null, offset: { x: mergedRect.x, y: mergedRect.y }, type: 'paint' },
      type: 'mergeCanvasLayersDown',
      upperLayerId,
    });

    // The (synchronously mirrored) dispatch invalidated the below cache and
    // dropped the upper one. Replace the below cache with a fresh union-sized
    // surface holding the merged pixels, placed at the merged rect, and shield it
    // from the async rasterize pass exactly like a fresh paint stroke.
    layerCache.delete(below.id);
    const targetEntry = layerCache.getOrCreateRect(below.id, mergedRect);
    targetEntry.surface.ctx.drawImage(merged.canvas, 0, 0);
    targetEntry.stale = false;
    notifyLayerPainted(below.id);
    bitmapStore.markLayerDirty(below.id);
    return true;
  };

  /** Guard against a pathological fold loop if a merge ever fails to remove a layer. */
  const MAX_MERGE_VISIBLE_STEPS = 256;

  const mergeVisibleRasterLayers = (): MergeVisibleResult => {
    if (!canEditDocument()) {
      return 'nothing';
    }
    // Mirrors the mergeLayerDown guard: no pixel work under an open gesture.
    if (pipeline.isGestureActive()) {
      return 'nothing';
    }
    const doc = mirror.getDocument();
    if (!doc) {
      return 'nothing';
    }
    const runs = planMergeVisibleRuns(doc.layers);
    if (runs.length === 0) {
      return 'nothing';
    }
    // Pre-flight the ENTIRE fold before touching any pixels, so a mid-fold
    // refusal can never leave a half-merged (non-undoable) stack:
    // 1. every participant's cache must be ready (not stale / mid-decode);
    // 2. every consecutive pair's merge matrix must be computable. Transforms
    //    are stable across the fold (a merge keeps the below transform verbatim
    //    and the merged layer then becomes the next pair's upper), so checking
    //    the original transforms covers every step exactly.
    const layerById = new Map(doc.layers.map((layer) => [layer.id, layer]));
    for (const run of runs) {
      for (const id of run) {
        const layer = layerById.get(id);
        if (!layer || !isLayerCacheReadyForOp(layer, doc)) {
          return 'not-ready';
        }
      }
      for (let i = 0; i + 1 < run.length; i += 1) {
        const upper = layerById.get(run[i] ?? '');
        const below = layerById.get(run[i + 1] ?? '');
        if (!upper || !below || !mergeDownMatrix(below.transform, upper.transform)) {
          return 'not-ready';
        }
      }
    }
    // Execute the fold: reorder the pair adjacent when interleaved (the reorder
    // only slides the upper past non-raster layers and/or hidden rasters, which
    // is render-neutral — the compositor draws by group rank), merge, re-plan
    // against the synchronously-updated document, repeat.
    for (let step = 0; step < MAX_MERGE_VISIBLE_STEPS; step += 1) {
      const liveDoc = mirror.getDocument();
      if (!liveDoc) {
        break;
      }
      const next = planNextMergeVisibleStep(liveDoc.layers);
      if (!next) {
        break;
      }
      if (next.orderedIds) {
        store.dispatch({ orderedIds: next.orderedIds, type: 'reorderCanvasLayers' });
      }
      if (!mergeLayerDown(next.upperId)) {
        // Defensive: should be unreachable after the pre-flight above.
        break;
      }
    }
    return 'merged';
  };

  /**
   * True when `layer` is a raster layer whose source the engine can rasterize
   * IGNORING its enabled state — the export gate (hidden layers are exported too,
   * so `isRenderableLayer`, which requires `isEnabled`, is too strict). `polygon`
   * shapes have no rasterizer, so they are excluded.
   */
  const isExportableRasterLayer = (layer: CanvasLayerContract): boolean => {
    if (layer.type !== 'raster') {
      return false;
    }
    switch (layer.source.type) {
      case 'image':
      case 'paint':
      case 'gradient':
      case 'text':
        return true;
      case 'shape':
        return layer.source.kind !== 'polygon';
      default:
        return false;
    }
  };

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

  const exportRasterLayersToPsd = async (fileName: string): Promise<PsdExportResult> => {
    const doc = mirror.getDocument();
    if (!doc) {
      return 'nothing';
    }
    const rasterLayers = doc.layers.filter(isExportableRasterLayer);
    if (rasterLayers.length === 0) {
      return 'nothing';
    }
    // Readiness pre-flight: a layer whose cache is in its current decode job would
    // bake blank/old pixels, so refuse the whole export (nothing partial). A
    // layer with no live cache entry is safe — the executor rasterizes it fresh
    // and synchronously. Transform/stroke sessions in progress export the current
    // committed cache state (acceptable).
    for (const layer of rasterLayers) {
      if (isEmpty(getSourceContentRect(layer, doc))) {
        continue;
      }
      if (isCurrentRasterizationJob(layer)) {
        return 'not-ready';
      }
    }
    const inputs: PsdExportLayerInput[] = rasterLayers.map((layer) => ({
      adjustments: layer.type === 'raster' ? layer.adjustments : undefined,
      blendMode: layer.blendMode,
      contentRect: getSourceContentRect(layer, doc),
      id: layer.id,
      isEnabled: layer.isEnabled,
      name: layer.name,
      opacity: layer.opacity,
      transform: layer.transform,
    }));
    const plan = planPsdExport(inputs);
    if (plan.status === 'empty') {
      return 'nothing';
    }
    if (plan.status === 'too-large') {
      return 'too-large';
    }
    const name = /\.psd$/i.test(fileName) ? fileName : `${fileName}.psd`;
    await executePsdExport(plan, name, { backend, getLayerSurface: getLayerSurfaceForPsd });
    return 'exported';
  };

  /**
   * Rasterizes a parametric (shape/gradient/text) layer to pixels: bakes its current
   * appearance (the cached parametric pixels drawn through the layer transform)
   * into a NEW content-sized paint layer at identity — the transformed content
   * bounds become the paint source's persisted offset — then converts the layer's
   * source to `paint` via `convertCanvasLayer`. The baked pixels are written into
   * the layer's cache and persisted through the normal bitmap-store dirty path
   * (mirrors mergeLayerDown / the transform bake).
   *
   * UNDOABLE via a composed history entry: undo re-converts the layer back to its
   * ORIGINAL parametric source (angle/stops/shape params). Because those pixels
   * regenerate deterministically from the params, undo needs NO pixel snapshot —
   * the source-ref change flips the mirror's sourceChanged bit, which invalidates
   * the cache so `ensureLayerCaches` re-rasterizes the parametric source. That is
   * the key asymmetry with a paint merge (whose inverse WOULD need the discarded
   * pixels): parametric params are the cheap, lossless source of truth.
   *
   * Returns `false` (a no-op) mid-gesture, with no document, or when the layer is
   * missing, not a raster layer, locked, or not a (rasterizable) parametric
   * source.
   */
  const rasterizeLayer = (layerId: string): boolean => {
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return false;
    }
    endNudgeBurst();
    const doc = mirror.getDocument();
    if (!doc) {
      return false;
    }
    const layer = doc.layers.find((candidate) => candidate.id === layerId);
    if (!layer || layer.type !== 'raster' || layer.isLocked) {
      return false;
    }
    const source = layer.source;
    if (source.type !== 'shape' && source.type !== 'gradient' && source.type !== 'text') {
      return false;
    }
    if (source.type === 'shape' && source.kind === 'polygon') {
      // Deferred: no polygon rasterizer, so nothing to bake.
      return false;
    }

    // The original parametric layer, captured for the undo re-conversion.
    const parametricLayer: CanvasLayerContract = structuredClone(layer);

    /**
     * Converts the CURRENT parametric layer to paint, re-baking its pixels from
     * the live params each call. This is BOTH the initial apply and the redo:
     * redo runs after an undo has restored the parametric source, so the layer is
     * parametric again and its params regenerate the pixels deterministically.
     * Re-baking (rather than capturing the doc-sized `baked` surface in a closure)
     * keeps the history entry's retained bytes tiny — the entry must not pin a
     * ~w×h×4 surface (a 4096² doc is ~64 MB) invisibly to the byte budget.
     */
    const applyForward = (): void => {
      const liveDoc = mirror.getDocument();
      const liveLayer = liveDoc?.layers.find((candidate) => candidate.id === layerId);
      if (!liveDoc || !liveLayer || liveLayer.type !== 'raster') {
        return;
      }
      const liveSource = liveLayer.source;
      if (liveSource.type !== 'shape' && liveSource.type !== 'gradient' && liveSource.type !== 'text') {
        return;
      }

      // Ensure the parametric cache holds fresh pixels. A parametric layer's
      // content rect is authoritative and origin-anchored, so it — not the cache's
      // possibly-stale `entry.rect` — drives the bake. This matters on redo: a
      // re-bake after undo finds the entry still carrying the PREVIOUS paint-bake
      // extent (an off-origin rect), and `getOrCreateRect` deliberately never
      // resizes an existing entry, so we reset it here before re-rasterizing.
      const contentRect = getSourceContentRect(liveLayer, liveDoc);
      const entry = layerCache.getOrCreateRect(layerId, contentRect);
      entry.rect = contentRect;
      entry.stale = true;
      // The shape/gradient/text rasterizers draw synchronously (they resolve
      // immediately), so the target surface is populated the moment the call
      // returns despite the async signature; the deferred `.then` only reconciles
      // `entry.rect` (a no-op for these origin-anchored parametric rects).
      void rasterizeSource(liveSource, rasterizeDeps(liveDoc), entry.surface).then((r) => {
        entry.rect = r.rect;
      });
      entry.stale = false;

      // Bake the parametric pixels through the layer transform into a surface sized
      // to the TRANSFORMED CONTENT bounds (content-sized, not document-clipped). The
      // layer transform resets to identity, so this doc-space rect becomes the new
      // paint layer's local content rect + persisted offset.
      const matrix = bakeMatrix(liveLayer.transform);
      const bakedRect = roundOut(transformBounds(matrix, contentRect));
      const baked = backend.createSurface(bakedRect.width, bakedRect.height);
      const bctx = baked.ctx;
      bctx.setTransform(1, 0, 0, 1, 0, 0);
      bctx.clearRect(0, 0, bakedRect.width, bakedRect.height);
      bctx.imageSmoothingEnabled = true;
      // Draw in baked-local space (subtract the baked origin from the bake matrix),
      // placing the parametric surface at its content origin.
      bctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - bakedRect.x, matrix.f - bakedRect.y);
      bctx.drawImage(entry.surface.canvas, contentRect.x, contentRect.y);
      bctx.setTransform(1, 0, 0, 1, 0, 0);

      const paintLayer: CanvasLayerContract = {
        ...liveLayer,
        source: { bitmap: null, offset: { x: bakedRect.x, y: bakedRect.y }, type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      };
      store.dispatch({ id: layerId, layer: paintLayer, targetType: 'raster', type: 'convertCanvasLayer' });

      // AFTER the convert dispatch (and the synchronous mirror invalidation it
      // triggers): replace the cache with a fresh content-sized surface holding the
      // baked pixels, placed at `bakedRect`, shielded from the async rasterize pass.
      layerCache.delete(layerId);
      const target = layerCache.getOrCreateRect(layerId, bakedRect);
      target.surface.ctx.drawImage(baked.canvas, 0, 0);
      target.stale = false;
      notifyLayerPainted(layerId);
      bitmapStore.markLayerDirty(layerId);
    };

    applyForward();
    history.push({
      // Tiny: the entry holds only the (pixel-free) parametric layer clone plus
      // closures — redo re-bakes from params, so no doc-sized surface is pinned.
      bytes: 256,
      label: 'Rasterize layer',
      redo: applyForward,
      // Re-convert to the parametric source; the mirror re-rasterizes it from
      // params (no pixel snapshot needed).
      undo: () =>
        store.dispatch({ id: layerId, layer: parametricLayer, targetType: 'raster', type: 'convertCanvasLayer' }),
    });
    return true;
  };

  const commitStructural = (label: string, forward: WorkbenchAction, inverse: WorkbenchAction): void => {
    // No-op while a pointer gesture is in progress. A structural edit injected
    // mid-stroke (a duplicate/delete/reorder hotkey mashed during a paint drag)
    // would replace the layer under the open session and interleave a history
    // entry inside the gesture. The move/bbox tools commit on pointer-UP, by
    // which point the pipeline has already cleared the gesture flag (it is reset
    // before the active tool's `onPointerUp` runs), so their commits still land.
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return;
    }
    endNudgeBurst();
    store.dispatch(forward);
    history.push(
      createDocumentPatchEntry({
        dispatch: (action) => store.dispatch(action),
        forward,
        inverse,
        label,
      })
    );
  };

  const applyStructuralPreview = (action: WorkbenchAction): boolean => {
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return false;
    }
    store.dispatch(action);
    return true;
  };

  /**
   * Nudges the selected layer by `(dx, dy)` document pixels as a structural,
   * undoable edit. A no-op with no selection or a locked/hidden selected layer.
   * A rapid burst of nudges on the same layer (within {@link NUDGE_COALESCE_MS})
   * coalesces into a single history entry whose inverse restores the original
   * position — so a burst undoes in one step, not one press at a time.
   */
  const nudgeSelectedLayer = (dx: number, dy: number): void => {
    // No-op mid-gesture: an arrow-key nudge during a paint drag would dispatch a
    // structural transform under the open stroke session (feeding the cache-wipe
    // path) and interleave a history entry inside the gesture. The pipeline
    // reports gesture state; while a stroke is in progress this is a no-op.
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return;
    }
    const doc = mirror.getDocument();
    if (!doc || !doc.selectedLayerId) {
      return;
    }
    const layer = doc.layers.find((candidate) => candidate.id === doc.selectedLayerId);
    if (!layer || layer.isLocked || !layer.isEnabled) {
      return;
    }
    const oldX = layer.transform.x;
    const oldY = layer.transform.y;
    const next = { x: oldX + dx, y: oldY + dy };

    const now = Date.now();
    const coalesce = !!nudgeBurst && nudgeBurst.layerId === layer.id && now < nudgeBurst.expiresAt;
    const origin = coalesce && nudgeBurst ? nudgeBurst.origin : { x: oldX, y: oldY };

    const forward: WorkbenchAction = {
      id: layer.id,
      patch: { transform: { x: next.x, y: next.y } },
      type: 'updateCanvasLayer',
    };
    const inverse: WorkbenchAction = {
      id: layer.id,
      patch: { transform: { x: origin.x, y: origin.y } },
      type: 'updateCanvasLayer',
    };
    store.dispatch(forward);
    const entry = createDocumentPatchEntry({
      dispatch: (action) => store.dispatch(action),
      forward,
      inverse,
      label: 'Nudge layer',
    });
    if (coalesce) {
      history.amendLast(entry);
    } else {
      history.push(entry);
    }
    nudgeBurst = { expiresAt: now + NUDGE_COALESCE_MS, layerId: layer.id, origin };
  };

  // ---- Transform session --------------------------------------------------
  //
  // The transform tool opens a session on one layer (start/live transform in
  // `stores.transformSession`, preview via `transformOverrides`) that outlives
  // individual pointer gestures. Apply commits — a param edit for image layers,
  // a pixel bake for paint layers — as ONE undoable entry; Cancel drops the
  // preview. The transform tool drives begin/update/cancel through the tool
  // context; React (numeric bar + Apply/Cancel buttons) drives the public API.

  const transformUnchanged = (a: LayerTransform, b: LayerTransform): boolean =>
    a.x === b.x && a.y === b.y && a.scaleX === b.scaleX && a.scaleY === b.scaleY && a.rotation === b.rotation;

  /** Drops the active session's preview override without clearing the session store. */
  const clearSessionOverride = (): void => {
    const session = stores.transformSession.get();
    if (session) {
      transformOverrides.delete(session.layerId);
    }
  };

  const beginTransformSession = (layerId: string): void => {
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (!doc || !layer || !layer.isEnabled || layer.isLocked || !hittableLayerSize(layer, doc)) {
      // Ineligible target: leave any existing session untouched.
      return;
    }
    // Switching layers: drop the outgoing layer's preview first.
    clearSessionOverride();
    const start: LayerTransform = { ...layer.transform };
    stores.transformSession.set({ layerId, startTransform: start, transform: start });
    transformOverrides.set(layerId, start);
    scheduler.invalidate({ layers: [layerId], overlay: true });
  };

  const updateTransformSession = (transform: LayerTransform): void => {
    const session = stores.transformSession.get();
    if (!session) {
      return;
    }
    stores.transformSession.set({ ...session, transform });
    transformOverrides.set(session.layerId, transform);
    scheduler.invalidate({ layers: [session.layerId], overlay: true });
  };

  const cancelTransform = (): void => {
    const session = stores.transformSession.get();
    if (!session) {
      return;
    }
    transformOverrides.delete(session.layerId);
    stores.transformSession.set(null);
    scheduler.invalidate({ layers: [session.layerId], overlay: true });
  };

  /**
   * A composed history entry for a paint-layer transform bake: undo restores the
   * OLD pixels (at the OLD content rect) AND the OLD transform; redo re-applies the
   * baked pixels (at the baked content rect) and the identity transform. Because
   * the pre- and post-bake caches occupy DIFFERENT rects, pixel writes go through
   * {@link restoreLayerCache} (whole-cache swap), not the grow-and-overlay
   * `applyImagePatch`.
   */
  const createTransformBakeEntry = (
    layerId: string,
    beforeRect: Rect,
    before: ImageData,
    afterRect: Rect,
    after: ImageData,
    oldTransform: LayerTransform,
    newTransform: LayerTransform,
    label: string
  ): HistoryEntry => ({
    bytes: before.data.byteLength + after.data.byteLength + 256,
    label,
    redo: () => {
      store.dispatch({ id: layerId, patch: { transform: newTransform }, type: 'updateCanvasLayer' });
      restoreLayerCache(layerId, afterRect, after);
    },
    undo: () => {
      store.dispatch({ id: layerId, patch: { transform: oldTransform }, type: 'updateCanvasLayer' });
      restoreLayerCache(layerId, beforeRect, before);
    },
  });

  const applyTransform = (): void => {
    if (!canEditDocument()) {
      return;
    }
    // No-op mid-gesture (guarded like commitStructural/merge): apply writes a
    // structural transform (and, for paint, pixels) that must not interleave with
    // an open pointer session. Apply lands on Enter/button, after pointer-up.
    if (pipeline.isGestureActive()) {
      return;
    }
    const session = stores.transformSession.get();
    if (!session) {
      return;
    }
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === session.layerId);
    const size = doc && layer ? hittableLayerSize(layer, doc) : null;
    // Ineligible now (removed / locked / unrenderable), or an unchanged transform:
    // drop the session with no commit.
    if (
      !doc ||
      !layer ||
      !size ||
      !isRenderableLayer(layer) ||
      layer.isLocked ||
      transformUnchanged(session.transform, session.startTransform)
    ) {
      cancelTransform();
      return;
    }
    const source = layer.type === 'raster' || layer.type === 'control' ? layer.source : null;

    // Parametric-or-image sources commit the transform as a param edit: the
    // source stays parametric (editable-forever), the compositor applies the new
    // transform at draw time. Only paint layers bake pixels (below). CANVAS_PLAN
    // Phase 5: "bake for paint, param for parametric".
    if (
      source?.type === 'image' ||
      source?.type === 'shape' ||
      source?.type === 'gradient' ||
      source?.type === 'text'
    ) {
      // Param commit: ONE structural entry with the new/old transform. No pixels.
      endNudgeBurst();
      transformOverrides.delete(session.layerId);
      stores.transformSession.set(null);
      const forward: WorkbenchAction = {
        id: session.layerId,
        patch: { transform: session.transform },
        type: 'updateCanvasLayer',
      };
      const inverse: WorkbenchAction = {
        id: session.layerId,
        patch: { transform: session.startTransform },
        type: 'updateCanvasLayer',
      };
      store.dispatch(forward);
      history.push(
        createDocumentPatchEntry({
          dispatch: (action) => store.dispatch(action),
          forward,
          inverse,
          label: 'Transform layer',
        })
      );
      scheduler.invalidate({ layers: [session.layerId], overlay: true });
      return;
    }

    if (source?.type !== 'paint') {
      cancelTransform();
      return;
    }

    // Bake for paint: render the old cache through the final matrix into a NEW
    // surface sized to the TRANSFORMED CONTENT bounds (no document clipping — the
    // Task-26 carryover fix), swap the cache, and reset the layer transform to
    // identity — pixels + transform composed into one history entry, persisted
    // through the normal bitmap-store dirty path (mirrors mergeLayerDown).
    const cache = layerCache.get(layer.id);
    if (!cache || isEmpty(cache.rect)) {
      cancelTransform();
      return;
    }
    endNudgeBurst();

    const beforeRect: Rect = { ...cache.rect };
    const before = cache.surface.ctx.getImageData(0, 0, beforeRect.width, beforeRect.height);

    // The transformed content bounds in document space become the baked paint
    // layer's local content rect (the transform resets to identity below).
    const matrix = bakeMatrix(session.transform);
    const afterRect = roundOut(transformBounds(matrix, beforeRect));

    const baked = backend.createSurface(afterRect.width, afterRect.height);
    const bctx = baked.ctx;
    bctx.setTransform(1, 0, 0, 1, 0, 0);
    bctx.clearRect(0, 0, afterRect.width, afterRect.height);
    // The bake is a one-shot resample onto the layer's permanent pixels — always
    // smooth it explicitly (quality over per-frame cost; no zoom to key off).
    bctx.imageSmoothingEnabled = true;
    // Draw in baked-local space: subtract the baked origin from the bake matrix and
    // place the cache surface at its content origin.
    bctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - afterRect.x, matrix.f - afterRect.y);
    bctx.drawImage(cache.surface.canvas, beforeRect.x, beforeRect.y);
    bctx.setTransform(1, 0, 0, 1, 0, 0);
    const after = baked.ctx.getImageData(0, 0, afterRect.width, afterRect.height);

    const oldTransform: LayerTransform = { ...session.startTransform };
    const identityTransform: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

    // Clear the preview before the structural dispatch so the composite shows the
    // reset transform + baked pixels (which are equal to the preview anyway).
    transformOverrides.delete(layer.id);
    stores.transformSession.set(null);

    // Reset the transform in the reducer (pixel-free), then swap the cache to the
    // baked, content-sized surface placed at the baked rect.
    store.dispatch({ id: layer.id, patch: { transform: identityTransform }, type: 'updateCanvasLayer' });
    layerCache.delete(layer.id);
    const target = layerCache.getOrCreateRect(layer.id, afterRect);
    target.surface.ctx.drawImage(baked.canvas, 0, 0);
    target.stale = false;
    notifyLayerPainted(layer.id);
    bitmapStore.markLayerDirty(layer.id);

    history.push(
      createTransformBakeEntry(
        layer.id,
        beforeRect,
        before,
        afterRect,
        after,
        oldTransform,
        identityTransform,
        'Transform layer'
      )
    );
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

  let textSessionId = 0;

  // A getter the React portal registers (in its ref callback) so the engine can
  // read the live contenteditable text at commit time WITHOUT per-keystroke
  // traffic. `null` when no portal is mounted (e.g. node tests inject their own).
  let textContentReader: (() => string) | null = null;

  const setTextEditContentReader = (reader: (() => string) | null): void => {
    textContentReader = reader;
  };

  const textSourceEqual = (a: TextSource, b: TextSource): boolean =>
    a.content === b.content &&
    a.fontFamily === b.fontFamily &&
    a.fontSize === b.fontSize &&
    a.fontWeight === b.fontWeight &&
    a.lineHeight === b.lineHeight &&
    a.align === b.align &&
    a.color === b.color;

  /** Builds a text source from the given content + the current text options. */
  const textSourceFromOptions = (content: string): TextSource => {
    const options = stores.textOptions.get();
    return {
      align: options.align,
      color: options.color,
      content,
      fontFamily: options.fontFamily,
      fontSize: options.fontSize,
      fontWeight: options.fontWeight,
      lineHeight: options.lineHeight,
      type: 'text',
    };
  };

  const openTextCreate = (docPoint: Vec2): void => {
    if (!mirror.getDocument()) {
      return;
    }
    stores.textEditSession.set({
      id: ++textSessionId,
      layerId: null,
      mode: 'create',
      source: textSourceFromOptions(''),
      startSource: null,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: Math.round(docPoint.x), y: Math.round(docPoint.y) },
    });
    // Create mode adds no layer yet, so the composite is unchanged; the portal
    // draws over it. Refresh the overlay for the tool cursor.
    scheduler.invalidate({ overlay: true });
  };

  const openTextEdit = (layerId: string): void => {
    const doc = mirror.getDocument();
    const layer = doc?.layers.find((candidate) => candidate.id === layerId);
    if (
      !doc ||
      !layer ||
      layer.type !== 'raster' ||
      layer.source.type !== 'text' ||
      !layer.isEnabled ||
      layer.isLocked
    ) {
      return;
    }
    stores.textEditSession.set({
      id: ++textSessionId,
      layerId,
      mode: 'edit',
      source: { ...layer.source },
      startSource: { ...layer.source },
      transform: { ...layer.transform },
    });
    // Recomposite so the layer is skipped (the portal shows its live text).
    scheduler.invalidate({ layers: [layerId] });
  };

  const updateTextEditStyle = (patch: Partial<TextToolOptions>): void => {
    const session = stores.textEditSession.get();
    if (!session) {
      return;
    }
    // The edit-mode layer is already skipped and create mode has no layer, so no
    // composite change is needed — the portal restyles itself from the store.
    stores.textEditSession.set({ ...session, source: { ...session.source, ...patch } });
  };

  const cancelTextEdit = (): void => {
    const session = stores.textEditSession.get();
    if (!session) {
      return;
    }
    stores.textEditSession.set(null);
    // Unskip: recompositing redraws an edit-mode layer's committed pixels.
    if (session.layerId) {
      scheduler.invalidate({ layers: [session.layerId] });
    } else {
      scheduler.invalidate({ overlay: true });
    }
  };

  const commitTextEdit = (content: string, styleChanges?: Partial<TextToolOptions>): void => {
    if (!canEditDocument()) {
      return;
    }
    // Defensive mid-gesture guard (commit is React-driven on blur/mod+enter, so
    // this normally never fires mid-stroke): a structural dispatch must not
    // interleave with an open pointer session. Mirrors commitStructural's guard.
    if (pipeline.isGestureActive()) {
      return;
    }
    const session = stores.textEditSession.get();
    if (!session) {
      return;
    }
    const finalSource: TextSource = { ...session.source, ...styleChanges, content };

    if (session.mode === 'create') {
      if (content.trim() === '') {
        // An empty create adds nothing (cancel semantics).
        cancelTextEdit();
        return;
      }
      const doc = mirror.getDocument();
      const layerId = createLayerId();
      const layer: CanvasLayerContract = {
        blendMode: 'normal',
        id: layerId,
        isEnabled: true,
        isLocked: false,
        name: `Text ${(doc?.layers.length ?? 0) + 1}`,
        opacity: 1,
        source: finalSource,
        transform: session.transform,
        type: 'raster',
      };
      stores.textEditSession.set(null);
      commitStructural(
        'Add text',
        { index: 0, layer, type: 'addCanvasLayer' },
        { ids: [layerId], type: 'removeCanvasLayers' }
      );
      scheduler.invalidate({ overlay: true });
      return;
    }

    // Edit mode: ONE updateCanvasLayerSource with the exact inverse (the source
    // captured at session open). No change → no dispatch (cancel semantics).
    const layerId = session.layerId;
    const before = session.startSource;
    if (!layerId || !before) {
      cancelTextEdit();
      return;
    }
    stores.textEditSession.set(null);
    if (textSourceEqual(before, finalSource)) {
      scheduler.invalidate({ layers: [layerId] });
      return;
    }
    commitStructural(
      'Edit text',
      { id: layerId, source: finalSource, type: 'updateCanvasLayerSource' },
      { id: layerId, source: before, type: 'updateCanvasLayerSource' }
    );
  };

  /**
   * Commits an open text-edit session, reading the live content from the portal's
   * registered reader (falling back to the session's own content when none is
   * registered — node tests, or a session with no mounted portal). Returns `true`
   * when a session was open and committed, `false` otherwise. This is the
   * engine-side, node-testable commit that a canvas pointerdown triggers (via the
   * pointer pipeline's `maybeCommitModalSession`): the reviewer's "click elsewhere
   * to commit" — the click's job is to close the open session, so the pipeline
   * swallows it rather than painting or opening a second session. It runs BEFORE
   * the pipeline sets `gestureActive`, so `commitTextEdit`'s mid-gesture guard
   * cannot swallow it. One commit per session close is preserved: `commitTextEdit`
   * clears the session, and the portal's `onBlur` re-commit then no-ops.
   */
  const commitOpenTextSession = (): boolean => {
    if (!canEditDocument()) {
      return false;
    }
    const session = stores.textEditSession.get();
    if (!session) {
      return false;
    }
    const content = textContentReader ? textContentReader() : session.source.content;
    commitTextEdit(content);
    return true;
  };

  // ---- Selection public API -----------------------------------------------

  /**
   * The bounded domain selectAll/invert operate over now that the document rect is
   * retired: `content ∪ bbox` — the same union `fitToView` fits. The bbox anchors
   * an empty canvas; any renderable layer beyond it is unioned in. The closest
   * coherent analogue of legacy's bounded canvas for the complement in `invert`.
   */
  const selectionDomain = (): Rect | null => {
    const doc = mirror.getDocument();
    if (!doc) {
      return null;
    }
    let bounds: Rect = { ...doc.bbox };
    for (const layer of doc.layers) {
      if (isRenderableLayer(layer)) {
        bounds = union(bounds, getSourceBounds(layer, doc));
      }
    }
    return roundOut(bounds);
  };

  const selectAll = (): void => {
    const rect = selectionDomain();
    if (rect) {
      selection.selectAll(rect);
    }
  };

  const deselect = (): void => {
    selection.clear();
  };

  const invertSelection = (): void => {
    const rect = selectionDomain();
    if (rect) {
      selection.invert(rect);
    }
  };

  /** Resolves the selected paint layer eligible for a masked fill/erase, or `null`. */
  const selectionPaintTarget = (): { layerId: string; transparencyLocked: boolean } | null => {
    const doc = mirror.getDocument();
    if (!doc || !doc.selectedLayerId) {
      return null;
    }
    const layer = doc.layers.find((candidate) => candidate.id === doc.selectedLayerId);
    // Paint layers only (image-source layers are a no-op this phase — rasterize
    // comes with the next task); locked/hidden targets are refused.
    if (!layer || layer.type !== 'raster' || layer.source.type !== 'paint' || layer.isLocked || !layer.isEnabled) {
      return null;
    }
    return { layerId: layer.id, transparencyLocked: layer.isTransparencyLocked === true };
  };

  /**
   * Fills or erases the selection region on the selected paint layer, as ONE
   * undoable image patch (mirrors a stroke: cache write + before/after history +
   * bitmap-store persistence). A no-op mid-gesture, with no selection, or with an
   * ineligible target.
   *
   * Content-sized: FILL may GROW the layer's cache to the selection bounds (a fill
   * in empty space must appear); ERASE only touches existing pixels (its region is
   * clamped to the layer's current content extent). The patch rect is layer-local
   * (= document space for identity paint layers).
   */
  const runMaskedSelectionEdit = (kind: 'fill' | 'erase'): void => {
    if (!canEditDocument()) {
      return;
    }
    if (pipeline.isGestureActive()) {
      return;
    }
    const doc = mirror.getDocument();
    const placedMask = selection.mask();
    const bounds = selection.bounds();
    if (!doc || !placedMask || !bounds) {
      return;
    }
    const target = selectionPaintTarget();
    if (!target) {
      return;
    }
    // Transparency lock (mirrors the brush/eraser tool policy, paintTool.ts): an
    // ERASE is refused outright (it would delete locked alpha), and a FILL is
    // constrained to existing pixels via `source-atop` — colour lands only on
    // already-opaque areas, never into transparent space.
    if (kind === 'erase' && target.transparencyLocked) {
      return;
    }
    const selRect = roundOut(bounds);
    let rect: Rect | null;
    let entry: ReturnType<LayerCacheStore['getOrCreateRect']>;
    if (kind === 'fill' && !target.transparencyLocked) {
      // Grow the cache to the selection bounds so a fill in empty space appears.
      rect = selRect;
      entry = layerCache.growToRect(target.layerId, selRect);
    } else {
      // Erase, OR a transparency-locked fill: both only affect EXISTING pixels, so
      // clamp to the current content extent (never grow into empty space).
      const existing = layerCache.get(target.layerId);
      if (!existing || isEmpty(existing.rect)) {
        return;
      }
      rect = intersect(selRect, existing.rect);
      entry = existing;
    }
    if (!rect || isEmpty(rect)) {
      return;
    }
    endNudgeBurst();
    const surface = entry.surface;
    const origin = { x: entry.rect.x, y: entry.rect.y };
    const before = surface.ctx.getImageData(rect.x - origin.x, rect.y - origin.y, rect.width, rect.height);
    if (kind === 'fill') {
      fillMaskedRegion({
        backend,
        color: stores.brushOptions.get().color,
        // Transparency lock: colour only on existing pixels (never fill holes).
        composite: target.transparencyLocked ? 'source-atop' : 'source-over',
        mask: placedMask.surface,
        maskOrigin: placedMask.rect,
        rect,
        target: surface,
        targetOrigin: origin,
      });
    } else {
      eraseMaskedRegion({
        backend,
        mask: placedMask.surface,
        maskOrigin: placedMask.rect,
        rect,
        target: surface,
        targetOrigin: origin,
      });
    }
    const after = surface.ctx.getImageData(rect.x - origin.x, rect.y - origin.y, rect.width, rect.height);
    notifyLayerPainted(target.layerId);
    bitmapStore.markLayerDirty(target.layerId);
    if (!history.isApplying()) {
      history.push(
        createImagePatchEntry({
          after,
          apply: applyImagePatch,
          before,
          label: kind === 'fill' ? 'Fill selection' : 'Erase selection',
          layerId: target.layerId,
          rect,
        })
      );
    }
  };

  const fillSelection = (): void => runMaskedSelectionEdit('fill');
  const eraseSelection = (): void => runMaskedSelectionEdit('erase');

  /** Clears a mask's live pixels and persisted bitmap while retaining prompts, fill, and modifiers. */
  const clearMask = (layerId: string): boolean => {
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return false;
    }
    const doc = mirror.getDocument();
    if (!doc) {
      return false;
    }
    const layer = doc.layers.find((candidate) => candidate.id === layerId);
    if (!layer || !isMaskLayer(layer) || layer.isLocked) {
      return false;
    }
    const originalBitmap = layer.mask.bitmap;
    const originalOffset = layer.mask.offset ?? { x: 0, y: 0 };
    const entry = isLayerCacheReadyForOp(layer, doc) ? layerCache.get(layerId) : undefined;
    const rect = entry && !isEmpty(entry.rect) ? { ...entry.rect } : null;
    const before = rect ? entry?.surface.ctx.getImageData(0, 0, rect.width, rect.height) : null;
    if (!originalBitmap && (!before || !rect)) {
      return false;
    }
    endNudgeBurst();
    const emptyRect = { height: 0, width: 0, x: 0, y: 0 };
    const dispatchMask = (bitmap: CanvasImageRef | null, offset: { x: number; y: number }): void => {
      store.dispatch({
        config: { layerType: layer.type, mask: { bitmap, offset } },
        id: layerId,
        type: 'updateCanvasLayerConfig',
      });
    };
    const applyClear = (): void => {
      bitmapStore.discardLayer(layerId);
      dispatchMask(null, { x: 0, y: 0 });
      layerCache.delete(layerId);
      adjustedSurfaceCache.delete(layerId);
      const empty = layerCache.getOrCreateRect(layerId, emptyRect);
      empty.stale = false;
      notifyLayerPainted(layerId);
    };
    const applyRestore = (): void => {
      bitmapStore.discardLayer(layerId);
      dispatchMask(originalBitmap, originalOffset);
      if (before && rect) {
        restoreLayerCache(layerId, rect, before);
      }
    };

    applyClear();
    history.push({
      bytes: (before?.data.byteLength ?? 0) + 256,
      label: 'Clear mask',
      redo: applyClear,
      undo: applyRestore,
    });
    return true;
  };

  /**
   * Inverts a mask layer's alpha in place, over the domain
   * `content ∪ liveCacheRect ∪ bbox` (the same bounded-domain decision the
   * selection invert uses: a bounded, meaningful region rather than the
   * unbounded plane). The live cache rect is unioned in alongside the contract's
   * persisted content rect because the latter lags the debounced bitmap-store
   * flush — a stroke painted moments ago is already reflected in `layerCache`
   * but not yet in `mask.bitmap`, so relying on content alone would silently
   * exclude it (and any of its pixels outside the bbox) from the invert.
   * Legacy parity (`hooks/useInvertMask.ts`): only the ALPHA channel is flipped
   * (`a = 255 - a`); RGB is irrelevant since the compositor colorizes by alpha.
   * ONE undoable image patch (pixel-exact round trip), persisted through the same
   * dirty path as a stroke. A no-op mid-gesture, or when the layer is missing, not
   * a mask, or locked/hidden. Returns whether it ran.
   */
  const invertMask = (layerId: string): boolean => {
    if (!canEditDocument() || pipeline.isGestureActive()) {
      return false;
    }
    const doc = mirror.getDocument();
    if (!doc) {
      return false;
    }
    const layer = doc.layers.find((candidate) => candidate.id === layerId);
    if (!layer || !isMaskLayer(layer) || layer.isLocked || !layer.isEnabled) {
      return false;
    }
    // Refuse while the mask's cache is stale or its bitmap decode is in flight:
    // inverting would read a blank/old surface (garbage `before` in history), and
    // the in-flight `rasterizeSource` completion would then redraw the surface,
    // erasing the invert entirely. A freshly-painted mask (live cache, not stale)
    // and an empty mask (no persisted bitmap) both pass and invert correctly.
    if (!isLayerCacheReadyForOp(layer, doc)) {
      return false;
    }
    // The contract's `getSourceContentRect` reads the persisted `mask.bitmap`,
    // which lags the live paint cache until the debounced flush runs — a stroke
    // painted moments ago (still in-flight in `layerCache`, not yet flushed to the
    // contract) would otherwise be silently excluded from the invert domain.
    // Union in the live cache's rect (when present) so an un-flushed stroke that
    // extends past the persisted content is still covered.
    const content = getSourceContentRect(layer, doc);
    const liveRect = layerCache.get(layerId)?.rect;
    const contentUnion =
      liveRect && !isEmpty(liveRect) ? (isEmpty(content) ? liveRect : union(content, liveRect)) : content;
    // `content`/`liveRect` are LAYER-LOCAL (the cache's own space, which is what
    // `getImageData` below reads); `doc.bbox` is DOCUMENT space. Unioning them
    // directly inverts the wrong region on a moved/transformed mask. Convert the
    // bbox into layer-local space (inverse of the layer transform, rotation-aware)
    // before the union; identity transforms are unchanged.
    const inverse = invertMatrix(bakeMatrix(layer.transform));
    const bbox = inverse ? roundOut(transformBounds(inverse, doc.bbox)) : doc.bbox;
    const domain = roundOut(isEmpty(contentUnion) ? bbox : union(contentUnion, bbox));
    if (isEmpty(domain)) {
      return false;
    }
    endNudgeBurst();
    // Grow the cache to the whole domain (content may be empty / smaller than the
    // bbox) so the previously-transparent region flips to opaque coverage.
    const entry = layerCache.growToRect(layerId, domain);
    const origin = { x: entry.rect.x, y: entry.rect.y };
    const surfaceCtx = entry.surface.ctx;
    const readX = domain.x - origin.x;
    const readY = domain.y - origin.y;
    const before = surfaceCtx.getImageData(readX, readY, domain.width, domain.height);
    // A second read of the same (still-pristine) pixels to mutate + write back —
    // avoids `new ImageData` (absent in the node test env). `work` becomes the
    // patch's `after`: after `putImageData` its bytes are exactly the surface's.
    const work = surfaceCtx.getImageData(readX, readY, domain.width, domain.height);
    const data = work.data;
    for (let i = 3; i < data.length; i += 4) {
      data[i] = 255 - (data[i] ?? 0);
    }
    surfaceCtx.putImageData(work, readX, readY);
    notifyLayerPainted(layerId);
    bitmapStore.markLayerDirty(layerId);
    if (!history.isApplying()) {
      history.push(
        createImagePatchEntry({
          after: work,
          apply: applyImagePatch,
          before,
          label: 'Invert mask',
          layerId,
          rect: domain,
        })
      );
    }
    return true;
  };

  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    clearOwnedFilterSession();
    clearOwnedSelectObjectSession();
    canvasOperations.dispose();
    cancelAllLayerRasterizations();
    detach();
    // Drop any open text-edit session (its layer belongs to a document this
    // engine no longer serves).
    stores.textEditSession.set(null);
    // Drop any guarded filter previews outright — the engine is going away, so
    // there's no render loop left to invalidate for them.
    filterPreviews.clear();
    filterPreviewTokens.clear();
    guardedFilterPreviewTokens.clear();
    antsAnimator.stop();
    selection.dispose();
    activeTool()?.onDeactivate?.(toolContext);
    unsubscribeViewport();
    unsubscribeBrushOptions();
    unsubscribeEraserOptions();
    unsubscribeCheckerboard();
    unsubscribeCheckerColors();
    unsubscribeShowGrid();
    unsubscribeBboxGrid();
    unsubscribeShowBbox();
    unsubscribeBboxOverlay();
    unsubscribeRuleOfThirds();
    unsubscribeHistory();
    unsubscribeProjectPreviewLifecycle();
    unsubscribeDocumentEditingLock();
    history.clear();
    bitmapStore.dispose();
    mirror.dispose();
    scheduler.dispose();
    layerCache.dispose();
    adjustedSurfaceCache.dispose();
    trackedImageNames.clear();
    stores.thumbnailStatus.clear();
    strokeListeners.clear();
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
    await bitmapStore.flushPendingUploads();
    const doc = mirror.getDocument();
    // Invalidate (mark stale → re-rasterize) every live layer cache and drop its
    // memoized adjusted surface; the next composite rebuilds them from source.
    for (const layer of doc?.layers ?? []) {
      invalidateLayerCache(layer.id);
      adjustedSurfaceCache.delete(layer.id);
    }
    // Drop the derived pattern tiles so they rebuild from the current fed colors.
    checkerboardTile = null;
    maskPatternTiles.clear();
    scheduler.invalidate({ all: true });
  };

  const clearHistory = (): void => {
    if (!canEditDocument()) {
      return;
    }
    // `history.clear` notifies subscribers, so `syncHistoryStores` refreshes
    // canUndo/canRedo — the header buttons disable in lockstep.
    endNudgeBurst();
    history.clear();
  };

  const logDebugInfo = (): void => {
    const doc = mirror.getDocument();
    // eslint-disable-next-line no-console
    console.info('[canvas-engine] debug info', {
      activeTool: activeToolId,
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

  return {
    applyStructuralPreview,
    applyTransform,
    booleanMergeRasterLayers,
    canvasOperations,
    captureLayerExportGuard: captureCurrentLayerExportGuard,
    clearCaches,
    clearHistory,
    clearMask,
    contextMenuLayerIdAt,
    logDebugInfo,
    attach,
    cancelSelectObjectSession,
    cancelFilterOperation,
    cancelTextEdit,
    cancelTransform,
    commitGeneratedImageResult,
    commitOpenTextSession,
    commitLayerConversion,
    commitLayerCopy,
    commitMaskImageResult,
    commitRasterFilterResult,
    commitFilterOperation,
    commitStructural,
    commitTextEdit,
    cropLayerToBbox,
    copyLayerToRaster,
    deselect,
    detach,
    dispose,
    disposeSelectObjectSession: cancelSelectObjectSession,
    drawLayerThumbnail,
    eraseSelection,
    extractMaskedArea,
    exportRasterLayersToPsd,
    exportBakedLayerBlob,
    exportBakedLayerPixels,
    exportLayerPixels: rasterizeLayerPixels,
    fillSelection,
    fitToView,
    flushPendingUploads: () => bitmapStore.flushPendingUploads(),
    getCompositeExecutorDeps: () => ({
      backend,
      getLayerSurface: getLayerSurfaceForExport,
      uploadImage: (blob) => uploadCanvasImage(blob, { isIntermediate: true }),
    }),
    getDocument: () => mirror.getDocument(),
    handleEscapePriority,
    hasExportableLayerContent,
    invertMask,
    invertSelection,
    isLayerExportGuardCurrent,
    mergeLayerDown,
    mergeVisibleRasterLayers,
    processSelectObjectSession,
    processFilterOperation,
    applySelectObjectSession,
    resetSelectObjectSession,
    resetFilterOperation,
    saveSelectObjectSession,
    startSelectObject,
    startFilterOperation,
    updateFilterOperation,
    updateSelectObjectSession,
    openTextCreate,
    openTextEdit,
    rasterizeLayer,
    requestLayerThumbnail,
    replaceSelectionFromImage,
    setTextEditContentReader,
    updateTextEditStyle,
    getViewport: () => viewport,
    onStrokeCommitted,
    projectId,
    selectAll,
    // Guarded against firing mid-stroke: an undo/redo during a live gesture would
    // put pixels under the open session, and the eventual commit would record a
    // before/after straddling the injected pixels. The pipeline reports gesture
    // state; while a stroke is in progress these are no-ops.
    nudgeSelectedLayer,
    redo: () => {
      if (!canEditDocument() || pipeline.isGestureActive()) {
        return;
      }
      endNudgeBurst();
      history.redo();
    },
    resize,
    setBboxGrid: (size) => stores.bboxGrid.set(size > 0 ? size : 1),
    setGuardedFilterPreview,
    setInteractionLocked,
    setStagedPreview,
    setTool,
    stepBrushSize: stepActiveBrushSize,
    stores,
    undo: () => {
      if (!canEditDocument() || pipeline.isGestureActive()) {
        return;
      }
      endNudgeBurst();
      history.undo();
    },
    updateTransformSession,
  };
};
