import type {
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasStagingCandidateContract,
  CanvasStateContractV2,
} from '@workbench/canvas-engine/contracts';
import type { StrokeCommittedEvent } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';

import type { NewRasterLayerResult } from './controllers/newRasterLayerController';
import type { CanvasEditGate } from './editGate';
import type {
  BboxToolOptions,
  BrushOptions,
  CheckerColors,
  EraserOptions,
  GradientToolOptions,
  LassoToolOptions,
  LayerThumbnailStatus,
  MarqueeToolOptions,
  ShapeToolOptions,
  TextEditSession,
  TextToolOptions,
  TransformSession,
} from './engineStores';
import type { RasterCompositeExportRequest, RasterCompositeExportResult } from './exportRasterComposite';
import type { CanvasProjectMutation } from './mutationContracts';
import type { Rect, ToolId, Vec2 } from './types';
import type { Viewport } from './viewport';

/** Opaque snapshot identity carried through asynchronous layer operations. */
export interface LayerExportGuard {
  readonly projectId: string;
  readonly layerId: string;
  readonly layer: CanvasLayerContract;
  readonly cacheVersion: number;
  readonly documentGeneration: number;
}

/**
 * Why a guarded layer mutation declined to touch the document. Every commit
 * that captures a permit, does async work, then re-checks before publishing
 * reports refusal with exactly these reasons — one shared vocabulary so callers
 * can handle them uniformly and new commits can't invent near-miss variants.
 *
 * `busy` — another edit owns the document, or a gesture started mid-flight.
 * `stale` — the guarded pixels no longer match the live layer.
 * `aborted` — the caller's signal fired.
 */
export type GuardedMutationRefusal = 'aborted' | 'busy' | 'locked' | 'missing' | 'stale' | 'unsupported';

export type CommitRasterFilterResult =
  | { status: 'committed'; layerId: string }
  | { status: GuardedMutationRefusal }
  | { status: 'failed'; message: string };
export interface RasterFilterSettings {
  type: string;
  settings: Record<string, unknown>;
}
export type RasterFilterCommitTarget = 'apply' | 'raster' | 'control';
export interface CommitRasterFilterOptions {
  guard: LayerExportGuard;
  image: CanvasImageRef;
  rect: Rect;
  mode: 'replace' | 'copy';
  filter?: RasterFilterSettings;
  target?: RasterFilterCommitTarget;
  requireExactImageDimensions?: boolean;
  signal?: AbortSignal;
}
export type MaskImageResultTarget = 'inpaint_mask' | 'regional_guidance';
export interface CommitMaskImageResultOptions {
  guard: LayerExportGuard;
  image: CanvasImageRef;
  rect: Rect;
  target: MaskImageResultTarget;
  signal?: AbortSignal;
}
export type CommitMaskImageResult = { status: 'committed'; layerId: string } | { status: GuardedMutationRefusal };

export interface CanvasInteractionState {
  activeTool: ToolId;
  bboxGrid: number;
  bboxOptions: BboxToolOptions;
  bboxOverlay: boolean;
  brushOptions: BrushOptions;
  canRedo: boolean;
  canUndo: boolean;
  checkerboard: boolean;
  checkerColors: CheckerColors;
  clipToBbox: boolean;
  documentEditingLocked: boolean;
  eraserOptions: EraserOptions;
  gradientOptions: GradientToolOptions;
  hasFloatingSelection: boolean;
  hasSelection: boolean;
  invertBrushSizeScroll: boolean;
  lassoOptions: LassoToolOptions;
  marqueeOptions: MarqueeToolOptions;
  /** Monotonic signal for live layer-pixel/cache content changes. */
  rasterContentEpoch: number;
  ruleOfThirds: boolean;
  shapeOptions: ShapeToolOptions;
  showBbox: boolean;
  showGrid: boolean;
  snapToGrid: boolean;
  textEditSession: TextEditSession | null;
  textOptions: TextToolOptions;
  transformSession: TransformSession | null;
  viewportReady: boolean;
  zoom: number;
}

export interface CanvasInteractionStateCapability {
  get<K extends keyof CanvasInteractionState>(key: K): CanvasInteractionState[K];
  set<K extends keyof CanvasInteractionState>(key: K, value: CanvasInteractionState[K]): void;
  subscribe<K extends keyof CanvasInteractionState>(key: K, listener: () => void): () => void;
  getLayerThumbnailStatus(layerId: string): LayerThumbnailStatus | 'idle';
  getLayerThumbnailVersion(layerId: string): number | undefined;
  subscribeLayerThumbnailStatus(layerId: string, listener: () => void): () => void;
  subscribeLayerThumbnailVersion(layerId: string, listener: () => void): () => void;
}

export interface CanvasCoreStoreCapability {
  readonly interaction: CanvasInteractionStateCapability;
}

export interface CanvasSurfaceCapability {
  attach(screenCanvas: HTMLCanvasElement, overlayCanvas: HTMLCanvasElement): void;
  detach(): void;
  resize(cssWidth: number, cssHeight: number, dpr: number): void;
}

export interface CanvasDocumentCapability {
  captureSnapshot(): CanvasDocumentSnapshot | null;
  getDocument(): CanvasDocumentContractV2 | null;
}

/** Immutable reducer canvas state captured at one engine document generation. */
export interface CanvasDocumentSnapshot {
  readonly canvas: CanvasStateContractV2;
  readonly documentGeneration: number;
}

export type PsdExportResult = 'exported' | 'nothing' | 'too-large' | 'not-ready' | 'over-budget' | 'stale' | 'aborted';

export interface CanvasPsdExportCapability {
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
}

export interface CanvasViewportCapability {
  getViewport(): Viewport;
  fitToView(): void;
  /**
   * Fits the document into view only the first time this engine is shown.
   * Surfaces attach on every re-show of a kept-alive widget; an unconditional
   * fit there would discard the user's zoom and pan.
   */
  fitToViewOnFirstShow(): void;
  setBboxGrid(size: number): void;
}

export interface CanvasToolCapability {
  setTool(toolId: ToolId, options?: { temporary?: boolean }): void;
  stepBrushSize(direction: 1 | -1): void;
}

export interface CanvasHistoryCapability {
  undo(): void;
  redo(): void;
  clearHistory(): void;
}

export type LayerThumbnailRequestResult = 'ready' | 'stale' | 'error' | 'missing' | 'unsupported' | 'over-budget';

export interface CanvasPreviewCapability {
  drawLayerThumbnail(layerId: string, target: HTMLCanvasElement, maxSize: number): boolean;
  requestLayerThumbnail(layerId: string): Promise<LayerThumbnailRequestResult>;
}

export interface CanvasExportCapability {
  captureLayerExportGuard(layerId: string): LayerExportGuard | null;
  exportBakedLayerBlob(layerId: string, options?: ExportBakedLayerPixelsOptions): Promise<ExportBakedLayerBlobResult>;
  exportRasterComposite(request: RasterCompositeExportRequest): Promise<RasterCompositeExportResult>;
  hasExportableLayerContent(layerId: string): boolean;
  isLayerExportGuardCurrent(guard: LayerExportGuard): boolean;
}

export type { RasterCompositeExportRequest, RasterCompositeExportResult } from './exportRasterComposite';

/**
 * The canvas mutation vocabulary. Declared under `canvas-engine` (see
 * `mutationContracts.ts`) and surfaced here so workbench callers reach it
 * through the public API instead of the engine importing upward for it.
 */
export type { CanvasLayerBasePatch, CanvasLayerConfigPatch, CanvasProjectMutation } from './mutationContracts';

export interface ExportLayerPixelsOptions {
  includeDisabled?: boolean;
  applyAdjustments?: boolean;
  signal?: AbortSignal;
}

export type ExportBakedLayerPixelsOptions = Omit<ExportLayerPixelsOptions, 'applyAdjustments'> & {
  applyAdjustments?: boolean;
};

export type ExportBakedLayerBlobResult =
  | { status: 'ok'; blob: Blob; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' | 'aborted' };

export interface CanvasSelectionCapability {
  deselect(): void;
  eraseSelection(): void;
  fillSelection(): void;
  getSelectionBounds(): Rect | null;
  getSelectionMaskRect(): Rect | null;
  invertSelection(): void;
  /**
   * Encodes the selection's pixels on the active layer as a PNG, or `null` when
   * there is nothing to copy. The engine deliberately stops at the blob: writing
   * to the system clipboard is a widget-layer concern (`canvas-engine` may not
   * reach `workbench/widgets`).
   */
  exportSelectionBlob(): Promise<Blob | null>;
  /** Inserts decoded pixels as a new raster layer above the active one. */
  pasteImage(pixels: ImageData, center?: Vec2): NewRasterLayerResult;
  /** Copies the selection's pixels into a new layer above the active one, leaving the source intact. */
  liftSelectionToLayer(): NewRasterLayerResult;
  replaceSelectionFromImage(
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal
  ): Promise<ReplaceSelectionFromImageResult>;
  selectAll(): void;
}

export type { NewRasterLayerResult };

export type ReplaceSelectionFromImageResult =
  | { status: 'selected' }
  | { status: GuardedMutationRefusal }
  | { status: 'failed'; message: string };

export interface CanvasLayerCapability {
  applyStructuralPreview(action: CanvasProjectMutation): boolean;
  canCommitStructural(): boolean;
  commitGeneratedImageResult(options: CommitGeneratedImageOptions): Promise<CommitGeneratedImageResult>;
  commitStagedImage(options: CommitStagedImageOptions): CommitStagedImageResult;
  commitStructural(label: string, forward: CanvasProjectMutation, inverse: CanvasProjectMutation): boolean;
  invertMask(layerId: string): boolean;
}

export interface CommitStagedImageOptions {
  candidate: CanvasStagingCandidateContract;
  /** Save a disabled layer without clearing or otherwise disturbing staging. */
  continueStaging?: boolean;
  selectedImageIndex: number;
}

export type CommitStagedImageResult =
  | { status: 'committed'; layerId: string }
  | { status: 'busy' | 'stale' | 'missing' };

export type GeneratedImageTarget = 'replace' | 'copy-raster' | 'copy-control';

export interface CommitGeneratedImageOptions {
  guard: LayerExportGuard;
  image: CanvasImageRef;
  origin: Vec2;
  target: GeneratedImageTarget;
  historyLabel?: string;
  copyLayerName?: string;
  signal?: AbortSignal;
}

export type CommitGeneratedImageResult =
  | { status: 'committed'; layerId: string }
  | { status: GuardedMutationRefusal }
  | { status: 'failed'; message: string };

export type CanvasLifecycleState = 'active' | 'cooling' | 'cool' | 'disposed';

export interface CanvasLifecycleCapability {
  activate(): void;
  beginCooldown(): Promise<'cooled' | 'dirty'>;
  dispose(): void;
  getLifecycleState(): CanvasLifecycleState;
  flushPendingUploads(): Promise<void>;
}

export type CanvasEditCapability = CanvasEditGate;

/**
 * The input to {@link CanvasEnginePreviewCapability.setGuardedFilterPreview}: a
 * persisted filter result (decoded via the engine's `imageResolver`) and the
 * document-space rect it occupies, tagged with the filter that produced it.
 */
export interface FilterPreviewInput {
  imageName: string;
  rect: Rect;
  filterType?: string;
}

/**
 * Result of {@link CanvasEngineLayerCapability.mergeVisibleRasterLayers}: `'merged'` when a new
 * composite layer was inserted, `'not-ready'` when a contributor could not be
 * rasterized consistently, `'over-budget'` when safe raster allocation was
 * refused, `'busy'` when another edit owns the document, and `'nothing'` when
 * fewer than two eligible rasters have content.
 */
export type MergeVisibleResult = 'merged' | 'not-ready' | 'over-budget' | 'busy' | 'nothing';
export interface DuplicateLayersResult {
  readonly duplicateIds: readonly string[];
  readonly selectedLayerId: string;
}
export type BooleanRasterResult = 'merged' | 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty';
export type ExtractMaskedAreaResult =
  | { status: 'extracted'; layerId: string }
  | { status: 'missing' | 'unsupported' | 'not-ready' | 'busy' | 'empty' };
export type CropLayerResult =
  | { status: 'cropped' }
  | { status: 'missing' | 'locked' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' | 'busy' }
  | { status: 'failed'; message: string };
export interface CanvasDiagnosticsCapability {
  clearCaches(): Promise<void>;
  getDiagnostics(): Readonly<CanvasDiagnosticsSnapshot>;
  logDebugInfo(): void;
}

export interface CanvasDiagnosticsSnapshot {
  readonly surfaceCreations: number;
  readonly surfaceResizes: number;
  readonly allocatedBaseBytes: number;
  readonly allocatedDerivedBytes: number;
  readonly imageDataReads: number;
  readonly imageDataWrites: number;
  readonly derivedCacheHits: number;
  readonly derivedCacheMisses: number;
  readonly derivedCacheEvictions: number;
  readonly layersConsidered: number;
  readonly layersCulled: number;
  readonly layersDrawn: number;
  readonly compositeFrames: number;
  readonly overlayFrames: number;
  readonly overBudgetVisibleBaseBytes: number;
}

export interface CanvasEngineToolCapability extends CanvasToolCapability {
  /**
   * Whether the canvas context menu may act on a layer. The menu targets the
   * document's SELECTED layer (never the layer under the pointer); this reports
   * only whether an in-progress gesture/session should suppress it.
   */
  canTargetLayerFromContextMenu(): boolean;
  handleEscapePriority(options: { gestureWasActive: boolean }): void;
  onStrokeCommitted(listener: (event: StrokeCommittedEvent) => void): () => void;
  /**
   * Arms the eyedropper for a single sample of the composited document,
   * resolving with the picked `#rrggbb` or `null` if the user cancels (Escape,
   * another tool, or the engine going away). Restores the previously active
   * tool either way. Backs the color picker's eyedropper button.
   */
  requestColorSample(): Promise<string | null>;
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
  duplicateLayers(layerIds: readonly string[]): DuplicateLayersResult | null;
  mergeLayerDown(upperLayerId: string): boolean;
  mergeSelectedRasterLayers(layerIds: readonly string[]): Promise<MergeVisibleResult>;
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
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
  extractMaskedArea(maskLayerId: string): Promise<ExtractMaskedAreaResult>;
}

export interface CanvasEnginePreviewCapability extends CanvasPreviewCapability {
  setGuardedFilterPreview(
    layerId: string,
    input: FilterPreviewInput,
    guard: LayerExportGuard
  ): Promise<'shown' | 'missing' | 'stale'>;
  setStagedPreview(input: StagedPreviewInput | null): void;
}

/** Public capability-only handle. Construction and mutable stores are intentionally absent. */
export interface CanvasEngine {
  readonly projectId: string;
  readonly surface: CanvasSurfaceCapability;
  readonly viewport: CanvasViewportCapability;
  readonly tools: CanvasEngineToolCapability;
  readonly history: CanvasHistoryCapability;
  readonly lifecycle: CanvasLifecycleCapability;
  readonly layers: CanvasEngineLayerCapability;
  readonly previews: CanvasEnginePreviewCapability;
  readonly selection: CanvasSelectionCapability;
  readonly edits: CanvasEditCapability;
  readonly document: CanvasDocumentCapability;
  readonly exports: CanvasEngineExportCapability;
  readonly diagnostics: CanvasDiagnosticsCapability;
  readonly interaction: CanvasInteractionStateCapability;
}

// Public Canvas-owned value contracts. These remain serializable and contain
// no engine implementation, mutable store, controller, or construction type.
export type * from './contracts';
export type BooleanRasterOperation = 'intersect' | 'cutout' | 'cutaway' | 'exclude';
export interface StagedPreviewPlacement extends Rect {
  opacity: number;
}
export type StagedPreviewInput =
  | { imageName: string; placement?: StagedPreviewPlacement }
  | { dataUrl: string; width: number; height: number };
export type {
  BboxToolOptions,
  BrushOptions,
  CheckerColors,
  EraserOptions,
  GradientStop,
  GradientToolOptions,
  LassoToolOptions,
  LayerThumbnailStatus,
  MarqueeToolOptions,
  ShapeToolOptions,
  TextEditSession,
  TextToolOptions,
  TransformSession,
} from './engineStores';
export {
  MAX_BRUSH_SIZE,
  MAX_SHAPE_STROKE_WIDTH,
  MAX_TEXT_FONT_SIZE,
  MIN_BRUSH_SIZE,
  MIN_TEXT_FONT_SIZE,
  TEXT_FONT_FAMILIES,
  TEXT_FONT_WEIGHTS,
} from './engineStores';
export type { LayerTransform } from './transform/transformMath';
export type { ImageResolver } from './render/rasterizers';
export type { Rect, SelectionOp, ToolId, Vec2 } from './types';
export { adjustmentsKey, buildCurveLut, DEFAULT_ADJUSTMENTS } from './render/adjustments';
export { DEFAULT_CHECKER_COLORS } from './render/compositor';
export {
  getBaseRasterContentBounds,
  getCompositeLayerBounds,
  planBaseRasterComposite,
  type CompositeEntry,
  type CompositeLayerRef,
} from './render/rasterComposite';
export {
  getSourceBounds,
  getSourceContentRect,
  isMergeableRasterLayer,
  isRenderableLayer,
  renderableSourceOf,
} from './document/sources';
export {
  areSelectedRasterLayersContiguous,
  canMergeSelectedRasters,
  canMergeVisibleRasters,
} from './document/mergeVisible';
export { documentToExportLocalSamPoint } from './samCoordinates';
export { bboxEquals, constrainBboxToRatio, roundBbox } from './tools/bboxHitTest';
export { isEmpty, union } from './math/rect';
export { ZOOM_PRESETS } from './math/snapping';
export { isLayerPixelEditEligible } from './editing/controlPixelEdit';
export { type HideableLayer, isHideableLayer, isLayerHidden } from './document/sources';
export {
  getLayerThumbnailFallbackRenderState,
  nextLayerThumbnailFallbackStage,
  resolveLayerThumbnailImageRef,
  type LayerThumbnailFallbackStage,
} from './render/thumbnail';
