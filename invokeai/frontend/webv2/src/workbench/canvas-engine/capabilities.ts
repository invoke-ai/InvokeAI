import type {
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasStagingCandidateContract,
  CanvasStateContractV3,
  CanvasTextFontRef,
} from '@workbench/canvas-engine/contracts';
import type { CanvasMutationOrigin } from '@workbench/canvas-engine/mutationContracts';
import type { StrokeCommittedEvent } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';

import type { BooleanRasterResult } from './controllers/booleanMergeController';
import type { CopyLayerToRasterResult } from './controllers/copyLayerController';
import type { CropLayerResult } from './controllers/cropLayerController';
import type { ExtractMaskedAreaResult } from './controllers/extractMaskedAreaController';
import type { MergeDownResult, MergeVisibleResult } from './controllers/mergeLayerController';
import type { NewRasterLayerResult } from './controllers/newRasterLayerController';
import type { RasterizeLayerResult } from './controllers/rasterizeLayerController';
import type { PreparedDocumentEdit } from './document-model/documentCommands';
import type { CanvasDocumentModel } from './document-model/documentModel';
import type { CanvasCommandRefusal } from './document/commandRefusal';
import type { CanvasNodeInsertionAnchor } from './document/insertionAnchors';
import type { LayerStackKind } from './document/layerStacks';
import type { CanvasTransactionOutcome, SubsetOf } from './editConcurrency';
import type { CanvasEditGate } from './editGate';
import type {
  TextStylePatch,
  ActiveColorPairState,
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
import type {
  CanvasFontReferenceGroup,
  CanvasFontReplacementSummary,
  CanvasFontReplacementTarget,
} from './fontReferences';
import type { CanvasLayerPreviewMutation, CanvasProjectMutation } from './mutationContracts';
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
 * Guarded mutations can decline as busy (another edit or gesture), stale (pixels changed), aborted (caller
 * signal), not-ready (pixels still loading), over-budget (too large to undo or prepare), or command-specific
 * refusals. Structural commits use {@link StructuralCommitResult}.
 */
export type GuardedMutationRefusal =
  | SubsetOf<CanvasTransactionOutcome, 'aborted' | 'busy' | 'stale' | 'not-ready' | 'over-budget'>
  | SubsetOf<CanvasCommandRefusal, 'locked' | 'missing' | 'unsupported'>;

/** A destructive mask edit (clear, invert); `nothing` when the mask holds no pixels to change. */
export type MaskEditResult =
  | { status: 'committed' }
  | { status: SubsetOf<CanvasTransactionOutcome, 'busy' | 'not-ready' | 'over-budget' | 'stale'> }
  | { status: SubsetOf<CanvasCommandRefusal, 'locked' | 'missing' | 'unsupported'> | 'nothing' };

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
  /** The mirrored workbench pair; the engine reads it at gesture start, never writes it. */
  colorPair: ActiveColorPairState;
  documentEditingLocked: boolean;
  eraserOptions: EraserOptions;
  gradientOptions: GradientToolOptions;
  hasFloatingSelection: boolean;
  hasSelection: boolean;
  /** Monotonic signal for engine history-stack mutations (see the History pane). */
  historyEpoch: number;
  invertBrushSizeScroll: boolean;
  lassoOptions: LassoToolOptions;
  marqueeOptions: MarqueeToolOptions;
  /** Monotonic signal: some layer published new cache pixels (thumbnails, overview). */
  layerPixelEpoch: number;
  /** Monotonic signal: the mirrored document changed (an edit, sync or swap), after the engine reacted to it. */
  documentEpoch: number;
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

/** Settings callers may write; every other interaction key is engine-owned state they only observe. */
export type CanvasWritableInteractionKey = Extract<
  keyof CanvasInteractionState,
  | 'bboxOptions'
  | 'bboxOverlay'
  | 'brushOptions'
  | 'checkerboard'
  | 'checkerColors'
  | 'clipToBbox'
  | 'colorPair'
  | 'eraserOptions'
  | 'gradientOptions'
  | 'invertBrushSizeScroll'
  | 'lassoOptions'
  | 'marqueeOptions'
  | 'ruleOfThirds'
  | 'shapeOptions'
  | 'showBbox'
  | 'showGrid'
  | 'snapToGrid'
  | 'textOptions'
>;

export interface CanvasInteractionStateCapability {
  get<K extends keyof CanvasInteractionState>(key: K): CanvasInteractionState[K];
  set<K extends CanvasWritableInteractionKey>(key: K, value: CanvasInteractionState[K]): void;
  subscribe<K extends keyof CanvasInteractionState>(key: K, listener: () => void): () => void;
  getLayerThumbnailStatus(layerId: string): LayerThumbnailStatus | 'idle';
  getLayerThumbnailVersion(layerId: string): number | undefined;
  subscribeLayerThumbnailStatus(layerId: string, listener: () => void): () => void;
  subscribeLayerThumbnailVersion(layerId: string, listener: () => void): () => void;
}

/**
 * Font resolution shared by Canvas rasterization and the editing portal.
 * Implementations may return a browser-only family alias, but that alias must
 * never be written into a persisted text source.
 */
export interface CanvasFontCapability {
  resolveFamily(source: Extract<CanvasLayerSourceContract, { type: 'text' }>): string;
  /** Starts/joins a live editor preview and isolates caller cancellation. */
  ensurePreview(source: Extract<CanvasLayerSourceContract, { type: 'text' }>, signal?: AbortSignal): Promise<string>;
  waitForReady(source: Extract<CanvasLayerSourceContract, { type: 'text' }>, signal?: AbortSignal): Promise<string>;
  subscribe(listener: () => void): () => void;
  /** Groups exact persisted custom references for missing-font recovery UI. */
  collectReferences(document: CanvasDocumentContractV3): readonly CanvasFontReferenceGroup[];
  /** Replaces every matching text source in one undoable structural edit. */
  replaceAllReferences(from: CanvasTextFontRef, target: CanvasFontReplacementTarget): CanvasFontReplacementResult;
}

export interface CanvasCoreStoreCapability {
  readonly interaction: CanvasInteractionStateCapability;
}

export interface CanvasSurfaceCapability {
  /** `keyboardRoot` is the focusable element whose focus gives the canvas its session keys; defaults to the overlay. */
  attach(screenCanvas: HTMLCanvasElement, overlayCanvas: HTMLCanvasElement, keyboardRoot?: HTMLElement): void;
  detach(): void;
  resize(cssWidth: number, cssHeight: number, dpr: number): void;
}

export interface CanvasDocumentCapability {
  captureSnapshot(): CanvasDocumentSnapshot | null;
  getDocument(): CanvasDocumentContractV3 | null;
  /**
   * Counts every reducer document identity change, including previews, selection and syncs, to reject stale
   * prepared edits. Unlike raster `documentGeneration` or wholesale-swap `documentRevision`, it advances on each
   * edit.
   */
  getEditRevision(): number;
  /** Where a new `stack` layer lands: above `aboveId` when it belongs to the stack, else the stack top. */
  captureInsertionAnchor(stack: LayerStackKind, aboveId: string | null): CanvasNodeInsertionAnchor;
  /** The anchor that restores `nodeId` between its current siblings; null when absent. */
  captureRestoreAnchor(nodeId: string): CanvasNodeInsertionAnchor | null;
  /** Swaps in a whole new document; the mirror treats it as a document swap and history clears. */
  replaceDocument(document: CanvasDocumentContractV3): boolean;
  /** The pure model over the current document; the same instance until the document changes. */
  model(): CanvasDocumentModel | null;
}

/** Immutable reducer canvas state captured at one engine document generation. */
export interface CanvasDocumentSnapshot {
  readonly canvas: CanvasStateContractV3;
  readonly documentGeneration: number;
}

export type PsdExportResult =
  | 'exported'
  | 'nothing'
  | 'too-large'
  | 'over-budget'
  | SubsetOf<CanvasTransactionOutcome, 'not-ready' | 'stale' | 'aborted'>;

export interface CanvasPsdExportCapability {
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
}

export interface CanvasViewportCapability {
  getViewport(): Viewport;
  fitToView(): void;
  /** Fit only on first show; reattaching a kept-alive widget must preserve zoom and pan. */
  fitToViewOnFirstShow(): void;
  setBboxGrid(size: number): void;
}

export interface CanvasToolCapability {
  setTool(toolId: ToolId, options?: { temporary?: boolean }): void;
  stepBrushSize(direction: 1 | -1): void;
}

/** Labels of the retained undo/redo steps, for the History pane. */
export interface CanvasHistoryEntries {
  /** Applied steps, oldest first; the last is what `undo()` reverts. */
  past: readonly string[];
  /** Undone steps, next-redo first. */
  future: readonly string[];
}

/**
 * How a replay ended: `applied`, nothing to replay (`empty`), another replay or edit in progress (`busy`/`refused`),
 * or `failed`, in which case the step stayed where it was and the engine reported why.
 */
export type CanvasHistoryReplayStatus = 'applied' | 'empty' | 'busy' | 'refused' | 'failed';

export interface CanvasHistoryCapability {
  undo(): Promise<CanvasHistoryReplayStatus>;
  redo(): Promise<CanvasHistoryReplayStatus>;
  clearHistory(): void;
  getEntries(): CanvasHistoryEntries;
  getHeldAssetRefs(): { images: readonly string[]; videos: readonly string[] };
  /** Replays `offset` steps, one at a time, stopping at the first that does not apply. */
  stepBy(offset: number): Promise<CanvasHistoryReplayStatus>;
}

/** Why the engine refused an edit the user started without a caller to report to (a stroke, a shape). */
export type CanvasEditRefusal = SubsetOf<
  CanvasTransactionOutcome,
  'busy' | 'gesture-active' | 'not-ready' | 'over-budget'
>;

export type LayerThumbnailRequestResult =
  | 'ready'
  | 'error'
  | 'over-budget'
  | SubsetOf<CanvasTransactionOutcome, 'stale'>
  | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'>;

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

export type {
  CanvasEditIntent,
  CanvasLayerBasePatch,
  CanvasLayerConfigPatch,
  CanvasMutationOrigin,
  CanvasProjectMutation,
} from './mutationContracts';
export { GROUP_PATCH_KEYS } from './mutationContracts';

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
  | {
      status:
        | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'>
        | SubsetOf<CanvasTransactionOutcome, 'not-ready' | 'aborted'>
        | 'disabled'
        | 'empty'
        | 'over-budget';
    };

export interface CanvasSelectionCapability {
  deselect(): void;
  eraseSelection(): void;
  fillSelection(): void;
  getSelectionBounds(): Rect | null;
  getSelectionMaskRect(): Rect | null;
  invertSelection(): void;
  /** Returns the active layer's selected pixels as PNG, or null if empty. The widget owns system clipboard writes. */
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

/**
 * The runtime outcome of a structural document commit. `busy`, `gesture-active`, `not-ready`, and
 * `stale` refuse before dispatch. `dispatch-rejected` means the reducer left the document unchanged,
 * which includes a forward that would not change anything. `postcondition-failed` means the reducer
 * accepted but the result could not be verified; `recovered` says how far the inverse got, and any
 * outcome short of `reverted` was reported. Exactly one history entry is recorded, and only for
 * `committed`.
 */
export type StructuralCommitResult =
  | { status: 'committed' }
  | {
      status:
        | SubsetOf<CanvasTransactionOutcome, 'busy' | 'gesture-active' | 'not-ready' | 'over-budget'>
        | 'dispatch-rejected';
    }
  | { status: SubsetOf<CanvasTransactionOutcome, 'stale'>; expectedRevision: number; actualRevision: number }
  | { status: 'postcondition-failed'; recovered: 'reverted' | 'reverted-unmirrored' | 'unreverted' };

export type CanvasFontReplacementResult =
  | (CanvasFontReplacementSummary & { status: 'committed' | 'unchanged' })
  | Exclude<StructuralCommitResult, { status: 'committed' }>;

export interface StructuralCommitOptions {
  /** The edit revision the edit was prepared against; a mismatch refuses as `stale`. */
  expectedRevision?: number;
  /** An extra reducer postcondition beyond "the document changed". */
  verify?: (document: CanvasDocumentContractV3) => boolean;
}

/** The narrowest engine surface a structural edit needs. */
export type { CanvasLayerPreviewMutation } from './mutationContracts';

export interface CanvasStructuralEngine {
  readonly layers: CanvasLayerCapability;
}

/**
 * An owned live preview of a structural edit. `apply` publishes previews at most once per frame and captures, before
 * the first one lands, the values it replaces: the session's baseline. `commit` records the gesture as one undo step
 * from its prepared baseline, without dispatching again when the preview already reached it, and returns to the
 * baseline when it is refused; `cancel` drops pending previews and restores the baseline, unrecorded. A newer
 * session, a commit from elsewhere, a history replay or disposal ends the session first, restoring its baseline and
 * dropping pending previews, so the previewed values never land over a replayed document or inside another step;
 * later calls on an ended session are refused (`commit` reports `busy`). While edits are locked, `apply` and
 * `commit` refuse.
 */
export interface StructuralPreviewSession {
  apply(action: CanvasLayerPreviewMutation): boolean;
  /** The values its previews replaced, as the mutation that restores them; null until it previews, and once ended. */
  baseline(): CanvasLayerPreviewMutation | null;
  cancel(): void;
  commit(label: string, edit: PreparedDocumentEdit): StructuralCommitResult;
  /** False once something other than its own commit or cancel ended it. */
  isActive(): boolean;
}

export interface CanvasLayerCapability {
  /** Starts a preview session, or null while edits are refused. */
  beginStructuralPreview(): StructuralPreviewSession | null;
  /** Ends an open preview session, restoring its baseline, so an edit prepared next sees the committed document. */
  endStructuralPreview(): void;
  canCommitStructural(): boolean;
  commitGeneratedImageResult(options: CommitGeneratedImageOptions): Promise<CommitGeneratedImageResult>;
  commitStagedImage(options: CommitStagedImageOptions): CommitStagedImageResult;
  commitStructural(
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation,
    options?: StructuralCommitOptions
  ): StructuralCommitResult;
  /** Runs a prepared flat edit through the transaction: refusals, dispatch, verification and history. */
  commitPrepared(label: string, edit: PreparedDocumentEdit, options?: PreparedCommitOptions): StructuralCommitResult;
  invertMask(layerId: string): MaskEditResult;
}

export interface PreparedCommitOptions {
  readonly origin?: CanvasMutationOrigin;
}

export interface CommitStagedImageOptions {
  candidate: CanvasStagingCandidateContract;
  /** Save a disabled layer without clearing or otherwise disturbing staging. */
  continueStaging?: boolean;
  selectedImageIndex: number;
}

export type CommitStagedImageResult =
  | { status: 'committed'; layerId: string }
  | {
      status:
        | SubsetOf<CanvasTransactionOutcome, 'busy' | 'stale' | 'not-ready' | 'over-budget'>
        | SubsetOf<CanvasCommandRefusal, 'missing'>;
    };

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
  /**
   * Persists unsaved pixels. With `waitForHeldPixels: false` it rejects instead of waiting while an open edit
   * holds dirty pixels.
   */
  flushPendingUploads(options?: { readonly waitForHeldPixels?: boolean }): Promise<void>;
}

export type CanvasEditCapability = CanvasEditGate;

/**
 * Persisted filter result for {@link CanvasEnginePreviewCapability.setGuardedFilterPreview}, decoded by
 * `imageResolver`, with its document-space rect and producing filter.
 */
export interface FilterPreviewInput {
  imageName: string;
  rect: Rect;
  filterType?: string;
}

export type DuplicateLayersResult =
  | { readonly status: 'duplicated'; readonly duplicateIds: readonly string[]; readonly selectedLayerId: string }
  | { readonly status: SubsetOf<CanvasTransactionOutcome, 'busy' | 'not-ready' | 'stale'> | 'nothing' | 'over-budget' };
/**
 * Layer pixel operations report a distinct outcome: refused before anything changed (`busy`, `not-ready`,
 * `over-budget` when the edit could never be undone), nothing to do, or `failed` when the document refused it.
 */
export type {
  BooleanRasterResult,
  CopyLayerToRasterResult,
  CropLayerResult,
  ExtractMaskedAreaResult,
  MergeDownResult,
  MergeVisibleResult,
  RasterizeLayerResult,
};
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
  readonly rasterOverageBytes: number;
  /** Group composite backing stores allocated or resized. */
  readonly groupSurfaceAllocations: number;
  /** Whole-surface group redraws (new content, geometry, or unknown member damage). */
  readonly groupSurfaceRebuilds: number;
  /** Group redraws limited to the region members damaged. */
  readonly groupSurfaceRefreshes: number;
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
  /** Refusals of edits that have no caller to report to, such as a stroke too large to undo. */
  onEditRefused(listener: (refusal: CanvasEditRefusal) => void): () => void;
  /**
   * Samples the composite once as `#rrggbb`, or null on Escape, tool change or disposal. Restores the previous
   * tool on either outcome.
   */
  requestColorSample(): Promise<string | null>;
  /**
   * Routes unclaimed eyedropper samples to the active color target, falling back to brush color if absent or
   * declined. Disposal removes only this router, preserving newer registrations.
   */
  setColorSampleRouter(router: (hex: string) => boolean): () => void;
  setInteractionLocked(locked: boolean): void;
}

export interface CanvasEngineLayerCapability extends CanvasLayerCapability {
  applyTransform(): void;
  booleanMergeRasterLayers(upperLayerId: string, operation: BooleanRasterOperation): Promise<BooleanRasterResult>;
  cancelTextEdit(): void;
  cancelTransform(): void;
  clearMask(layerId: string): MaskEditResult;
  commitLayerConversion(label: string, expectedLiveLayer: CanvasLayerContract, after: CanvasLayerContract): boolean;
  commitLayerCopy(
    label: string,
    sourceLayerId: string,
    layer: CanvasLayerContract,
    anchor: CanvasNodeInsertionAnchor
  ): boolean;
  commitMaskImageResult(options: CommitMaskImageResultOptions): Promise<CommitMaskImageResult>;
  commitOpenTextSession(): boolean;
  commitRasterFilterResult(options: CommitRasterFilterOptions): Promise<CommitRasterFilterResult>;
  /** `null` when there was nothing to commit: an empty creation or an unchanged edit. */
  commitTextEdit(content: string, styleChanges?: Partial<TextToolOptions>): StructuralCommitResult | null;
  copyLayerToRaster(layerId: string): Promise<CopyLayerToRasterResult>;
  cropLayerToBbox(layerId: string): Promise<CropLayerResult>;
  duplicateLayers(layerIds: readonly string[]): Promise<DuplicateLayersResult>;
  mergeLayerDown(upperLayerId: string): MergeDownResult;
  mergeSelectedRasterLayers(layerIds: readonly string[]): Promise<MergeVisibleResult>;
  mergeVisibleRasterLayers(): Promise<MergeVisibleResult>;
  /** `dispatch-rejected` also covers a selection with nothing eligible to move. */
  nudgeSelectedLayer(dx: number, dy: number): StructuralCommitResult;
  openTextCreate(docPoint: Vec2): void;
  openTextEdit(layerId: string): void;
  rasterizeLayer(layerId: string): RasterizeLayerResult;
  setTextEditContentReader(reader: (() => string) | null): void;
  updateTextEditStyle(patch: TextStylePatch): void;
  updateTransformSession(transform: LayerTransform): void;
}

export interface CanvasEngineExportCapability extends CanvasExportCapability {
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
  extractMaskedArea(maskLayerId: string): Promise<ExtractMaskedAreaResult>;
}

export interface CanvasEnginePreviewCapability extends CanvasPreviewCapability {
  /**
   * Sizes `target` and draws the whole document fitted to `maxSizePx`. Returns its drawn rect, or null without a
   * document.
   */
  drawDocumentOverview(target: HTMLCanvasElement, maxSizePx: number): Rect | null;
  preloadStagedPreview(imageName: string): void;
  setGuardedFilterPreview(
    layerId: string,
    input: FilterPreviewInput,
    guard: LayerExportGuard
  ): Promise<'shown' | SubsetOf<CanvasCommandRefusal, 'missing'> | SubsetOf<CanvasTransactionOutcome, 'stale'>>;
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
  readonly fonts: CanvasFontCapability;
}

export type * from './contracts';
export { CANVAS_COLOR_LABELS, CANVAS_MAX_NODE_COUNT, CANVAS_MAX_NODE_DEPTH } from './contracts';
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
  ShapeToolKind,
  ShapeToolOptions,
  ShapeToolTarget,
  TextEditSession,
  TextToolOptions,
  TransformSession,
} from './engineStores';
export {
  DEFAULT_BRUSH_OPTIONS,
  DEFAULT_ERASER_OPTIONS,
  DEFAULT_GRADIENT_OPTIONS,
  DEFAULT_SHAPE_OPTIONS,
  DEFAULT_TEXT_OPTIONS,
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
export { adjustmentsKey, buildCurveLut } from './render/adjustments';
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
  isEmptyPolygonShape,
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
export { isLeafPixelEditEligible } from './editing/controlPixelEdit';
export {
  getSiblingOrder,
  haveSameStructure,
  isOverlayStack,
  LAYER_STACK_ORDER,
  LAYER_STACKS_TOP_FIRST,
  layerStackOf,
  type LayerStackKind,
  type LayerStackMoveKind,
  type OverlayStackKind,
  type ReorderSiblingsCommand,
} from './document/layerStacks';
export {
  childrenOf,
  collectSubtree,
  collectSubtreeLeaves,
  createEmptyStacks,
  isGroupNode,
  isLeafNode,
  subtreeDepth,
} from './document/documentTree';
export {
  getDocumentIndex,
  getDocumentLayer,
  getDocumentLeaves,
  getDocumentNode,
  hasDocumentNode,
  isSelfOrAncestor,
  outermostNodes,
  type CanvasDocumentIndex,
  type CanvasNodeEntry,
} from './document/documentIndex';
export {
  type HideableLayer,
  isHideableLayer,
  isLayerContributing,
  isLayerEditable,
  isLayerHidden,
  isLayerPaintable,
  isMergeableRasterLayer,
  isLayerTransparencyLocked,
  isNodeHidden,
  isPixelBackedLayer,
} from './document/layerEligibility';
export {
  type DocumentCommand,
  type DocumentRefusal,
  type InvalidTargetReason,
  type MergeDownEligibility,
  type PreparedDocumentEdit,
  type PrepareEditResult,
} from './document-model/documentCommands';
export {
  compileContributingLayers,
  compileDocumentNodes,
  lookupDocumentNodeState,
  compileDocumentLeaves,
  createDocumentModel,
  type CanvasDocumentModel,
  lookupDocumentLayer,
  lookupDocumentLeaf,
  lookupDocumentNode,
  lookupLayerBelow,
  mergeDownEligibility,
} from './document-model/documentModel';
export { checkEditPostconditions, type EditPostcondition } from './document-model/postconditions';
export {
  isLeafIsolated,
  planScreenComposition,
  type CanvasScreenViewState,
  type ScreenCompositionPlan,
} from './document-model/screenComposition';
export { type SemanticLeaf } from './document-model/semanticLeaf';
export { type SemanticNode } from './document-model/semanticNode';
export {
  captureInsertionAnchor,
  captureRestoreAnchor,
  resolveInsertionTarget,
  type CanvasNodeInsertion,
  type CanvasNodeInsertionAnchor,
  type CanvasNodeMove,
  type InsertionAnchorCapture,
} from './document/insertionAnchors';
export { isExportableRasterLayer } from './layerExportGuards';
export {
  getLayerThumbnailFallbackRenderState,
  nextLayerThumbnailFallbackStage,
  resolveLayerThumbnailImageRef,
  type LayerThumbnailFallbackStage,
} from './render/thumbnail';
