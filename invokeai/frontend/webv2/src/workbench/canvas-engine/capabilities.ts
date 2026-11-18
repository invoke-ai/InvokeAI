import type {
  CanvasControlDetectionContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasStagingCandidateContract,
  CanvasStateContractV3,
  CanvasTextFontRef,
  ControlType,
} from '@workbench/canvas-engine/contracts';
import type { CanvasMutationOrigin } from '@workbench/canvas-engine/mutationContracts';
import type { StrokeCommittedEvent } from '@workbench/canvas-engine/tools/tool';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';

import type { BooleanRasterResult } from './controllers/booleanMergeController';
import type { CopyLayerResult } from './controllers/copyLayerController';
import type { CropAllLayersResult, CropLayerResult } from './controllers/cropLayerController';
import type { ExtractMaskedAreaResult } from './controllers/extractMaskedAreaController';
import type { ConvertLayerToControlResult } from './controllers/layerMutationController';
import type { MergeDownResult, MergeInpaintMasksResult, MergeVisibleResult } from './controllers/mergeLayerController';
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
  ControlLayerView,
  BboxToolOptions,
  BrushOptions,
  CheckerColors,
  EraserOptions,
  GradientToolOptions,
  LassoToolOptions,
  LayerAdjustmentStatus,
  LayerQuickSwitchState,
  LayerThumbnailStatus,
  MarqueeToolOptions,
  ShapeToolOptions,
  TextEditSession,
  TextToolOptions,
  TransformOptions,
  TransformSession,
} from './engineStores';
import type { RasterCompositeExportRequest, RasterCompositeExportResult } from './exportRasterComposite';
import type {
  CanvasFontReferenceGroup,
  CanvasFontReplacementSummary,
  CanvasFontReplacementTarget,
} from './fontReferences';
import type { LayerUnreadable } from './layerExportGuards';
import type { CanvasLayerPreviewMutation, CanvasProjectMutation } from './mutationContracts';
import type { AdjustmentRefusal } from './render/exactAdjustments';
import type { Rect, ToolId, Vec2 } from './types';
import type { Viewport } from './viewport';

export interface LayerExportGuard {
  readonly projectId: string;
  readonly layerId: string;
  readonly layer: CanvasLayerContract;
  readonly cacheVersion: number;
  readonly documentGeneration: number;
}

/** Refusals shared by guarded mutations; structural commits use {@link StructuralCommitResult}. */
export type GuardedMutationRefusal =
  | SubsetOf<CanvasTransactionOutcome, 'aborted' | 'busy' | 'stale' | 'not-ready' | 'over-budget'>
  | SubsetOf<CanvasCommandRefusal, 'locked' | 'missing' | 'unsupported'>;

/** `nothing` when the mask holds no pixels to change. */
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
  /** `target: 'control'` only: null makes an untyped copy; either way the copy keeps no derivation of its source. */
  controlCopy?: { controlType: ControlType | null };
  requireExactImageDimensions?: boolean;
  signal?: AbortSignal;
}

export interface ControlDetectionGuard {
  readonly projectId: string;
  readonly layerId: string;
  readonly documentRevision: number;
  /** `controlDetectionInputKey` of the layer when the run started. */
  readonly inputKey: string;
}

export type ControlLayerPatchResult =
  | { status: 'committed' }
  | {
      status: 'stale' | 'missing' | 'locked' | 'busy' | 'not-ready' | 'over-budget' | 'failed';
      message?: string;
    };

export type ControlMapCommitResult =
  | { status: 'committed' }
  | {
      status:
        | 'stale'
        | 'missing'
        | 'locked'
        | 'busy'
        | 'not-ready'
        | 'over-budget'
        | 'aborted'
        | 'unrepresentable'
        | 'failed';
      message?: string;
    };

/** Neither commit takes the operation edit lease: both refuse `busy` while anything else holds the document. */
export interface CanvasControlDerivationCapability {
  /** One user-origin entry, `stale` unless the layer equals `expected`; a repeated `coalesceKey` replaces it. */
  commitControlLayerPatch(options: {
    label: string;
    expected: CanvasControlLayerContract;
    next: CanvasControlLayerContract;
    coalesceKey?: string;
  }): ControlLayerPatchResult;
  /** Null when the layer is missing or not a control layer. */
  captureControlDetectionGuard(layerId: string): ControlDetectionGuard | null;
  /** Installs a detected map as `source`, keeping the original, as a system-origin entry. */
  commitControlMap(options: {
    guard: ControlDetectionGuard;
    image: CanvasImageRef;
    rect: Rect;
    detection: CanvasControlDetectionContract;
    /** Awaited once the guard holds, so a refused or stale result is never promoted; rejection refuses `failed`. */
    makeDurable(): Promise<void>;
    signal?: AbortSignal;
  }): Promise<ControlMapCommitResult>;
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

export type PsdExportStatus = 'idle' | 'preparing' | 'serializing' | 'downloading' | 'cancelling';

export interface CanvasInteractionState {
  psdExportStatus: PsdExportStatus;
  activeTool: ToolId;
  bboxGrid: number;
  bboxOptions: BboxToolOptions;
  bboxOverlay: boolean;
  /** Null when no Frame tool drag is under way. */
  bboxPreview: Rect | null;
  brushOptions: BrushOptions;
  canRedo: boolean;
  canUndo: boolean;
  checkerboard: boolean;
  checkerColors: CheckerColors;
  clipToBbox: boolean;
  /** The engine reads it at gesture start, never writes it. */
  colorPair: ActiveColorPairState;
  /** Null shows every map. */
  controlLayerView: ControlLayerView | null;
  documentEditingLocked: boolean;
  eraserOptions: EraserOptions;
  gradientOptions: GradientToolOptions;
  hasFloatingSelection: boolean;
  hasSelection: boolean;
  historyEpoch: number;
  invertBrushSizeScroll: boolean;
  lassoOptions: LassoToolOptions;
  /** Callers write the bookmark; the engine maintains the previous selection. */
  layerQuickSwitch: LayerQuickSwitchState;
  marqueeOptions: MarqueeToolOptions;
  layerPixelEpoch: number;
  /** Advances after the engine reacted to the document change. */
  documentEpoch: number;
  ruleOfThirds: boolean;
  shapeOptions: ShapeToolOptions;
  showBbox: boolean;
  showGrid: boolean;
  snapToGrid: boolean;
  /** Whether a staged preview hides control layers, masks and regions; an edit of one shows them meanwhile. */
  stagingHidesOverlays: boolean;
  textEditSession: TextEditSession | null;
  textOptions: TextToolOptions;
  transformOptions: TransformOptions;
  transformSession: TransformSession | null;
  /** Waits for fresh pixels to persist; clears on tool switch, selection change, Escape, removal or failure. */
  pendingTransformLayerId: string | null;
  viewportReady: boolean;
  zoom: number;
}

/** Every other interaction key is engine-owned and read-only to callers. */
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
  | 'layerQuickSwitch'
  | 'marqueeOptions'
  | 'ruleOfThirds'
  | 'shapeOptions'
  | 'showBbox'
  | 'showGrid'
  | 'snapToGrid'
  | 'textOptions'
  | 'transformOptions'
>;

export interface CanvasInteractionStateCapability {
  get<K extends keyof CanvasInteractionState>(key: K): CanvasInteractionState[K];
  set<K extends CanvasWritableInteractionKey>(key: K, value: CanvasInteractionState[K]): void;
  subscribe<K extends keyof CanvasInteractionState>(key: K, listener: () => void): () => void;
  getLayerThumbnailStatus(layerId: string): LayerThumbnailStatus | 'idle';
  getLayerThumbnailVersion(layerId: string): number | undefined;
  subscribeLayerThumbnailStatus(layerId: string, listener: () => void): () => void;
  subscribeLayerThumbnailVersion(layerId: string, listener: () => void): () => void;
  getLayerAdjustmentStatus(layerId: string): LayerAdjustmentStatus | 'idle';
  subscribeLayerAdjustmentStatus(layerId: string, listener: () => void): () => void;
}

/** Implementations may return a browser-only family alias, which must never be persisted in a text source. */
export interface CanvasFontCapability {
  resolveFamily(source: Extract<CanvasLayerSourceContract, { type: 'text' }>): string;
  ensurePreview(source: Extract<CanvasLayerSourceContract, { type: 'text' }>, signal?: AbortSignal): Promise<string>;
  waitForReady(source: Extract<CanvasLayerSourceContract, { type: 'text' }>, signal?: AbortSignal): Promise<string>;
  subscribe(listener: () => void): () => void;
  collectReferences(document: CanvasDocumentContractV3): readonly CanvasFontReferenceGroup[];
  replaceAllReferences(from: CanvasTextFontRef, target: CanvasFontReplacementTarget): CanvasFontReplacementResult;
}

export interface CanvasCoreStoreCapability {
  readonly interaction: CanvasInteractionStateCapability;
}

/** Owner of a key as the host's focus judge decides; `companion` acts on this canvas's layers. */
export type CanvasKeyOwner = 'surface' | 'companion' | 'elsewhere';

export interface CanvasSurfaceAttachOptions {
  /** Asked only on keydown, after the pipeline's own refusals; without it ownership comes from focus alone. */
  readonly resolveKeyOwner?: (event: KeyboardEvent) => CanvasKeyOwner;
}

export interface CanvasSurfaceCapability {
  /** `keyboardRoot` gets the canvas its session keys while focused; defaults to the overlay. */
  attach(
    screenCanvas: HTMLCanvasElement,
    overlayCanvas: HTMLCanvasElement,
    keyboardRoot?: HTMLElement,
    options?: CanvasSurfaceAttachOptions
  ): void;
  detach(): void;
  resize(cssWidth: number, cssHeight: number, dpr: number): void;
}

export interface CanvasDocumentCapability {
  captureSnapshot(): CanvasDocumentSnapshot | null;
  getDocument(): CanvasDocumentContractV3 | null;
  /** Advances on every reducer document identity change (previews, selection, syncs) to reject stale prepared edits. */
  getEditRevision(): number;
  captureInsertionAnchor(stack: LayerStackKind, aboveId: string | null): CanvasNodeInsertionAnchor;
  captureRestoreAnchor(nodeId: string): CanvasNodeInsertionAnchor | null;
  /** History clears. */
  replaceDocument(document: CanvasDocumentContractV3): boolean;
  /** A document swap; `keepCurrentAs` keeps the replaced canvas as a point. False when nothing changed. */
  restoreSnapshot(snapshotId: string, keepCurrentAs?: { id: string; name: string; createdAt: string }): boolean;
  /** The same instance until the document changes. */
  model(): CanvasDocumentModel | null;
}

export interface CanvasDocumentSnapshot {
  readonly canvas: CanvasStateContractV3;
  readonly documentGeneration: number;
}

export type PsdExportResult =
  | AdjustmentRefusal
  | LayerUnreadable
  | 'busy'
  | 'exported'
  | 'nothing'
  | 'too-large'
  | 'over-budget'
  | SubsetOf<CanvasTransactionOutcome, 'not-ready' | 'stale' | 'aborted'>;

export interface CanvasPsdExportCapability {
  cancelPsdExport(): void;
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
}

export interface CanvasViewportCapability {
  getViewport(): Viewport;
  fitToView(): void;
  fitToViewOnFirstShow(): void;
  setBboxGrid(size: number): void;
}

export interface CanvasToolCapability {
  setTool(toolId: ToolId, options?: { temporary?: boolean }): void;
  stepBrushSize(direction: 1 | -1): void;
}

export interface CanvasHistoryEntries {
  /** Oldest first; the last is what `undo()` reverts. */
  past: readonly string[];
  /** Next-redo first. */
  future: readonly string[];
  /** Oldest steps dropped for the entry cap or memory budget since the last clear. */
  evicted: number;
  /** Steps the previous engine held at disposal that this one could not take over; present while nonzero. */
  dropped?: number;
}

/** `failed` leaves the step where it was; the engine reported why. */
export type CanvasHistoryReplayStatus = 'applied' | 'empty' | 'busy' | 'refused' | 'failed';

export interface CanvasHistoryCapability {
  undo(): Promise<CanvasHistoryReplayStatus>;
  redo(): Promise<CanvasHistoryReplayStatus>;
  clearHistory(): void;
  getEntries(): CanvasHistoryEntries;
  getHeldAssetRefs(): { images: readonly string[]; videos: readonly string[] };
  /** Stops at the first step that does not apply. */
  stepBy(offset: number): Promise<CanvasHistoryReplayStatus>;
}

/** `commit` and `cancel` fire once per pick; `cancel` carries the last previewed color. */
export type ColorSamplePhase = 'preview' | 'commit' | 'cancel';

/** Names the layer and why it cannot take the edit; `transform-session` names the session Fill and Erase wait for. */
export interface CanvasLayerEditRefusal {
  readonly status:
    | 'layer-hidden'
    | 'layer-display-hidden'
    | 'layer-locked'
    | 'layer-alpha-locked'
    | 'layer-loading'
    | 'layer-unpaintable'
    | 'erase-unsupported'
    | 'selection-erase-unsupported'
    | 'gradient-unsupported'
    | 'transform-session';
  readonly layerId: string;
  readonly layerName: string;
  /** A new paint layer above this image layer can take the stroke instead. */
  readonly canPaintNewLayer?: boolean;
}

export interface CanvasNoTargetRefusal {
  readonly status: 'cutout-no-target' | 'erase-no-target' | 'selection-erase-no-target';
}

/** Edits the user started with no caller to report a refusal to (a stroke, a shape). */
export type CanvasEditRefusal =
  | SubsetOf<CanvasTransactionOutcome, 'busy' | 'gesture-active' | 'not-ready' | 'over-budget'>
  | AdjustmentRefusal
  | CanvasLayerEditRefusal
  | CanvasNoTargetRefusal;

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
    }
  | LayerUnreadable
  | AdjustmentRefusal;

export interface CanvasSelectionCapability {
  /** Otherwise reports why through `tools.onEditRefused`; Cut asks before it writes the clipboard. */
  admitSelectionErase(): boolean;
  deselect(): void;
  eraseSelection(): void;
  fillSelection(): void;
  getSelectionBounds(): Rect | null;
  getSelectionMaskRect(): Rect | null;
  invertSelection(): void;
  /** Null if empty. The widget owns system clipboard writes. */
  exportSelectionBlob(): Promise<Blob | null>;
  pasteImage(pixels: ImageData, center?: Vec2): NewRasterLayerResult;
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

/** `dispatch-rejected` includes a no-op forward; one history entry is recorded, only for `committed`. */
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
  /** A mismatch refuses as `stale`. */
  expectedRevision?: number;
  verify?: (document: CanvasDocumentContractV3) => boolean;
}

export type { CanvasLayerPreviewMutation } from './mutationContracts';

export interface CanvasStructuralEngine {
  readonly layers: CanvasLayerCapability;
}

/** Previews apply at most once per frame; ending the session for any reason restores the baseline, unrecorded. */
export interface StructuralPreviewSession {
  apply(action: CanvasLayerPreviewMutation): boolean;
  /** Null until it previews, and once ended. */
  baseline(): CanvasLayerPreviewMutation | null;
  cancel(): void;
  commit(label: string, edit: PreparedDocumentEdit): StructuralCommitResult;
  isActive(): boolean;
}

export interface CanvasLayerCapability {
  /** Null while edits are refused. */
  beginStructuralPreview(): StructuralPreviewSession | null;
  /** Restores the baseline, so an edit prepared next sees the committed document. */
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
  /** With `waitForHeldPixels: false` it rejects instead of waiting on an open edit's dirty pixels. */
  flushPendingUploads(options?: { readonly waitForHeldPixels?: boolean }): Promise<void>;
}

export type CanvasEditCapability = CanvasEditGate;

/** Decoded by `imageResolver`. */
export interface FilterPreviewInput {
  imageName: string;
  rect: Rect;
  filterType?: string;
}

export type DuplicateLayersResult =
  | { readonly status: 'duplicated'; readonly duplicateIds: readonly string[]; readonly selectedLayerId: string }
  | {
      readonly status: SubsetOf<CanvasTransactionOutcome, 'busy' | 'not-ready' | 'stale'> | 'nothing' | 'over-budget';
    }
  | LayerUnreadable;
export type {
  BooleanRasterResult,
  ConvertLayerToControlResult,
  CopyLayerResult,
  CropAllLayersResult,
  CropLayerResult,
  ExtractMaskedAreaResult,
  MergeDownResult,
  MergeInpaintMasksResult,
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
  readonly groupSurfaceAllocations: number;
  /** Whole-surface redraws (new content, geometry, or unknown member damage). */
  readonly groupSurfaceRebuilds: number;
  readonly groupSurfaceRefreshes: number;
}

export interface CanvasEngineToolCapability extends CanvasToolCapability {
  /** The menu targets the selected layer, never the one under the pointer; this only reports gesture suppression. */
  canTargetLayerFromContextMenu(): boolean;
  handleEscapePriority(options: { gestureWasActive: boolean }): void;
  onStrokeCommitted(listener: (event: StrokeCommittedEvent) => void): () => void;
  onEditRefused(listener: (refusal: CanvasEditRefusal) => void): () => void;
  /** Null on Escape, tool change or disposal; restores the previous tool either way. */
  requestColorSample(): Promise<string | null>;
  /** A declined `preview` falls back to the brush color; disposal removes only this router. */
  setColorSampleRouter(router: (hex: string, phase: ColorSamplePhase) => boolean): () => void;
}

/** Refusals leave the selection and the active tool unchanged. */
export type BeginTransformResult =
  | 'started'
  | SubsetOf<CanvasCommandRefusal, 'missing' | 'locked'>
  | 'ineligible'
  | SubsetOf<CanvasTransactionOutcome, 'busy'>;

export interface CanvasEngineLayerCapability extends CanvasLayerCapability {
  applyTransform(): void;
  /** Opens a session even when transform is already active; another layer's session is cancelled first. */
  beginTransform(layerId: string): BeginTransformResult;
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
  /** Null when there was nothing to commit: an empty creation or an unchanged edit. */
  commitTextEdit(content: string, styleChanges?: Partial<TextToolOptions>): StructuralCommitResult | null;
  /** Bakes adjustments (exactly, or refused) and transform into paint pixels; undo restores the raster layer. */
  convertLayerToControl(label: string, expectedLiveLayer: CanvasLayerContract): Promise<ConvertLayerToControlResult>;
  /** An empty raster layer's copy is a plain clone of its source and transform. */
  copyLayerToControl(layerId: string): Promise<CopyLayerResult>;
  copyLayerToRaster(layerId: string): Promise<CopyLayerResult>;
  /** One undo step, hidden and disabled leaves included; locked leaves are skipped and counted. */
  cropAllLayersToBbox(): Promise<CropAllLayersResult>;
  cropLayerToBbox(layerId: string): Promise<CropLayerResult>;
  duplicateLayers(label: string, layerIds: readonly string[]): Promise<DuplicateLayersResult>;
  mergeInpaintMasks(layerIds: readonly string[]): Promise<MergeInpaintMasksResult>;
  mergeLayerDown(upperLayerId: string): Promise<MergeDownResult>;
  mergeSelectedRasterLayers(layerIds: readonly string[]): Promise<MergeVisibleResult>;
  mergeVisibleRasterLayers(): Promise<MergeVisibleResult>;
  /** `dispatch-rejected` also covers a selection with nothing eligible to move. */
  nudgeSelectedLayer(dx: number, dy: number): StructuralCommitResult;
  openTextCreate(docPoint: Vec2): void;
  openTextEdit(layerId: string): void;
  rasterizeLayer(layerId: string): Promise<RasterizeLayerResult>;
  setTextEditContentReader(reader: (() => string) | null): void;
  updateTextEditStyle(patch: TextStylePatch): void;
  updateTransformSession(transform: LayerTransform): void;
}

export interface CanvasEngineExportCapability extends CanvasExportCapability, CanvasPsdExportCapability {
  extractMaskedArea(maskLayerId: string): Promise<ExtractMaskedAreaResult>;
}

/** Default `map`. */
export type ControlLayerViewMode = 'original' | 'map' | 'overlay';

export type ControlLayerViewOutcome = 'shown' | 'missing' | 'locked' | 'stale' | 'over-budget' | 'failed';

export interface CanvasEnginePreviewCapability extends CanvasPreviewCapability {
  /** Draws the `fitToView` extent at most 1:1 and returns that document rect; null without a document or 2D context. */
  drawDocumentOverview(target: HTMLCanvasElement, maxSizePx: number): Rect | null;
  preloadStagedPreview(imageName: string): void;
  setGuardedFilterPreview(
    layerId: string,
    input: FilterPreviewInput,
    guard: LayerExportGuard
  ): Promise<'shown' | SubsetOf<CanvasCommandRefusal, 'missing'> | SubsetOf<CanvasTransactionOutcome, 'stale'>>;
  setStagedPreview(input: StagedPreviewInput | null): void;
  /** Overlay chrome only, absolute document rects; an equal list repaints nothing. */
  setStagedRegions(regions: readonly StagedRegion[]): void;
  /** Display-only, one layer at a time; never recorded in history, persisted or exported. */
  setControlLayerView(layerId: string, view: ControlLayerViewMode): Promise<ControlLayerViewOutcome>;
}

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
export {
  CANVAS_COLOR_LABELS,
  CANVAS_MAX_NODE_COUNT,
  CANVAS_MAX_NODE_DEPTH,
  CANVAS_REGIONAL_LAYOUT_KINDS,
} from './contracts';
export type BooleanRasterOperation = 'intersect' | 'cutout' | 'cutaway' | 'exclude';
export interface StagedPreviewPlacement extends Rect {
  opacity: number;
}
/** Lands at document coordinates pinned at submission; `hidden` keeps the screen as if the candidate showed. */
export type StagedPreviewInput =
  | { imageName: string; placement: StagedPreviewPlacement; hidden?: boolean }
  | { dataUrl: string; placement: StagedPreviewPlacement };
export interface StagedRegion {
  readonly rect: Rect;
  readonly selected: boolean;
}
export type {
  BboxToolOptions,
  BrushOptions,
  CheckerColors,
  ControlLayerView,
  EraserOptions,
  GradientStop,
  GradientToolOptions,
  LassoToolOptions,
  LayerAdjustmentStatus,
  LayerQuickSwitchState,
  LayerThumbnailStatus,
  MarqueeToolOptions,
  ShapeToolKind,
  ShapeToolOptions,
  ShapeToolTarget,
  TextEditSession,
  TextToolOptions,
  TransformOptions,
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
export { CanvasImageMissingError, type ImageResolver } from './render/rasterizers';
export type { Rect, SelectionOp, ToolId, Vec2 } from './types';
export { adjustmentsKey, buildCurveLut, isIdentityAdjustments } from './render/adjustments';
export { isAdjustmentRefusal, type AdjustmentRefusal, type AdjustmentRefusalReason } from './render/exactAdjustments';
export { DEFAULT_CHECKER_COLORS } from './render/compositor';
export {
  getBaseRasterContentBounds,
  getCompositeLayerBounds,
  isBaseRasterContentLayer,
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
  inpaintMaskMergeEligibility,
  type InpaintMaskMergeEligibility,
} from './document/mergeVisible';
export { documentToExportLocalSamPoint } from './samCoordinates';
export { bboxEquals, bboxMinSize, roundBbox } from './tools/bboxHitTest';
export { isEmpty, union } from './math/rect';
export { ZOOM_PRESETS } from './math/snapping';
export {
  isLeafPixelEditEligible,
  selectionPixelEditRefusal,
  type SelectionPixelEditRefusal,
} from './editing/controlPixelEdit';
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
export { isExportableRasterLayer, isLayerUnreadable, type LayerUnreadable } from './layerExportGuards';
export {
  getLayerThumbnailFallbackRenderState,
  nextLayerThumbnailFallbackStage,
  resolveLayerThumbnailImageRef,
  type LayerThumbnailFallbackStage,
} from './render/thumbnail';
export { HISTORY_MAX_ENTRIES } from './history/history';
