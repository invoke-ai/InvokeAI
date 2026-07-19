import type {
  CommitGeneratedImageResult,
  CanvasDocumentSnapshot,
  CanvasStateContractV2,
  LayerExportGuard,
  ReplaceSelectionFromImageResult,
} from '@workbench/canvas-engine/api';
import type { CanvasDocumentContractV2, CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { CommitRasterFilterResult } from '@workbench/canvas-engine/controllers/filterResultController';
import type {
  CommitMaskImageResult,
  MaskImageResultTarget,
} from '@workbench/canvas-engine/controllers/maskResultController';
import type { CanvasEditGate } from '@workbench/canvas-engine/editGate';
import type { ExportBakedLayerBlobResult, ExportLayerPixelsResult } from '@workbench/canvas-engine/engine';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';
import type {
  CompositeEntryResult,
  CompositeResult,
  MaskCompositeResult,
} from '@workbench/canvas-operations/compositeForGeneration';
import type { CompositeEntry, CompositePlan } from '@workbench/canvas-operations/generationContracts';
import type {
  FilterCommitTarget,
  FilterOperationSessionState,
  LayerFilterSettings,
  SamInput,
  SamModel,
  SamSessionError,
  SamSessionSnapshot,
} from '@workbench/canvas-operations/operationTypes';
import type { BackendGraphContract } from '@workbench/graphContracts';
import type { WorkbenchState } from '@workbench/projectContracts';

export type StartSelectObjectSessionResult =
  | 'started'
  | 'missing'
  | 'disabled'
  | 'locked'
  | 'unsupported'
  | 'not-ready';
export type StartFilterOperationResult = 'started' | 'missing' | 'disabled' | 'locked' | 'unsupported' | 'not-ready';
export type CanvasOperationActionResult = 'completed' | 'blocked' | 'stale';
export type FilterCommitOperationResult = 'committed' | 'blocked' | 'stale';
export type CanvasOperationMutationResult = 'updated' | 'blocked' | 'stale';
export type SelectObjectSaveTarget = 'selection' | 'raster' | 'control' | MaskImageResultTarget;
export type SaveSelectObjectSessionResult =
  | CommitGeneratedImageResult
  | CommitMaskImageResult
  | ReplaceSelectionFromImageResult;
export interface SelectObjectSessionUpdate {
  input?: SamInput;
  pointLabel?: 'include' | 'exclude';
  model?: SamModel;
  invert?: boolean;
  applyPolygonRefinement?: boolean;
  autoProcess?: boolean;
  isolatedPreview?: boolean;
}

/** Intent-oriented application operations attached to one Canvas engine. */
export interface CanvasOperationCapability {
  captureCompositeTransaction(
    snapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options: { signal: AbortSignal }
  ): Promise<CanvasCompositeTransactionResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
  getOperationState(): CanvasOperationState;
  subscribeOperation(listener: () => void): () => void;
  getFilterSessionState(): FilterOperationSessionState | null;
  subscribeFilterSession(listener: () => void): () => void;
  getSamSessionState(): SamSessionSnapshot | null;
  subscribeSamSession(listener: () => void): () => void;
  startSelectObject(layerId: string): StartSelectObjectSessionResult;
  startFilterOperation(layerId: string, recommendedFilterType?: string | null): StartFilterOperationResult;
  updateFilterOperation(draft: LayerFilterSettings): CanvasOperationMutationResult;
  setFilterOperationAutoProcess(value: boolean): CanvasOperationMutationResult;
  processFilterOperation(): Promise<CanvasOperationActionResult>;
  resetFilterOperation(settings: Record<string, unknown>): CanvasOperationMutationResult;
  commitFilterOperation(
    target: FilterCommitTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<FilterCommitOperationResult>;
  cancelFilterOperation(): void;
  updateSelectObjectSession(changes: SelectObjectSessionUpdate): CanvasOperationMutationResult;
  processSelectObjectSession(): Promise<SelectObjectSessionProcessResult>;
  applySelectObjectSession(makeImageDurable: (imageName: string) => Promise<void>): Promise<CommitGeneratedImageResult>;
  saveSelectObjectSession(
    target: SelectObjectSaveTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<SaveSelectObjectSessionResult>;
  resetSelectObjectSession(): CanvasOperationMutationResult;
  cancelSelectObjectSession(): void;
}

/** Opaque generation transaction; pixels, surfaces, caches, and reservations remain Canvas-owned. */
export interface CanvasCompositeTransaction {
  readonly canvas: CanvasStateContractV2;
  executePlan(plan: CompositePlan): Promise<CompositeResult>;
  executeControl(entry: CompositeEntry): Promise<CompositeEntryResult>;
  executeMask(entry: CompositeEntry): Promise<MaskCompositeResult>;
  executeRegionalMask(entry: CompositeEntry): Promise<CompositeEntryResult>;
  commit(): void;
  release(): void;
}

export type CanvasCompositeTransactionResult =
  | { status: 'ok'; transaction: CanvasCompositeTransaction }
  | { status: 'stale' | 'aborted' | 'not-ready' | 'over-budget' };

/** Private composition shape; mutable stores and controller never cross the Canvas interface. */
export interface CanvasOperationImplementation extends CanvasOperationCapability {
  readonly controller: CanvasOperationController;
  readonly stores: CanvasApplicationOperationStores;
}

export type CanvasOperationIdentity =
  | { kind: 'select-object'; projectId: string; layerId: string }
  | { kind: 'filter'; projectId: string; layerId: string };
export type CanvasOperationState =
  | { status: 'idle' }
  | { status: 'active'; identity: CanvasOperationIdentity; phase: 'ready' | 'running' | 'error'; error: string | null };
export type CanvasOperationRunResult = 'published' | 'stale' | 'error';
export interface CanvasOperationSession {
  run<T>(
    work: (signal: AbortSignal) => Promise<T>,
    commitPreview: (result: T) => undefined
  ): Promise<CanvasOperationRunResult>;
  reset(): void;
  interruptProcessing(): void;
  cancel(): void;
}
export interface CanvasOperationController {
  getSnapshot(): CanvasOperationState;
  subscribe(listener: () => void): () => void;
  start(options: {
    identity: CanvasOperationIdentity;
    guard: LayerExportGuard;
    cleanupPreview(): void;
  }): CanvasOperationSession | null;
  reset(): void;
  cancel(): void;
  invalidateSource(projectId: string, layerId: string): void;
  invalidateLayer(projectId: string, layerId: string): void;
  invalidateProject(projectId: string): void;
  invalidateDocument(projectId: string): void;
  dispose(): void;
}

export interface CanvasApplicationScalarStore<T> {
  get(): T;
  set(value: T): void;
  subscribe(listener: () => void): () => void;
}

export interface CanvasApplicationOperationStores {
  readonly filterSession: CanvasApplicationScalarStore<FilterOperationSessionState | null>;
  readonly samSession: CanvasApplicationScalarStore<SamSessionSnapshot | null>;
}

export interface CanvasFilterCoordinatorDeps {
  readonly stores: CanvasApplicationOperationStores;
  readonly controller: CanvasOperationController;
  isInteractionLocked(): boolean;
  getDocument(): CanvasDocumentContractV2 | null;
  captureGuard(layerId: string): LayerExportGuard | null;
  selectLayer(layerId: string): void;
  clearOtherOperation(): void;
  clearPreview(layerId: string): void;
  setViewTool(): void;
  encodeSurface(surface: RasterSurface): Promise<Blob>;
  runFilterGraph(options: {
    graph: BackendGraphContract;
    outputNodeId?: string;
    signal?: AbortSignal;
  }): Promise<{ height: number; imageName: string; width: number }>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ imageName: string }>;
  createSession?(options: CreateFilterSessionOptions): FilterOperationSession | null;
  getSessionDeps(
    layerId: string
  ): Omit<
    CreateFilterSessionOptions['deps'],
    'canCommit' | 'clearPreview' | 'controller' | 'isDraftValid' | 'makeDurable' | 'runFilter'
  >;
}

export interface CanvasFilterOperationCoordinator {
  start(layerId: string, recommendedFilterType?: string | null): StartFilterOperationResult;
  updateDraft(draft: LayerFilterSettings): CanvasOperationMutationResult;
  setAutoProcess(value: boolean): CanvasOperationMutationResult;
  process(): Promise<CanvasOperationActionResult>;
  reset(settings: Record<string, unknown>): CanvasOperationMutationResult;
  commit(
    target: FilterCommitTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<FilterCommitOperationResult>;
  cancel(): void;
  interruptAndBlock(): void;
  isActive(): boolean;
  dispose(): void;
}

export type SelectObjectStartContext =
  | { status: Exclude<StartSelectObjectSessionResult, 'started'> }
  | {
      status: 'ready';
      guard: LayerExportGuard;
      layerId: string;
      layerName: string;
      layerType: 'raster' | 'control';
      sourceRect: Rect;
    };

export interface CanvasSelectObjectCoordinatorDeps {
  readonly stores: CanvasApplicationOperationStores;
  readonly controller: CanvasOperationController;
  readonly projectId: string;
  isInteractionLocked(): boolean;
  prepareStart(layerId: string): SelectObjectStartContext;
  selectLayer(layerId: string): void;
  clearOtherOperation(): void;
  setSamTool(): void;
  setViewTool(): void;
  replaceTemporaryRestoreTool(): void;
  isSamToolActive(): boolean;
  captureGuard(layerId: string): LayerExportGuard | null;
  isGuardCurrent(guard: LayerExportGuard): boolean;
  exportSource(layerId: string): Promise<ExportBakedLayerBlobResult>;
  uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ height: number; imageName: string; width: number }>;
  runGraph(options: {
    graph: BackendGraphContract;
    outputNodeId?: string;
    signal?: AbortSignal;
  }): Promise<CanvasUtilityGraphResult>;
  decodePreview(result: SelectObjectReadyResult, signal: AbortSignal): Promise<RasterSurface>;
  publishPreview(preview: SelectObjectSessionPreview<RasterSurface>): void;
  clearPreview(): void;
  invalidateOverlay(): void;
  commitGenerated(
    preview: SelectObjectSessionPreview<RasterSurface>,
    options: { mode: 'replace' | 'copy-raster' | 'copy-control'; signal: AbortSignal }
  ): Promise<CommitGeneratedImageResult>;
  commitMask(
    preview: SelectObjectSessionPreview<RasterSurface>,
    target: MaskImageResultTarget,
    signal: AbortSignal
  ): Promise<CommitMaskImageResult>;
  replaceSelection(
    preview: SelectObjectSessionPreview<RasterSurface>,
    signal: AbortSignal
  ): Promise<ReplaceSelectionFromImageResult>;
  createSession?(options: CreateSelectObjectSessionOptions<RasterSurface>): SelectObjectSession<RasterSurface>;
}

export interface CanvasSelectObjectOperationCoordinator {
  start(layerId: string): StartSelectObjectSessionResult;
  update(changes: SelectObjectSessionUpdate): CanvasOperationMutationResult;
  process(): Promise<SelectObjectSessionProcessResult>;
  apply(makeImageDurable: (imageName: string) => Promise<void>): Promise<CommitGeneratedImageResult>;
  save(
    target: SelectObjectSaveTarget,
    makeImageDurable: (imageName: string) => Promise<void>
  ): Promise<SaveSelectObjectSessionResult>;
  reset(): CanvasOperationMutationResult;
  cancel(): void;
  interruptAndBlock(): void;
  isActive(): boolean;
  dispose(): void;
}

export interface FilterOperationSession {
  getSnapshot(): FilterOperationSessionState;
  subscribe(listener: () => void): () => void;
  updateDraft(draft: LayerFilterSettings): void;
  setAutoProcess(value: boolean): void;
  process(): Promise<CanvasOperationRunResult>;
  interruptProcessing(): void;
  reset(settings: Record<string, unknown>): void;
  commit(target: FilterCommitTarget): Promise<'committed' | 'blocked' | 'stale'>;
  blockCommit(): void;
  cancel(): void;
  dispose(): void;
}

export type SelectObjectSessionStatus =
  | 'ready'
  | 'scheduled'
  | 'preparing-source'
  | 'uploading'
  | 'processing-sam'
  | 'rendering-preview'
  | 'error';
export interface SelectObjectSessionPreview<T> {
  data: T;
  guard: LayerExportGuard;
  image: CanvasImageRef;
  inputHash: string;
  previewId: number;
  isolated: boolean;
  rect: Rect;
  sourceImageName: string;
}
export interface SelectObjectSessionState<T> {
  input: SamInput;
  model: SamModel;
  invert: boolean;
  applyPolygonRefinement: boolean;
  autoProcess: boolean;
  isolatedPreview: boolean;
  status: SelectObjectSessionStatus;
  error: SamSessionError | null;
  preview: SelectObjectSessionPreview<T> | null;
  sourceGuard: LayerExportGuard | null;
}
export type SelectObjectSessionProcessResult = CanvasOperationRunResult | 'blocked' | 'deduped' | 'invalid';
export interface SelectObjectSession<T> {
  getSnapshot(): SelectObjectSessionState<T>;
  subscribe(listener: () => void): () => void;
  update(
    changes: Partial<
      Pick<
        SelectObjectSessionState<T>,
        'applyPolygonRefinement' | 'autoProcess' | 'input' | 'invert' | 'isolatedPreview' | 'model'
      >
    >
  ): void;
  process(): Promise<SelectObjectSessionProcessResult>;
  interruptProcessing(): void;
  reportError(error: SamSessionError | string): void;
  reset(): void;
  cancel(): void;
  dispose(): void;
}

export interface CanvasUtilityGraphResult {
  imageName: string;
  width: number;
  height: number;
}
export interface LayerFilterResult {
  imageName: string;
  width: number;
  height: number;
  origin: { x: number; y: number };
}

export interface CreateFilterSessionOptions {
  deps: {
    controller: CanvasOperationController;
    exportPixels(): Promise<ExportLayerPixelsResult>;
    runFilter(options: {
      filterType: string;
      input: { surface: RasterSurface; rect: Rect };
      settings: Record<string, unknown>;
      signal: AbortSignal;
    }): Promise<LayerFilterResult>;
    publishPreview(
      imageName: string,
      rect: Rect,
      guard: LayerExportGuard,
      filterType: string
    ): Promise<'shown' | 'missing' | 'stale'>;
    clearPreview(): void;
    isGuardCurrent(guard: LayerExportGuard): boolean;
    isDraftValid(draft: LayerFilterSettings): boolean;
    makeDurable(imageName: string): Promise<void>;
    canCommit(): boolean;
    commit(options: {
      draft: LayerFilterSettings;
      guard: LayerExportGuard;
      image: { imageName: string; width: number; height: number };
      origin: { x: number; y: number };
      rect: Rect;
      signal: AbortSignal;
      target: FilterCommitTarget;
    }): Promise<CommitRasterFilterResult>;
  };
  guard: LayerExportGuard;
  initialFilter: LayerFilterSettings | null;
  initialDraft?: LayerFilterSettings;
  layerName?: string;
  layerType: 'raster' | 'control';
}

export interface SelectObjectReadyResult {
  status: 'ready';
  image: CanvasImageRef;
  rect: Rect;
  guard: LayerExportGuard;
}
export interface CreateSelectObjectSessionOptions<T> {
  deps: {
    captureGuard(): LayerExportGuard | null;
    controller: CanvasOperationController;
    exportSource(): Promise<ExportBakedLayerBlobResult>;
    uploadIntermediate(blob: Blob, signal?: AbortSignal): Promise<{ height: number; imageName: string; width: number }>;
    runGraph(options: {
      graph: BackendGraphContract;
      outputNodeId?: string;
      signal?: AbortSignal;
    }): Promise<CanvasUtilityGraphResult>;
    decodePreview(result: SelectObjectReadyResult, signal: AbortSignal): Promise<T>;
    publishPreview(preview: SelectObjectSessionPreview<T>): undefined;
    cleanupPreview(): void;
    isGuardCurrent(guard: LayerExportGuard): boolean;
  };
  operation: CanvasOperationSession;
}

export interface CanvasApplicationPort {
  createFilterCoordinator(deps: CanvasFilterCoordinatorDeps): CanvasFilterOperationCoordinator;
  createSelectObjectCoordinator(deps: CanvasSelectObjectCoordinatorDeps): CanvasSelectObjectOperationCoordinator;
  createOperationStores(): CanvasApplicationOperationStores;
  createOperationController(deps: {
    edits: CanvasEditGate;
    isGuardCurrent(guard: LayerExportGuard): boolean;
  }): CanvasOperationController;
  runGraph(options: {
    graph: BackendGraphContract;
    outputNodeId?: string;
    signal?: AbortSignal;
  }): Promise<CanvasUtilityGraphResult>;
  uploadImage(
    blob: Blob,
    options?: { isIntermediate?: boolean; signal?: AbortSignal }
  ): Promise<{ height: number; imageName: string; width: number }>;
  getSelectedModelBase(state: WorkbenchState, projectId: string): string | null;
}
