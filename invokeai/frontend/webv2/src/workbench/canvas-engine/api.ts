import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type {
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasLayerContract,
  CanvasStagingCandidateContract,
  CanvasStateContractV2,
} from '@workbench/types';

import type { CanvasEditGate } from './editGate';
import type { EngineStores } from './engineStores';
import type { RasterCompositeExportRequest, RasterCompositeExportResult } from './exportRasterComposite';
import type { RasterSurface } from './render/raster';
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

export type CanvasCoreStores = EngineStores;

export interface CanvasCoreStoreCapability {
  readonly stores: CanvasCoreStores;
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

export interface CanvasDetachedLayerSurface {
  readonly rect: Rect;
  readonly surface: RasterSurface;
}

/** Caller-owned frozen pixels paired with the exact canvas contract that planned them. */
export interface CanvasRasterSnapshot extends CanvasDocumentSnapshot {
  /** Requested layers confirmed to have no current live or persisted pixels. */
  readonly emptyLayerIds: ReadonlySet<string>;
  readonly layerSurfaces: ReadonlyMap<string, CanvasDetachedLayerSurface>;
  /** Idempotently releases all detached-pixel memory owned by this snapshot. */
  release(): void;
}

export type CaptureRasterSnapshotResult =
  | { status: 'ok'; snapshot: CanvasRasterSnapshot }
  | { status: 'stale' | 'aborted' | 'not-ready' | 'over-budget' };

export type PsdExportResult = 'exported' | 'nothing' | 'too-large' | 'not-ready' | 'over-budget' | 'stale' | 'aborted';

export interface CanvasPsdExportCapability {
  exportRasterLayersToPsd(fileName: string): Promise<PsdExportResult>;
}

export interface CanvasViewportCapability {
  getViewport(): Viewport;
  fitToView(): void;
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
  captureRasterSnapshot(
    documentSnapshot: CanvasDocumentSnapshot,
    layerIds: readonly string[],
    options?: { signal?: AbortSignal; includeDisabled?: boolean }
  ): Promise<CaptureRasterSnapshotResult>;
  captureLayerExportGuard(layerId: string): LayerExportGuard | null;
  exportBakedLayerBlob(layerId: string, options?: ExportBakedLayerPixelsOptions): Promise<ExportBakedLayerBlobResult>;
  exportRasterComposite(request: RasterCompositeExportRequest): Promise<RasterCompositeExportResult>;
  getCompositeExecutorDeps(): CanvasCompositeExecutorDeps;
  hasExportableLayerContent(layerId: string): boolean;
  isLayerExportGuardCurrent(guard: LayerExportGuard): boolean;
}

export type {
  RasterCompositeExportRequest,
  RasterCompositeExportResult,
  RasterCompositeExportSnapshot,
} from './exportRasterComposite';

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

export interface CanvasCompositeExecutorDeps {
  backend: {
    createSurface(width: number, height: number): RasterSurface;
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect }>;
  reserve?(
    bytes: number
  ):
    | { status: 'ok'; lease: { release(): void } }
    | { status: 'over-budget'; requestedBytes: number; availableBytes: number };
  uploadImage(blob: Blob): Promise<{ imageName: string; width: number; height: number }>;
}

export interface CanvasSelectionCapability {
  deselect(): void;
  eraseSelection(): void;
  fillSelection(): void;
  getSelectionBounds(): Rect | null;
  getSelectionMaskRect(): Rect | null;
  invertSelection(): void;
  replaceSelectionFromImage(
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal
  ): Promise<ReplaceSelectionFromImageResult>;
  selectAll(): void;
}

export type ReplaceSelectionFromImageResult =
  | { status: 'selected' }
  | { status: 'aborted' | 'missing' | 'locked' | 'stale' | 'unsupported' | 'busy' }
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
  | { status: 'missing' | 'locked' | 'stale' | 'unsupported' | 'busy' | 'aborted' }
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
