import type { CanvasDocumentContractV2, CanvasImageRef, CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

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
  getDocument(): CanvasDocumentContractV2 | null;
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

export type LayerThumbnailRequestResult = 'ready' | 'stale' | 'error' | 'missing' | 'unsupported';

export interface CanvasPreviewCapability {
  drawLayerThumbnail(layerId: string, target: HTMLCanvasElement, maxSize: number): boolean;
  requestLayerThumbnail(layerId: string): Promise<LayerThumbnailRequestResult>;
}

export interface CanvasExportCapability {
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
}

export type ExportBakedLayerPixelsOptions = Omit<ExportLayerPixelsOptions, 'applyAdjustments'> & {
  applyAdjustments?: boolean;
};

export type ExportBakedLayerBlobResult =
  | { status: 'ok'; blob: Blob; rect: Rect; guard: LayerExportGuard }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' };

export interface CanvasCompositeExecutorDeps {
  backend: {
    createSurface(width: number, height: number): RasterSurface;
    encodeSurface(surface: RasterSurface, type?: string): Promise<Blob>;
  };
  getLayerSurface(layerId: string): Promise<{ surface: RasterSurface; rect: Rect }>;
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
  applyStructuralPreview(action: WorkbenchAction): boolean;
  commitGeneratedImageResult(options: CommitGeneratedImageOptions): Promise<CommitGeneratedImageResult>;
  commitStructural(label: string, forward: WorkbenchAction, inverse: WorkbenchAction): void;
  invertMask(layerId: string): boolean;
}

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
