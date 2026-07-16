import type {
  CanvasCompositeExecutorDeps,
  CommitGeneratedImageOptions,
  CommitGeneratedImageResult,
  ExportBakedLayerBlobResult,
  ExportLayerPixelsOptions,
  LayerExportGuard,
  ReplaceSelectionFromImageResult,
} from '@workbench/canvas-engine/api';
import type {
  CommitRasterFilterOptions,
  CommitRasterFilterResult,
} from '@workbench/canvas-engine/controllers/filterResultController';
import type {
  CommitMaskImageResult,
  CommitMaskImageResultOptions,
} from '@workbench/canvas-engine/controllers/maskResultController';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { SamInteractionState, SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasDocumentContractV2, CanvasImageRef } from '@workbench/types';

export type SelectObjectStartContext =
  | { status: 'missing' | 'disabled' | 'locked' | 'unsupported' | 'not-ready' }
  | {
      status: 'ready';
      guard: LayerExportGuard;
      layerId: string;
      layerName: string;
      layerType: 'raster' | 'control';
      sourceRect: Rect;
    };

/** Narrow core primitives used only by application-level canvas coordinators. */
export interface CanvasApplicationHost {
  captureGuard(layerId: string): LayerExportGuard | null;
  clearFilterPreview(layerId: string): void;
  clearSamPreview(): void;
  commitFilter(options: CommitRasterFilterOptions): Promise<CommitRasterFilterResult>;
  commitGenerated(options: CommitGeneratedImageOptions): Promise<CommitGeneratedImageResult>;
  commitMask(options: CommitMaskImageResultOptions): Promise<CommitMaskImageResult>;
  decodeSelectObjectPreview(result: { image: CanvasImageRef; rect: Rect }, signal: AbortSignal): Promise<RasterSurface>;
  encodeSurface(surface: RasterSurface): Promise<Blob>;
  exportBakedLayerBlob(layerId: string): Promise<ExportBakedLayerBlobResult>;
  exportLayerPixels(
    layerId: string,
    options?: ExportLayerPixelsOptions
  ): Promise<
    | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
    | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' }
  >;
  getCompositeExecutorDeps(): CanvasCompositeExecutorDeps;
  getDocument(): CanvasDocumentContractV2 | null;
  isGuardCurrent(guard: LayerExportGuard): boolean;
  isInteractionLocked(): boolean;
  isSamToolActive(): boolean;
  prepareSelectObjectStart(layerId: string): SelectObjectStartContext;
  publishFilterPreview(
    layerId: string,
    imageName: string,
    rect: Rect,
    guard: LayerExportGuard,
    filterType?: string
  ): Promise<'shown' | 'missing' | 'stale'>;
  publishSamPreview(preview: {
    data: RasterSurface;
    guard: LayerExportGuard;
    isolated: boolean;
    rect: Rect;
  }): undefined;
  replaceSelection(
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal
  ): Promise<ReplaceSelectionFromImageResult>;
  replaceTemporaryRestoreTool(): void;
  selectLayer(layerId: string): void;
  setSamInputHandler(handler: ((input: SamVisualInput) => void) | null): void;
  setEscapeHandler(handler: ((gestureWasActive: boolean) => boolean) | null): void;
  setSamInteraction(state: SamInteractionState | null): void;
  setSamTool(): void;
  setViewTool(): void;
  subscribeToolChanges(listener: (change: { from: string; to: string; temporary: boolean }) => void): () => void;
}
