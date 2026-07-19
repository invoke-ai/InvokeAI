import type { LayerExportGuard, ReplaceSelectionFromImageResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV2, CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect } from '@workbench/canvas-engine/types';

export interface SelectionImageControllerOptions<Permit, Owner = symbol> {
  readonly capturePermit: (owner?: Owner) => Permit | null;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly decodeImage: (
    image: CanvasImageRef,
    options: {
      signal?: AbortSignal;
      isCurrent?: () => boolean;
      scaleToImage?: boolean;
      validateDecoded?: (width: number, height: number) => void;
    }
  ) => Promise<DecodeImageResult>;
  readonly isGestureActive: () => boolean;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly isPermitCurrent: (permit: Permit) => boolean;
  readonly selection: SelectionState;
}

/** Decodes guarded application results into the transient selection mask. */
export class SelectionImageController<Permit, Owner = symbol> {
  constructor(private readonly options: SelectionImageControllerOptions<Permit, Owner>) {}

  async replace(
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal,
    owner?: Owner
  ): Promise<ReplaceSelectionFromImageResult> {
    const permit = this.options.capturePermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    try {
      const decoded = await this.options.decodeImage(image, {
        isCurrent: () => this.options.isPermitCurrent(permit),
        signal,
      });
      if (decoded.status !== 'ok') {
        return { status: decoded.status === 'aborted' ? 'aborted' : 'busy' };
      }
      const pixels = decoded.surface;
      const document = this.options.getDocument();
      if (!document) {
        return { status: 'missing' };
      }
      const layer = document.layers.find((candidate) => candidate.id === guard.layerId);
      if (!layer) {
        return { status: 'missing' };
      }
      if (layer.isLocked) {
        return { status: 'locked' };
      }
      if (layer.type !== 'raster' && layer.type !== 'control') {
        return { status: 'unsupported' };
      }
      if (!this.options.isPermitCurrent(permit) || this.options.isGestureActive()) {
        return { status: 'busy' };
      }
      if (!this.options.isGuardCurrent(guard)) {
        return { status: 'stale' };
      }
      if (signal?.aborted) {
        return { status: 'aborted' };
      }
      this.options.selection.replaceMask({ rect: { ...rect }, surface: pixels });
      return { status: 'selected' };
    } catch (error) {
      if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        return { status: 'aborted' };
      }
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  }

  dispose(): void {}
}
