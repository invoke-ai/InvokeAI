import type { LayerExportGuard, ReplaceSelectionFromImageResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasImageRef } from '@workbench/canvas-engine/contracts';
import type { DecodeImageResult } from '@workbench/canvas-engine/controllers/rasterController';
import type { CanvasEditConcurrency } from '@workbench/canvas-engine/editConcurrency';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { Rect } from '@workbench/canvas-engine/types';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';

export interface SelectionImageControllerOptions {
  readonly concurrency: CanvasEditConcurrency;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly decodeImage: (
    image: CanvasImageRef,
    options: {
      signal?: AbortSignal;
      isCurrent?: () => boolean;
      scaleToImage?: boolean;
      validateDecoded?: (width: number, height: number) => void;
    }
  ) => Promise<DecodeImageResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly selection: SelectionState;
}

/** Decodes guarded application results into the transient selection mask. */
export class SelectionImageController {
  constructor(private readonly options: SelectionImageControllerOptions) {}

  async replace(
    guard: LayerExportGuard,
    image: CanvasImageRef,
    rect: Rect,
    signal?: AbortSignal,
    owner?: symbol
  ): Promise<ReplaceSelectionFromImageResult> {
    const permit = this.options.concurrency.capturePermit(owner);
    if (!permit) {
      return { status: 'busy' };
    }
    if (signal?.aborted) {
      return { status: 'aborted' };
    }
    try {
      const decoded = await this.options.decodeImage(image, {
        isCurrent: () => this.options.concurrency.isPermitCurrent(permit),
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
      const layer = getDocumentLayer(document, guard.layerId);
      if (!layer) {
        return { status: 'missing' };
      }
      if (layer.isLocked) {
        return { status: 'locked' };
      }
      if (layer.type !== 'raster' && layer.type !== 'control') {
        return { status: 'unsupported' };
      }
      if (!this.options.concurrency.isPermitCurrent(permit) || this.options.concurrency.isGestureActive()) {
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
