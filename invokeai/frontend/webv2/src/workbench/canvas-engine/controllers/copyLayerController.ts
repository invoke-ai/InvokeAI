import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';

import type { CanvasMutationContext } from './mutationContext';

import {
  addedLayerHistoryBytes,
  layerEditRefusal,
  layerOperationStatus,
  paintLayerAt,
  publishAddedRasterLayer,
  rgbaBytes,
  type LayerStepContext,
  type LayerEditRefusal,
} from './editSteps';

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };

export type CopyLayerToRasterResult =
  | { status: 'copied'; layerId: string }
  | { status: LayerEditRefusal | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'> | 'empty' | 'failed' };

export interface CopyLayerControllerOptions {
  readonly ctx: LayerStepContext &
    Pick<
      CanvasMutationContext,
      | 'begin'
      | 'capturePermit'
      | 'captureInsertionAnchor'
      | 'createLayerId'
      | 'getDocument'
      | 'getReducerDocument'
      | 'isPermitCurrent'
    >;
  readonly backend: RasterBackend;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
}

/** Owns guarded baked copies into new raster paint layers. */
export class CopyLayerController {
  private disposed = false;
  constructor(private readonly deps: CopyLayerControllerOptions) {}

  async copyToRaster(layerId: string): Promise<CopyLayerToRasterResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return { status: 'busy' };
    }
    const sourceLayer = getDocumentLayer(ctx.getDocument(), layerId);
    if (!sourceLayer) {
      return { status: 'missing' };
    }
    const baked = await this.deps.exportBaked(layerId);
    if (baked.status !== 'ok') {
      return { status: baked.status === 'disabled' ? 'not-ready' : baked.status };
    }
    try {
      if (!ctx.isPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      // A preview-tolerant recheck; see `CanvasMutationContext.getReducerDocument`.
      const liveDocument = ctx.getReducerDocument();
      if (
        !liveDocument ||
        getDocumentLayer(liveDocument, sourceLayer.id) !== sourceLayer ||
        baked.guard.layer !== sourceLayer ||
        !this.deps.isGuardCurrent(baked.guard)
      ) {
        return { status: 'not-ready' };
      }
      const txn = ctx.begin({ historyBytes: addedLayerHistoryBytes(baked.rect) });
      if (!('publish' in txn)) {
        return { status: layerEditRefusal(txn.status) };
      }
      try {
        // The entry owns a copy: the leased cache keeps changing once the lease ends.
        if (!txn.reserveRaster(rgbaBytes(baked.rect))) {
          return { status: 'over-budget' };
        }
        const pixels = this.deps.backend.createSurface(baked.rect.width, baked.rect.height);
        pixels.ctx.drawImage(baked.surface.canvas, 0, 0);
        const layer = paintLayerAt(ctx.createLayerId(), `${sourceLayer.name} copy`, baked.rect);
        const status = layerOperationStatus(
          publishAddedRasterLayer(ctx, txn, {
            anchor: ctx.captureInsertionAnchor('raster', layerId),
            label: 'Copy layer to raster',
            layer,
            pixels,
            rect: baked.rect,
            selectedLayerId: liveDocument.selectedLayerId,
          })
        );
        return status === 'committed' ? { layerId: layer.id, status: 'copied' } : { status };
      } finally {
        txn.end();
      }
    } finally {
      baked.release();
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
