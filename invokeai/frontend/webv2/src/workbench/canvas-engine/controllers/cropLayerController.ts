import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { intersect, isEmpty, roundOut } from '@workbench/canvas-engine/math/rect';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import {
  layerEditRefusal,
  layerOperationStatus,
  rgbaBytes,
  withReplayReservation,
  type LayerEditRefusal,
  replaceLayerStep,
  type LayerPixels,
} from './editSteps';

export type CropLayerResult =
  | { status: 'cropped' }
  | { status: 'missing' | 'locked' | 'unsupported' | 'empty' | LayerEditRefusal }
  | { status: 'failed'; message: string };

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };

export interface CropLayerControllerOptions {
  readonly ctx: Pick<
    CanvasMutationContext,
    | 'applyStep'
    | 'begin'
    | 'capturePermit'
    | 'getDocument'
    | 'getReducerDocument'
    | 'installPrepared'
    | 'isPermitCurrent'
    | 'preparePixels'
    | 'reserveRaster'
  >;
  readonly backend: RasterBackend;
  readonly isSupportedSource: (source: CanvasLayerSourceContract) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
  readonly captureCache: (
    layer: CanvasLayerContract,
    document: CanvasDocumentContractV3,
    admit?: (rect: Rect) => boolean
  ) => LayerPixels | null | 'not-ready' | 'over-budget';
  readonly discardPersisted: (layerId: string) => void;
}

const failedCrop: CropLayerResult = { message: 'The crop could not be applied.', status: 'failed' };

/** Owns guarded crop-to-bbox conversion and replayable pixel snapshots. */
export class CropLayerController {
  private disposed = false;
  constructor(private readonly deps: CropLayerControllerOptions) {}

  async crop(layerId: string): Promise<CropLayerResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return { status: 'busy' };
    }
    const document = ctx.getDocument();
    const layer = getDocumentLayer(document, layerId);
    if (!document || !layer) {
      return { status: 'missing' };
    }
    if (layer.isLocked) {
      return { status: 'locked' };
    }
    const source = renderableSourceOf(layer);
    if (!source || !this.deps.isSupportedSource(source)) {
      return { status: 'unsupported' };
    }
    try {
      const exported = await this.deps.exportBaked(layerId);
      if (exported.status !== 'ok') {
        return { status: exported.status === 'disabled' ? 'not-ready' : exported.status };
      }
      try {
        if (!ctx.isPermitCurrent(permit)) {
          return { status: 'busy' };
        }
        const liveDocument = ctx.getDocument();
        const liveLayer = getDocumentLayer(liveDocument, layerId);
        if (!liveDocument || !liveLayer) {
          return { status: 'missing' };
        }
        if (liveLayer.isLocked) {
          return { status: 'locked' };
        }
        if (!this.deps.isGuardCurrent(exported.guard)) {
          return { status: 'not-ready' };
        }
        const liveSource = renderableSourceOf(liveLayer);
        if (!liveSource || !this.deps.isSupportedSource(liveSource)) {
          return { status: 'unsupported' };
        }
        const overlap = intersect(exported.rect, roundOut(liveDocument.bbox));
        if (!overlap || isEmpty(overlap)) {
          return { status: 'empty' };
        }
        const cropRect = roundOut(overlap);
        const txn = ctx.begin({ historyBytes: rgbaBytes(cropRect) + HISTORY_ENTRY_OVERHEAD_BYTES });
        if (!('publish' in txn)) {
          return { status: layerEditRefusal(txn.status) };
        }
        try {
          // The cropped pixels and their prepared cache replacement.
          if (!txn.reserveRaster(rgbaBytes(cropRect) * 2)) {
            return { status: 'over-budget' };
          }
          const beforePixels = this.deps.captureCache(liveLayer, liveDocument, (rect) =>
            txn.growHistory(rgbaBytes(rect))
          );
          if (beforePixels === 'over-budget') {
            return { status: 'over-budget' };
          }
          if (!beforePixels || beforePixels === 'not-ready') {
            return { status: 'not-ready' };
          }
          const before = structuredClone(liveLayer);
          const cropped = this.deps.backend.createSurface(cropRect.width, cropRect.height);
          cropped.ctx.setTransform(1, 0, 0, 1, 0, 0);
          cropped.ctx.clearRect(0, 0, cropRect.width, cropRect.height);
          cropped.ctx.drawImage(exported.surface.canvas, exported.rect.x - cropRect.x, exported.rect.y - cropRect.y);
          const identity = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
          const paint = { bitmap: null, offset: { x: cropRect.x, y: cropRect.y }, type: 'paint' } as const;
          let after: CanvasLayerContract;
          if (before.type === 'raster') {
            const { adjustments: _adjustments, ...rest } = before;
            after = { ...rest, source: paint, transform: identity };
          } else if (before.type === 'control') {
            const { filter: _filter, ...rest } = before;
            after = { ...rest, source: paint, transform: identity };
          } else {
            after = { ...before, mask: { ...before.mask, bitmap: null, offset: paint.offset }, transform: identity };
          }
          const afterPixels: LayerPixels = { pixels: cropped, rect: cropRect };
          /** Replaces the layer's contract and cache together; a failed replacement restores the current contract. */
          const replaceStep = (contract: CanvasLayerContract, snapshot: LayerPixels): EditStep =>
            replaceLayerStep(ctx, contract, ctx.preparePixels(layerId, snapshot.rect, snapshot.pixels), {
              beforeInstall: () => {
                try {
                  this.deps.discardPersisted(layerId);
                } catch {
                  // The installed pixels are marked dirty and persist on their own.
                }
              },
              restore: getDocumentLayer(ctx.getReducerDocument(), layerId) ?? undefined,
            });
          const result = txn.publish('Crop layer to bbox', replaceStep(after, afterPixels), {
            bytes: rgbaBytes(beforePixels.rect) + rgbaBytes(cropRect) + HISTORY_ENTRY_OVERHEAD_BYTES,
            heldAssetRefs: collectHistoryMediaRefs(before, after),
            redo: () =>
              withReplayReservation(ctx, rgbaBytes(cropRect), () => ctx.applyStep(replaceStep(after, afterPixels))),
            undo: () =>
              withReplayReservation(ctx, rgbaBytes(beforePixels.rect), () =>
                ctx.applyStep(replaceStep(before, beforePixels))
              ),
          });
          const status = layerOperationStatus(result);
          return status === 'committed' ? { status: 'cropped' } : status === 'failed' ? failedCrop : { status };
        } finally {
          txn.end();
        }
      } finally {
        exported.release();
      }
    } catch (error) {
      return { message: error instanceof Error ? error.message : String(error), status: 'failed' };
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
