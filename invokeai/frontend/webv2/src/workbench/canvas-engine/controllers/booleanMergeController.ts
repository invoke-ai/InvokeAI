import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { lookupLayerBelow, mergeDownEligibility } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { isEmpty, roundOut, union } from '@workbench/canvas-engine/math/rect';

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

export type BooleanRasterOperation = 'intersect' | 'cutout' | 'cutaway' | 'exclude';
export type BooleanRasterResult = 'merged' | 'missing' | 'unsupported' | 'empty' | 'failed' | LayerEditRefusal;

type ExportResult =
  | { status: 'ok'; surface: RasterSurface; rect: Rect; guard: LayerExportGuard; release(): void }
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'over-budget' };

export interface BooleanMergeControllerOptions {
  readonly ctx: LayerStepContext &
    Pick<
      CanvasMutationContext,
      'begin' | 'capturePermit' | 'captureInsertionAnchor' | 'createLayerId' | 'getDocument' | 'isPermitCurrent'
    >;
  readonly backend: RasterBackend;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
}

const modes: Record<BooleanRasterOperation, GlobalCompositeOperation> = {
  cutaway: 'source-out',
  cutout: 'destination-in',
  exclude: 'xor',
  intersect: 'source-in',
};

/** Owns guarded two-layer boolean compositing and atomic stack history. */
export class BooleanMergeController {
  private disposed = false;

  constructor(private readonly deps: BooleanMergeControllerOptions) {}

  async merge(upperLayerId: string, operation: BooleanRasterOperation): Promise<BooleanRasterResult> {
    const { ctx } = this.deps;
    const permit = ctx.capturePermit();
    if (this.disposed || !permit) {
      return 'busy';
    }
    const document = ctx.getDocument();
    if (!document) {
      return 'missing';
    }
    const eligibility = mergeDownEligibility(document, upperLayerId);
    if (eligibility.status !== 'eligible') {
      return eligibility.status === 'missing' ||
        (eligibility.status === 'invalid-target' && eligibility.reason === 'no-layer-below')
        ? 'missing'
        : 'unsupported';
    }
    const upper = getDocumentLayer(document, eligibility.upperId)!;
    const below = getDocumentLayer(document, eligibility.lowerId)!;
    if (!this.deps.isCacheReady(upper, document) || !this.deps.isCacheReady(below, document)) {
      return 'not-ready';
    }
    const owned: Extract<ExportResult, { status: 'ok' }>[] = [];
    const acquire = async (layerId: string): Promise<ExportResult> => {
      const result = await this.deps.exportBaked(layerId);
      if (result.status === 'ok') {
        owned.push(result);
      }
      return result;
    };
    try {
      const settled = await Promise.allSettled([acquire(upper.id), acquire(below.id)]);
      const rejected = settled.find((result) => result.status === 'rejected');
      if (rejected?.status === 'rejected') {
        throw rejected.reason instanceof Error ? rejected.reason : new Error(String(rejected.reason));
      }
      const [upperPixels, belowPixels] = settled.map(
        (result) => (result as PromiseFulfilledResult<ExportResult>).value
      );
      if (!ctx.isPermitCurrent(permit)) {
        return 'busy';
      }
      if (upperPixels.status !== 'ok' || belowPixels.status !== 'ok') {
        if (upperPixels.status === 'over-budget' || belowPixels.status === 'over-budget') {
          return 'over-budget';
        }
        if (upperPixels.status === 'not-ready' || belowPixels.status === 'not-ready') {
          return 'not-ready';
        }
        if (
          upperPixels.status === 'disabled' ||
          upperPixels.status === 'unsupported' ||
          belowPixels.status === 'disabled' ||
          belowPixels.status === 'unsupported'
        ) {
          return 'unsupported';
        }
        return 'empty';
      }
      if (
        upperPixels.guard.layer !== upper ||
        belowPixels.guard.layer !== below ||
        !this.deps.isGuardCurrent(upperPixels.guard) ||
        !this.deps.isGuardCurrent(belowPixels.guard)
      ) {
        return 'not-ready';
      }
      const liveDocument = ctx.getDocument();
      if (
        !liveDocument ||
        getDocumentLayer(liveDocument, upperLayerId) !== upper ||
        lookupLayerBelow(liveDocument, upperLayerId) !== below
      ) {
        return 'not-ready';
      }
      const resultRect = roundOut(union(upperPixels.rect, belowPixels.rect));
      if (isEmpty(resultRect)) {
        return 'empty';
      }
      const txn = ctx.begin({ historyBytes: addedLayerHistoryBytes(resultRect) });
      if (!('publish' in txn)) {
        return layerEditRefusal(txn.status);
      }
      try {
        if (!txn.reserveRaster(rgbaBytes(resultRect))) {
          return 'over-budget';
        }
        const pixels = this.deps.backend.createSurface(resultRect.width, resultRect.height);
        pixels.ctx.setTransform(1, 0, 0, 1, 0, 0);
        pixels.ctx.clearRect(0, 0, resultRect.width, resultRect.height);
        pixels.ctx.globalAlpha = below.opacity;
        pixels.ctx.globalCompositeOperation = 'source-over';
        pixels.ctx.drawImage(
          belowPixels.surface.canvas,
          belowPixels.rect.x - resultRect.x,
          belowPixels.rect.y - resultRect.y
        );
        pixels.ctx.globalAlpha = upper.opacity;
        pixels.ctx.globalCompositeOperation = modes[operation];
        pixels.ctx.drawImage(
          upperPixels.surface.canvas,
          upperPixels.rect.x - resultRect.x,
          upperPixels.rect.y - resultRect.y
        );
        const layer = paintLayerAt(ctx.createLayerId(), `${upper.name} ${operation}`, resultRect);
        const status = layerOperationStatus(
          publishAddedRasterLayer(ctx, txn, {
            anchor: ctx.captureInsertionAnchor('raster', upper.id),
            enabled: {
              after: [upper, below].map(({ id }) => ({ id, isEnabled: false })),
              before: [upper, below].map(({ id, isEnabled }) => ({ id, isEnabled })),
            },
            label: `Boolean ${operation}`,
            layer,
            pixels,
            rect: resultRect,
            selectedLayerId: liveDocument.selectedLayerId,
          })
        );
        return status === 'committed' ? 'merged' : status;
      } finally {
        txn.end();
      }
    } finally {
      for (const result of owned) {
        result.release();
      }
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
