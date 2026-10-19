import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { RasterizeDeps } from '@workbench/canvas-engine/render/rasterizers';
import type { Rect } from '@workbench/canvas-engine/types';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect, isEmptyPolygonShape } from '@workbench/canvas-engine/document/sources';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { rasterizeSource } from '@workbench/canvas-engine/render/rasterizers';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import {
  layerEditRefusal,
  layerOperationStatus,
  rgbaBytes,
  withReplayReservation,
  type LayerEditRefusal,
} from './editSteps';

export type RasterizeLayerResult =
  | 'rasterized'
  | SubsetOf<CanvasCommandRefusal, 'missing' | 'locked' | 'unsupported'>
  | LayerEditRefusal
  | 'failed';

export interface RasterizeLayerControllerOptions {
  readonly ctx: Pick<
    CanvasMutationContext,
    'applyStep' | 'begin' | 'getDocument' | 'installPrepared' | 'preparePixels' | 'reserveRaster'
  >;
  readonly backend: RasterBackend;
  readonly rasterizeDeps: (document: CanvasDocumentContractV3) => RasterizeDeps;
}

/** Owns conversion of parametric raster layers into persisted paint pixels. */
export class RasterizeLayerController {
  private disposed = false;

  constructor(private readonly deps: RasterizeLayerControllerOptions) {}

  /**
   * Bakes the layer's source and transform into paint pixels as one undo step. Undo restores the parametric
   * contract, which rerasterizes from its source; redo reinstalls the baked pixels, never a rerender.
   */
  rasterize(layerId: string): RasterizeLayerResult {
    const { ctx } = this.deps;
    if (this.disposed) {
      return 'not-ready';
    }
    const document = ctx.getDocument();
    const layer = getDocumentLayer(document, layerId);
    if (!document || !layer) {
      return 'missing';
    }
    if (layer.isLocked) {
      return 'locked';
    }
    if (layer.type !== 'raster') {
      return 'unsupported';
    }
    const source = layer.source;
    if (
      (source.type !== 'shape' && source.type !== 'gradient' && source.type !== 'text') ||
      (source.type === 'shape' && isEmptyPolygonShape(source))
    ) {
      return 'unsupported';
    }
    const contentRect = getSourceContentRect(layer, document);
    const matrix = bakeMatrix(layer.transform);
    const bakedRect = roundOut(transformBounds(matrix, contentRect));
    const txn = ctx.begin({ historyBytes: rgbaBytes(bakedRect) + HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return layerEditRefusal(txn.status);
    }
    try {
      // The source render, the baked pixels the entry keeps, and their prepared cache replacement.
      if (!txn.reserveRaster(rgbaBytes(contentRect) + rgbaBytes(bakedRect) * 2)) {
        return 'over-budget';
      }
      const scratch = this.deps.backend.createSurface(contentRect.width, contentRect.height);
      void rasterizeSource(source, this.deps.rasterizeDeps(document), scratch);
      const baked = this.deps.backend.createSurface(bakedRect.width, bakedRect.height);
      baked.ctx.setTransform(1, 0, 0, 1, 0, 0);
      baked.ctx.clearRect(0, 0, bakedRect.width, bakedRect.height);
      baked.ctx.imageSmoothingEnabled = true;
      baked.ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - bakedRect.x, matrix.f - bakedRect.y);
      baked.ctx.drawImage(scratch.canvas, contentRect.x, contentRect.y);
      baked.ctx.setTransform(1, 0, 0, 1, 0, 0);
      const parametric = structuredClone(layer);
      const paint: CanvasLayerContract = {
        ...parametric,
        source: { bitmap: null, offset: { x: bakedRect.x, y: bakedRect.y }, type: 'paint' },
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      };
      const status = layerOperationStatus(
        txn.publish('Rasterize layer', this.convertStep(layerId, paint, parametric, bakedRect, baked), {
          bytes: rgbaBytes(bakedRect) + HISTORY_ENTRY_OVERHEAD_BYTES,
          heldAssetRefs: collectHistoryMediaRefs(parametric),
          redo: () =>
            withReplayReservation(ctx, rgbaBytes(bakedRect), () =>
              ctx.applyStep(this.convertStep(layerId, paint, parametric, bakedRect, baked))
            ),
          undo: () => ctx.applyStep(this.convertStep(layerId, parametric, paint)),
        })
      );
      return status === 'committed' ? 'rasterized' : status;
    } finally {
      txn.end();
    }
  }

  dispose(): void {
    this.disposed = true;
  }

  /** Converts the layer to `target`, installing `pixels` with it; a failed conversion converts back to `previous`. */
  private convertStep(
    layerId: string,
    target: CanvasLayerContract,
    previous: CanvasLayerContract,
    rect?: Rect,
    pixels?: RasterSurface
  ): EditStep {
    const { ctx } = this.deps;
    const prepared = rect && pixels ? ctx.preparePixels(layerId, rect, pixels) : null;
    // Conversion clones the contract, so it is matched by value.
    const holds = (contract: CanvasLayerContract) => (document: CanvasDocumentContractV3 | null) =>
      areJsonValuesStructurallyEqual(getDocumentLayer(document, layerId), contract);
    return {
      accepted: holds(target),
      install: prepared ? () => ctx.installPrepared(prepared) : undefined,
      mutation: { id: layerId, layer: target, targetType: 'raster', type: 'convertCanvasLayer' },
      rollback: {
        mutation: { id: layerId, layer: previous, targetType: 'raster', type: 'convertCanvasLayer' },
        restored: holds(previous),
      },
    };
  }
}
