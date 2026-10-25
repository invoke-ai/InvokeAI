import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasCommandRefusal } from '@workbench/canvas-engine/document/commandRefusal';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { SubsetOf } from '@workbench/canvas-engine/editConcurrency';
import type { AdjustmentRefusal } from '@workbench/canvas-engine/render/exactAdjustments';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createControlLayerFrom } from '@workbench/canvas-engine/document/layerFactories';

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
  | { status: 'missing' | 'disabled' | 'unsupported' | 'empty' | 'not-ready' | 'unreadable' | 'over-budget' }
  | AdjustmentRefusal;

/** The stack a baked copy lands in: a raster paint layer, or a control layer with a default adapter. */
export type CopyLayerTarget = 'raster' | 'control';

export type CopyLayerResult =
  | { status: 'copied'; layerId: string }
  | {
      status:
        | LayerEditRefusal
        | SubsetOf<CanvasCommandRefusal, 'missing' | 'unsupported'>
        | 'empty'
        | 'unreadable'
        | 'failed';
    }
  | AdjustmentRefusal;

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
  /** Adds `layer` as a plain contract copy of the source, with the source's live cache if it has one. */
  readonly commitLayerCopy: (
    label: string,
    sourceLayerId: string,
    layer: CanvasLayerContract,
    anchor: CanvasNodeInsertionAnchor
  ) => boolean;
  readonly exportBaked: (layerId: string) => Promise<ExportResult>;
  /** The main model's base and its default control model, for a control copy's adapter. */
  readonly getControlModel: () => { readonly base: string | null; readonly model: string | null };
  readonly isGuardCurrent: (guard: LayerExportGuard) => boolean;
}

const COPY_LABELS: Record<CopyLayerTarget, string> = {
  control: 'Copy layer to control',
  raster: 'Copy layer to raster',
};

/**
 * Owns guarded baked copies into new paint layers. The copy holds the layer's pixels as displayed: its own adjustment
 * stack evaluated exactly (spatial entries included, or refused) and its transform, at an identity transform. An
 * empty raster image or paint layer has no pixels to bake, so its control copy is a plain clone of its source and
 * transform, without the stack.
 */
export class CopyLayerController {
  private disposed = false;
  constructor(private readonly deps: CopyLayerControllerOptions) {}

  async copy(layerId: string, target: CopyLayerTarget): Promise<CopyLayerResult> {
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
    if (baked.status === 'adjustment-refused') {
      return baked;
    }
    if (
      baked.status === 'empty' &&
      target === 'control' &&
      sourceLayer.type === 'raster' &&
      (sourceLayer.source.type === 'image' || sourceLayer.source.type === 'paint')
    ) {
      if (!ctx.isPermitCurrent(permit)) {
        return { status: 'busy' };
      }
      // A preview-tolerant recheck, as for baked copies below.
      if (getDocumentLayer(ctx.getReducerDocument(), layerId) !== sourceLayer) {
        return { status: 'not-ready' };
      }
      const layer = createControlLayerFrom(sourceLayer, {
        ...this.deps.getControlModel(),
        content: { source: sourceLayer.source },
        id: ctx.createLayerId(),
        name: `${sourceLayer.name} copy`,
      });
      const anchor = ctx.captureInsertionAnchor('control', layerId);
      return this.deps.commitLayerCopy(COPY_LABELS.control, layerId, layer, anchor)
        ? { layerId: layer.id, status: 'copied' }
        : { status: 'not-ready' };
    }
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
        const id = ctx.createLayerId();
        const name = `${sourceLayer.name} copy`;
        const layer =
          target === 'control'
            ? createControlLayerFrom(sourceLayer, {
                ...this.deps.getControlModel(),
                content: { rect: baked.rect },
                id,
                name,
              })
            : paintLayerAt(id, name, baked.rect);
        const status = layerOperationStatus(
          publishAddedRasterLayer(ctx, txn, {
            anchor: ctx.captureInsertionAnchor(target, layerId),
            label: COPY_LABELS[target],
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
