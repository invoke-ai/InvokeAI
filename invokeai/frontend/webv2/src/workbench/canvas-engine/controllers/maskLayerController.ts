import type { MaskEditResult } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasImageRef, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { Rect } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
import { getSourceContentRect, isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { collectHistoryMediaRefs, HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { invert as invertMatrix } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

import type { CanvasMutationContext, EditStep } from './mutationContext';

import { guardedResultRefusal, layerEditRefusal, rgbaBytes, withReplayReservation } from './editSteps';

export interface MaskLayerControllerOptions {
  readonly ctx: Pick<CanvasMutationContext, 'applyStep' | 'begin' | 'getDocument' | 'reserveRaster'>;
  readonly layers: LayerCacheStore;
  readonly applyImagePatch: ImagePatchApply;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => boolean;
  readonly discardPersisted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
  readonly deleteDerived: (layerId: string) => void;
  readonly notifyPainted: (layerId: string) => void;
  readonly restoreCache: (layerId: string, rect: Rect, pixels: ImageData) => void;
}

type MaskLayer = Extract<CanvasLayerContract, { mask: unknown }>;

/** Owns destructive mask pixel operations as admitted, failure-atomic undo steps. */
export class MaskLayerController {
  private disposed = false;

  constructor(private readonly deps: MaskLayerControllerOptions) {}

  clear(layerId: string): MaskEditResult {
    const { ctx } = this.deps;
    if (this.disposed) {
      return { status: 'not-ready' };
    }
    const document = ctx.getDocument();
    const layer = getDocumentLayer(document, layerId);
    if (!document || !layer) {
      return { status: 'missing' };
    }
    if (!isMaskLayer(layer)) {
      return { status: 'unsupported' };
    }
    if (layer.isLocked) {
      return { status: 'locked' };
    }
    const originalBitmap = layer.mask.bitmap;
    const originalOffset = layer.mask.offset ?? { x: 0, y: 0 };
    const entry = this.deps.isCacheReady(layer, document) ? this.deps.layers.get(layerId) : undefined;
    const rect = entry && !isEmpty(entry.rect) ? { ...entry.rect } : null;
    if (!originalBitmap && !rect) {
      return { status: 'nothing' };
    }
    const txn = ctx.begin({ historyBytes: (rect ? rgbaBytes(rect) : 0) + HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return { status: layerEditRefusal(txn.status) };
    }
    try {
      const before = rect && entry ? entry.surface.ctx.getImageData(0, 0, rect.width, rect.height) : null;
      const cleared = this.maskStep(layer, null, { x: 0, y: 0 }, originalBitmap, originalOffset, () => {
        this.deps.layers.delete(layerId);
        this.deps.deleteDerived(layerId);
        this.deps.layers.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 }).stale = false;
        this.deps.notifyPainted(layerId);
      });
      const result = txn.publish('Clear mask', cleared, {
        bytes: (before?.data.byteLength ?? 0) + HISTORY_ENTRY_OVERHEAD_BYTES,
        heldAssetRefs: collectHistoryMediaRefs(originalBitmap),
        redo: () => ctx.applyStep(cleared),
        undo: () => {
          withReplayReservation(ctx, rect ? rgbaBytes(rect) : 0, () =>
            ctx.applyStep(
              this.maskStep(layer, originalBitmap, originalOffset, null, { x: 0, y: 0 }, () => {
                if (before && rect) {
                  this.deps.restoreCache(layerId, rect, before);
                }
              })
            )
          );
        },
      });
      return result.status === 'committed' ? { status: 'committed' } : { status: guardedResultRefusal(result) };
    } finally {
      txn.end();
    }
  }

  invert(layerId: string): MaskEditResult {
    const { ctx } = this.deps;
    if (this.disposed) {
      return { status: 'not-ready' };
    }
    const document = ctx.getDocument();
    const leaf = lookupDocumentLeaf(document, layerId);
    const layer = leaf?.layer;
    if (!document || !layer) {
      return { status: 'missing' };
    }
    if (!isMaskLayer(layer)) {
      return { status: 'unsupported' };
    }
    if (!isLeafEditable(leaf)) {
      return { status: 'locked' };
    }
    if (!this.deps.isCacheReady(layer, document)) {
      return { status: 'not-ready' };
    }
    const content = getSourceContentRect(layer, document);
    const liveRect = this.deps.layers.get(layerId)?.rect;
    const contentUnion =
      liveRect && !isEmpty(liveRect) ? (isEmpty(content) ? liveRect : union(content, liveRect)) : content;
    const inverse = invertMatrix(bakeMatrix(layer.transform));
    const bbox = inverse ? roundOut(transformBounds(inverse, document.bbox)) : document.bbox;
    const domain = roundOut(isEmpty(contentUnion) ? bbox : union(contentUnion, bbox));
    if (isEmpty(domain)) {
      return { status: 'nothing' };
    }
    const txn = ctx.begin({ historyBytes: rgbaBytes(domain) * 2 + HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return { status: layerEditRefusal(txn.status) };
    }
    try {
      if (!txn.reserveRaster(rgbaBytes(domain))) {
        return { status: 'over-budget' };
      }
      const originalRect = this.deps.layers.peek(layerId)?.rect;
      const restoreExtent = originalRect ? { ...originalRect } : null;
      // Growing changes no visible pixel; a refused publication only has to return the cache to its extent.
      const entry = this.deps.layers.growToRect(layerId, domain);
      const readX = domain.x - entry.rect.x;
      const readY = domain.y - entry.rect.y;
      const before = entry.surface.ctx.getImageData(readX, readY, domain.width, domain.height);
      const after = entry.surface.ctx.getImageData(readX, readY, domain.width, domain.height);
      for (let index = 3; index < after.data.length; index += 4) {
        after.data[index] = 255 - (after.data[index] ?? 0);
      }
      const result = txn.publish(
        'Invert mask',
        {
          install: () => entry.surface.ctx.putImageData(after, readX, readY),
          notify: () => {
            this.deps.notifyPainted(layerId);
            this.deps.markDirty(layerId);
          },
        },
        createImagePatchEntry({
          after,
          apply: this.deps.applyImagePatch,
          before,
          label: 'Invert mask',
          layerId,
          rect: domain,
        })
      );
      if (result.status === 'committed') {
        return { status: 'committed' };
      }
      if (restoreExtent) {
        this.deps.layers.shrinkToRect(layerId, restoreExtent);
      } else {
        this.deps.layers.delete(layerId);
      }
      return { status: guardedResultRefusal(result) };
    } finally {
      txn.end();
    }
  }

  dispose(): void {
    this.disposed = true;
  }

  /** Sets the mask's persisted bitmap; once accepted, drops pending uploads and installs the matching pixels. */
  private maskStep(
    layer: MaskLayer,
    bitmap: CanvasImageRef | null,
    offset: { x: number; y: number },
    restoreBitmap: CanvasImageRef | null,
    restoreOffset: { x: number; y: number },
    install: () => void
  ): EditStep {
    const configured = (expected: CanvasImageRef | null) => (document: CanvasDocumentContractV3 | null) => {
      const current = getDocumentLayer(document, layer.id);
      return !!current && isMaskLayer(current) && current.mask.bitmap === expected;
    };
    return {
      accepted: configured(bitmap),
      install: () => {
        this.deps.discardPersisted(layer.id);
        install();
      },
      mutation: {
        config: { layerType: layer.type, mask: { bitmap, offset } },
        id: layer.id,
        type: 'updateCanvasLayerConfig',
      },
      rollback: {
        mutation: {
          config: { layerType: layer.type, mask: { bitmap: restoreBitmap, offset: restoreOffset } },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        },
        restored: configured(restoreBitmap),
      },
    };
  }
}
