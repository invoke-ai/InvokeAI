import type { CanvasDocumentContractV2, CanvasImageRef, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { Rect } from '@workbench/canvas-engine/types';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { getSourceContentRect, isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { invert as invertMatrix } from '@workbench/canvas-engine/math/mat2d';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { bakeMatrix } from '@workbench/canvas-engine/transform/transformMath';

export interface MaskLayerControllerOptions {
  readonly layers: LayerCacheStore;
  readonly history: History;
  readonly applyImagePatch: ImagePatchApply;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly isCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV2) => boolean;
  readonly endBurst: () => void;
  readonly discardPersisted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
  readonly dispatch: (action: CanvasProjectMutation) => void;
  readonly deleteDerived: (layerId: string) => void;
  readonly notifyPainted: (layerId: string) => void;
  readonly restoreCache: (layerId: string, rect: Rect, pixels: ImageData) => void;
}

/** Owns destructive mask pixel operations and their history/persistence lifecycle. */
export class MaskLayerController {
  private disposed = false;

  constructor(private readonly deps: MaskLayerControllerOptions) {}

  clear(layerId: string): boolean {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return false;
    }
    const document = this.deps.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer || !isMaskLayer(layer) || layer.isLocked) {
      return false;
    }
    const originalBitmap = layer.mask.bitmap;
    const originalOffset = layer.mask.offset ?? { x: 0, y: 0 };
    const entry = this.deps.isCacheReady(layer, document) ? this.deps.layers.get(layerId) : undefined;
    const rect = entry && !isEmpty(entry.rect) ? { ...entry.rect } : null;
    const before = rect ? entry?.surface.ctx.getImageData(0, 0, rect.width, rect.height) : null;
    if (!originalBitmap && (!before || !rect)) {
      return false;
    }
    this.deps.endBurst();
    const dispatchMask = (bitmap: CanvasImageRef | null, offset: { x: number; y: number }): void => {
      this.deps.dispatch({
        config: { layerType: layer.type, mask: { bitmap, offset } },
        id: layerId,
        type: 'updateCanvasLayerConfig',
      });
    };
    const applyClear = (): void => {
      this.deps.discardPersisted(layerId);
      dispatchMask(null, { x: 0, y: 0 });
      this.deps.layers.delete(layerId);
      this.deps.deleteDerived(layerId);
      const empty = this.deps.layers.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 });
      empty.stale = false;
      this.deps.notifyPainted(layerId);
    };
    const applyRestore = (): void => {
      this.deps.discardPersisted(layerId);
      dispatchMask(originalBitmap, originalOffset);
      if (before && rect) {
        this.deps.restoreCache(layerId, rect, before);
      }
    };
    applyClear();
    this.deps.history.push({
      bytes: (before?.data.byteLength ?? 0) + 256,
      label: 'Clear mask',
      redo: applyClear,
      undo: applyRestore,
    });
    return true;
  }

  invert(layerId: string): boolean {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return false;
    }
    const document = this.deps.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer || !isMaskLayer(layer) || layer.isLocked || !layer.isEnabled) {
      return false;
    }
    if (!this.deps.isCacheReady(layer, document)) {
      return false;
    }
    const content = getSourceContentRect(layer, document);
    const liveRect = this.deps.layers.get(layerId)?.rect;
    const contentUnion =
      liveRect && !isEmpty(liveRect) ? (isEmpty(content) ? liveRect : union(content, liveRect)) : content;
    const inverse = invertMatrix(bakeMatrix(layer.transform));
    const bbox = inverse ? roundOut(transformBounds(inverse, document.bbox)) : document.bbox;
    const domain = roundOut(isEmpty(contentUnion) ? bbox : union(contentUnion, bbox));
    if (isEmpty(domain)) {
      return false;
    }
    this.deps.endBurst();
    const entry = this.deps.layers.growToRect(layerId, domain);
    const readX = domain.x - entry.rect.x;
    const readY = domain.y - entry.rect.y;
    const before = entry.surface.ctx.getImageData(readX, readY, domain.width, domain.height);
    const work = entry.surface.ctx.getImageData(readX, readY, domain.width, domain.height);
    for (let index = 3; index < work.data.length; index += 4) {
      work.data[index] = 255 - (work.data[index] ?? 0);
    }
    entry.surface.ctx.putImageData(work, readX, readY);
    this.deps.notifyPainted(layerId);
    this.deps.markDirty(layerId);
    if (!this.deps.history.isApplying()) {
      this.deps.history.push(
        createImagePatchEntry({
          after: work,
          apply: this.deps.applyImagePatch,
          before,
          label: 'Invert mask',
          layerId,
          rect: domain,
        })
      );
    }
    return true;
  }

  dispose(): void {
    this.disposed = true;
  }
}
