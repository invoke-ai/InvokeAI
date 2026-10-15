import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { PixelEditTransaction } from '@workbench/canvas-engine/tools/tool';
import type { Rect } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
import { getSourceBounds } from '@workbench/canvas-engine/document/sources';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { intersect, isEmpty, roundOut } from '@workbench/canvas-engine/math/rect';
import { eraseMaskedRegion, fillMaskedRegion } from '@workbench/canvas-engine/selection/selectionOps';

import type { CanvasMutationContext, EditTransaction } from './mutationContext';

type PixelTarget =
  | { kind: 'raster'; layerId: string; transparencyLocked: boolean }
  | { kind: 'control'; transaction: PixelEditTransaction; transparencyLocked: false };

export interface SelectionPixelControllerOptions {
  readonly selection: SelectionState;
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly ctx: Pick<CanvasMutationContext, 'begin'>;
  readonly reportRefusal: (refusal: CanvasEditRefusal) => void;
  readonly applyImagePatch: ImagePatchApply;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly beginPixelEdit: (layerId: string) => PixelEditTransaction | null;
  readonly canEdit: () => boolean;
  readonly isGestureActive: () => boolean;
  readonly getFillColor: () => string;
  readonly deleteDerived: (layerId: string) => void;
  readonly invalidateLayer: (layerId: string) => void;
  readonly isRasterCacheReady: (layer: CanvasLayerContract, document: CanvasDocumentContractV3) => boolean;
  readonly notifyPainted: (layerId: string) => void;
  readonly requestRasterization: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
}

const imageDataEqual = (left: ImageData, right: ImageData): boolean => {
  if (left.width !== right.width || left.height !== right.height || left.data.length !== right.data.length) {
    return false;
  }
  for (let index = 0; index < left.data.length; index += 1) {
    if (left.data[index] !== right.data[index]) {
      return false;
    }
  }
  return true;
};

/** Owns selection-driven fill/erase pixel transactions. */
export class SelectionPixelController {
  private disposed = false;

  constructor(private readonly deps: SelectionPixelControllerOptions) {}

  private target(): PixelTarget | null {
    const document = this.deps.getDocument();
    if (!document?.selectedLayerId) {
      return null;
    }
    const leaf = lookupDocumentLeaf(document, document.selectedLayerId);
    const layer = leaf?.layer;
    if (leaf && layer?.type === 'raster' && layer.source.type === 'paint' && isLeafEditable(leaf)) {
      return { kind: 'raster', layerId: layer.id, transparencyLocked: layer.isTransparencyLocked === true };
    }
    if (layer?.type === 'control') {
      const transaction = this.deps.beginPixelEdit(layer.id);
      return transaction ? { kind: 'control', transaction, transparencyLocked: false } : null;
    }
    return null;
  }

  run(kind: 'fill' | 'erase'): void {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return;
    }
    const document = this.deps.getDocument();
    const placedMask = this.deps.selection.mask();
    const bounds = this.deps.selection.bounds();
    if (!document || !placedMask || !bounds) {
      return;
    }
    const selectionRect = roundOut(bounds);
    const selectedLayer = getDocumentLayer(document, document.selectedLayerId);
    if (
      selectedLayer?.type === 'raster' &&
      selectedLayer.source.type === 'paint' &&
      !this.deps.isRasterCacheReady(selectedLayer, document)
    ) {
      this.deps.requestRasterization(selectedLayer.id);
      return;
    }
    if (
      kind === 'erase' &&
      selectedLayer?.type === 'control' &&
      !intersect(selectionRect, roundOut(getSourceBounds(selectedLayer, document)))
    ) {
      return;
    }
    const target = this.target();
    if (!target) {
      return;
    }
    if (kind === 'erase' && target.transparencyLocked) {
      return;
    }
    if (target.kind === 'control') {
      this.edit(kind, target.transaction.layerId, target, selectionRect, placedMask);
      return;
    }
    const txn = this.deps.ctx.begin({ historyBytes: 0 });
    if (!('publish' in txn)) {
      this.deps.reportRefusal(txn.status);
      return;
    }
    try {
      this.edit(kind, target.layerId, { ...target, txn }, selectionRect, placedMask);
    } finally {
      txn.end();
    }
  }

  /**
   * Fills or erases the selection on one target. Only the touched region is captured: it is admitted before any
   * pixel changes, and a refused or failed publication restores it and the cache's original extent.
   */
  private edit(
    kind: 'fill' | 'erase',
    layerId: string,
    target:
      | Exclude<PixelTarget, { kind: 'raster' }>
      | (Extract<PixelTarget, { kind: 'raster' }> & { txn: EditTransaction }),
    selectionRect: Rect,
    placedMask: NonNullable<ReturnType<SelectionState['mask']>>
  ): void {
    const { deps } = this;
    const live = target.kind === 'control' ? target.transaction : null;
    const existing = deps.layers.get(layerId);
    const grows = kind === 'fill' && !target.transparencyLocked;
    const rect = grows
      ? selectionRect
      : existing && !isEmpty(existing.rect)
        ? intersect(selectionRect, existing.rect)
        : null;
    if (!rect || isEmpty(rect)) {
      live?.cancel();
      return;
    }
    const footprint = rect.width * rect.height * 8;
    if (!(live ? live.grow(footprint) : target.kind === 'raster' && target.txn.growHistory(footprint))) {
      if (!live) {
        deps.reportRefusal('over-budget');
      }
      live?.cancel();
      return;
    }
    const originalRect = existing ? { ...existing.rect } : null;
    let before: ImageData | null = null;
    const rollback = (): void => {
      const entry = deps.layers.get(layerId);
      if (entry && before) {
        entry.surface.ctx.putImageData(before, rect.x - entry.rect.x, rect.y - entry.rect.y);
      }
      if (!originalRect) {
        deps.layers.delete(layerId);
      } else {
        deps.layers.shrinkToRect(layerId, originalRect);
      }
      deps.deleteDerived(layerId);
      deps.invalidateLayer(layerId);
    };
    let published = false;
    try {
      const entry = grows ? deps.layers.growToRect(layerId, rect) : existing!;
      const surface = entry.surface;
      const origin = { x: entry.rect.x, y: entry.rect.y };
      before = surface.ctx.getImageData(rect.x - origin.x, rect.y - origin.y, rect.width, rect.height);
      if (kind === 'fill') {
        fillMaskedRegion({
          backend: deps.backend,
          color: deps.getFillColor(),
          composite: target.transparencyLocked ? 'source-atop' : 'source-over',
          mask: placedMask.surface,
          maskOrigin: placedMask.rect,
          rect,
          target: surface,
          targetOrigin: origin,
        });
      } else {
        eraseMaskedRegion({
          backend: deps.backend,
          mask: placedMask.surface,
          maskOrigin: placedMask.rect,
          rect,
          target: surface,
          targetOrigin: origin,
        });
      }
      const after = surface.ctx.getImageData(rect.x - origin.x, rect.y - origin.y, rect.width, rect.height);
      const label = kind === 'fill' ? 'Fill selection' : 'Erase selection';
      if (live) {
        published = live.commitPatch(label, { after, before, rect });
        return;
      }
      if (target.kind !== 'raster' || imageDataEqual(before, after)) {
        return;
      }
      const result = target.txn.publish(
        label,
        {
          notify: () => {
            deps.notifyPainted(layerId);
            deps.markDirty(layerId);
          },
        },
        createImagePatchEntry({ after, apply: deps.applyImagePatch, before, label, layerId, rect }),
        { origin: 'system' }
      );
      if (result.status === 'over-budget') {
        deps.reportRefusal('over-budget');
      }
      published = result.status === 'committed';
    } finally {
      if (!published) {
        try {
          rollback();
        } finally {
          live?.cancel();
        }
      }
    }
  }

  dispose(): void {
    this.disposed = true;
  }
}
