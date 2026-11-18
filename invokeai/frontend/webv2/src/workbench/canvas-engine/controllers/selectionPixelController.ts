import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { SelectionPixelEditRefusal } from '@workbench/canvas-engine/editing/controlPixelEdit';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { PixelEditTransaction } from '@workbench/canvas-engine/tools/tool';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer, getDocumentNode } from '@workbench/canvas-engine/document/documentIndex';
import { isLayerTransparencyLocked } from '@workbench/canvas-engine/document/layerEligibility';
import { isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { selectionPixelEditRefusal } from '@workbench/canvas-engine/editing/controlPixelEdit';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { invert } from '@workbench/canvas-engine/math/mat2d';
import { intersect, isEmpty, roundOut, transformBounds } from '@workbench/canvas-engine/math/rect';
import { eraseMaskedRegion, fillMaskedRegion } from '@workbench/canvas-engine/selection/selectionOps';
import { hittableLayerRect, layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';
import { MASK_PAINT_COLOR } from '@workbench/canvas-engine/tools/paintConstants';

import type { CanvasMutationContext, EditTransaction } from './mutationContext';

/**
 * A paint layer or mask is edited in its own local space, the selection mapped into it. A control layer or raster
 * image edits through a pixel-edit transaction, which bakes its transform first, so the document is its space.
 */
type PixelTarget =
  | { kind: 'direct'; color: string | null; toLocal: Mat2d; transparencyLocked: boolean; txn: EditTransaction }
  | { kind: 'transaction'; transaction: PixelEditTransaction; transparencyLocked: boolean };

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
  /** The layer an open transform session previews, whose placement the selection would not match until it lands. */
  readonly getTransformSessionLayerId: () => string | null;
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

  /**
   * Says why the selection cannot be edited, as the brush and eraser would. Removing the selection's pixels (Erase,
   * Delete, Cut) names what it could not act on; Fill is only offered for a layer it can fill.
   */
  private reportTargetRefusal(
    document: CanvasDocumentContractV3,
    kind: 'fill' | 'erase',
    refusal: SelectionPixelEditRefusal
  ): void {
    const node = getDocumentNode(document, document.selectedLayerId);
    if (refusal === 'no-target' || refusal === 'unsupported') {
      if (kind === 'erase') {
        this.deps.reportRefusal(
          node
            ? { layerId: node.id, layerName: node.name, status: 'selection-erase-unsupported' }
            : { status: 'selection-erase-no-target' }
        );
      }
      return;
    }
    if (node) {
      this.deps.reportRefusal({ layerId: node.id, layerName: node.name, status: refusal });
    }
  }

  /**
   * The selection and the layer a Fill or Erase would edit now, or null after reporting why not. Waiting pixels are
   * requested, so a later try can edit them.
   */
  private admit(kind: 'fill' | 'erase') {
    if (this.disposed || !this.deps.canEdit() || this.deps.isGestureActive()) {
      return null;
    }
    const document = this.deps.getDocument();
    const placedMask = this.deps.selection.mask();
    const bounds = this.deps.selection.bounds();
    if (!document || !placedMask || !bounds) {
      return null;
    }
    const node = getDocumentNode(document, document.selectedLayerId);
    const refusal = selectionPixelEditRefusal(node ? (lookupDocumentLeaf(document, node.id) ?? 'group') : null, kind);
    const layer = getDocumentLayer(document, document.selectedLayerId);
    if (refusal || !layer) {
      this.reportTargetRefusal(document, kind, refusal ?? 'no-target');
      return null;
    }
    const sessionLayerId = this.deps.getTransformSessionLayerId();
    if (sessionLayerId !== null) {
      // The session previews a placement the document does not hold yet; editing now would land elsewhere.
      const sessionLayer = getDocumentNode(document, sessionLayerId);
      this.deps.reportRefusal({
        layerId: sessionLayerId,
        layerName: sessionLayer?.name ?? '',
        status: 'transform-session',
      });
      return null;
    }
    if (!this.deps.isRasterCacheReady(layer, document)) {
      // Editing now would grow a transparent cache over pixels still on their way.
      this.deps.requestRasterization(layer.id);
      this.deps.reportRefusal({ layerId: layer.id, layerName: layer.name, status: 'layer-loading' });
      return null;
    }
    return { document, layer, placedMask, selectionRect: roundOut(bounds) };
  }

  /** Whether an Erase may run now; otherwise it reports why, as the Erase itself would. Cut asks before it copies. */
  admitErase(): boolean {
    return this.admit('erase') !== null;
  }

  run(kind: 'fill' | 'erase'): void {
    const admitted = this.admit(kind);
    if (!admitted) {
      return;
    }
    const { document, layer, placedMask, selectionRect } = admitted;
    const transparencyLocked = isLayerTransparencyLocked(layer);
    if (layer.type === 'control' || (layer.type === 'raster' && layer.source.type === 'image')) {
      // Nothing to erase is no reason to bake the layer's transform. The extent includes pixels not saved yet.
      const content = hittableLayerRect(layer, document, this.deps.layers.get(layer.id)?.rect);
      if (
        kind === 'erase' &&
        (!content || !intersect(selectionRect, roundOut(transformBounds(layerMatrix(layer.transform), content))))
      ) {
        return;
      }
      const transaction = this.deps.beginPixelEdit(layer.id);
      if (transaction) {
        this.edit(
          kind,
          transaction.layerId,
          { kind: 'transaction', transaction, transparencyLocked },
          selectionRect,
          placedMask
        );
      }
      return;
    }
    const toLocal = invert(layerMatrix(layer.transform));
    if (!toLocal) {
      // A zero-scale layer has no pixels under the selection.
      return;
    }
    const txn = this.deps.ctx.begin({ historyBytes: 0 });
    if (!('publish' in txn)) {
      this.deps.reportRefusal(txn.status);
      return;
    }
    try {
      this.edit(
        kind,
        layer.id,
        { color: isMaskLayer(layer) ? MASK_PAINT_COLOR : null, kind: 'direct', toLocal, transparencyLocked, txn },
        roundOut(transformBounds(toLocal, selectionRect)),
        placedMask
      );
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
    target: PixelTarget,
    /** The selection's bounds in the target's space. */
    selectionRect: Rect,
    placedMask: NonNullable<ReturnType<SelectionState['mask']>>
  ): void {
    const { deps } = this;
    const live = target.kind === 'transaction' ? target.transaction : null;
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
    if (!(live ? live.grow(footprint) : target.kind === 'direct' && target.txn.growHistory(footprint))) {
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
      const maskToLocal = target.kind === 'direct' ? target.toLocal : undefined;
      if (kind === 'fill') {
        fillMaskedRegion({
          backend: deps.backend,
          color: (target.kind === 'direct' ? target.color : null) ?? deps.getFillColor(),
          composite: target.transparencyLocked ? 'source-atop' : 'source-over',
          mask: placedMask.surface,
          maskOrigin: placedMask.rect,
          maskToLocal,
          rect,
          target: surface,
          targetOrigin: origin,
        });
      } else {
        eraseMaskedRegion({
          backend: deps.backend,
          mask: placedMask.surface,
          maskOrigin: placedMask.rect,
          maskToLocal,
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
      if (target.kind !== 'direct' || imageDataEqual(before, after)) {
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
