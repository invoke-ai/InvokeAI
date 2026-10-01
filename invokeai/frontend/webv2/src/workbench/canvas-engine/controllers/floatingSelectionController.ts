import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { FloatingSelection, FloatLiftResult } from '@workbench/canvas-engine/selection/floatingSelection';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { getSourceContentRect, renderableSourceOf } from '@workbench/canvas-engine/document/sources';
import { isLeafPixelEditEligible } from '@workbench/canvas-engine/editing/controlPixelEdit';
import { HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { renderLayerDisplayEffect } from '@workbench/canvas-engine/render/layerDisplayEffect';
import {
  floatDocumentMatrix,
  liftRegion,
  liftSelectedPixels,
  transformPlacedMask,
} from '@workbench/canvas-engine/selection/floatingSelection';
import { eraseMaskedRegion } from '@workbench/canvas-engine/selection/selectionOps';
import { layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';
import { bakeMatrix, IDENTITY_TRANSFORM } from '@workbench/canvas-engine/transform/transformMath';

import type { CanvasMutationContext, EditPublishResult, EditTransaction } from './mutationContext';

import { rgbaBytes } from './editSteps';

export interface FloatingSelectionControllerOptions {
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly selection: SelectionState;
  readonly ctx: Pick<CanvasMutationContext, 'begin'>;
  readonly applyImagePatch: ImagePatchApply;
  /** Holds back persistence of a layer until the returned release; the cut hole must never upload. */
  readonly suspendPersistence: (layerId: string) => () => void;
  readonly reportRefusal: (refusal: CanvasEditRefusal) => void;
  readonly getDocument: () => CanvasDocumentContractV3 | null;
  readonly notifyPainted: (layerId: string) => void;
  readonly markDirty: (layerId: string) => void;
  readonly invalidateLayer: (layerId: string) => void;
  /** Called whenever a float appears or disappears (the engine mirrors it to a store). */
  readonly onChange: () => void;
}

const isIdentity = (transform: LayerTransform): boolean =>
  transform.x === 0 &&
  transform.y === 0 &&
  transform.scaleX === 1 &&
  transform.scaleY === 1 &&
  transform.rotation === 0;

const pixelCount = (rect: Rect): number => (isEmpty(rect) ? 0 : rect.width * rect.height);

/**
 * Undo bytes a commit at `transform` records, from geometry alone: pre-lift and landed pixels over the union of the
 * hole and the landing, the selection alpha before and after, and the entry overhead.
 */
const commitHistoryBytes = (
  region: Rect,
  maskRect: Rect,
  layerTransform: LayerTransform,
  transform: LayerTransform
): number => {
  const matrix = bakeMatrix(transform);
  const patch = union(region, roundOut(transformBounds(matrix, region)));
  const documentMatrix = floatDocumentMatrix(layerMatrix(layerTransform), matrix);
  const movedMask = documentMatrix ? roundOut(transformBounds(documentMatrix, maskRect)) : maskRect;
  return rgbaBytes(patch) * 2 + pixelCount(maskRect) + pixelCount(movedMask) + HISTORY_ENTRY_OVERHEAD_BYTES;
};

type PublishedStatus = EditPublishResult['status'];
const isRefusal = (status: PublishedStatus): status is CanvasEditRefusal =>
  status === 'busy' || status === 'gesture-active' || status === 'not-ready' || status === 'over-budget';

/**
 * A float is admitted before it cuts anything and keeps that admission until it lands or is dropped; each transform
 * grows it to cover the commit there, so an accepted transform always fits its undo entry. The cut layer's
 * persistence is suspended meanwhile, so the hole never uploads. Commit records pre-lift and post-bake pixels across
 * the hole/landing union with both selections as one undo entry; a commit that cannot be recorded puts the pixels
 * back instead. Cancellation restores pixels without history.
 */
export class FloatingSelectionController {
  private float: FloatingSelection | null = null;
  /** The float's admission, held from lift until the float lands or is dropped. */
  private txn: EditTransaction | null = null;
  private admittedBytes = 0;
  private refusalReported = false;
  private releasePersistence: (() => void) | null = null;
  private disposed = false;

  constructor(private readonly deps: FloatingSelectionControllerOptions) {}

  get(): FloatingSelection | null {
    return this.float;
  }

  has(): boolean {
    return this.float !== null;
  }

  /** The float's layer, or `null` — resolved fresh so a deleted layer reads as absent. */
  private layerOf(float: FloatingSelection): CanvasLayerContract | null {
    const document = this.deps.getDocument();
    return getDocumentLayer(document, float.layerId) ?? null;
  }

  /**
   * Lifts selected pixels once the edit is admitted. `unavailable` means there is nothing to lift (no selection,
   * an ineligible layer, no overlap); `refused` means the edit was refused and the refusal reported.
   */
  lift(layerId: string): FloatLiftResult {
    if (this.disposed || this.float) {
      return 'unavailable';
    }
    const document = this.deps.getDocument();
    const leaf = lookupDocumentLeaf(document, layerId);
    const layer = leaf?.layer;
    if (!document || !layer || !isLeafPixelEditEligible(leaf)) {
      return 'unavailable';
    }
    const mask = this.deps.selection.mask();
    if (!mask) {
      return 'unavailable';
    }
    const entry = this.deps.layers.get(layerId);
    if (!entry) {
      // Pixels the cache has not rebuilt yet are still the layer's; moving the whole layer instead would surprise.
      if (renderableSourceOf(layer) && !isEmpty(getSourceContentRect(layer, document))) {
        this.deps.reportRefusal('not-ready');
        return 'refused';
      }
      return 'unavailable';
    }
    const matrix = layerMatrix(layer.transform);
    const region = liftRegion(entry.rect, matrix, mask.rect);
    if (!region) {
      return 'unavailable';
    }

    const historyBytes = commitHistoryBytes(region, mask.rect, layer.transform, IDENTITY_TRANSFORM);
    // The lift belongs to the drag that starts it.
    const txn = this.deps.ctx.begin({ gesture: true, historyBytes });
    if (!('publish' in txn)) {
      this.deps.reportRefusal(txn.status);
      return 'refused';
    }

    let before: ImageData | null = null;
    let releasePersistence: (() => void) | null = null;
    try {
      const lifted = liftSelectedPixels({
        backend: this.deps.backend,
        cache: { rect: entry.rect, surface: entry.surface },
        layerMatrix: matrix,
        mask,
      });
      if (!lifted) {
        txn.end();
        return 'unavailable';
      }
      // Bake darkness dropout once so the float matches the rendered control map.
      const display = renderLayerDisplayEffect(this.deps.backend, layer, lifted.pixels.surface);
      before = entry.surface.ctx.getImageData(
        region.x - entry.rect.x,
        region.y - entry.rect.y,
        region.width,
        region.height
      );
      releasePersistence = this.deps.suspendPersistence(layerId);
      eraseMaskedRegion({
        backend: this.deps.backend,
        mask: lifted.localMask.surface,
        maskOrigin: lifted.localMask.rect,
        rect: region,
        target: entry.surface,
        targetOrigin: entry.rect,
      });
      this.float = {
        before: { data: before, rect: region },
        display,
        layerId,
        mask: { rect: mask.rect, surface: mask.surface },
        pixels: lifted.pixels,
        transform: { ...IDENTITY_TRANSFORM },
      };
    } catch (error) {
      try {
        if (before && releasePersistence) {
          entry.surface.ctx.putImageData(before, region.x - entry.rect.x, region.y - entry.rect.y);
          releasePersistence();
        }
      } finally {
        txn.end();
      }
      throw error;
    }
    this.txn = txn;
    this.admittedBytes = historyBytes;
    this.refusalReported = false;
    this.releasePersistence = releasePersistence;
    // Bump the cache version so the hole composites, WITHOUT marking dirty.
    this.deps.notifyPainted(layerId);
    this.deps.invalidateLayer(layerId);
    this.deps.onChange();
    return 'lifted';
  }

  /**
   * Accepts the float's live (layer-local) transform once the admission covers a commit there. A transform that
   * would not fit keeps the last accepted one and reports the refusal once per float.
   */
  setTransform(transform: LayerTransform): boolean {
    const float = this.float;
    const txn = this.txn;
    if (!float || !txn) {
      return false;
    }
    const layer = this.layerOf(float);
    if (layer) {
      const required = commitHistoryBytes(float.pixels.rect, float.mask.rect, layer.transform, transform);
      if (required > this.admittedBytes) {
        if (!txn.growHistory(required - this.admittedBytes)) {
          if (!this.refusalReported) {
            this.refusalReported = true;
            this.deps.reportRefusal('over-budget');
          }
          return false;
        }
        this.admittedBytes = required;
      }
    }
    float.transform = { ...transform };
    this.deps.invalidateLayer(float.layerId);
    return true;
  }

  /**
   * Bakes pixels and moves the selection in one undo entry under the float's admission. An unmoved float restores
   * its cut pixels without adding history.
   */
  commit(): void {
    const float = this.float;
    if (!float || !this.txn) {
      return;
    }
    const layer = this.layerOf(float);
    if (!layer) {
      // The layer went away under the float; the pixels have nowhere to land.
      this.settle();
      return;
    }
    if (isIdentity(float.transform)) {
      this.cancel();
      return;
    }
    const txn = this.currentTransaction();
    if (!txn) {
      return;
    }
    try {
      this.land(float, layer, txn);
    } catch (error) {
      this.cancel();
      throw error;
    }
  }

  /**
   * The retained admission, renewed once when its permit went stale (an operation locked and released the document
   * meanwhile). A float that can no longer be admitted is put back and the refusal reported.
   */
  private currentTransaction(): EditTransaction | null {
    const retained = this.txn!;
    if (retained.isCurrent()) {
      return retained;
    }
    // Hand the capacity back first; holding both would need the float's bytes twice.
    retained.end();
    const renewed = this.deps.ctx.begin({ gesture: true, historyBytes: this.admittedBytes });
    if (!('publish' in renewed)) {
      this.cancel();
      this.deps.reportRefusal(renewed.status);
      return null;
    }
    this.txn = renewed;
    return renewed;
  }

  private land(float: FloatingSelection, layer: CanvasLayerContract, txn: EditTransaction): void {
    const matrix = bakeMatrix(float.transform);
    // The patch spans both the hole left behind and where the pixels land, so a
    // single undo restores both halves of the move.
    const patchRect = union(float.before.rect, roundOut(transformBounds(matrix, float.pixels.rect)));
    const selectionBefore = this.deps.selection.snapshot();
    const existing = this.deps.layers.get(float.layerId);
    const originalRect = existing ? { ...existing.rect } : null;
    const entry = this.deps.layers.growToRect(float.layerId, patchRect);

    let before: ImageData | null = null;
    let committed = false;
    try {
      // Reconstruct the pre-lift pixels over the patch: the live cache is correct
      // everywhere except the hole, which `before` refills exactly.
      const scratch = this.deps.backend.createSurface(patchRect.width, patchRect.height);
      const sctx = scratch.ctx;
      sctx.setTransform(1, 0, 0, 1, 0, 0);
      sctx.globalCompositeOperation = 'source-over';
      sctx.globalAlpha = 1;
      sctx.drawImage(entry.surface.canvas, entry.rect.x - patchRect.x, entry.rect.y - patchRect.y);
      sctx.putImageData(float.before.data, float.before.rect.x - patchRect.x, float.before.rect.y - patchRect.y);
      before = sctx.getImageData(0, 0, patchRect.width, patchRect.height);

      // Land the float through its transform, in layer-local space.
      const ctx = entry.surface.ctx;
      ctx.save();
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1;
      ctx.imageSmoothingEnabled = true;
      ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e - entry.rect.x, matrix.f - entry.rect.y);
      ctx.drawImage(float.pixels.surface.canvas, float.pixels.rect.x, float.pixels.rect.y);
      ctx.restore();
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      const after = ctx.getImageData(
        patchRect.x - entry.rect.x,
        patchRect.y - entry.rect.y,
        patchRect.width,
        patchRect.height
      );

      // The ants travel with the pixels inside the same step, so one undo puts
      // both back; the raw selection records nothing of its own here.
      this.moveSelectionWithFloat(layer, matrix, float);
      const selectionAfter = this.deps.selection.snapshot();

      const patch = createImagePatchEntry({
        after,
        apply: this.deps.applyImagePatch,
        before,
        label: 'Move selection',
        layerId: float.layerId,
        rect: patchRect,
      });
      const result = txn.publish(
        patch.label,
        {
          notify: () => {
            this.deps.notifyPainted(float.layerId);
            this.deps.markDirty(float.layerId);
            this.deps.invalidateLayer(float.layerId);
          },
        },
        {
          bytes:
            patch.bytes +
            (selectionBefore.alpha?.byteLength ?? 0) +
            (selectionAfter.alpha?.byteLength ?? 0) +
            HISTORY_ENTRY_OVERHEAD_BYTES,
          heldAssetRefs: patch.heldAssetRefs,
          redo: async () => {
            await patch.redo();
            this.deps.selection.restore(selectionAfter);
          },
          undo: async () => {
            await patch.undo();
            this.deps.selection.restore(selectionBefore);
          },
        },
        { origin: 'system' }
      );
      committed = result.status === 'committed';
      if (isRefusal(result.status)) {
        this.deps.reportRefusal(result.status);
      }
    } finally {
      if (!committed) {
        // Nothing was recorded: return the layer and the ants to their pre-lift state.
        const live = this.deps.layers.get(float.layerId);
        if (live && before) {
          live.surface.ctx.putImageData(before, patchRect.x - live.rect.x, patchRect.y - live.rect.y);
        }
        if (originalRect) {
          this.deps.layers.shrinkToRect(float.layerId, originalRect);
        } else {
          this.deps.layers.delete(float.layerId);
        }
        this.deps.selection.restore(selectionBefore);
        this.deps.notifyPainted(float.layerId);
        this.deps.invalidateLayer(float.layerId);
      }
    }
    this.settle();
  }

  /** Drops the float, lets its layer persist again and returns its unused admission. */
  private settle(): void {
    const txn = this.txn;
    const release = this.releasePersistence;
    this.float = null;
    this.txn = null;
    this.admittedBytes = 0;
    this.releasePersistence = null;
    try {
      release?.();
    } finally {
      try {
        txn?.end();
      } finally {
        this.deps.onChange();
      }
    }
  }

  private moveSelectionWithFloat(
    layer: CanvasLayerContract,
    floatMatrix: ReturnType<typeof bakeMatrix>,
    float: FloatingSelection
  ): void {
    const documentMatrix = floatDocumentMatrix(layerMatrix(layer.transform), floatMatrix);
    if (!documentMatrix) {
      return;
    }
    const moved = transformPlacedMask(this.deps.backend, float.mask, documentMatrix);
    if (moved) {
      this.deps.selection.replaceMask(moved);
    }
  }

  /** Puts the lifted pixels back untouched and drops the float. Pushes no history. */
  cancel(): void {
    const float = this.float;
    if (!float) {
      return;
    }
    let restored = true;
    try {
      const restoreRect: Rect = float.before.rect;
      if (this.layerOf(float) && !isEmpty(restoreRect)) {
        restored = false;
        const entry = this.deps.layers.growToRect(float.layerId, restoreRect);
        entry.surface.ctx.putImageData(float.before.data, restoreRect.x - entry.rect.x, restoreRect.y - entry.rect.y);
        restored = true;
        this.deps.notifyPainted(float.layerId);
        this.deps.invalidateLayer(float.layerId);
      }
    } finally {
      if (!restored) {
        // A hole that could not be refilled must never persist; the layer stays suspended.
        this.releasePersistence = null;
      }
      this.settle();
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    // Restore the float's uniquely held pixels before disposal.
    this.cancel();
  }
}
