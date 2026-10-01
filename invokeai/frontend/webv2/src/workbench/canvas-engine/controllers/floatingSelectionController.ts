import type { CanvasEditRefusal } from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { isLeafPixelEditEligible } from '@workbench/canvas-engine/editing/controlPixelEdit';
import { createImagePatchEntry } from '@workbench/canvas-engine/history/imagePatch';
import { isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { renderLayerDisplayEffect } from '@workbench/canvas-engine/render/layerDisplayEffect';
import {
  floatDocumentMatrix,
  liftSelectedPixels,
  transformPlacedMask,
} from '@workbench/canvas-engine/selection/floatingSelection';
import { eraseMaskedRegion } from '@workbench/canvas-engine/selection/selectionOps';
import { layerMatrix } from '@workbench/canvas-engine/tools/moveHitTest';
import { bakeMatrix, IDENTITY_TRANSFORM } from '@workbench/canvas-engine/transform/transformMath';

import type { CanvasMutationContext } from './mutationContext';

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
  readonly canEdit: () => boolean;
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

/**
 * Lifts pixels without history or dirtying, suspending the layer's persistence so the temporary hole never uploads.
 * Commit is admitted before it lands anything and records pre-lift and post-bake pixels across the hole/landing union
 * as one undo entry; a commit that cannot be recorded puts the pixels back instead. Cancellation restores pixels
 * without history.
 */
export class FloatingSelectionController {
  private float: FloatingSelection | null = null;
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

  /** Lifts selected pixels; returns false for absent selection, ineligible layers or no overlap. */
  lift(layerId: string): boolean {
    if (this.disposed || this.float || !this.deps.canEdit()) {
      return false;
    }
    const document = this.deps.getDocument();
    const leaf = lookupDocumentLeaf(document, layerId);
    const layer = leaf?.layer;
    if (!document || !layer || !isLeafPixelEditEligible(leaf)) {
      return false;
    }
    const mask = this.deps.selection.mask();
    const entry = this.deps.layers.get(layerId);
    if (!mask || !entry || isEmpty(entry.rect)) {
      return false;
    }

    const lifted = liftSelectedPixels({
      backend: this.deps.backend,
      cache: { rect: entry.rect, surface: entry.surface },
      layerMatrix: layerMatrix(layer.transform),
      mask,
    });
    if (!lifted) {
      return false;
    }

    this.releasePersistence = this.deps.suspendPersistence(layerId);
    const region = lifted.pixels.rect;
    const before = entry.surface.ctx.getImageData(
      region.x - entry.rect.x,
      region.y - entry.rect.y,
      region.width,
      region.height
    );

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
      // Bake darkness dropout once so the float matches the rendered control map.
      display: renderLayerDisplayEffect(this.deps.backend, layer, lifted.pixels.surface),
      layerId,
      mask: { rect: mask.rect, surface: mask.surface },
      pixels: lifted.pixels,
      transform: { ...IDENTITY_TRANSFORM },
    };
    // Bump the cache version so the hole composites, WITHOUT marking dirty.
    this.deps.notifyPainted(layerId);
    this.deps.invalidateLayer(layerId);
    this.deps.onChange();
    return true;
  }

  /** Updates the float's live (layer-local) transform. */
  setTransform(transform: LayerTransform): void {
    if (!this.float) {
      return;
    }
    this.float.transform = { ...transform };
    this.deps.invalidateLayer(this.float.layerId);
  }

  /**
   * Bakes pixels and moves the selection in one undo entry. An unmoved float restores its cut pixels without
   * adding history.
   */
  commit(): void {
    const float = this.float;
    if (this.disposed || !float) {
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

    const matrix = bakeMatrix(float.transform);
    const landing = roundOut(transformBounds(matrix, float.pixels.rect));
    // The patch spans both the hole left behind and where the pixels land, so a
    // single undo restores both halves of the move.
    const patchRect = union(float.before.rect, landing);
    const selectionBefore = this.deps.selection.snapshot();
    const selectionBytes = (selectionBefore.alpha?.byteLength ?? 0) * 2;
    // Banking a float is part of whatever the user moved on to, even mid-gesture.
    const txn = this.deps.ctx.begin({
      gesture: true,
      historyBytes: patchRect.width * patchRect.height * 8 + selectionBytes,
    });
    if (!('publish' in txn)) {
      this.cancel();
      this.deps.reportRefusal(txn.status);
      return;
    }
    try {
      this.land(float, layer, matrix, patchRect, selectionBefore, txn);
    } finally {
      txn.end();
    }
  }

  private land(
    float: FloatingSelection,
    layer: CanvasLayerContract,
    matrix: ReturnType<typeof bakeMatrix>,
    patchRect: Rect,
    selectionBefore: ReturnType<SelectionState['snapshot']>,
    txn: Extract<ReturnType<CanvasMutationContext['begin']>, { publish: unknown }>
  ): void {
    const existing = this.deps.layers.get(float.layerId);
    const originalRect = existing ? { ...existing.rect } : null;
    const entry = this.deps.layers.growToRect(float.layerId, patchRect);

    // Reconstruct the pre-lift pixels over the patch: the live cache is correct
    // everywhere except the hole, which `before` refills exactly.
    const scratch = this.deps.backend.createSurface(patchRect.width, patchRect.height);
    const sctx = scratch.ctx;
    sctx.setTransform(1, 0, 0, 1, 0, 0);
    sctx.globalCompositeOperation = 'source-over';
    sctx.globalAlpha = 1;
    sctx.drawImage(entry.surface.canvas, entry.rect.x - patchRect.x, entry.rect.y - patchRect.y);
    sctx.putImageData(float.before.data, float.before.rect.x - patchRect.x, float.before.rect.y - patchRect.y);
    const before = sctx.getImageData(0, 0, patchRect.width, patchRect.height);

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
        bytes: patch.bytes + (selectionBefore.alpha?.byteLength ?? 0) + (selectionAfter.alpha?.byteLength ?? 0),
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
    if (result.status !== 'committed') {
      // Nothing was recorded: return the layer and the ants to their pre-lift state.
      const live = this.deps.layers.get(float.layerId);
      live?.surface.ctx.putImageData(before, patchRect.x - live.rect.x, patchRect.y - live.rect.y);
      if (originalRect) {
        this.deps.layers.shrinkToRect(float.layerId, originalRect);
      } else {
        this.deps.layers.delete(float.layerId);
      }
      this.deps.selection.restore(selectionBefore);
      this.deps.notifyPainted(float.layerId);
      this.deps.invalidateLayer(float.layerId);
      if (result.status === 'over-budget') {
        this.deps.reportRefusal('over-budget');
      }
    }
    this.settle();
  }

  /** Drops the float and lets its layer persist again. */
  private settle(): void {
    this.float = null;
    const release = this.releasePersistence;
    this.releasePersistence = null;
    try {
      release?.();
    } finally {
      this.deps.onChange();
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
    const restoreRect: Rect = float.before.rect;
    if (this.layerOf(float) && !isEmpty(restoreRect)) {
      const entry = this.deps.layers.growToRect(float.layerId, restoreRect);
      entry.surface.ctx.putImageData(float.before.data, restoreRect.x - entry.rect.x, restoreRect.y - entry.rect.y);
      this.deps.notifyPainted(float.layerId);
      this.deps.invalidateLayer(float.layerId);
    }
    this.settle();
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    // Restore the float's uniquely held pixels before disposal.
    this.cancel();
    this.disposed = true;
  }
}
