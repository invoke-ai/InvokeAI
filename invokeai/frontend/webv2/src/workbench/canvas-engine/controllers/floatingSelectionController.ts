import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { History } from '@workbench/canvas-engine/history/history';
import type { ImagePatchApply } from '@workbench/canvas-engine/history/imagePatch';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { FloatingSelection } from '@workbench/canvas-engine/selection/floatingSelection';
import type { SelectionState } from '@workbench/canvas-engine/selection/selectionState';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';

import { isLayerPixelEditEligible } from '@workbench/canvas-engine/editing/controlPixelEdit';
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

export interface FloatingSelectionControllerOptions {
  readonly backend: RasterBackend;
  readonly layers: LayerCacheStore;
  readonly selection: SelectionState;
  readonly history: History;
  readonly applyImagePatch: ImagePatchApply;
  readonly getDocument: () => CanvasDocumentContractV2 | null;
  readonly canEdit: () => boolean;
  readonly endBurst: () => void;
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
 * Owns the floating-selection lifecycle: cutting selected pixels off a layer,
 * holding them in flight, and either baking them back as ONE undo entry or
 * putting them back untouched.
 *
 * **The lift is deliberately not a history entry.** Only `commit` pushes, with
 * `before` = the pre-lift pixels and `after` = the post-bake pixels over the
 * union of the hole and the landing region. So one undo restores the layer
 * exactly, no matter how many drags happened in between — and a cancelled float
 * leaves the history untouched rather than adding a pair of entries that undo
 * each other.
 *
 * The cache is mutated on lift (the hole) but deliberately NOT marked dirty:
 * persisting a holed bitmap mid-drag would upload a state the user never asked
 * for, and a slow network could land it after the commit. Only `commit` marks
 * dirty.
 */
export class FloatingSelectionController {
  private float: FloatingSelection | null = null;
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
    return document?.layers.find((layer) => layer.id === float.layerId) ?? null;
  }

  /**
   * Cuts the selection's pixels out of `layerId` into a new float. Returns false
   * when there is nothing to lift (no selection, an ineligible layer, or no
   * overlap between the two).
   */
  lift(layerId: string): boolean {
    if (this.disposed || this.float || !this.deps.canEdit()) {
      return false;
    }
    const document = this.deps.getDocument();
    const layer = document?.layers.find((candidate) => candidate.id === layerId);
    if (!document || !layer || !isLayerPixelEditEligible(layer)) {
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

    // Close any coalescing burst before mutating pixels, so a pending nudge does
    // not absorb this edit.
    this.deps.endBurst();
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
      // Baked now, not per frame: floating a control map must show the same
      // darkness-dropped-out pixels the layer itself shows, never its raw
      // opaque background.
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
   * Bakes the float back into its layer as one undoable entry, and carries the
   * selection outline along with it. A float whose transform never moved is
   * still put back (it was cut out), but without a history entry — nothing
   * changed.
   */
  commit(): void {
    const float = this.float;
    if (this.disposed || !float) {
      return;
    }
    const layer = this.layerOf(float);
    if (!layer) {
      // The layer went away under the float; the pixels have nowhere to land.
      this.float = null;
      this.deps.onChange();
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

    this.deps.endBurst();
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

    this.float = null;
    this.deps.onChange();
    this.deps.notifyPainted(float.layerId);
    this.deps.markDirty(float.layerId);
    this.deps.invalidateLayer(float.layerId);

    if (!this.deps.history.isApplying()) {
      this.deps.history.push(
        createImagePatchEntry({
          after,
          apply: this.deps.applyImagePatch,
          before,
          label: 'Move selection',
          layerId: float.layerId,
          rect: patchRect,
        })
      );
    }

    this.moveSelectionWithFloat(layer, matrix, float);
  }

  /**
   * Re-places the document-space selection mask by the float's motion, so the
   * marching ants end up around the pixels' new home rather than their old one.
   */
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
    this.float = null;
    this.deps.onChange();
    if (!this.layerOf(float)) {
      return;
    }
    const restoreRect: Rect = float.before.rect;
    if (isEmpty(restoreRect)) {
      return;
    }
    const entry = this.deps.layers.growToRect(float.layerId, restoreRect);
    entry.surface.ctx.putImageData(float.before.data, restoreRect.x - entry.rect.x, restoreRect.y - entry.rect.y);
    this.deps.notifyPainted(float.layerId);
    this.deps.invalidateLayer(float.layerId);
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    // A live float holds pixels that exist nowhere else; put them back rather
    // than dropping them on the floor.
    this.cancel();
    this.disposed = true;
  }
}
