/**
 * A coverage edit draws a tool's coverage (a shape, a fill region) straight into one layer cache and records the last
 * draw as one stroke step. A tool may redraw as often as it likes within the edit: each draw first puts back the
 * pixels the previous one replaced, so only the last draw remains and the step's `before` is always the original
 * pixels. The edit owns the layer-local dirty rect, admission growth, the pristine pixels (kept over every rect drawn
 * so far), drawing through the layer inverse with the selection mask and rotated-layer clip, and putting the cache's
 * pixels and extent back on cancel, refused growth or a refused commit.
 */

import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, PlacedSurface, Rect } from '@workbench/canvas-engine/types';

import { identity, multiply, translate } from '@workbench/canvas-engine/math/mat2d';
import { intersect, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';

import type { StrokeCommittedEvent, StrokeEdit, ToolContext } from './tool';

/** How the coverage lands on the cache: paint over, paint inside existing alpha (transparency lock), or erase. */
export type CoverageOperation = 'source-over' | 'source-atop' | 'destination-out';

/** Draws coverage into a scratch context whose transform maps document space, so callers draw at document coords. */
export type CoverageDraw = (draw: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D) => void;

export interface CoverageEditTarget {
  readonly layerId: string;
  /** Document → layer-local: the inverse of the layer's matrix. */
  readonly toLocal: Mat2d;
  /** The document-space selection mask every draw keeps only the inside of. */
  readonly clipMask: PlacedSurface | null;
  /** The document-space stroke clip; exact on rotated or sheared layers, where the dirty rect only bounds it. */
  readonly clipRect: Rect | null;
  readonly operation: CoverageOperation;
}

export interface CoverageEdit {
  /**
   * Replaces the previous draw with `draw` over `docRect` (document space, already clipped by the caller). False when
   * the admission could not grow: the cache is back to its original pixels and extent and the edit is cancelled.
   */
  apply(docRect: Rect, draw: CoverageDraw): boolean;
  /** Records the last draw as one step; with nothing drawn, or when refused, it cancels and returns false. */
  commit(fields: Pick<StrokeCommittedEvent, 'tool' | 'createdLayer'>): boolean;
  /** Puts the original pixels and extent back and ends the edit unrecorded. */
  cancel(): void;
}

/** The part of a document rect the selection mask's bounds and the stroke clip leave; null when nothing is left. */
export const clipCoverageRect = (
  docRect: Rect,
  { clipMask, clipRect }: Pick<CoverageEditTarget, 'clipMask' | 'clipRect'>
): Rect | null => {
  let rect: Rect | null = docRect;
  if (rect && clipMask) {
    rect = intersect(rect, clipMask.rect);
  }
  if (rect && clipRect) {
    rect = intersect(rect, clipRect);
  }
  return rect;
};

/** The integer layer-local rect a document rect covers, or null when it covers no pixel. */
export const coverageDirtyRect = (toLocal: Mat2d, docRect: Rect): Rect | null => {
  const rect = roundOut(transformBounds(toLocal, docRect));
  return rect.width < 1 || rect.height < 1 ? null : rect;
};

/** The undo footprint of a coverage over `rect`: its before and after RGBA pixels. */
export const coverageEditBytes = (rect: Rect): number => rect.width * rect.height * 8;

const sameRect = (a: Rect, b: Rect): boolean =>
  a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

/** Copies `source` (covering `from`) into `target` (covering `to`) over their shared `rect`, which both contain. */
const blit = (source: ImageData, from: Rect, target: ImageData, to: Rect, rect: Rect): void => {
  const rowBytes = rect.width * 4;
  for (let row = 0; row < rect.height; row++) {
    const read = ((rect.y - from.y + row) * from.width + (rect.x - from.x)) * 4;
    const write = ((rect.y - to.y + row) * to.width + (rect.x - to.x)) * 4;
    target.data.set(source.data.subarray(read, read + rowBytes), write);
  }
};

/**
 * Opens a coverage edit on `target.layerId` under an already admitted `edit`. `admittedBytes` is what that admission
 * already covers, so the edit grows it only past that.
 */
export const createCoverageEdit = (
  ctx: Pick<ToolContext, 'backend' | 'invalidate' | 'layers' | 'notifyLayerPainted'>,
  target: CoverageEditTarget,
  edit: StrokeEdit,
  admittedBytes: number
): CoverageEdit => {
  const { clipMask, clipRect, layerId, operation, toLocal } = target;
  const original = ctx.layers.peek(layerId);
  const originalRect = original ? { ...original.rect } : null;
  let admitted = admittedBytes;
  /** Pristine pixels over every rect drawn so far (layer-local `beforeRect`). */
  let before: ImageData | null = null;
  let beforeRect: Rect | null = null;
  /** The rect the cache currently holds a draw in. */
  let applied: Rect | null = null;
  /** Whether the cache may differ from its original pixels or extent. */
  let touched = false;
  let closed = false;
  /** Where each draw composes before it lands, reused across redraws and freed when the edit closes. */
  let scratch: RasterSurface | null = null;

  const releaseScratch = (): void => {
    scratch?.resize(0, 0);
    scratch = null;
  };

  /** Puts the pristine pixels back over `rect`, which lies within `beforeRect`. */
  const putBefore = (rect: Rect): void => {
    const entry = ctx.layers.peek(layerId);
    if (!entry || !before || !beforeRect) {
      return;
    }
    entry.surface.ctx.putImageData(
      before,
      beforeRect.x - entry.rect.x,
      beforeRect.y - entry.rect.y,
      rect.x - beforeRect.x,
      rect.y - beforeRect.y,
      rect.width,
      rect.height
    );
  };

  const cancel = (): void => {
    if (closed) {
      return;
    }
    closed = true;
    releaseScratch();
    if (touched) {
      if (beforeRect) {
        putBefore(beforeRect);
      }
      if (originalRect) {
        ctx.layers.shrinkToRect(layerId, originalRect);
      } else {
        ctx.layers.delete(layerId);
      }
      ctx.notifyLayerPainted(layerId);
    }
    edit.cancel();
  };

  /** Publishes a live draw's layer-local damage without touching thumbnails, which follow the commit. */
  const publish = (damage: Rect): void => {
    const entry = ctx.layers.peek(layerId);
    if (!entry) {
      return;
    }
    ctx.layers.publishPixels(layerId, { ...damage, x: damage.x - entry.rect.x, y: damage.y - entry.rect.y });
    ctx.invalidate({ damage: { layerId, rect: damage }, layers: [layerId] });
  };

  const apply = (docRect: Rect, draw: CoverageDraw): boolean => {
    if (closed) {
      return false;
    }
    const dirty = coverageDirtyRect(toLocal, docRect);
    const nextBeforeRect = dirty && beforeRect ? union(beforeRect, dirty) : dirty;
    if (nextBeforeRect) {
      // Admitted before any pixel changes: before and after over everything this edit has drawn.
      const footprint = coverageEditBytes(nextBeforeRect);
      if (footprint > admitted) {
        if (!edit.grow(footprint - admitted)) {
          cancel();
          return false;
        }
        admitted = footprint;
      }
    }
    try {
      const previous = applied;
      if (previous) {
        putBefore(previous);
        applied = null;
      }
      if (!dirty || !nextBeforeRect) {
        if (previous) {
          publish(previous);
        }
        return true;
      }
      touched = true;
      const entry = ctx.layers.growToRect(layerId, nextBeforeRect);
      const surfaceCtx = entry.surface.ctx;
      const ox = entry.rect.x;
      const oy = entry.rect.y;
      if (!before || !beforeRect) {
        before = surfaceCtx.getImageData(dirty.x - ox, dirty.y - oy, dirty.width, dirty.height);
      } else if (!sameRect(beforeRect, nextBeforeRect)) {
        // The cache is pristine again everywhere, but the pixels already held are exact: carry them forward.
        const grown = surfaceCtx.getImageData(
          nextBeforeRect.x - ox,
          nextBeforeRect.y - oy,
          nextBeforeRect.width,
          nextBeforeRect.height
        );
        blit(before, beforeRect, grown, nextBeforeRect, beforeRect);
        before = grown;
      }
      beforeRect = nextBeforeRect;

      // Draw into dirty-rect-local scratch so clipping and transparency lock apply in one composite; map through
      // layer inverse then scratch offset.
      if (!scratch) {
        scratch = ctx.backend.createSurface(dirty.width, dirty.height);
      } else if (scratch.width !== dirty.width || scratch.height !== dirty.height) {
        scratch.resize(dirty.width, dirty.height);
      } else {
        scratch.ctx.setTransform(1, 0, 0, 1, 0, 0);
        scratch.ctx.globalCompositeOperation = 'source-over';
        scratch.ctx.clearRect(0, 0, dirty.width, dirty.height);
      }
      const scratchCtx = scratch.ctx;
      const toScratch = multiply(translate(identity(), { x: -dirty.x, y: -dirty.y }), toLocal);
      scratchCtx.setTransform(toScratch.a, toScratch.b, toScratch.c, toScratch.d, toScratch.e, toScratch.f);
      scratchCtx.save();
      draw(scratchCtx);
      scratchCtx.restore();
      scratchCtx.globalCompositeOperation = 'destination-in';
      if (clipMask) {
        // The mask sits in document space, so it goes through the same mapping.
        scratchCtx.drawImage(clipMask.surface.canvas, clipMask.rect.x, clipMask.rect.y);
      }
      if (clipRect && (toLocal.b !== 0 || toLocal.c !== 0)) {
        // The dirty-rect clamp only bounds the AABB on a rotated/sheared layer;
        // keep exactly the pixels inside the document-space rect.
        scratchCtx.fillStyle = '#000';
        scratchCtx.beginPath();
        scratchCtx.rect(clipRect.x, clipRect.y, clipRect.width, clipRect.height);
        scratchCtx.fill();
      }
      surfaceCtx.save();
      surfaceCtx.setTransform(1, 0, 0, 1, 0, 0);
      surfaceCtx.globalCompositeOperation = operation;
      surfaceCtx.drawImage(scratch.canvas, dirty.x - ox, dirty.y - oy);
      surfaceCtx.restore();
      applied = dirty;
      publish(previous ? union(previous, dirty) : dirty);
      return true;
    } catch (error) {
      cancel();
      throw error;
    }
  };

  const commit = (fields: Pick<StrokeCommittedEvent, 'tool' | 'createdLayer'>): boolean => {
    if (closed) {
      return false;
    }
    const entry = ctx.layers.peek(layerId);
    if (!applied || !before || !beforeRect || !entry) {
      cancel();
      return false;
    }
    try {
      const dirtyRect = applied;
      let beforeImageData = before;
      if (!sameRect(beforeRect, dirtyRect)) {
        beforeImageData = entry.surface.ctx.createImageData(dirtyRect.width, dirtyRect.height);
        blit(before, beforeRect, beforeImageData, dirtyRect, dirtyRect);
      }
      const afterImageData = entry.surface.ctx.getImageData(
        dirtyRect.x - entry.rect.x,
        dirtyRect.y - entry.rect.y,
        dirtyRect.width,
        dirtyRect.height
      );
      if (edit.commit({ ...fields, afterImageData, beforeImageData, dirtyRect, layerId })) {
        closed = true;
        releaseScratch();
        return true;
      }
    } catch (error) {
      cancel();
      throw error;
    }
    cancel();
    return false;
  };

  return { apply, cancel, commit };
};
