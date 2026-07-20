/**
 * Pure canvas generation-mode detection.
 *
 * Given only plain geometry/coverage facts the engine + composite executor
 * already computed, {@link detectCanvasMode} reproduces the legacy canvas
 * behavior of picking txt2img / img2img / inpaint / outpaint. It touches no
 * pixels and imports nothing from the engine — every input is precomputed and
 * passed in, so the whole decision table is exhaustively node-testable.
 *
 * Rules (legacy parity):
 * - No enabled raster content intersects the bbox → `txt2img`.
 * - Content opaquely covers the whole bbox, no active inpaint mask → `img2img`.
 * - Content opaquely covers the whole bbox, active inpaint mask → `inpaint`.
 * - The bbox extends past content, or has transparent holes inside it (i.e.
 *   content intersects but does not fully cover) → `outpaint`.
 */

import type { CanvasGenerationMode, Rect } from './types';

/** The precomputed facts {@link detectCanvasMode} decides from. */
export interface CanvasModeInput {
  /** The generation bounding box, in document space. */
  bbox: Rect;
  /**
   * Union of enabled raster-layer content bounds in document space, or `null`
   * when no enabled raster content exists.
   */
  contentBounds: Rect | null;
  /**
   * Whether enabled raster content opaquely covers the entire bbox (no
   * transparent holes) — the alpha scan of the composited bbox surface.
   */
  bboxFullyCovered: boolean;
  /**
   * Whether an enabled inpaint-mask layer with content exists. Always `false`
   * in Phase 4.2 (the contract type exists but no mask is produced yet).
   */
  hasActiveInpaintMask: boolean;
}

/**
 * True when two rects share interior area. Edge-touching or a shared corner is
 * NOT an intersection (strict `<`), matching the engine's `rect.intersect`
 * semantics, so a 1px edge overlap counts but a flush edge does not.
 *
 * Exported so the invoke orchestrator can run a bounds-only pre-pass: content
 * that doesn't overlap the bbox means txt2img, which needs no composite upload.
 */
export const rectsIntersect = (a: Rect, b: Rect): boolean => {
  if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) {
    return false;
  }
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
};

/** Resolves the generation mode for a canvas invoke from precomputed facts. */
export const detectCanvasMode = (input: CanvasModeInput): CanvasGenerationMode => {
  const { bbox, bboxFullyCovered, contentBounds, hasActiveInpaintMask } = input;

  // A degenerate bbox can never contain content: nothing to reference.
  if (bbox.width <= 0 || bbox.height <= 0) {
    return 'txt2img';
  }

  // No enabled raster content touches the bbox: pure text-to-image.
  if (!contentBounds || !rectsIntersect(contentBounds, bbox)) {
    return 'txt2img';
  }

  // Content opaquely fills the whole bbox: img2img, or inpaint when masked.
  if (bboxFullyCovered) {
    return hasActiveInpaintMask ? 'inpaint' : 'img2img';
  }

  // Content intersects but leaves the bbox partly transparent (extends past
  // content, or has holes): outpaint.
  return 'outpaint';
};
