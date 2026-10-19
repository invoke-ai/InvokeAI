/**
 * Mode detection uses precomputed geometry/coverage only. No intersection means txt2img; full opaque coverage
 * means img2img or inpaint when a mask is active; partial coverage or holes means outpaint.
 */

import type { CanvasGenerationMode, Rect } from './types';

/** The precomputed facts {@link detectCanvasMode} decides from. */
export interface CanvasModeInput {
  /** The generation bounding box, in document space. */
  bbox: Rect;
  /** Document-space union; null means no enabled content. */
  contentBounds: Rect | null;
  /** Opaque coverage is alpha-scanned and contains no holes. */
  bboxFullyCovered: boolean;
  /** True when an active mask contains inpaint content. */
  hasActiveInpaintMask: boolean;
}

/** Require interior overlap. The txt2img prepass avoids uploading unused content. */
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

  return 'outpaint';
};
