/**
 * Pure sizing math for layer thumbnails.
 *
 * A layer's cache surface can be any size (up to the document dimensions); a
 * thumbnail must fit within a small square box while preserving aspect ratio
 * and never upscaling past the source. Extracted from the engine's
 * `drawLayerThumbnail` so the arithmetic is unit-testable without a canvas.
 *
 * Zero React, zero DOM, zero import-time side effects.
 */

/** A thumbnail's integer pixel dimensions after fitting into a `maxSize` box. */
export interface ThumbnailSize {
  width: number;
  height: number;
}

/**
 * Scales a `srcW`x`srcH` surface to fit inside a `maxSize`x`maxSize` box,
 * preserving aspect ratio and never upscaling (scale is clamped to ≤ 1). Both
 * returned dimensions are at least 1px. Returns a zero size for a degenerate
 * (non-positive) source.
 */
export const fitThumbnailSize = (srcW: number, srcH: number, maxSize: number): ThumbnailSize => {
  if (srcW <= 0 || srcH <= 0 || maxSize <= 0) {
    return { height: 0, width: 0 };
  }
  const scale = Math.min(1, maxSize / srcW, maxSize / srcH);
  return {
    height: Math.max(1, Math.round(srcH * scale)),
    width: Math.max(1, Math.round(srcW * scale)),
  };
};
