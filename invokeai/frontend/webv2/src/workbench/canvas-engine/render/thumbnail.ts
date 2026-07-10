/**
 * Pure helpers for layer thumbnails.
 *
 * A layer's cache surface can be any size (up to the document dimensions); a
 * thumbnail must fit within a small square box while preserving aspect ratio
 * and never upscaling past the source. Extracted from the engine's
 * `drawLayerThumbnail` so the arithmetic is unit-testable without a canvas.
 *
 * Zero React, zero DOM, zero import-time side effects.
 */

import type { CanvasImageRef, CanvasLayerContract } from '@workbench/types';

import { renderableSourceOf } from '@workbench/canvas-engine/document/sources';

import { adjustmentsKey } from './adjustments';

/** A thumbnail's integer pixel dimensions after fitting into a `maxSize` box. */
export interface ThumbnailSize {
  width: number;
  height: number;
}

export type LayerThumbnailFallbackStage = 'thumbnail' | 'full' | 'failed';

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

/**
 * Resolves the persisted image that can stand in for a layer while its engine
 * cache is unavailable. Image and paint sources are handled uniformly through
 * `renderableSourceOf`, including the synthetic paint source exposed by masks.
 * Empty paint/mask and parametric sources have no persisted image fallback.
 */
export const resolveLayerThumbnailImageRef = (layer: CanvasLayerContract): CanvasImageRef | null => {
  try {
    const source = renderableSourceOf(layer);
    if (source?.type === 'image') {
      return source.image;
    }
    if (source?.type === 'paint') {
      return source.bitmap;
    }
    return null;
  } catch {
    // Invalid persisted contracts should degrade to the explicit placeholder.
    return null;
  }
};

/** Display-only properties that alter a layer thumbnail without changing its raster cache. */
export const getLayerThumbnailDisplayKey = (layer: CanvasLayerContract): string => {
  if (layer.type === 'raster') {
    return `raster:${adjustmentsKey(layer.adjustments)}`;
  }
  if (layer.type === 'control') {
    return `control:${layer.withTransparencyEffect ? 'transparent' : 'opaque'}`;
  }
  return `mask:${layer.mask.fill.style}:${layer.mask.fill.color}`;
};

/** Advances the persisted-image fallback after an image element load failure. */
export const nextLayerThumbnailFallbackStage = (current: LayerThumbnailFallbackStage): LayerThumbnailFallbackStage =>
  current === 'thumbnail' ? 'full' : 'failed';
