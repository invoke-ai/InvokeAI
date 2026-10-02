import type { CanvasImageRef, CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { renderableSourceOf } from '@workbench/canvas-engine/document/sources';

import { adjustmentsKey } from './adjustments';

/** A thumbnail's integer pixel dimensions after fitting into a `maxSize` box. */
export interface ThumbnailSize {
  width: number;
  height: number;
}

export type LayerThumbnailFallbackStage = 'thumbnail' | 'full' | 'failed';

/**
 * Aspect-preserving fit without upscaling; positive sources return at least 1px dimensions, degenerate sources
 * return zero size.
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

/** Persisted image fallback for image/paint and synthetic mask sources. Empty or parametric sources have none. */
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
    return `raster:${layer.opacity}:${adjustmentsKey(layer.adjustments)}`;
  }
  if (layer.type === 'control') {
    return `control:${layer.opacity}:${layer.withTransparencyEffect ? 'transparent' : 'opaque'}`;
  }
  return `mask:${layer.opacity}:${layer.mask.fill.style}:${layer.mask.fill.color}`;
};

/** Advances the persisted-image fallback after an image element load failure. */
export const nextLayerThumbnailFallbackStage = (current: LayerThumbnailFallbackStage): LayerThumbnailFallbackStage =>
  current === 'thumbnail' ? 'full' : 'failed';

/** Pure render policy shared by the thumbnail fallback content and retry overlay. */
export const getLayerThumbnailFallbackRenderState = (
  drawn: boolean,
  fallbackStage: LayerThumbnailFallbackStage,
  hasRequestError: boolean
): { showFallback: boolean; showRetry: boolean } => ({
  showFallback: !drawn,
  showRetry: !drawn && (hasRequestError || fallbackStage === 'failed'),
});
