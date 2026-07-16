/**
 * A cached, non-destructive "adjusted surface" per raster layer.
 *
 * Raster adjustments (brightness/contrast/saturation + curves) must be applied at
 * composite time WITHOUT recomputing every frame (the plan explicitly forbids a
 * third per-frame recompute alongside the mask-colorize and control-transparency
 * smells). This store memoizes each layer's adjusted pixels, keyed by the source
 * cache's `version` AND the adjustments' identity ({@link adjustmentsKey}), and
 * rebuilds only when either changes — so a steady frame reuses the cached surface,
 * an edit to the layer (cache version bump) invalidates it, and an unrelated
 * layer's edit does not.
 *
 * Everything flows through the injected {@link RasterBackend} seam, so it runs
 * unchanged in node tests. Zero React, zero import-time side effects.
 */

import type { CanvasAdjustmentsContract } from '@workbench/types';

import type { LayerCacheEntry } from './layerCache';
import type { RasterBackend, RasterSurface } from './raster';

import { adjustmentsKey, applyAdjustments, isIdentityAdjustments } from './adjustments';
import { createDerivedSurfaceCache, type DerivedSurfaceCache } from './derivedSurfaceCache';

/** The imperative adjusted-surface store returned by {@link createAdjustedSurfaceCache}. */
export interface AdjustedSurfaceCache {
  /**
   * Returns a surface holding `entry`'s pixels with `adjustments` applied, or
   * `null` when the adjustments are identity (the caller should draw the original
   * cache surface). The returned surface is memoized: an unchanged
   * `(entry.version, adjustmentsKey)` reuses it with zero pixel work.
   */
  get(
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface | null;
  /** Drops a layer's memoized adjusted surface (e.g. on layer delete). */
  delete(layerId: string): void;
  /** Number of memoized adjusted surfaces (for tests / accounting). */
  size(): number;
  /** Bytes held by adjusted surfaces. */
  byteSize(): number;
  /** Releases all memoized surfaces. */
  dispose(): void;
}

/** Creates an {@link AdjustedSurfaceCache} backed by the given {@link RasterBackend}. */
export const createAdjustedSurfaceCache = (
  backend: RasterBackend,
  cache: DerivedSurfaceCache = createDerivedSurfaceCache()
): AdjustedSurfaceCache => {
  const get = (
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface | null => {
    if (isIdentityAdjustments(adjustments)) {
      // Identity: nothing to cache — drop any stale slot and let the caller draw
      // the original surface.
      cache.delete(layerId, 'adjustments');
      return null;
    }
    const { height, width } = entry.surface;
    if (width <= 0 || height <= 0) {
      return null;
    }
    const key = adjustmentsKey(adjustments);
    return cache.get({
      create: (target) => {
        const surface = target ?? backend.createSurface(width, height);
        if (surface.width !== width || surface.height !== height) {
          surface.resize(width, height);
        }
        const ctx = surface.ctx;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, width, height);
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = 'source-over';
        ctx.drawImage(entry.surface.canvas, 0, 0);
        const imageData = ctx.getImageData(0, 0, width, height);
        applyAdjustments(imageData, adjustments);
        ctx.putImageData(imageData, 0, 0);
        return surface;
      },
      kind: 'adjustments',
      layerId,
      paramsKey: key,
      source: entry.surface,
      sourceVersion: entry.version,
    });
  };

  return {
    byteSize: cache.byteSize,
    delete: (layerId) => {
      cache.delete(layerId, 'adjustments');
    },
    dispose: () => {
      cache.dispose();
    },
    get,
    size: cache.size,
  };
};
