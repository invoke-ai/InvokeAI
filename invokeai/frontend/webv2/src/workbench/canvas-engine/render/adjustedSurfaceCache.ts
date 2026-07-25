import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { intersect, roundOut } from '@workbench/canvas-engine/math/rect';

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
  cache: DerivedSurfaceCache = createDerivedSurfaceCache(),
  /**
   * Reports the surface-local region a layer's cache was written in since a
   * given version — {@link LayerCacheStore.damageSince}. Omit it and every
   * rebuild is wholesale, which is what callers without a layer cache (tests,
   * one-off derivations) want.
   */
  layerDamageSince?: (layerId: string, version: number) => Rect | null
): AdjustedSurfaceCache => {
  /** The damaged region clamped to the surface, or `null` to rebuild in full. */
  const damageSince = (layerId: string, version: number, width: number, height: number): Rect | null => {
    const damaged = layerDamageSince?.(layerId, version) ?? null;
    if (!damaged) {
      return null;
    }
    const clamped = intersect(roundOut(damaged), { height, width, x: 0, y: 0 });
    return clamped && clamped.width > 0 && clamped.height > 0 ? clamped : null;
  };

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
      create: (target, reusableFromVersion) => {
        const surface = target ?? backend.createSurface(width, height);
        const resized = surface.width !== width || surface.height !== height;
        if (resized) {
          surface.resize(width, height);
        }
        // Rebuilding this costs a full-surface readback plus a per-pixel JS pass
        // — 35ms on a 2600x2100 layer, which is where a live stroke on an
        // adjusted layer spends its entire frame. A stroke bumps the source
        // version every tick, so the memo misses every frame; refreshing only
        // the region the stroke actually wrote turns that into a fraction of the
        // work. A resize invalidates the retained pixels, so it forces the full
        // path regardless of what the trail says.
        const refresh =
          resized || reusableFromVersion === null ? null : damageSince(layerId, reusableFromVersion, width, height);
        const region = refresh ?? { height, width, x: 0, y: 0 };
        const ctx = surface.ctx;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(region.x, region.y, region.width, region.height);
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = 'source-over';
        ctx.drawImage(
          entry.surface.canvas,
          region.x,
          region.y,
          region.width,
          region.height,
          region.x,
          region.y,
          region.width,
          region.height
        );
        const imageData = ctx.getImageData(region.x, region.y, region.width, region.height);
        applyAdjustments(imageData, adjustments);
        ctx.putImageData(imageData, region.x, region.y);
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
