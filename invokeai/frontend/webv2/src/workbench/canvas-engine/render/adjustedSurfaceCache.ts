import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import type { LayerCacheEntry } from './layerCache';
import type { RasterBackend, RasterSurface } from './raster';

import { adjustmentsKey, hasSpatialAdjustments, isIdentityAdjustments } from './adjustments';
import { createDerivedSurfaceCache, type DerivedSurfaceCache } from './derivedSurfaceCache';
import {
  clampDamage,
  createSpatialDisplay,
  type DisplayedPixelsState,
  paintAdjustedRegion,
  type SpatialDisplayOptions,
  type SpatialDisplayTarget,
} from './spatialDisplay';

export interface AdjustedSurfaceCacheOptions {
  /** Surface-local {@link LayerCacheStore.damageSince}; without it every content change rebuilds whole surfaces. */
  readonly damageSince?: (layerId: string, version: number) => Rect | null;
  /** Without it, spatial entries are left out of the display (the per-pixel part only). */
  readonly spatial?: SpatialDisplayOptions;
}

/** The imperative adjusted-surface store returned by {@link createAdjustedSurfaceCache}. */
export interface AdjustedSurfaceCache {
  /**
   * Returns the surface the display shows for `entry`'s pixels with `adjustments`, or `null` when the adjustments are
   * identity (the caller should draw the original cache surface). Memoized: an unchanged
   * `(entry.version, adjustmentsKey)` reuses it with zero pixel work. A stack with spatial entries shows an interim
   * until its asynchronous evaluation lands (see `SpatialDisplayOptions.onSettle`).
   */
  get(
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface | null;
  /** The memoized adjusted surface when it is current and exact for `entry` and `adjustments`; never builds one. */
  peek(
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface | null;
  /** What the display currently shows for `entry` and `adjustments`, exact or interim; never builds one. */
  peekDisplayed(
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface | null;
  /** How final what `get` shows for `entry` and `adjustments` is; a stack without spatial entries is exact. */
  state(
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): DisplayedPixelsState;
  /** Drops a layer's memoized adjusted surface and abandons its spatial work (e.g. on layer delete). */
  delete(layerId: string): void;
  /**
   * Abandons running and scheduled spatial jobs, dropping the interims they would have replaced; exact and failed
   * spatial surfaces keep their pixels and state.
   */
  cancelSpatial(): void;
  /**
   * Drops the spatial state and surface of every layer `keep` rejects, e.g. one whose stack lost its spatial entries
   * or that was removed while not drawn, so no stale status outlives its stack.
   */
  retainSpatial(keep: (layerId: string) => boolean): void;
  /** Number of memoized adjusted surfaces (for tests / accounting). */
  size(): number;
  /** Bytes held by adjusted surfaces. */
  byteSize(): number;
  /** Releases every memoized derived surface and abandons all spatial work; the cache stays usable. */
  dispose(): void;
}

/** Creates an {@link AdjustedSurfaceCache} backed by the given {@link RasterBackend}. */
export const createAdjustedSurfaceCache = (
  backend: RasterBackend,
  cache: DerivedSurfaceCache = createDerivedSurfaceCache(),
  options: AdjustedSurfaceCacheOptions = {}
): AdjustedSurfaceCache => {
  const display = options.spatial ? createSpatialDisplay(options.spatial, 'adjusted:') : null;

  const getPerPixel = (
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface => {
    const { height, width } = entry.surface;
    return cache.get({
      create: (target, reusableFromVersion) => {
        const surface = target ?? backend.createSurface(width, height);
        const resized = surface.width !== width || surface.height !== height;
        if (resized) {
          surface.resize(width, height);
        }
        // Source versions change every stroke tick. Refresh only reported damage to avoid full-surface readback
        // and adjustment work; resized surfaces always rebuild.
        const refresh =
          resized || reusableFromVersion === null
            ? null
            : clampDamage(options.damageSince?.(layerId, reusableFromVersion) ?? null, width, height);
        if (refresh !== 'none') {
          // A layer's stack works in layer-local pixels; surface (0,0) sits at `entry.rect`'s origin.
          paintAdjustedRegion(
            surface,
            entry.surface,
            refresh ?? { height, width, x: 0, y: 0 },
            adjustments,
            entry.rect
          );
        }
        return surface;
      },
      kind: 'adjustments',
      layerId,
      paramsKey: adjustmentsKey(adjustments),
      source: entry.surface,
      sourceVersion: entry.version,
    });
  };

  /** Forgets a layer's spatial state with the surface it describes, so no later request reuses that surface blind. */
  const dropSlot = (layerId: string): void => {
    if (display?.drop(layerId)) {
      cache.delete(layerId, 'adjustments');
    }
  };

  const dropSlots = (layerIds: readonly string[]): void => {
    for (const layerId of layerIds) {
      dropSlot(layerId);
    }
  };

  /** A spatial stack: an interim now, the exact pixels when the worker lands them. */
  const getSpatial = (
    spatial: NonNullable<typeof display>,
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract,
    key: string
  ): RasterSurface => {
    const source = entry.surface;
    const version = entry.version;
    const target: SpatialDisplayTarget = {
      adjustments,
      damageSince: (from) => options.damageSince?.(layerId, from) ?? null,
      hold: entry.hasPublishedPixels ? null : 'placeholder',
      key,
      liveVersion: () => (entry.surface === source ? entry.version : null),
      origin: { x: entry.rect.x, y: entry.rect.y },
      shown: () => cache.peek({ kind: 'adjustments', layerId, paramsKey: key, source, sourceVersion: version }),
      source,
      version,
    };
    const { height, width } = source;
    let rebuilt = false;
    const surface = cache.get({
      create: (reusable) => {
        const surface = reusable ?? backend.createSurface(width, height);
        const resized = surface.width !== width || surface.height !== height;
        if (resized) {
          surface.resize(width, height);
        }
        rebuilt = spatial.paintInterim(layerId, surface, reusable !== null && !resized, target);
        return surface;
      },
      kind: 'adjustments',
      layerId,
      paramsKey: key,
      source,
      sourceVersion: version,
    });
    spatial.request(layerId, target, rebuilt);
    return surface;
  };

  const get = (
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface | null => {
    if (isIdentityAdjustments(adjustments)) {
      dropSlot(layerId);
      cache.delete(layerId, 'adjustments');
      return null;
    }
    if (entry.surface.width <= 0 || entry.surface.height <= 0) {
      return null;
    }
    if (display && hasSpatialAdjustments(adjustments)) {
      return getSpatial(display, layerId, entry, adjustments!, adjustmentsKey(adjustments));
    }
    dropSlot(layerId);
    return getPerPixel(layerId, entry, adjustments);
  };

  const peekDisplayed = (
    layerId: string,
    entry: LayerCacheEntry,
    adjustments: CanvasAdjustmentsContract | undefined
  ): RasterSurface | null =>
    isIdentityAdjustments(adjustments)
      ? null
      : cache.peek({
          kind: 'adjustments',
          layerId,
          paramsKey: adjustmentsKey(adjustments),
          source: entry.surface,
          sourceVersion: entry.version,
        });

  const isExact = (layerId: string, entry: LayerCacheEntry, adjustments: CanvasAdjustmentsContract | undefined) =>
    !display ||
    !hasSpatialAdjustments(adjustments) ||
    display.isExact(layerId, { key: adjustmentsKey(adjustments), source: entry.surface, version: entry.version });

  return {
    byteSize: cache.byteSize,
    delete: (layerId) => {
      dropSlot(layerId);
      cache.delete(layerId, 'adjustments');
    },
    cancelSpatial: () => {
      for (const layerId of display?.cancel() ?? []) {
        cache.delete(layerId, 'adjustments');
      }
    },
    dispose: () => {
      dropSlots(display?.ownerIds() ?? []);
      cache.dispose();
    },
    retainSpatial: (keep) => dropSlots(display?.ownerIds().filter((layerId) => !keep(layerId)) ?? []),
    get,
    peek: (layerId, entry, adjustments) => {
      const displayed = peekDisplayed(layerId, entry, adjustments);
      return displayed && isExact(layerId, entry, adjustments) ? displayed : null;
    },
    peekDisplayed,
    size: cache.size,
    state: (layerId, entry, adjustments) =>
      display && hasSpatialAdjustments(adjustments)
        ? display.state(layerId, { key: adjustmentsKey(adjustments), source: entry.surface, version: entry.version })
        : 'exact',
  };
};
