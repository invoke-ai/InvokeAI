/**
 * Per-layer raster cache store.
 *
 * Each layer's rendered pixels are cached on their own {@link RasterSurface}
 * so the compositor can redraw the document without re-rasterizing every
 * layer each frame. Caches are always re-rasterizable, so eviction is safe:
 * a hidden layer's cache can be dropped to stay under a memory budget and
 * rebuilt on demand when it becomes visible again.
 *
 * All surface allocation flows through the injected {@link RasterBackend}
 * seam, so the store runs unchanged in node tests. Zero React, zero
 * import-time side effects.
 */

import type { Rect } from '@workbench/canvas-engine/types';

import type { RasterBackend, RasterSurface } from './raster';

/** Default cache budget: ~512 MB of surface pixels before hidden caches are evicted. */
export const DEFAULT_CACHE_BUDGET_BYTES = 512 * 1024 * 1024;

const BYTES_PER_PIXEL = 4;

/** A single layer's cache entry. */
export interface LayerCacheEntry {
  readonly layerId: string;
  /** The backing surface holding the layer's rasterized pixels. */
  surface: RasterSurface;
  /** True only after real pixels have been published into this allocation. */
  hasPublishedPixels: boolean;
  /**
   * The surface's content bounds in the layer's LOCAL coordinate space. The
   * surface holds `rect.width`×`rect.height` pixels; surface pixel `(sx, sy)`
   * maps to layer-local `(rect.x + sx, rect.y + sy)`. The origin can be negative
   * (a paint layer grown up/left of its start). The compositor draws the surface
   * at `transform × rect.origin`. An empty rect (0 width/height) marks an empty
   * layer (a brand-new or cleared paint layer) — safe to skip everywhere.
   */
  rect: Rect;
  /** Bumped every time the cache is invalidated; thumbnails/subscribers watch this. */
  version: number;
  /** True when the cached pixels are known to be out of date and need re-rasterizing. */
  stale: boolean;
  /** Monotonic access tick, used to order LRU eviction. */
  lastUsed: number;
}

/**
 * A fully-rasterized cache replacement that has not been published to the live
 * cache map yet. Preparation may allocate/draw and therefore may fail; install
 * performs only in-memory version/map bookkeeping.
 */
export interface PreparedLayerCacheReplacement {
  readonly layerId: string;
  readonly rect: Rect;
  readonly surface: RasterSurface;
}

export interface LayerCacheStoreOptions {
  /** Called after a cache identity or pixels change; fresh allocations stay silent. */
  onVersionChange?(layerId: string): void;
}

/** The imperative store returned by {@link createLayerCacheStore}. */
export interface LayerCacheStore {
  /** Returns an entry without changing LRU order. */
  peek(layerId: string): LayerCacheEntry | undefined;
  /** Returns the existing cache entry for a layer, or `undefined`. Touches LRU order. */
  get(layerId: string): LayerCacheEntry | undefined;
  /**
   * Returns the cache entry for a layer, creating (or resizing) its surface to
   * `width`x`height` at the LOCAL origin `(0, 0)`. The returned entry identity is
   * stable across calls that don't change the size, so callers can hold onto it
   * between frames. Use this for origin-anchored layers (image / shape / text /
   * gradient); paint caches, which can grow off-origin, use {@link getOrCreateRect}
   * / {@link growToRect}.
   */
  getOrCreate(layerId: string, width: number, height: number): LayerCacheEntry;
  /**
   * Like {@link getOrCreate} but places the surface at an arbitrary layer-local
   * `rect` origin (which may be negative). Creates the entry when absent; when it
   * already exists the surface is left untouched (the caller — a rasterizer or a
   * growth op — owns resizing), only ensuring existence. Used for paint layers
   * whose content rect is not origin-anchored.
   */
  getOrCreateRect(layerId: string, rect: Rect): LayerCacheEntry;
  /**
   * Grows (never shrinks) a layer's cache to cover `rect` in layer-local space,
   * preserving the existing pixels at their new offset. A no-op when `rect` is
   * already within the current extent. Creates the entry (surface sized to `rect`)
   * when absent. Returns the entry.
   */
  growToRect(layerId: string, rect: Rect): LayerCacheEntry;
  /** Clones `pixels` into a detached replacement without mutating the live cache. */
  prepareReplacement(layerId: string, rect: Rect, pixels: RasterSurface): PreparedLayerCacheReplacement;
  /** Publishes a detached replacement without allocating, resizing, or drawing. */
  installReplacement(prepared: PreparedLayerCacheReplacement): LayerCacheEntry;
  /** Marks directly-written pixels current, bumps the version, and notifies observers. */
  publishPixels(layerId: string): LayerCacheEntry | undefined;
  /** Marks a layer's cache stale and bumps its `version`. */
  invalidate(layerId: string): void;
  /** Drops a layer's cache entry entirely. */
  delete(layerId: string): void;
  /** The current `version` for a layer (0 if it has no cache yet). */
  version(layerId: string): number;
  /** Total bytes held across all cache surfaces (w*h*4 each). */
  byteSize(): number;
  /**
   * Evicts hidden-layer caches (those whose id is not in `visibleIds`) in
   * least-recently-used order until the total {@link byteSize} is within
   * `budgetBytes`. Visible layers are never evicted. Returns the evicted ids.
   */
  evictHidden(visibleIds: Iterable<string>, budgetBytes?: number): string[];
  /** Releases every cache entry. Cache surfaces are GC'd with the store. */
  dispose(): void;
}

const surfaceBytes = (surface: RasterSurface): number => surface.width * surface.height * BYTES_PER_PIXEL;

/** Creates a per-layer raster cache store backed by the given {@link RasterBackend}. */
export const createLayerCacheStore = (
  backend: RasterBackend,
  options: LayerCacheStoreOptions = {}
): LayerCacheStore => {
  const entries = new Map<string, LayerCacheEntry>();
  // Per-id version FLOOR: the highest version each layer id has ever reached,
  // retained across delete/recreate (delete, LRU eviction). A recreated entry
  // (undo→redo, transform bake, merge, evict→re-show) starts ABOVE this floor
  // rather than resetting to 0 — so version-keyed dependents (the adjusted-surface
  // cache, thumbnail state) never mistake fresh pixels for a stale version they
  // already have cached. Only ever grows; a plain number per id (negligible).
  const versionFloors = new Map<string, number>();
  let tick = 0;

  const notifyVersionChange = (layerId: string): void => {
    try {
      options.onVersionChange?.(layerId);
    } catch {
      // Cache mutation is already complete; observers cannot roll it back.
    }
  };

  const touch = (entry: LayerCacheEntry): void => {
    tick += 1;
    entry.lastUsed = tick;
  };

  /** The version a freshly-created entry for `layerId` must start at (monotonic). */
  const initialVersion = (layerId: string): number => {
    const floor = versionFloors.get(layerId);
    return floor === undefined ? 0 : floor + 1;
  };

  /** Records a to-be-dropped entry's version as the id's floor, so a recreate exceeds it. */
  const rememberFloor = (entry: LayerCacheEntry): void => {
    const prev = versionFloors.get(entry.layerId) ?? -1;
    if (entry.version > prev) {
      versionFloors.set(entry.layerId, entry.version);
    }
  };

  const get = (layerId: string): LayerCacheEntry | undefined => {
    const entry = entries.get(layerId);
    if (entry) {
      touch(entry);
    }
    return entry;
  };

  const peek = (layerId: string): LayerCacheEntry | undefined => entries.get(layerId);

  const getOrCreate = (layerId: string, width: number, height: number): LayerCacheEntry => {
    const existing = entries.get(layerId);
    if (existing) {
      const changed =
        existing.surface.width !== width ||
        existing.surface.height !== height ||
        existing.rect.x !== 0 ||
        existing.rect.y !== 0;
      const hadPublishedPixels = existing.hasPublishedPixels;
      if (existing.surface.width !== width || existing.surface.height !== height) {
        existing.surface.resize(width, height);
        existing.hasPublishedPixels = false;
        existing.stale = true;
      }
      // Origin-anchored: this variant always places the surface at (0, 0).
      existing.rect = { height, width, x: 0, y: 0 };
      touch(existing);
      if (changed && hadPublishedPixels) {
        existing.version += 1;
        notifyVersionChange(layerId);
      }
      return existing;
    }
    const entry: LayerCacheEntry = {
      hasPublishedPixels: false,
      lastUsed: 0,
      layerId,
      rect: { height, width, x: 0, y: 0 },
      stale: true,
      surface: backend.createSurface(width, height),
      version: initialVersion(layerId),
    };
    touch(entry);
    entries.set(layerId, entry);
    return entry;
  };

  const getOrCreateRect = (layerId: string, rect: Rect): LayerCacheEntry => {
    const existing = entries.get(layerId);
    if (existing) {
      touch(existing);
      return existing;
    }
    const width = Math.max(0, Math.round(rect.width));
    const height = Math.max(0, Math.round(rect.height));
    const entry: LayerCacheEntry = {
      hasPublishedPixels: false,
      lastUsed: 0,
      layerId,
      rect: { height, width, x: rect.x, y: rect.y },
      stale: true,
      surface: backend.createSurface(width, height),
      version: initialVersion(layerId),
    };
    touch(entry);
    entries.set(layerId, entry);
    return entry;
  };

  const growToRect = (layerId: string, rect: Rect): LayerCacheEntry => {
    const existing = entries.get(layerId);
    const targetRect: Rect = {
      height: Math.max(0, Math.round(rect.height)),
      width: Math.max(0, Math.round(rect.width)),
      x: Math.round(rect.x),
      y: Math.round(rect.y),
    };
    if (!existing) {
      const entry: LayerCacheEntry = {
        hasPublishedPixels: false,
        lastUsed: 0,
        layerId,
        rect: targetRect,
        stale: false,
        surface: backend.createSurface(targetRect.width, targetRect.height),
        version: initialVersion(layerId),
      };
      touch(entry);
      entries.set(layerId, entry);
      return entry;
    }
    const cur = existing.rect;
    const curEmpty = cur.width <= 0 || cur.height <= 0;
    // Union of the current extent and the requested rect (in layer-local space).
    const minX = curEmpty ? targetRect.x : Math.min(cur.x, targetRect.x);
    const minY = curEmpty ? targetRect.y : Math.min(cur.y, targetRect.y);
    const maxX = curEmpty
      ? targetRect.x + targetRect.width
      : Math.max(cur.x + cur.width, targetRect.x + targetRect.width);
    const maxY = curEmpty
      ? targetRect.y + targetRect.height
      : Math.max(cur.y + cur.height, targetRect.y + targetRect.height);
    const newRect: Rect = { height: maxY - minY, width: maxX - minX, x: minX, y: minY };
    if (newRect.x === cur.x && newRect.y === cur.y && newRect.width === cur.width && newRect.height === cur.height) {
      // Already covers the request — no realloc.
      touch(existing);
      return existing;
    }
    // Snapshot the old pixels BEFORE resizing (resize clears the backing canvas),
    // then blit them back at their new offset within the grown surface. Skip the
    // copy when the old extent was empty (nothing to preserve).
    const surface = existing.surface;
    let snapshot: ImageData | null = null;
    if (!curEmpty && cur.width > 0 && cur.height > 0) {
      snapshot = surface.ctx.getImageData(0, 0, cur.width, cur.height);
    }
    surface.resize(newRect.width, newRect.height);
    if (snapshot) {
      surface.ctx.putImageData(snapshot, cur.x - newRect.x, cur.y - newRect.y);
    }
    existing.rect = newRect;
    touch(existing);
    if (existing.hasPublishedPixels) {
      existing.version += 1;
      notifyVersionChange(layerId);
    }
    return existing;
  };

  const prepareReplacement = (layerId: string, rect: Rect, pixels: RasterSurface): PreparedLayerCacheReplacement => {
    const normalizedRect: Rect = {
      height: Math.max(0, Math.round(rect.height)),
      width: Math.max(0, Math.round(rect.width)),
      x: rect.x,
      y: rect.y,
    };
    const surface = backend.createSurface(normalizedRect.width, normalizedRect.height);
    if (normalizedRect.width > 0 && normalizedRect.height > 0) {
      surface.ctx.clearRect(0, 0, normalizedRect.width, normalizedRect.height);
      surface.ctx.drawImage(pixels.canvas, 0, 0);
    }
    return { layerId, rect: normalizedRect, surface };
  };

  const installReplacement = (prepared: PreparedLayerCacheReplacement): LayerCacheEntry => {
    const existing = entries.get(prepared.layerId);
    if (existing) {
      rememberFloor(existing);
    }
    const entry: LayerCacheEntry = {
      hasPublishedPixels: true,
      lastUsed: 0,
      layerId: prepared.layerId,
      rect: prepared.rect,
      stale: false,
      surface: prepared.surface,
      // Directly-published pixels receive the same extra version bump that the
      // old create-then-notify path applied, while remaining monotonic on swap.
      version: initialVersion(prepared.layerId) + 1,
    };
    touch(entry);
    entries.set(prepared.layerId, entry);
    notifyVersionChange(prepared.layerId);
    return entry;
  };

  const publishPixels = (layerId: string): LayerCacheEntry | undefined => {
    const entry = entries.get(layerId);
    if (!entry) {
      return undefined;
    }
    entry.hasPublishedPixels = true;
    entry.stale = false;
    entry.version += 1;
    notifyVersionChange(layerId);
    return entry;
  };

  const invalidate = (layerId: string): void => {
    const entry = entries.get(layerId);
    if (entry) {
      entry.version += 1;
      entry.stale = true;
      notifyVersionChange(layerId);
    }
  };

  const del = (layerId: string): void => {
    const entry = entries.get(layerId);
    if (entry) {
      rememberFloor(entry);
      entries.delete(layerId);
      notifyVersionChange(layerId);
    }
  };

  const version = (layerId: string): number => entries.get(layerId)?.version ?? 0;

  const byteSize = (): number => {
    let total = 0;
    for (const entry of entries.values()) {
      total += surfaceBytes(entry.surface);
    }
    return total;
  };

  const evictHidden = (visibleIds: Iterable<string>, budgetBytes: number = DEFAULT_CACHE_BUDGET_BYTES): string[] => {
    const visible = new Set(visibleIds);
    const evicted: string[] = [];
    let total = byteSize();
    if (total <= budgetBytes) {
      return evicted;
    }
    // Hidden entries, least-recently-used first.
    const candidates = [...entries.values()]
      .filter((entry) => !visible.has(entry.layerId))
      .sort((a, b) => a.lastUsed - b.lastUsed);
    for (const entry of candidates) {
      if (total <= budgetBytes) {
        break;
      }
      // Retain the version floor so a re-shown (re-rasterized) layer resumes ABOVE
      // its pre-eviction version, not from 0 — otherwise the adjusted-surface /
      // thumbnail caches would serve stale pixels keyed to the recycled version.
      rememberFloor(entry);
      entries.delete(entry.layerId);
      evicted.push(entry.layerId);
      total -= surfaceBytes(entry.surface);
      notifyVersionChange(entry.layerId);
    }
    return evicted;
  };

  const dispose = (): void => {
    entries.clear();
  };

  return {
    byteSize,
    delete: del,
    dispose,
    evictHidden,
    get,
    getOrCreate,
    getOrCreateRect,
    growToRect,
    installReplacement,
    invalidate,
    peek,
    prepareReplacement,
    publishPixels,
    version,
  };
};
