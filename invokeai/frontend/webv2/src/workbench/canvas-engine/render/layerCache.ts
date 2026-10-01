/**
 * Per-layer raster surfaces avoid repeated rasterization during compositing. Allocation uses injected {@link
 * RasterBackend}; callers protect pixels that cannot yet be reconstructed before eviction.
 */

import type { Rect } from '@workbench/canvas-engine/types';

import { intersect, isEmpty, union } from '@workbench/canvas-engine/math/rect';

import type { RasterBackend, RasterSurface } from './raster';

/** Default cache budget: ~512 MB of surface pixels before hidden caches are evicted. */
export const DEFAULT_CACHE_BUDGET_BYTES = 512 * 1024 * 1024;

const BYTES_PER_PIXEL = 4;

/** Bound damage history between derived-surface refreshes. Overrunning the window safely forces a full rebuild. */
const DAMAGE_TRAIL_LIMIT = 16;

/** One recorded write: the version it produced, and where it landed (`null` = everywhere). */
interface DamageStep {
  version: number;
  rect: Rect | null;
}
const READBACK_SURFACE_OPTIONS = { willReadFrequently: true } as const;

/** A single layer's cache entry. */
export interface LayerCacheEntry {
  readonly layerId: string;
  /** The backing surface holding the layer's rasterized pixels. */
  surface: RasterSurface;
  /** True only after real pixels have been published into this allocation. */
  hasPublishedPixels: boolean;
  /**
   * Local content bounds map surface (sx,sy) to (rect.x+sx,rect.y+sy), allowing negative origins. Composite at
   * transform*origin; zero-sized rects represent empty layers.
   */
  rect: Rect;
  /** Bumped every time the cache is invalidated; thumbnails/subscribers watch this. */
  version: number;
  /** True when the cached pixels are known to be out of date and need re-rasterizing. */
  stale: boolean;
  /** Monotonic access tick, used to order LRU eviction. */
  lastUsed: number;
  /**
   * Browser family alias used for the most recent text rasterization. The alias
   * identifies the exact account-scoped face/variation in the Canvas font
   * runtime, so output can reject pixels rendered with a fallback family after
   * the real face becomes available.
   */
  renderedFontFamily?: string;
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
  /** Receives the running byte total after every allocation, resize, transfer and release. */
  onBytesChange?(bytes: number): void;
}

/** A complete entry state, captured before a transaction and reinstated if it rolls back. */
export interface LayerCacheEntryState {
  readonly surface: RasterSurface;
  readonly rect: Rect;
  readonly hasPublishedPixels: boolean;
  readonly stale: boolean;
  readonly version: number;
}

/** The imperative store returned by {@link createLayerCacheStore}. */
export interface LayerCacheStore {
  /** Returns an entry without changing LRU order. */
  peek(layerId: string): LayerCacheEntry | undefined;
  /** Ids of every layer that holds a cache entry. */
  layerIds(): string[];
  /** Returns the existing cache entry for a layer, or `undefined`. Touches LRU order. */
  get(layerId: string): LayerCacheEntry | undefined;
  /**
   * Create/resize origin-anchored caches with stable entry identity while size matches. Off-origin paint uses
   * {@link getOrCreateRect} or {@link growToRect}.
   */
  getOrCreate(layerId: string, width: number, height: number): LayerCacheEntry;
  /**
   * Ensure a cache at an arbitrary local rect, including negative origins. Existing surfaces remain untouched;
   * rasterization/growth owns resizing.
   */
  getOrCreateRect(layerId: string, rect: Rect): LayerCacheEntry;
  /**
   * Grow to include a local rect while preserving pixels and offset; never shrink. Creates missing entries and
   * reuses already-covering ones.
   */
  growToRect(layerId: string, rect: Rect): LayerCacheEntry;
  /**
   * Intersect-crop local bounds, retaining the entry even when empty so undo can grow it again. Changed bounds
   * always bump version to fence stale raster jobs. Equal bounds are a no-op; missing caches return undefined.
   */
  shrinkToRect(layerId: string, rect: Rect): LayerCacheEntry | undefined;
  /** Clones `pixels` into a detached replacement without mutating the live cache. */
  prepareReplacement(layerId: string, rect: Rect, pixels: RasterSurface): PreparedLayerCacheReplacement;
  /** Publishes a detached replacement without allocating, resizing, or drawing. */
  installReplacement(prepared: PreparedLayerCacheReplacement): LayerCacheEntry;
  /** Copies rasterized `pixels` into the live entry at `rect` and publishes them. */
  publishRasterized(layerId: string, rect: Rect, pixels: RasterSurface, renderedFontFamily?: string): LayerCacheEntry;
  /** The entry's current state, or null when absent; the surface is shared, not copied. */
  captureState(layerId: string): LayerCacheEntryState | null;
  /**
   * Reinstates a captured state exactly (null deletes the entry), version included, so guards captured before a
   * rolled-back transaction stay current. The caller discards artifacts derived from versions issued since.
   */
  restoreState(layerId: string, state: LayerCacheEntryState | null): void;
  /**
   * Publish direct writes as current, bump version and notify. Optional surface-local damage enables partial
   * derived refresh; omission means the whole surface changed.
   */
  publishPixels(layerId: string, damage?: Rect | null): LayerCacheEntry | undefined;
  /**
   * Union damage since version, or null for unknown writes, expired history or reallocation. Null requires full
   * refresh.
   */
  damageSince(layerId: string, version: number): Rect | null;
  /** Marks a layer's cache stale and bumps its `version`. */
  invalidate(layerId: string): void;
  /** Drops a layer's cache entry entirely. */
  delete(layerId: string): void;
  /** The current `version` for a layer (0 if it has no cache yet). */
  version(layerId: string): number;
  /** Advances whenever any cache's published pixels, bounds or identity change; equal values mean unchanged caches. */
  revision(): number;
  /** Total bytes held across all cache surfaces (w*h*4 each). */
  byteSize(): number;
  /** Evicts unprotected entries least-recently-used first until within `budgetBytes`. Returns evicted ids. */
  evict(isProtected: (layerId: string) => boolean, budgetBytes: number): string[];
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
  const accountedBytes = new Map<string, number>();
  let totalBytes = 0;

  const reportBytes = (): void => options.onBytesChange?.(totalBytes);

  /** Brings the running total in line with the entry's current surface. */
  const account = (entry: LayerCacheEntry): void => {
    const next = surfaceBytes(entry.surface);
    const previous = accountedBytes.get(entry.layerId) ?? 0;
    accountedBytes.set(entry.layerId, next);
    if (next !== previous) {
      totalBytes += next - previous;
      reportBytes();
    }
  };

  const unaccount = (layerId: string): void => {
    const previous = accountedBytes.get(layerId);
    if (previous === undefined) {
      return;
    }
    accountedBytes.delete(layerId);
    if (previous !== 0) {
      totalBytes -= previous;
      reportBytes();
    }
  };
  /** Per-layer trail of recent writes, oldest first. See {@link DAMAGE_TRAIL_LIMIT}. */
  const damageTrails = new Map<string, DamageStep[]>();

  /**
   * Appends a write to a layer's damage trail. A `null` rect is recorded rather
   * than dropped: it is what tells {@link damageSince} that a write of unknown
   * extent happened, so a derived surface spanning it must rebuild in full.
   */
  const recordDamage = (entry: LayerCacheEntry, rect: Rect | null): void => {
    const trail = damageTrails.get(entry.layerId) ?? [];
    trail.push({ rect, version: entry.version });
    if (trail.length > DAMAGE_TRAIL_LIMIT) {
      trail.splice(0, trail.length - DAMAGE_TRAIL_LIMIT);
    }
    damageTrails.set(entry.layerId, trail);
  };
  // Retain per-id version floors across deletion/eviction. Recreated entries start above prior versions so derived
  // caches cannot mistake new pixels for old ones.
  const versionFloors = new Map<string, number>();
  let tick = 0;
  let revision = 0;

  const notifyVersionChange = (layerId: string): void => {
    revision += 1;
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
        damageTrails.delete(layerId);
        existing.hasPublishedPixels = false;
        existing.stale = true;
      }
      // Origin-anchored: this variant always places the surface at (0, 0).
      existing.rect = { height, width, x: 0, y: 0 };
      touch(existing);
      account(existing);
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
      surface: backend.createSurface(width, height, READBACK_SURFACE_OPTIONS),
      version: initialVersion(layerId),
    };
    touch(entry);
    entries.set(layerId, entry);
    account(entry);
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
      surface: backend.createSurface(width, height, READBACK_SURFACE_OPTIONS),
      version: initialVersion(layerId),
    };
    touch(entry);
    entries.set(layerId, entry);
    account(entry);
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
        surface: backend.createSurface(targetRect.width, targetRect.height, READBACK_SURFACE_OPTIONS),
        version: initialVersion(layerId),
      };
      touch(entry);
      entries.set(layerId, entry);
      account(entry);
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
    // The surface origin moves with the grown rect, so every surface-local rect
    // recorded so far is void. Dropping the trail makes derived surfaces rebuild
    // wholesale for one frame rather than refresh the wrong pixels.
    damageTrails.delete(layerId);
    // Grow by adopting a fresh backing store and blitting old pixels at the new offset. Ordinary resize clears,
    // and CPU readback/upload would add work on the stroke hot path.
    const surface = existing.surface;
    if (!curEmpty && cur.width > 0 && cur.height > 0) {
      surface.resizePreserving(newRect.width, newRect.height, cur.x - newRect.x, cur.y - newRect.y);
    } else {
      surface.resize(newRect.width, newRect.height);
    }
    existing.rect = newRect;
    touch(existing);
    account(existing);
    if (existing.hasPublishedPixels) {
      existing.version += 1;
      notifyVersionChange(layerId);
    }
    return existing;
  };

  const shrinkToRect = (layerId: string, rect: Rect): LayerCacheEntry | undefined => {
    const existing = entries.get(layerId);
    if (!existing) {
      return undefined;
    }
    const cur = existing.rect;
    const requested: Rect = {
      height: Math.max(0, Math.round(rect.height)),
      width: Math.max(0, Math.round(rect.width)),
      x: Math.round(rect.x),
      y: Math.round(rect.y),
    };
    // Intersect rather than adopt, so a request reaching outside is clamped inward.
    const clamped = isEmpty(cur) || isEmpty(requested) ? null : intersect(cur, requested);
    const newRect: Rect = clamped ?? { height: 0, width: 0, x: cur.x, y: cur.y };
    if (newRect.x === cur.x && newRect.y === cur.y && newRect.width === cur.width && newRect.height === cur.height) {
      touch(existing);
      return existing;
    }
    // The origin moved, so recorded surface-local rects are void (as in `growToRect`).
    damageTrails.delete(layerId);
    const surface = existing.surface;
    if (isEmpty(newRect)) {
      surface.resize(0, 0);
    } else {
      // A negative offset IS the crop: `drawImage` clips what falls outside the
      // smaller surface, so this is one GPU blit with no CPU round trip.
      surface.resizePreserving(newRect.width, newRect.height, cur.x - newRect.x, cur.y - newRect.y);
    }
    existing.rect = newRect;
    touch(existing);
    account(existing);
    existing.version += 1;
    notifyVersionChange(layerId);
    return existing;
  };

  const prepareReplacement = (layerId: string, rect: Rect, pixels: RasterSurface): PreparedLayerCacheReplacement => {
    const normalizedRect: Rect = {
      height: Math.max(0, Math.round(rect.height)),
      width: Math.max(0, Math.round(rect.width)),
      x: rect.x,
      y: rect.y,
    };
    const surface = backend.createSurface(normalizedRect.width, normalizedRect.height, READBACK_SURFACE_OPTIONS);
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
      // Direct publication preserves the extra create-then-notify version bump and monotonic swaps.
      version: initialVersion(prepared.layerId) + 1,
    };
    touch(entry);
    damageTrails.delete(prepared.layerId);
    entries.set(prepared.layerId, entry);
    account(entry);
    notifyVersionChange(prepared.layerId);
    return entry;
  };

  const publishPixels = (layerId: string, damage?: Rect | null): LayerCacheEntry | undefined => {
    const entry = entries.get(layerId);
    if (!entry) {
      return undefined;
    }
    entry.hasPublishedPixels = true;
    entry.stale = false;
    entry.version += 1;
    account(entry);
    recordDamage(entry, damage ?? null);
    notifyVersionChange(layerId);
    return entry;
  };

  const publishRasterized = (
    layerId: string,
    rect: Rect,
    pixels: RasterSurface,
    renderedFontFamily?: string
  ): LayerCacheEntry => {
    const entry = entries.get(layerId) ?? getOrCreateRect(layerId, rect);
    if (entry.surface.width !== rect.width || entry.surface.height !== rect.height) {
      entry.surface.resize(rect.width, rect.height);
    }
    const ctx = entry.surface.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!isEmpty(rect)) {
      ctx.drawImage(pixels.canvas, 0, 0);
    }
    entry.renderedFontFamily = renderedFontFamily;
    entry.rect = { ...rect };
    damageTrails.delete(layerId);
    publishPixels(layerId);
    return entry;
  };

  const captureState = (layerId: string): LayerCacheEntryState | null => {
    const entry = entries.get(layerId);
    return entry
      ? {
          hasPublishedPixels: entry.hasPublishedPixels,
          rect: { ...entry.rect },
          stale: entry.stale,
          surface: entry.surface,
          version: entry.version,
        }
      : null;
  };

  const restoreState = (layerId: string, state: LayerCacheEntryState | null): void => {
    if (!state) {
      del(layerId);
      return;
    }
    const restored: LayerCacheEntry = entries.get(layerId) ?? {
      ...state,
      lastUsed: 0,
      layerId,
      rect: { ...state.rect },
    };
    Object.assign(restored, { ...state, rect: { ...state.rect } });
    touch(restored);
    damageTrails.delete(layerId);
    entries.set(layerId, restored);
    account(restored);
    notifyVersionChange(layerId);
  };

  const damageSince = (layerId: string, version: number): Rect | null => {
    const entry = entries.get(layerId);
    if (!entry || version >= entry.version) {
      return null;
    }
    const trail = damageTrails.get(layerId);
    // The trail must reach back far enough to account for EVERY version since
    // the caller's: a gap means an unrecorded write, so nothing can be assumed.
    if (!trail || trail.length === 0 || trail[0]!.version > version + 1) {
      return null;
    }
    let accumulated: Rect | null = null;
    for (const step of trail) {
      if (step.version <= version) {
        continue;
      }
      if (!step.rect) {
        return null;
      }
      accumulated = accumulated ? union(accumulated, step.rect) : step.rect;
    }
    return accumulated;
  };

  const invalidate = (layerId: string): void => {
    const entry = entries.get(layerId);
    if (entry) {
      damageTrails.delete(layerId);
      entry.version += 1;
      entry.stale = true;
      notifyVersionChange(layerId);
    }
  };

  function del(layerId: string): void {
    const entry = entries.get(layerId);
    if (entry) {
      rememberFloor(entry);
      entries.delete(layerId);
      damageTrails.delete(layerId);
      unaccount(layerId);
      notifyVersionChange(layerId);
    }
  }

  const version = (layerId: string): number => entries.get(layerId)?.version ?? 0;

  const evict = (isProtected: (layerId: string) => boolean, budgetBytes: number): string[] => {
    const evicted: string[] = [];
    if (totalBytes <= budgetBytes) {
      return evicted;
    }
    const candidates = [...entries.values()]
      .filter((entry) => !isProtected(entry.layerId))
      .sort((a, b) => a.lastUsed - b.lastUsed);
    for (const entry of candidates) {
      if (totalBytes <= budgetBytes) {
        break;
      }
      // Retain version floors across eviction to prevent stale adjusted surfaces and thumbnails after recreation.
      rememberFloor(entry);
      entries.delete(entry.layerId);
      damageTrails.delete(entry.layerId);
      unaccount(entry.layerId);
      evicted.push(entry.layerId);
      notifyVersionChange(entry.layerId);
    }
    return evicted;
  };

  const dispose = (): void => {
    for (const entry of entries.values()) {
      rememberFloor(entry);
    }
    revision += 1;
    entries.clear();
    damageTrails.clear();
    accountedBytes.clear();
    if (totalBytes !== 0) {
      totalBytes = 0;
      reportBytes();
    }
  };

  return {
    byteSize: () => totalBytes,
    captureState,
    delete: del,
    dispose,
    evict,
    get,
    getOrCreate,
    getOrCreateRect,
    growToRect,
    installReplacement,
    damageSince,
    invalidate,
    layerIds: () => [...entries.keys()],
    peek,
    prepareReplacement,
    publishPixels,
    publishRasterized,
    restoreState,
    revision: () => revision,
    shrinkToRect,
    version,
  };
};
