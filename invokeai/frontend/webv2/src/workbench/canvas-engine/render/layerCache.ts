/** Callers protect pixels that cannot yet be reconstructed before eviction. */

import type { Rect } from '@workbench/canvas-engine/types';

import { intersect, isEmpty, union } from '@workbench/canvas-engine/math/rect';

import type { RasterBackend, RasterSurface } from './raster';

/** Hidden caches are evicted beyond this. */
export const DEFAULT_CACHE_BUDGET_BYTES = 512 * 1024 * 1024;

const BYTES_PER_PIXEL = 4;

/** Overrunning the window safely forces a full rebuild. */
const DAMAGE_TRAIL_LIMIT = 16;

/** `null` means everywhere. */
interface DamageStep {
  version: number;
  rect: Rect | null;
}
const READBACK_SURFACE_OPTIONS = { willReadFrequently: true } as const;

export interface LayerCacheEntry {
  readonly layerId: string;
  surface: RasterSurface;
  hasPublishedPixels: boolean;
  /** Layer-local: surface (sx,sy) is (rect.x+sx, rect.y+sy), origin possibly negative; zero size means empty. */
  rect: Rect;
  version: number;
  stale: boolean;
  lastUsed: number;
  /** Names the exact account-scoped face, so pixels rendered with a fallback family can be rejected later. */
  renderedFontFamily?: string;
}

/** Preparation may allocate and fail; install only does in-memory bookkeeping. */
export interface PreparedLayerCacheReplacement {
  readonly layerId: string;
  readonly rect: Rect;
  readonly surface: RasterSurface;
}

export interface LayerCacheStoreOptions {
  /** Fresh allocations stay silent. */
  onVersionChange?(layerId: string): void;
  onBytesChange?(bytes: number): void;
}

export interface LayerCacheEntryState {
  readonly surface: RasterSurface;
  readonly rect: Rect;
  readonly hasPublishedPixels: boolean;
  readonly stale: boolean;
  readonly version: number;
}

export interface LayerCacheStore {
  /** Does not change LRU order. */
  peek(layerId: string): LayerCacheEntry | undefined;
  layerIds(): string[];
  get(layerId: string): LayerCacheEntry | undefined;
  /** Origin-anchored; off-origin paint uses {@link getOrCreateRect} or {@link growToRect}. */
  getOrCreate(layerId: string, width: number, height: number): LayerCacheEntry;
  /** Existing surfaces stay untouched; rasterization and growth own resizing. */
  getOrCreateRect(layerId: string, rect: Rect): LayerCacheEntry;
  /** Preserves pixels and offset; never shrinks. */
  growToRect(layerId: string, rect: Rect): LayerCacheEntry;
  /** Keeps an emptied entry so undo can grow it; changed bounds bump version to fence stale raster jobs. */
  shrinkToRect(layerId: string, rect: Rect): LayerCacheEntry | undefined;
  prepareReplacement(layerId: string, rect: Rect, pixels: RasterSurface): PreparedLayerCacheReplacement;
  /** Does not allocate, resize or draw. */
  installReplacement(prepared: PreparedLayerCacheReplacement): LayerCacheEntry;
  publishRasterized(layerId: string, rect: Rect, pixels: RasterSurface, renderedFontFamily?: string): LayerCacheEntry;
  /** The surface is shared, not copied. */
  captureState(layerId: string): LayerCacheEntryState | null;
  /** Version included, so guards captured before a rolled-back transaction stay current. */
  restoreState(layerId: string, state: LayerCacheEntryState | null): void;
  /** Omitted damage means the whole surface changed. */
  publishPixels(layerId: string, damage?: Rect | null): LayerCacheEntry | undefined;
  /** Null (unknown writes, expired history or reallocation) requires a full refresh. */
  damageSince(layerId: string, version: number): Rect | null;
  invalidate(layerId: string): void;
  delete(layerId: string): void;
  /** 0 without a cache. */
  version(layerId: string): number;
  /** Equal values mean unchanged caches. */
  revision(): number;
  byteSize(): number;
  evict(isProtected: (layerId: string) => boolean, budgetBytes: number): string[];
  dispose(): void;
}

const surfaceBytes = (surface: RasterSurface): number => surface.width * surface.height * BYTES_PER_PIXEL;

export const createLayerCacheStore = (
  backend: RasterBackend,
  options: LayerCacheStoreOptions = {}
): LayerCacheStore => {
  const entries = new Map<string, LayerCacheEntry>();
  const accountedBytes = new Map<string, number>();
  let totalBytes = 0;

  const reportBytes = (): void => options.onBytesChange?.(totalBytes);

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
  const damageTrails = new Map<string, DamageStep[]>();

  /** A `null` rect is recorded, telling {@link damageSince} a write of unknown extent happened. */
  const recordDamage = (entry: LayerCacheEntry, rect: Rect | null): void => {
    const trail = damageTrails.get(entry.layerId) ?? [];
    trail.push({ rect, version: entry.version });
    if (trail.length > DAMAGE_TRAIL_LIMIT) {
      trail.splice(0, trail.length - DAMAGE_TRAIL_LIMIT);
    }
    damageTrails.set(entry.layerId, trail);
  };
  // Version floors outlive deletion and eviction, so recreated entries never reuse old versions.
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

  const initialVersion = (layerId: string): number => {
    const floor = versionFloors.get(layerId);
    return floor === undefined ? 0 : floor + 1;
  };

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
      touch(existing);
      return existing;
    }
    // The surface origin moved, so recorded surface-local rects are void.
    damageTrails.delete(layerId);
    // Blit old pixels at the new offset on the GPU; CPU readback would slow the stroke hot path.
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
    const clamped = isEmpty(cur) || isEmpty(requested) ? null : intersect(cur, requested);
    const newRect: Rect = clamped ?? { height: 0, width: 0, x: cur.x, y: cur.y };
    if (newRect.x === cur.x && newRect.y === cur.y && newRect.width === cur.width && newRect.height === cur.height) {
      touch(existing);
      return existing;
    }
    damageTrails.delete(layerId);
    const surface = existing.surface;
    if (isEmpty(newRect)) {
      surface.resize(0, 0);
    } else {
      // A negative offset is the crop: `drawImage` clips the rest in one GPU blit.
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
      // Keeps the extra create-then-notify version bump.
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
    // A gap in the trail means an unrecorded write.
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
