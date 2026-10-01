import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';

import type { RasterSurface } from './raster';

export type DerivedSurfaceKind = 'adjustments' | 'mask-fill' | 'region-fill' | 'control-transparency';

export interface DerivedSurfaceRequest {
  layerId: string;
  sourceVersion: number;
  kind: DerivedSurfaceKind;
  paramsKey: string;
  source: RasterSurface;
  /**
   * Reuse `target` only for the same source. `reusableFromVersion` permits damage-based refresh; null requires a
   * wholesale build.
   */
  create(target: RasterSurface | null, reusableFromVersion: number | null): RasterSurface;
}

export interface DerivedSurfaceCache {
  get(request: DerivedSurfaceRequest): RasterSurface;
  delete(layerId: string, kind: DerivedSurfaceKind): void;
  deleteLayer(layerId: string): void;
  byteSize(): number;
  size(): number;
  /** The access clock; an entry read at or after a captured tick is in use by that frame. */
  tick(): number;
  /** Evicts entries least-recently-used first until within budget, keeping those `isRequired` accepts. */
  evict(budgetBytes: number, isRequired: (layerId: string, lastUsed: number) => boolean): string[];
  dispose(): void;
}

interface CacheEntry {
  readonly layerId: string;
  readonly kind: DerivedSurfaceKind;
  paramsKey: string;
  source: RasterSurface;
  sourceVersion: number;
  surface: RasterSurface;
  bytes: number;
  lastUsed: number;
}

const BYTES_PER_PIXEL = 4;
const entryKey = (layerId: string, kind: DerivedSurfaceKind): string => `${layerId}\u0000${kind}`;
const surfaceBytes = (surface: RasterSurface): number => surface.width * surface.height * BYTES_PER_PIXEL;

export const createDerivedSurfaceCache = (
  diagnostics?: CanvasDiagnostics,
  onBytesChange?: (bytes: number) => void
): DerivedSurfaceCache => {
  const entries = new Map<string, CacheEntry>();
  let tick = 0;
  let totalBytes = 0;

  const adjustBytes = (delta: number): void => {
    if (delta === 0) {
      return;
    }
    totalBytes += delta;
    onBytesChange?.(totalBytes);
  };

  const remove = (key: string, entry: CacheEntry): void => {
    entries.delete(key);
    adjustBytes(-entry.bytes);
  };

  const get = (request: DerivedSurfaceRequest): RasterSurface => {
    const key = entryKey(request.layerId, request.kind);
    const existing = entries.get(key);
    tick += 1;
    if (
      existing &&
      existing.source === request.source &&
      existing.sourceVersion === request.sourceVersion &&
      existing.paramsKey === request.paramsKey
    ) {
      diagnostics?.increment('derivedCacheHits');
      existing.lastUsed = tick;
      return existing.surface;
    }

    diagnostics?.increment('derivedCacheMisses');
    const canReuse = existing?.source === request.source;
    // Only a reusable surface built with the SAME parameters can be refreshed in
    // part; different parameters mean every pixel is wrong, not just the changed
    // ones.
    const reusableFromVersion = canReuse && existing.paramsKey === request.paramsKey ? existing.sourceVersion : null;
    const surface = request.create(canReuse ? existing.surface : null, reusableFromVersion);
    const bytes = surfaceBytes(surface);
    if (!canReuse) {
      diagnostics?.add('allocatedDerivedBytes', bytes);
    }
    adjustBytes(bytes - (existing?.bytes ?? 0));
    entries.set(key, {
      bytes,
      kind: request.kind,
      lastUsed: tick,
      layerId: request.layerId,
      paramsKey: request.paramsKey,
      source: request.source,
      sourceVersion: request.sourceVersion,
      surface,
    });
    return surface;
  };

  return {
    byteSize: () => totalBytes,
    delete: (layerId, kind) => {
      const key = entryKey(layerId, kind);
      const entry = entries.get(key);
      if (entry) {
        remove(key, entry);
      }
    },
    deleteLayer: (layerId) => {
      for (const [key, entry] of entries) {
        if (entry.layerId === layerId) {
          remove(key, entry);
        }
      }
    },
    dispose: () => {
      entries.clear();
      adjustBytes(-totalBytes);
    },
    evict: (budgetBytes, isRequired) => {
      const evicted: string[] = [];
      if (totalBytes <= budgetBytes) {
        return evicted;
      }
      const oldestFirst = [...entries.entries()]
        .filter(([, entry]) => !isRequired(entry.layerId, entry.lastUsed))
        .sort((a, b) => a[1].lastUsed - b[1].lastUsed);
      for (const [key, entry] of oldestFirst) {
        if (totalBytes <= budgetBytes) {
          break;
        }
        remove(key, entry);
        evicted.push(entry.layerId);
        diagnostics?.increment('derivedCacheEvictions');
      }
      return evicted;
    },
    get,
    size: () => entries.size,
    tick: () => tick,
  };
};
