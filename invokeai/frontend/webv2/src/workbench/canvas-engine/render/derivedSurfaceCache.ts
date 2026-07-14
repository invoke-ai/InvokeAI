import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';

import type { RasterSurface } from './raster';

export type DerivedSurfaceKind = 'adjustments' | 'mask-fill' | 'control-transparency';

export interface DerivedSurfaceRequest {
  layerId: string;
  sourceVersion: number;
  kind: DerivedSurfaceKind;
  paramsKey: string;
  source: RasterSurface;
  create(target: RasterSurface | null): RasterSurface;
}

export interface DerivedSurfaceCache {
  get(request: DerivedSurfaceRequest): RasterSurface;
  delete(layerId: string, kind: DerivedSurfaceKind): void;
  deleteLayer(layerId: string): void;
  byteSize(): number;
  size(): number;
  evictToBudget(budgetBytes: number): string[];
  dispose(): void;
}

interface CacheEntry {
  readonly layerId: string;
  readonly kind: DerivedSurfaceKind;
  paramsKey: string;
  source: RasterSurface;
  sourceVersion: number;
  surface: RasterSurface;
  lastUsed: number;
}

const BYTES_PER_PIXEL = 4;
const entryKey = (layerId: string, kind: DerivedSurfaceKind): string => `${layerId}\u0000${kind}`;
const surfaceBytes = (surface: RasterSurface): number => surface.width * surface.height * BYTES_PER_PIXEL;

export const createDerivedSurfaceCache = (diagnostics?: CanvasDiagnostics): DerivedSurfaceCache => {
  const entries = new Map<string, CacheEntry>();
  let tick = 0;

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
    const surface = request.create(canReuse ? existing.surface : null);
    if (!canReuse) {
      diagnostics?.add('allocatedDerivedBytes', surfaceBytes(surface));
    }
    entries.set(key, {
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
    byteSize: () => {
      let bytes = 0;
      for (const entry of entries.values()) {
        bytes += surfaceBytes(entry.surface);
      }
      return bytes;
    },
    delete: (layerId, kind) => {
      entries.delete(entryKey(layerId, kind));
    },
    deleteLayer: (layerId) => {
      for (const [key, entry] of entries) {
        if (entry.layerId === layerId) {
          entries.delete(key);
        }
      }
    },
    dispose: () => entries.clear(),
    evictToBudget: (budgetBytes) => {
      let bytes = 0;
      for (const entry of entries.values()) {
        bytes += surfaceBytes(entry.surface);
      }
      const evicted: string[] = [];
      const oldestFirst = [...entries.entries()].sort((a, b) => a[1].lastUsed - b[1].lastUsed);
      for (const [key, entry] of oldestFirst) {
        if (bytes <= budgetBytes) {
          break;
        }
        entries.delete(key);
        bytes -= surfaceBytes(entry.surface);
        evicted.push(entry.layerId);
        diagnostics?.increment('derivedCacheEvictions');
      }
      return evicted;
    },
    get,
    size: () => entries.size,
  };
};
