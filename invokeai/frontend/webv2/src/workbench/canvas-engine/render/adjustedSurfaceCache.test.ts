import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import type { LayerCacheEntry } from './layerCache';
import type { StubRasterSurface } from './raster.testStub';

import { createAdjustedSurfaceCache } from './adjustedSurfaceCache';
import { createTestStubRasterBackend } from './raster.testStub';

const BRIGHTEN: CanvasAdjustmentsContract = { brightness: 0.5, contrast: 0, saturation: 0 };
const CONTRAST: CanvasAdjustmentsContract = { brightness: 0, contrast: 0.5, saturation: 0 };

const makeEntry = (layerId: string, surface: StubRasterSurface, version = 0): LayerCacheEntry => ({
  hasPublishedPixels: true,
  lastUsed: 0,
  layerId,
  rect: { height: surface.height, width: surface.width, x: 0, y: 0 },
  stale: false,
  surface,
  version,
});

const drawImageCount = (surface: StubRasterSurface): number =>
  surface.callLog.filter((entry) => entry.op === 'drawImage').length;

describe('createAdjustedSurfaceCache', () => {
  it('returns null (and caches nothing) for identity adjustments', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));
    expect(cache.get('a', entry, undefined)).toBeNull();
    expect(cache.get('a', entry, { brightness: 0, contrast: 0, saturation: 0 })).toBeNull();
    expect(cache.size()).toBe(0);
  });

  it('builds and reuses the adjusted surface while version + adjustments are unchanged', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));

    const first = cache.get('a', entry, BRIGHTEN);
    expect(first).not.toBeNull();
    const stub = first as StubRasterSurface;
    expect(drawImageCount(stub)).toBe(1);

    // Same version + key: reuse the same surface, no rebuild.
    const second = cache.get('a', entry, BRIGHTEN);
    expect(second).toBe(first);
    expect(drawImageCount(stub)).toBe(1);
    expect(cache.size()).toBe(1);
    expect(cache.byteSize()).toBe(10 * 10 * 4);
  });

  it('rebuilds when the adjustments value changes', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));

    const first = cache.get('a', entry, BRIGHTEN) as StubRasterSurface;
    expect(drawImageCount(first)).toBe(1);
    // Different adjustments → rebuild (draws again into the reused surface).
    cache.get('a', entry, CONTRAST);
    expect(drawImageCount(first)).toBe(2);
  });

  it('invalidates only the edited layer when its cache version bumps', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entryA = makeEntry('a', backend.createSurface(10, 10), 0);
    const entryB = makeEntry('b', backend.createSurface(10, 10), 0);

    const surfaceA = cache.get('a', entryA, BRIGHTEN) as StubRasterSurface;
    const surfaceB = cache.get('b', entryB, BRIGHTEN) as StubRasterSurface;
    expect(drawImageCount(surfaceA)).toBe(1);
    expect(drawImageCount(surfaceB)).toBe(1);

    // Layer 'a' is edited: its cache version bumps → its adjusted surface rebuilds.
    const editedA = makeEntry('a', surfaceA, 1);
    const rebuiltA = cache.get('a', editedA, BRIGHTEN) as StubRasterSurface;
    expect(rebuiltA).not.toBe(surfaceA);
    expect(drawImageCount(rebuiltA)).toBe(1);

    // Layer 'b' is untouched (same version) → reused, no rebuild.
    const reusedB = cache.get('b', entryB, BRIGHTEN);
    expect(reusedB).toBe(surfaceB);
    expect(drawImageCount(surfaceB)).toBe(1);
  });

  it('drops a layer slot on delete and reverting to identity', () => {
    const backend = createTestStubRasterBackend();
    const cache = createAdjustedSurfaceCache(backend);
    const entry = makeEntry('a', backend.createSurface(10, 10));
    cache.get('a', entry, BRIGHTEN);
    expect(cache.size()).toBe(1);
    // Reverting to identity clears the slot.
    expect(cache.get('a', entry, undefined)).toBeNull();
    expect(cache.size()).toBe(0);
    // Re-cache then delete.
    cache.get('a', entry, BRIGHTEN);
    cache.delete('a');
    expect(cache.size()).toBe(0);
    expect(cache.byteSize()).toBe(0);
  });
});
