import { describe, expect, it, vi } from 'vitest';

import type { StubRasterSurface } from './raster.testStub';

import { createLayerCacheStore } from './layerCache';
import { createTestStubRasterBackend } from './raster.testStub';

describe('createLayerCacheStore', () => {
  it('notifies version changes for pixel publication, invalidation, replacement, growth, and deletion', () => {
    const backend = createTestStubRasterBackend();
    const onVersionChange = vi.fn();
    const store = createLayerCacheStore(backend, { onVersionChange });

    store.getOrCreate('layer-1', 10, 10);
    expect(onVersionChange).not.toHaveBeenCalled();

    store.publishPixels('layer-1');
    store.growToRect('layer-1', { height: 20, width: 20, x: 0, y: 0 });
    store.invalidate('layer-1');
    store.installReplacement(
      store.prepareReplacement('layer-1', { height: 5, width: 5, x: 0, y: 0 }, backend.createSurface(5, 5))
    );
    store.delete('layer-1');

    expect(onVersionChange).toHaveBeenCalledTimes(5);
    expect(onVersionChange).toHaveBeenCalledWith('layer-1');
  });

  it('preserves published pixels through invalidation so stale previews remain drawable', () => {
    const backend = createTestStubRasterBackend();
    const store = createLayerCacheStore(backend);
    const entry = store.installReplacement(
      store.prepareReplacement('layer-1', { height: 10, width: 10, x: 0, y: 0 }, backend.createSurface(10, 10))
    );

    store.invalidate('layer-1');

    expect(entry.stale).toBe(true);
    expect(entry.hasPublishedPixels).toBe(true);
  });

  it('resizes the surface (and marks stale) when the requested size changes', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.getOrCreate('layer-1', 100, 50);
    entry.stale = false;
    const resized = store.getOrCreate('layer-1', 200, 80);
    expect(resized).toBe(entry);
    expect(resized.surface.width).toBe(200);
    expect(resized.surface.height).toBe(80);
    expect(resized.stale).toBe(true);
  });

  it('evict removes least-recently-used unprotected caches until within budget, never protected ones', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    // Each surface is 100*100*4 = 40_000 bytes.
    store.getOrCreate('old-hidden', 100, 100);
    store.getOrCreate('visible', 100, 100);
    store.getOrCreate('new-hidden', 100, 100);
    store.get('new-hidden');

    const budget = 100 * 100 * 4 * 2;
    const evicted = store.evict((id) => id === 'visible', budget);

    expect(evicted).toEqual(['old-hidden']);
    expect(store.get('old-hidden')).toBeUndefined();
    expect(store.get('visible')).toBeDefined();
    expect(store.get('new-hidden')).toBeDefined();
    expect(store.byteSize()).toBeLessThanOrEqual(budget);
  });

  it('keeps a running byte total through allocation, resize, transfer and release', () => {
    const backend = createTestStubRasterBackend();
    const reported: number[] = [];
    const store = createLayerCacheStore(backend, { onBytesChange: (bytes) => reported.push(bytes) });

    store.getOrCreateRect('a', { height: 10, width: 10, x: 0, y: 0 });
    store.growToRect('a', { height: 10, width: 20, x: 0, y: 0 });
    store.shrinkToRect('a', { height: 5, width: 20, x: 0, y: 0 });
    const before = store.captureState('a');
    store.installReplacement(
      store.prepareReplacement('a', { height: 4, width: 4, x: 0, y: 0 }, backend.createSurface(4, 4))
    );
    expect(store.byteSize()).toBe(64);
    store.restoreState('a', before);
    expect(store.byteSize()).toBe(400);
    expect(store.peek('a')).toMatchObject({ rect: before!.rect, surface: before!.surface, version: before!.version });
    store.publishRasterized('b', { height: 2, width: 3, x: 0, y: 0 }, backend.createSurface(3, 2));
    store.delete('a');

    expect(reported).toEqual([400, 800, 400, 64, 400, 424, 24]);
    expect(store.byteSize()).toBe(24);
    store.dispose();
    expect(reported.at(-1)).toBe(0);
  });

  it('publishRasterized resizes, replaces and publishes the live pixels at a copied rect', () => {
    const backend = createTestStubRasterBackend();
    const store = createLayerCacheStore(backend);
    const entry = store.getOrCreateRect('a', { height: 4, width: 4, x: 0, y: 0 });
    const rect = { height: 8, width: 6, x: 1, y: 2 };

    const published = store.publishRasterized('a', rect, backend.createSurface(6, 8), 'face-1');

    expect(published).toBe(entry);
    expect([entry.surface.width, entry.surface.height]).toEqual([6, 8]);
    expect(entry.rect).toEqual(rect);
    expect(entry.rect).not.toBe(rect);
    expect(entry).toMatchObject({ hasPublishedPixels: true, renderedFontFamily: 'face-1', stale: false });
    expect(store.byteSize()).toBe(6 * 8 * 4);
  });
});

describe('getOrCreateRect (content-sized, off-origin placement)', () => {
  it('NEVER resizes or re-places an existing entry (a grown paint cache must survive)', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.getOrCreateRect('L', { height: 40, width: 60, x: 5, y: 5 });
    entry.stale = false;
    const again = store.getOrCreateRect('L', { height: 10, width: 10, x: 0, y: 0 });
    expect(again).toBe(entry);
    expect(again.rect).toEqual({ height: 40, width: 60, x: 5, y: 5 });
    expect(again.surface.width).toBe(60);
    expect(again.stale).toBe(false);
  });
});

describe('growToRect (paint caches grow with strokes)', () => {
  it('grows to the UNION of the current extent and the request, preserving old pixels at their shifted origin', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.growToRect('L', { height: 20, width: 20, x: 10, y: 10 });
    const surfaceBefore = entry.surface;

    const grown = store.growToRect('L', { height: 20, width: 20, x: -10, y: -10 });
    expect(grown).toBe(entry);
    // Union of [10,30)² and [-10,10)² = [-10,30)² → 40×40 at (-10,-10).
    expect(grown.rect).toEqual({ height: 40, width: 40, x: -10, y: -10 });
    expect(grown.surface).toBe(surfaceBefore);
    expect(grown.surface.width).toBe(40);
    expect(grown.surface.height).toBe(40);

    const log = (grown.surface as StubRasterSurface).callLog;
    expect(log.filter((e) => e.op === 'resizePreserving').map((e) => e.args)).toEqual([[40, 40, 20, 20]]);
    expect(log.map((e) => e.op)).not.toContain('getImageData');
    expect(log.map((e) => e.op)).not.toContain('putImageData');
  });

  it('is a no-op (no resize, no blit) when the current extent already covers the request', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.growToRect('L', { height: 100, width: 100, x: 0, y: 0 });
    const logLength = (entry.surface as StubRasterSurface).callLog.length;

    const again = store.growToRect('L', { height: 10, width: 10, x: 20, y: 20 });
    expect(again).toBe(entry);
    expect(again.rect).toEqual({ height: 100, width: 100, x: 0, y: 0 });
    expect((again.surface as StubRasterSurface).callLog.length).toBe(logLength);
  });

  it('adopts the target rect from an EMPTY extent without snapshotting stale pixels', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const empty = store.getOrCreateRect('L', { height: 0, width: 0, x: 0, y: 0 });
    empty.stale = false;

    const grown = store.growToRect('L', { height: 30, width: 30, x: 50, y: 60 });
    expect(grown).toBe(empty);
    expect(grown.rect).toEqual({ height: 30, width: 30, x: 50, y: 60 });
    const log = (grown.surface as StubRasterSurface).callLog;
    expect(log.some((e) => e.op === 'getImageData')).toBe(false);
    expect(log.some((e) => e.op === 'putImageData')).toBe(false);
  });
});

describe('shrinkToRect (paint caches trim back to their visible pixels)', () => {
  const published = () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.growToRect('L', { height: 40, width: 40, x: 10, y: 10 });
    store.publishPixels('L');
    return { entry, store };
  };

  it('crops to the target rect with a NEGATIVE blit offset, preserving surface identity', () => {
    const { entry, store } = published();
    const trimmed = store.shrinkToRect('L', { height: 10, width: 10, x: 25, y: 30 });

    expect(trimmed).toBe(entry);
    expect(trimmed?.rect).toEqual({ height: 10, width: 10, x: 25, y: 30 });
    expect(trimmed?.surface).toBe(entry.surface);
    // Old origin (10,10) minus new origin (25,30) = (-15,-20), in one GPU blit.
    const log = (entry.surface as StubRasterSurface).callLog;
    expect(log.filter((e) => e.op === 'resizePreserving').map((e) => e.args)).toEqual([[10, 10, -15, -20]]);
    expect(log.map((e) => e.op)).not.toContain('getImageData');
    expect(log.map((e) => e.op)).not.toContain('putImageData');
  });
});

describe('damageSince', () => {
  const seeded = () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    store.getOrCreate('a', 100, 100);
    return store;
  };
  const rect = (x: number, y: number, size = 10) => ({ height: size, width: size, x, y });

  it('unions the regions written since a version', () => {
    const store = seeded();
    const base = store.version('a');
    store.publishPixels('a', rect(10, 10));
    store.publishPixels('a', rect(50, 40));
    expect(store.damageSince('a', base)).toEqual({ height: 40, width: 50, x: 10, y: 10 });
  });

  it('gives up when any write in the span did not name its damage', () => {
    const store = seeded();
    const base = store.version('a');
    store.publishPixels('a', rect(10, 10));
    store.publishPixels('a', null);
    store.publishPixels('a', rect(60, 60));
    expect(store.damageSince('a', base)).toBeNull();
  });
});
