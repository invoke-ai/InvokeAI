import { describe, expect, it, vi } from 'vitest';

import type { StubRasterSurface } from './raster.testStub';

import { createLayerCacheStore, DEFAULT_CACHE_BUDGET_BYTES } from './layerCache';
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

  it('does not notify for ordinary fresh or stale allocation', () => {
    const onVersionChange = vi.fn();
    const store = createLayerCacheStore(createTestStubRasterBackend(), { onVersionChange });

    store.getOrCreate('origin', 10, 10);
    store.getOrCreateRect('rect', { height: 10, width: 10, x: 2, y: 3 });
    store.growToRect('grown', { height: 10, width: 10, x: 2, y: 3 });

    expect(onVersionChange).not.toHaveBeenCalled();
  });

  it('returns a stable entry identity for the same layer + size', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const a = store.getOrCreate('layer-1', 100, 50);
    const b = store.getOrCreate('layer-1', 100, 50);
    expect(b).toBe(a);
    expect(store.get('layer-1')).toBe(a);
  });

  it('marks new allocations as never published', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());

    expect(store.getOrCreate('origin', 10, 10).hasPublishedPixels).toBe(false);
    expect(store.getOrCreateRect('rect', { height: 10, width: 10, x: 2, y: 3 }).hasPublishedPixels).toBe(false);
    expect(store.growToRect('grown', { height: 10, width: 10, x: 2, y: 3 }).hasPublishedPixels).toBe(false);
  });

  it('marks installed raster replacements as published', () => {
    const backend = createTestStubRasterBackend();
    const store = createLayerCacheStore(backend);
    const prepared = store.prepareReplacement(
      'layer-1',
      { height: 10, width: 10, x: 0, y: 0 },
      backend.createSurface(10, 10)
    );

    expect(store.installReplacement(prepared).hasPublishedPixels).toBe(true);
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

  it('resets publication readiness when a resize destroys the old pixels', () => {
    const backend = createTestStubRasterBackend();
    const store = createLayerCacheStore(backend);
    const entry = store.installReplacement(
      store.prepareReplacement('layer-1', { height: 10, width: 10, x: 0, y: 0 }, backend.createSurface(10, 10))
    );

    store.getOrCreate('layer-1', 20, 20);

    expect(entry.hasPublishedPixels).toBe(false);
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

  it('invalidate bumps the version and marks the cache stale', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.getOrCreate('layer-1', 10, 10);
    entry.stale = false;
    expect(store.version('layer-1')).toBe(0);

    store.invalidate('layer-1');
    expect(store.version('layer-1')).toBe(1);
    expect(entry.version).toBe(1);
    expect(entry.stale).toBe(true);

    store.invalidate('layer-1');
    expect(store.version('layer-1')).toBe(2);
  });

  it('version is 0 for an unknown layer', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    expect(store.version('nope')).toBe(0);
  });

  it('a recreated entry (after delete) resumes ABOVE the old version, never resetting to 0', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const first = store.getOrCreate('a', 10, 10);
    store.invalidate('a');
    store.invalidate('a');
    expect(first.version).toBe(2);

    // Delete + recreate (undo→redo / transform bake / merge / evict→re-show): the
    // fresh entry must start above 2 so version-keyed caches (adjusted surface,
    // thumbnails) can't mistake its new pixels for a version they already hold.
    store.delete('a');
    const recreated = store.getOrCreate('a', 10, 10);
    expect(recreated.version).toBeGreaterThan(2);
  });

  it('an evicted-then-re-shown entry also resumes above its pre-eviction version', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    store.getOrCreate('hidden', 100, 100);
    store.invalidate('hidden'); // version 1
    const budget = 100 * 100 * 4; // room for one surface
    store.getOrCreate('visible', 100, 100);
    const evicted = store.evictHidden(['visible'], budget);
    expect(evicted).toContain('hidden');

    const reshown = store.getOrCreate('hidden', 100, 100);
    expect(reshown.version).toBeGreaterThan(1);
  });

  it('byteSize sums w*h*4 across all cache surfaces', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    store.getOrCreate('a', 100, 100); // 40_000
    store.getOrCreate('b', 10, 10); // 400
    expect(store.byteSize()).toBe(100 * 100 * 4 + 10 * 10 * 4);
  });

  it('delete drops a cache entry', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    store.getOrCreate('a', 10, 10);
    store.delete('a');
    expect(store.get('a')).toBeUndefined();
    expect(store.byteSize()).toBe(0);
  });

  it('evictHidden does nothing when already within budget', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    store.getOrCreate('a', 10, 10);
    const evicted = store.evictHidden([], DEFAULT_CACHE_BUDGET_BYTES);
    expect(evicted).toEqual([]);
    expect(store.get('a')).toBeDefined();
  });

  it('evictHidden evicts least-recently-used hidden caches until within budget, never visible ones', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    // Each surface is 100*100*4 = 40_000 bytes.
    store.getOrCreate('old-hidden', 100, 100);
    store.getOrCreate('visible', 100, 100);
    store.getOrCreate('new-hidden', 100, 100);
    // Touch 'new-hidden' so 'old-hidden' is the LRU among hidden entries.
    store.get('new-hidden');

    // Budget allows exactly two surfaces; three exist -> one hidden must go.
    const budget = 100 * 100 * 4 * 2;
    const evicted = store.evictHidden(['visible'], budget);

    expect(evicted).toEqual(['old-hidden']);
    expect(store.get('old-hidden')).toBeUndefined();
    expect(store.get('visible')).toBeDefined();
    expect(store.get('new-hidden')).toBeDefined();
    expect(store.byteSize()).toBeLessThanOrEqual(budget);
  });

  it('evictHidden will not evict visible caches even if still over budget', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    store.getOrCreate('visible', 100, 100);
    const evicted = store.evictHidden(['visible'], 1);
    expect(evicted).toEqual([]);
    expect(store.get('visible')).toBeDefined();
  });

  it('dispose clears caches', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    store.getOrCreate('a', 10, 10);

    store.dispose();
    expect(store.byteSize()).toBe(0);
  });
});

describe('getOrCreateRect (content-sized, off-origin placement)', () => {
  it('creates a new surface placed at an off-origin (negative) layer-local rect', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.getOrCreateRect('L', { height: 40, width: 60, x: -15, y: -25 });
    expect(entry.rect).toEqual({ height: 40, width: 60, x: -15, y: -25 });
    expect(entry.surface.width).toBe(60);
    expect(entry.surface.height).toBe(40);
    expect(entry.stale).toBe(true);
  });

  it('NEVER resizes or re-places an existing entry (a grown paint cache must survive)', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.getOrCreateRect('L', { height: 40, width: 60, x: 5, y: 5 });
    entry.stale = false;
    // A later request with the (smaller/different) contract rect returns the
    // live entry untouched: sizing belongs to the rasterizer/grow paths.
    const again = store.getOrCreateRect('L', { height: 10, width: 10, x: 0, y: 0 });
    expect(again).toBe(entry);
    expect(again.rect).toEqual({ height: 40, width: 60, x: 5, y: 5 });
    expect(again.surface.width).toBe(60);
    expect(again.stale).toBe(false);
  });

  it('supports a zero-rect entry (brand-new empty paint layer)', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.getOrCreateRect('L', { height: 0, width: 0, x: 0, y: 0 });
    expect(entry.rect).toEqual({ height: 0, width: 0, x: 0, y: 0 });
    expect(entry.surface.width).toBe(0);
    expect(entry.surface.height).toBe(0);
  });
});

describe('growToRect (paint caches grow with strokes)', () => {
  it('creates a fresh non-stale entry at the rounded target rect when none exists', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.growToRect('L', { height: 19.2, width: 10.6, x: 3.4, y: -2.7 });
    expect(entry.rect).toEqual({ height: 19, width: 11, x: 3, y: -3 });
    expect(entry.stale).toBe(false);
  });

  it('grows to the UNION of the current extent and the request, preserving old pixels at their shifted origin', () => {
    const store = createLayerCacheStore(createTestStubRasterBackend());
    const entry = store.growToRect('L', { height: 20, width: 20, x: 10, y: 10 });
    const surfaceBefore = entry.surface;

    const grown = store.growToRect('L', { height: 20, width: 20, x: -10, y: -10 });
    expect(grown).toBe(entry);
    // Union of [10,30)² and [-10,10)² = [-10,30)² → 40×40 at (-10,-10).
    expect(grown.rect).toEqual({ height: 40, width: 40, x: -10, y: -10 });
    // Resized in place: the surface identity is preserved (open sessions hold it).
    expect(grown.surface).toBe(surfaceBefore);
    expect(grown.surface.width).toBe(40);
    expect(grown.surface.height).toBe(40);

    // The old pixels were snapshotted before the (clearing) resize and blitted
    // back at their layer-local position within the grown surface: old origin
    // (10,10) minus new origin (-10,-10) = surface (20,20).
    const log = (grown.surface as StubRasterSurface).callLog;
    const snapshot = log.find((e) => e.op === 'getImageData');
    expect(snapshot?.args).toEqual([0, 0, 20, 20]);
    const resizeIndex = log.findIndex((e) => e.op === 'resize');
    const putIndex = log.findIndex((e) => e.op === 'putImageData');
    expect(resizeIndex).toBeGreaterThan(-1);
    expect(putIndex).toBeGreaterThan(resizeIndex);
    expect(log[putIndex]?.args.slice(1)).toEqual([20, 20]);
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
    // No union with the empty rect's (0,0) origin: the first stroke's bounds
    // become the extent verbatim (no needless 0,0-anchored surface).
    expect(grown.rect).toEqual({ height: 30, width: 30, x: 50, y: 60 });
    const log = (grown.surface as StubRasterSurface).callLog;
    expect(log.some((e) => e.op === 'getImageData')).toBe(false);
    expect(log.some((e) => e.op === 'putImageData')).toBe(false);
  });
});
