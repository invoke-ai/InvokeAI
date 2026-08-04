import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

/**
 * A real cache store on the stub backend, whose readback is zeroed — so every
 * published surface reads as transparent, making this the home for the `emptied`
 * case and the guard matrix. Real cropping is asserted in the browser suite.
 */
const harness = (options: { busy?: boolean } = {}) => {
  const store = createLayerCacheStore(createTestStubRasterBackend());
  const isLayerBusy = vi.fn(() => options.busy ?? false);
  return { deps: { isLayerBusy, layers: store }, isLayerBusy, store };
};

/** A published 40x40 cache at (10,10) — the shape a chunk-padded stroke leaves. */
const publish = (store: ReturnType<typeof createLayerCacheStore>) => {
  const entry = store.growToRect('L', { height: 40, width: 40, x: 10, y: 10 });
  store.publishPixels('L');
  return entry;
};

describe('trimPaintCacheToAlpha', () => {
  it('keeps a layer with no cache at all', () => {
    const { deps, isLayerBusy } = harness();
    expect(trimPaintCacheToAlpha(deps, 'missing')).toBe('kept');
    expect(isLayerBusy).not.toHaveBeenCalled();
  });

  it('keeps an already-empty cache without probing pixels', () => {
    const { deps, store } = harness();
    const entry = store.getOrCreateRect('L', { height: 0, width: 0, x: 0, y: 0 });
    entry.stale = false;

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('kept');
    expect((entry.surface as StubRasterSurface).callLog.map((e) => e.op)).not.toContain('getImageData');
  });

  it('DEFERS while pixels have never been published — the pre-rasterize window', () => {
    const { deps, store } = harness();
    const entry = store.getOrCreateRect('L', { height: 40, width: 40, x: 0, y: 0 });
    entry.stale = false;
    expect(entry.hasPublishedPixels).toBe(false);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('deferred');
    expect(entry.rect).toEqual({ height: 40, width: 40, x: 0, y: 0 });
    expect((entry.surface as StubRasterSurface).callLog.map((e) => e.op)).not.toContain('getImageData');
  });

  it('DEFERS while the cache is stale', () => {
    const { deps, store } = harness();
    const entry = publish(store);
    entry.stale = true;

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('deferred');
    expect(entry.rect).toEqual({ height: 40, width: 40, x: 10, y: 10 });
  });

  it('DEFERS while the layer is busy, without reading pixels', () => {
    const { deps, isLayerBusy, store } = harness({ busy: true });
    const entry = publish(store);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('deferred');
    expect(isLayerBusy).toHaveBeenCalledWith('L');
    expect(entry.rect).toEqual({ height: 40, width: 40, x: 10, y: 10 });
    expect((entry.surface as StubRasterSurface).callLog.map((e) => e.op)).not.toContain('getImageData');
  });

  it('EMPTIES a published cache holding no visible alpha, collapsing it to a zero rect', () => {
    const { deps, store } = harness();
    const entry = publish(store);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('emptied');
    expect(entry.rect).toEqual({ height: 0, width: 0, x: 10, y: 10 });
    expect(entry.surface.width).toBe(0);
    expect(entry.surface.height).toBe(0);
    expect((entry.surface as StubRasterSurface).callLog.filter((e) => e.op === 'resize').map((e) => e.args)).toEqual([
      [0, 0],
    ]);
  });

  it('leaves the ENTRY alive after emptying, so an undo can re-grow into it', () => {
    const { deps, store } = harness();
    publish(store);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('emptied');
    expect(store.peek('L')).toBeDefined();
  });

  it('reads the cache RECT, not the whole surface, when probing', () => {
    const { deps, store } = harness();
    const entry = publish(store);
    store.shrinkToRect('L', { height: 12, width: 20, x: 15, y: 18 });

    trimPaintCacheToAlpha(deps, 'L');
    const reads = (entry.surface as StubRasterSurface).callLog.filter((e) => e.op === 'getImageData');
    expect(reads.map((e) => e.args)).toEqual([[0, 0, 20, 12]]);
  });

  it('is idempotent — a second pass over an emptied cache keeps it', () => {
    const { deps, store } = harness();
    publish(store);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('emptied');
    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('kept');
  });
});
