import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

const harness = (options: { busy?: boolean; pinned?: boolean } = {}) => {
  const store = createLayerCacheStore(createTestStubRasterBackend({ readbackAlpha: 0 }));
  const isLayerBusy = vi.fn(() => options.busy ?? false);
  return { deps: { isLayerBusy, isLayerPinned: () => options.pinned ?? false, layers: store }, isLayerBusy, store };
};

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

  it.each([
    {
      name: 'pixels have never been published — the pre-rasterize window',
      options: {},
      prepare: (store: ReturnType<typeof createLayerCacheStore>) => {
        const entry = store.getOrCreateRect('L', { height: 40, width: 40, x: 0, y: 0 });
        entry.stale = false;
        expect(entry.hasPublishedPixels).toBe(false);
        return entry;
      },
      rect: { height: 40, width: 40, x: 0, y: 0 },
    },
    {
      name: 'the cache is stale',
      options: {},
      prepare: (store: ReturnType<typeof createLayerCacheStore>) => {
        const entry = publish(store);
        entry.stale = true;
        return entry;
      },
      rect: { height: 40, width: 40, x: 10, y: 10 },
    },
    {
      name: 'the layer is busy',
      options: { busy: true },
      prepare: publish,
      rect: { height: 40, width: 40, x: 10, y: 10 },
    },
  ])('DEFERS without reading pixels while $name', ({ options, prepare, rect }) => {
    const { deps, store } = harness(options);
    const entry = prepare(store);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('deferred');
    expect(entry.rect).toEqual(rect);
    expect((entry.surface as StubRasterSurface).callLog.map((e) => e.op)).not.toContain('getImageData');
  });

  it('KEEPS a pinned cache untrimmed so persistence never waits on a read lease', () => {
    const { deps, store } = harness({ pinned: true });
    const entry = publish(store);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('kept');
    expect(entry.rect).toEqual({ height: 40, width: 40, x: 10, y: 10 });
    expect((entry.surface as StubRasterSurface).callLog.map((e) => e.op)).not.toContain('getImageData');
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
