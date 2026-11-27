import type { Rect } from '@workbench/canvas-engine/types';

import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

const START: Rect = { height: 200, width: 200, x: 100, y: 100 };

const published = () => {
  const store = createLayerCacheStore(createDomRasterBackend());
  const entry = store.growToRect('L', START);
  store.publishPixels('L');
  return { deps: { isLayerBusy: () => false, isLayerPinned: () => false, layers: store }, entry, store };
};

const mark = (
  entry: { rect: Rect; surface: { ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D } },
  x: number,
  y: number,
  fill: string
): void => {
  entry.surface.ctx.fillStyle = fill;
  entry.surface.ctx.fillRect(x - entry.rect.x, y - entry.rect.y, 10, 10);
};

describe('trimPaintCacheToAlpha with real pixels', () => {
  it('crops to the mark, preserving its pixels at the new origin', () => {
    const { deps, entry } = published();
    mark(entry, 150, 160, 'rgb(220,30,90)');
    const surfaceBefore = entry.surface;

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('trimmed');

    expect(entry.rect).toEqual({ height: 10, width: 10, x: 150, y: 160 });
    expect(entry.surface).toBe(surfaceBefore);
    expect(entry.surface.width).toBe(10);
    expect(entry.surface.height).toBe(10);
    const pixel = entry.surface.ctx.getImageData(5, 5, 1, 1).data;
    expect([pixel[0], pixel[1], pixel[2], pixel[3]]).toEqual([220, 30, 90, 255]);
  });

  it('EMPTIES a cache erased with destination-out — the reported bug', () => {
    const { deps, entry } = published();
    mark(entry, 150, 150, 'rgb(10,120,240)');
    entry.surface.ctx.globalCompositeOperation = 'destination-out';
    entry.surface.ctx.fillStyle = 'rgb(0,0,0)';
    entry.surface.ctx.fillRect(0, 0, entry.rect.width, entry.rect.height);
    entry.surface.ctx.globalCompositeOperation = 'source-over';

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('emptied');
    expect(isEmpty(entry.rect)).toBe(true);
    expect(entry.surface.width).toBe(0);
  });

  it('EMPTIES a cleared cache and keeps the entry so undo can re-grow into it', () => {
    const { deps, entry, store } = published();
    mark(entry, 150, 150, 'rgb(10,120,240)');
    entry.surface.ctx.clearRect(0, 0, entry.rect.width, entry.rect.height);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('emptied');
    expect(store.peek('L')).toBeDefined();

    const regrown = store.growToRect('L', { height: 20, width: 20, x: 140, y: 140 });
    expect(regrown.rect).toEqual({ height: 20, width: 20, x: 140, y: 140 });
  });

  it('PRESERVES a barely-visible mark: strict-zero never discards faint user pixels', () => {
    // A soft eraser leaves `a_dst * (1 - a_src)`, and antialiased edges a few units.
    const { deps, entry } = published();
    mark(entry, 180, 180, 'rgba(40,180,220,0.02)');
    const sampled = entry.surface.ctx.getImageData(185 - START.x, 185 - START.y, 1, 1).data;
    expect(sampled[3]).toBeGreaterThan(0);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('trimmed');
    expect(entry.rect).toEqual({ height: 10, width: 10, x: 180, y: 180 });
  });
});
