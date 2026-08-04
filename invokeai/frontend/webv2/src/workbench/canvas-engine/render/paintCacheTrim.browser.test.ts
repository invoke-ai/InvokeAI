import type { Rect } from '@workbench/canvas-engine/types';

import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { trimPaintCacheToAlpha } from '@workbench/canvas-engine/render/paintCacheTrim';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

/**
 * The trim decides whether a layer still has content by reading its alpha, and it
 * crops the cache by blitting the surviving window to a new origin. Both are real
 * pixel work: the node stub invents a uniform readback and only records the blit's
 * arguments, so it can cover the guard matrix but not the actual verdict.
 *
 * These assert the pixels — including the `destination-out` erase that produced the
 * reported bug, and the premultiplied readback of a faint mark, which is exactly
 * where a threshold-based emptiness test would wrongly discard user content.
 */

/** A published cache at a non-zero origin — the shape a chunk-padded stroke leaves. */
const START: Rect = { height: 200, width: 200, x: 100, y: 100 };

const published = () => {
  const store = createLayerCacheStore(createDomRasterBackend());
  const entry = store.growToRect('L', START);
  store.publishPixels('L');
  return { deps: { isLayerBusy: () => false, layers: store }, entry, store };
};

/** Fills a 10x10 mark addressed in LAYER-LOCAL space. */
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
    // Derived-surface caches key on the surface OBJECT, so the crop must resize in
    // place rather than swap the surface out from under them.
    expect(entry.surface).toBe(surfaceBefore);
    expect(entry.surface.width).toBe(10);
    expect(entry.surface.height).toBe(10);
    const pixel = entry.surface.ctx.getImageData(5, 5, 1, 1).data;
    expect([pixel[0], pixel[1], pixel[2], pixel[3]]).toEqual([220, 30, 90, 255]);
  });

  it('spans every mark rather than stopping at the first', () => {
    const { deps, entry } = published();
    mark(entry, 120, 250, 'rgb(255,0,0)');
    mark(entry, 270, 130, 'rgb(0,255,0)');

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('trimmed');
    // Bounding box of [120,130) x [250,260) and [270,280) x [130,140).
    expect(entry.rect).toEqual({ height: 130, width: 160, x: 120, y: 130 });
  });

  it('EMPTIES a cache erased with destination-out — the reported bug', () => {
    const { deps, entry } = published();
    mark(entry, 150, 150, 'rgb(10,120,240)');
    // Exactly what the eraser does: composite the stroke out of the cache.
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
    // A soft eraser leaves `a_dst * (1 - a_src)` behind, and an antialiased edge
    // leaves a couple of units. Treating those as empty would silently delete
    // content, so the verdict is strict-zero — which only real premultiplied
    // readback can exercise.
    const { deps, entry } = published();
    mark(entry, 180, 180, 'rgba(40,180,220,0.02)');
    const sampled = entry.surface.ctx.getImageData(185 - START.x, 185 - START.y, 1, 1).data;
    expect(sampled[3]).toBeGreaterThan(0);

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('trimmed');
    expect(entry.rect).toEqual({ height: 10, width: 10, x: 180, y: 180 });
  });

  it('KEEPS an already-tight cache without reallocating', () => {
    const { deps, entry } = published();
    entry.surface.ctx.fillStyle = 'rgb(0,0,0)';
    entry.surface.ctx.fillRect(0, 0, entry.rect.width, entry.rect.height);
    const versionBefore = entry.version;

    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('kept');
    expect(entry.rect).toEqual(START);
    expect(entry.version).toBe(versionBefore);
  });

  it('converges: trim, undo-style repaint, trim again', () => {
    // The full round trip the engine performs — erase → flush empties → undo writes
    // pixels back → flush finds them again. Drift in either direction would show up
    // as a rect that never settles.
    const { deps, entry, store } = published();
    mark(entry, 150, 150, 'rgb(255,255,255)');
    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('trimmed');
    expect(entry.rect).toEqual({ height: 10, width: 10, x: 150, y: 150 });

    entry.surface.ctx.clearRect(0, 0, entry.rect.width, entry.rect.height);
    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('emptied');

    const regrown = store.growToRect('L', { height: 40, width: 40, x: 130, y: 130 });
    store.publishPixels('L');
    mark(regrown, 145, 145, 'rgb(255,255,255)');
    expect(trimPaintCacheToAlpha(deps, 'L')).toBe('trimmed');
    expect(regrown.rect).toEqual({ height: 10, width: 10, x: 145, y: 145 });
  });
});
