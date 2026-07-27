import type { Rect } from '@workbench/canvas-engine/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

/**
 * Growing a layer cache is the one place the engine moves a whole surface's
 * pixels, and it happens repeatedly mid-stroke. It preserves them by adopting a
 * fresh backing store and blitting the old canvas into it (`resizePreserving`)
 * rather than round-tripping through `getImageData`/`putImageData`. That is
 * measurably faster, but it also means the copy is real GPU work with a real
 * offset — a wrong `dx`/`dy`, or a blit that silently does nothing, would
 * corrupt layer content on any stroke that paints outward.
 *
 * The node tests can only assert the ARGUMENTS, because the stub surface holds
 * no pixels. These assert the pixels.
 */

/** Distinctly-coloured marks, addressed in layer-local space. */
const MARKS: { color: [number, number, number, number]; local: { x: number; y: number } }[] = [
  { color: [255, 0, 0, 255], local: { x: 105, y: 105 } },
  { color: [0, 255, 0, 255], local: { x: 285, y: 105 } },
  { color: [0, 0, 255, 255], local: { x: 105, y: 285 } },
  { color: [255, 255, 0, 255], local: { x: 285, y: 285 } },
  // A semi-transparent mark: a blit that un-premultiplies and re-premultiplies
  // would shift these channels, and a stroke's "before" snapshot is only
  // pristine if they survive intact.
  { color: [40, 180, 220, 128], local: { x: 200, y: 200 } },
];

const START: Rect = { height: 200, width: 200, x: 100, y: 100 };

const growthCases: { grow: Rect; label: string }[] = [
  { grow: { height: 200, width: 200, x: -50, y: -50 }, label: 'up and to the left (origin moves)' },
  { grow: { height: 200, width: 200, x: 250, y: 250 }, label: 'down and to the right (origin fixed)' },
  { grow: { height: 100, width: 800, x: -300, y: 150 }, label: 'sideways only' },
  { grow: { height: 900, width: 100, x: 150, y: -400 }, label: 'upward only' },
];

describe('growToRect preserves pixels', () => {
  it.each(growthCases)('$label', ({ grow }) => {
    const store = createLayerCacheStore(createDomRasterBackend());
    const entry = store.growToRect('L', START);
    const before = entry.surface;

    for (const mark of MARKS) {
      const [r, g, b, a] = mark.color;
      entry.surface.ctx.fillStyle = `rgba(${r},${g},${b},${a / 255})`;
      entry.surface.ctx.fillRect(mark.local.x - START.x, mark.local.y - START.y, 10, 10);
    }

    // Sample what is actually on the surface rather than the nominal fill
    // colour: a semi-transparent fill is stored premultiplied, so it reads back
    // a step off what was asked for whether or not it is ever moved. Comparing
    // against this isolates the growth, which is what is under test.
    const priorRect = entry.rect;
    const prior = MARKS.map(
      (mark) =>
        entry.surface.ctx.getImageData(mark.local.x - priorRect.x + 5, mark.local.y - priorRect.y + 5, 1, 1).data
    );

    const grown = store.growToRect('L', grow);

    // Derived-surface caches key on the surface OBJECT, so swapping the backing
    // canvas must not swap the surface out from under them.
    expect(grown.surface).toBe(before);
    expect(grown.surface.width).toBe(grown.rect.width);
    expect(grown.surface.height).toBe(grown.rect.height);

    MARKS.forEach((mark, i) => {
      const sx = mark.local.x - grown.rect.x;
      const sy = mark.local.y - grown.rect.y;
      const pixel = grown.surface.ctx.getImageData(sx + 5, sy + 5, 1, 1).data;
      const expected = prior[i]!;
      expect({
        at: `${mark.color.join(',')}`,
        px: [pixel[0], pixel[1], pixel[2], pixel[3]],
      }).toEqual({ at: `${mark.color.join(',')}`, px: [expected[0], expected[1], expected[2], expected[3]] });
    });
  });

  it('leaves the area the growth exposed transparent', () => {
    const store = createLayerCacheStore(createDomRasterBackend());
    const entry = store.growToRect('L', START);
    entry.surface.ctx.fillStyle = '#ffffff';
    entry.surface.ctx.fillRect(0, 0, START.width, START.height);

    const grown = store.growToRect('L', { height: 200, width: 200, x: -50, y: -50 });
    // (0,0) in the grown surface is layer-local (-50,-50) — outside the old
    // extent, so it must be untouched, not a smeared edge of the blit.
    const pixel = grown.surface.ctx.getImageData(0, 0, 1, 1).data;
    expect([pixel[0], pixel[1], pixel[2], pixel[3]]).toEqual([0, 0, 0, 0]);
  });

  it('keeps the pixels across a run of successive growths', () => {
    // A single stroke crosses many chunk boundaries, so the pixels are copied
    // again and again. Any per-growth drift (a rounding slip in the offset, or a
    // blit that resamples) compounds; one growth is not enough to catch it.
    const store = createLayerCacheStore(createDomRasterBackend());
    store.growToRect('L', START);
    const entry = store.get('L')!;
    entry.surface.ctx.fillStyle = 'rgb(220,30,90)';
    entry.surface.ctx.fillRect(100, 100, 10, 10); // layer-local (200,200)
    // A semi-transparent mark too: this is the one that could drift, since a
    // copy that un-premultiplies and re-premultiplies loses a step. It must
    // settle rather than lose a step per growth — a stroke's "before" snapshot
    // is read off this surface, so drift here would slowly lift the pixels the
    // gesture is supposed to leave pristine.
    entry.surface.ctx.fillStyle = 'rgba(40,180,220,0.5)';
    entry.surface.ctx.fillRect(20, 20, 10, 10); // layer-local (120,120)
    const settled = entry.surface.ctx.getImageData(25, 25, 1, 1).data;

    for (let i = 1; i <= 12; i++) {
      store.growToRect('L', { height: 64, width: 64, x: 100 - i * 37, y: 100 - i * 53 });
    }

    const grown = store.get('L')!;
    const opaque = grown.surface.ctx.getImageData(205 - grown.rect.x, 205 - grown.rect.y, 1, 1).data;
    expect([opaque[0], opaque[1], opaque[2], opaque[3]]).toEqual([220, 30, 90, 255]);
    const faint = grown.surface.ctx.getImageData(125 - grown.rect.x, 125 - grown.rect.y, 1, 1).data;
    expect([faint[0], faint[1], faint[2], faint[3]]).toEqual([settled[0], settled[1], settled[2], settled[3]]);
  });
});
