import type { Rect } from '@workbench/canvas-engine/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

const MARKS: { color: [number, number, number, number]; local: { x: number; y: number } }[] = [
  { color: [255, 0, 0, 255], local: { x: 105, y: 105 } },
  { color: [0, 255, 0, 255], local: { x: 285, y: 105 } },
  { color: [0, 0, 255, 255], local: { x: 105, y: 285 } },
  { color: [255, 255, 0, 255], local: { x: 285, y: 285 } },
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

    // Sample what is actually on the surface rather than the nominal fill colour: a semi-transparent fill is stored
    // premultiplied, so it reads back a step off what was asked for whether or not it is ever moved. Comparing against
    // this isolates the growth, which is what is under test.
    const priorRect = entry.rect;
    const prior = MARKS.map(
      (mark) =>
        entry.surface.ctx.getImageData(mark.local.x - priorRect.x + 5, mark.local.y - priorRect.y + 5, 1, 1).data
    );

    const grown = store.growToRect('L', grow);

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
});
