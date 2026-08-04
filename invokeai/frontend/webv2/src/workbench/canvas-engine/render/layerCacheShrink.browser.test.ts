import type { Rect } from '@workbench/canvas-engine/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { describe, expect, it } from 'vitest';

/**
 * `shrinkToRect` crops by blitting the surviving window with a NEGATIVE
 * `resizePreserving` offset, letting `drawImage` clip the rest. A sign slip would
 * keep the wrong pixels; a no-op blit would lose the layer entirely. The node tests
 * assert the arguments — these assert the pixels.
 */

const START: Rect = { height: 200, width: 200, x: 100, y: 100 };

/** Layer-local coordinates of a 10x10 probe mark and the colour it carries. */
const PROBES: { color: [number, number, number]; local: { x: number; y: number } }[] = [
  { color: [255, 0, 0], local: { x: 110, y: 110 } },
  { color: [0, 255, 0], local: { x: 280, y: 110 } },
  { color: [0, 0, 255], local: { x: 110, y: 280 } },
  { color: [255, 255, 0], local: { x: 280, y: 280 } },
  { color: [255, 0, 255], local: { x: 195, y: 195 } },
];

const seeded = () => {
  const store = createLayerCacheStore(createDomRasterBackend());
  const entry = store.growToRect('L', START);
  store.publishPixels('L');
  for (const probe of PROBES) {
    const [r, g, b] = probe.color;
    entry.surface.ctx.fillStyle = `rgb(${r},${g},${b})`;
    entry.surface.ctx.fillRect(probe.local.x - START.x, probe.local.y - START.y, 10, 10);
  }
  return { entry, store };
};

/** Reads the centre of the 10x10 mark that sits at layer-local `local`. */
const sample = (
  entry: { rect: Rect; surface: { ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D } },
  local: { x: number; y: number }
): number[] => {
  const data = entry.surface.ctx.getImageData(local.x - entry.rect.x + 5, local.y - entry.rect.y + 5, 1, 1).data;
  return [data[0]!, data[1]!, data[2]!, data[3]!];
};

const shrinkCases: { label: string; retain: Rect; keeps: number[] }[] = [
  // `keeps` indexes PROBES; the retained rect must hold exactly those marks.
  {
    keeps: [0],
    label: 'to the top-left corner (origin fixed, dx/dy zero)',
    retain: { height: 40, width: 40, x: 100, y: 100 },
  },
  {
    keeps: [3],
    label: 'to the bottom-right corner (both axes move)',
    retain: { height: 40, width: 40, x: 260, y: 260 },
  },
  { keeps: [0, 1], label: 'to a horizontal band (only y moves)', retain: { height: 30, width: 200, x: 100, y: 100 } },
  { keeps: [0, 2], label: 'to a vertical band (only x moves)', retain: { height: 200, width: 30, x: 100, y: 100 } },
  { keeps: [4], label: 'to a window in the middle', retain: { height: 20, width: 20, x: 190, y: 190 } },
];

describe('shrinkToRect preserves the retained pixels', () => {
  it.each(shrinkCases)('$label', ({ keeps, retain }) => {
    const { entry, store } = seeded();
    const surfaceBefore = entry.surface;

    const shrunk = store.shrinkToRect('L', retain)!;

    expect(shrunk.rect).toEqual(retain);
    // Derived-surface caches key on the surface OBJECT, so this must resize in place.
    expect(shrunk.surface).toBe(surfaceBefore);
    expect(shrunk.surface.width).toBe(retain.width);
    expect(shrunk.surface.height).toBe(retain.height);

    for (const index of keeps) {
      const probe = PROBES[index]!;
      expect({ mark: probe.color.join(','), px: sample(shrunk, probe.local) }).toEqual({
        mark: probe.color.join(','),
        px: [...probe.color, 255],
      });
    }
  });

  it('re-grows to transparent pixels after a crop, rather than resurrecting the discarded ones', () => {
    const { entry, store } = seeded();
    store.shrinkToRect('L', { height: 40, width: 40, x: 100, y: 100 });

    const regrown = store.growToRect('L', START);
    // The kept mark survives the round trip...
    expect(sample(regrown, PROBES[0]!.local)).toEqual([255, 0, 0, 255]);
    // ...but a discarded one is gone, not smeared back in by the blit.
    expect(sample(regrown, PROBES[3]!.local)).toEqual([0, 0, 0, 0]);
    expect(entry.surface.width).toBe(START.width);
  });

  it('collapses to a 0x0 surface without throwing', () => {
    const { store } = seeded();
    const emptied = store.shrinkToRect('L', { height: 0, width: 0, x: 100, y: 100 })!;
    expect(emptied.surface.width).toBe(0);
    expect(emptied.surface.height).toBe(0);
  });
});
