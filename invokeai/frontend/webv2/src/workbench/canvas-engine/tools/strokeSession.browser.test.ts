import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, Rect } from '@workbench/canvas-engine/types';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createStrokeSession } from '@workbench/canvas-engine/tools/strokeSession';
import { describe, expect, it, vi } from 'vitest';

const pointer = (x: number, y: number): PointerInput => ({
  buttons: 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp: 0,
});

/**
 * Paints `path` through a session, delivering it in batches of `batchSize`, and
 * returns the committed pixels plus the rect they cover.
 */
const paint = (
  path: PointerInput[],
  batchSize: number,
  opts: { opacity: number; size: number; thinning: number }
): { pixels: Uint8ClampedArray; rect: Rect } => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  layers.getOrCreate('L', 0, 0);
  const ctx = {
    backend,
    createPath2D: (d?: string) => new Path2D(d),
    emitStrokeCommitted: vi.fn(),
    invalidate: vi.fn(),
    layers,
    notifyLayerPainted: vi.fn(),
  } as unknown as ToolContext;
  const session = createStrokeSession({
    color: '#3b82f6',
    composite: 'source-over',
    ctx,
    layerId: 'L',
    opacity: opts.opacity,
    size: opts.size,
    pressureOpacity: false,
    thinning: opts.thinning,
    tool: 'brush',
  });
  for (let i = 0; i < path.length; i += batchSize) {
    session.addPoints(path.slice(i, i + batchSize));
  }
  const event = session.commit()!;
  return { pixels: event.afterImageData.data, rect: event.dirtyRect };
};

/**
 * The largest per-channel difference attributable to premultiplied-alpha
 * rounding. Structural errors (a gap, a seam, a double-composite) are one to two
 * orders of magnitude above this.
 */
const ROUNDING_TOLERANCE = 6;

const sweep = (): PointerInput[] => {
  const points: PointerInput[] = [];
  for (let i = 0; i < 240; i++) {
    points.push(pointer(200 + i * 6 + Math.sin(i / 9) * 40, 400 + Math.cos(i / 7) * 160));
  }
  return points;
};

describe('incremental compositing produces the same coverage as recompositing everything', () => {
  // The session restores and composites only the band where the outline moved,
  // and grows its "before" snapshot from the strips the region gains. If either
  // bound were too tight the stroke would come out with gaps, seams or doubled
  // opacity.
  //
  // The assertion is about COVERAGE, not bit-equality. Whether a given edge
  // pixel's antialiasing was laid down on this frame or three frames ago can
  // shift it by a subpixel, so a handful of boundary pixels legitimately differ
  // from a single-batch paint. What must not differ is the interior: any gap,
  // seam or double-composite shows up as an opaque pixel disagreeing by far more
  // than a rounding step, which is what these bounds catch.
  //
  // Calibrated by deliberately shrinking the band: 10px still passes (the vertex
  // diff carries that much natural slack at these sample rates), 30px fails all
  // four cases.
  const cases: { batchSize: number; label: string; opacity: number; size: number; thinning: number }[] = [
    { batchSize: 1, label: 'one sample per batch, opaque', opacity: 1, size: 220, thinning: 0 },
    { batchSize: 1, label: 'one sample per batch, semi-transparent', opacity: 0.4, size: 220, thinning: 0 },
    { batchSize: 3, label: 'coalesced batches with pressure thinning', opacity: 0.75, size: 400, thinning: 0.5 },
    { batchSize: 7, label: 'large brush, coarse batches', opacity: 1, size: 900, thinning: 0 },
  ];

  it.each(cases)('$label', ({ batchSize, opacity, size, thinning }) => {
    const path = sweep();
    const incremental = paint(path, batchSize, { opacity, size, thinning });
    const wholesale = paint(path, path.length, { opacity, size, thinning });

    expect(incremental.rect).toEqual(wholesale.rect);
    expect(incremental.pixels.length).toBe(wholesale.pixels.length);

    // Interior = a pixel both paints agree is essentially opaque. A gap or a
    // double-composite lands here; antialiasing never does.
    let interiorDiffering = 0;
    let interiorWorst = 0;
    let edgeDiffering = 0;
    for (let i = 0; i < incremental.pixels.length; i += 4) {
      const alphaA = incremental.pixels[i + 3]!;
      const alphaB = wholesale.pixels[i + 3]!;
      let delta = 0;
      for (let channel = 0; channel < 4; channel++) {
        delta = Math.max(delta, Math.abs(incremental.pixels[i + channel]! - wholesale.pixels[i + channel]!));
      }
      if (delta === 0) {
        continue;
      }
      if (alphaA > 250 && alphaB > 250) {
        interiorDiffering++;
        interiorWorst = Math.max(interiorWorst, delta);
      } else {
        edgeDiffering++;
      }
    }
    const total = incremental.pixels.length / 4;
    // Interior pixels may drift by a rounding step — compositing at an opacity
    // rounds premultiplied channels, and which frame did it changes nothing
    // else. What this rules out is structural error, which is nowhere near this
    // scale: an unpainted gap reads as a 255 difference and a double-composite
    // at opacity 0.4 as ~100.
    expect(interiorWorst).toBeLessThanOrEqual(ROUNDING_TOLERANCE);
    expect(interiorDiffering / total).toBeLessThan(0.001);
    // The antialiased boundary may land a subpixel differently, but only there.
    expect(edgeDiffering / total).toBeLessThan(0.005);
  });
});
