import type { CoverageOperation } from '@workbench/canvas-engine/tools/coverageEdit';
import type { Mat2d, PlacedSurface, Rect } from '@workbench/canvas-engine/types';

import { applyToPoint, identity, invert, rotate } from '@workbench/canvas-engine/math/mat2d';
import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createCoverageEdit } from '@workbench/canvas-engine/tools/coverageEdit';
import { createRecordingStrokeEdit } from '@workbench/canvas-engine/tools/strokeEdit.testStub';
import { describe, expect, it, vi } from 'vitest';

/** Real pixels: a redraw must leave only the original plus the last draw, and a cancel must leave the original. */

const LAYER = 'layer';
const ORIGINAL: Rect = { height: 40, width: 40, x: 0, y: 0 };

/** A distinct opaque color per layer-local pixel, so a misplaced or stale pixel shows. */
const patternAt = (x: number, y: number): [number, number, number, number] => [x * 6, y * 6, (x + y) * 3, 255];

const harness = (
  options: { cached?: boolean; admitBytes?: number; clipMask?: PlacedSurface; clipRect?: Rect } = {}
) => {
  const backend = createDomRasterBackend();
  const layers = createLayerCacheStore(backend);
  if (options.cached ?? true) {
    const entry = layers.growToRect(LAYER, ORIGINAL);
    entry.stale = false;
    const image = entry.surface.ctx.createImageData(ORIGINAL.width, ORIGINAL.height);
    for (let y = 0; y < ORIGINAL.height; y++) {
      for (let x = 0; x < ORIGINAL.width; x++) {
        image.data.set(patternAt(x, y), (y * ORIGINAL.width + x) * 4);
      }
    }
    entry.surface.ctx.putImageData(image, 0, 0);
  }
  const strokeEdit = createRecordingStrokeEdit(options.admitBytes);
  const ctx = { backend, invalidate: vi.fn(), layers, notifyLayerPainted: vi.fn() };
  const open = (operation: CoverageOperation = 'source-over', toLocal: Mat2d = identity()) =>
    createCoverageEdit(
      ctx,
      { clipMask: options.clipMask ?? null, clipRect: options.clipRect ?? null, layerId: LAYER, operation, toLocal },
      strokeEdit.edit,
      0
    );
  /** RGBA at a layer-local pixel; transparent outside the cache. */
  const pixel = (x: number, y: number): number[] => {
    const entry = layers.peek(LAYER);
    if (!entry || x < entry.rect.x || y < entry.rect.y) {
      return [0, 0, 0, 0];
    }
    return [...entry.surface.ctx.getImageData(x - entry.rect.x, y - entry.rect.y, 1, 1).data];
  };
  return { ctx, layers, open, pixel, record: strokeEdit.record, strokeEdit };
};

/** Fills everything the scratch shows with `color`, in document space. */
const fill =
  (color: string) =>
  (draw: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D): void => {
    draw.fillStyle = color;
    draw.fillRect(-1000, -1000, 3000, 3000);
  };

const inside = (rect: Rect, x: number, y: number): boolean =>
  x >= rect.x && y >= rect.y && x < rect.x + rect.width && y < rect.y + rect.height;

const originalAt = (x: number, y: number): number[] => (inside(ORIGINAL, x, y) ? patternAt(x, y) : [0, 0, 0, 0]);

/** The RGBA of `image` (covering `rect`) at layer-local (x, y). */
const imagePixel = (image: ImageData, rect: Rect, x: number, y: number): number[] => {
  const offset = ((y - rect.y) * rect.width + (x - rect.x)) * 4;
  return [...image.data.subarray(offset, offset + 4)];
};

const BLUE = [0, 0, 255, 255];
const GREEN = [0, 255, 0, 255];

describe('coverage edit', () => {
  it('records the original and the drawn pixels over the dirty rect', () => {
    const h = harness();
    const edit = h.open();

    expect(edit.apply({ height: 10, width: 10, x: 5, y: 5 }, fill('#0000ff'))).toBe(true);
    expect(edit.commit({ tool: 'paintBucket' })).toBe(true);

    const event = h.record.commits[0]!;
    expect(event).toMatchObject({ dirtyRect: { height: 10, width: 10, x: 5, y: 5 }, layerId: LAYER });
    expect(event.tool).toBe('paintBucket');
    expect(imagePixel(event.beforeImageData, event.dirtyRect, 7, 12)).toEqual(patternAt(7, 12));
    expect(imagePixel(event.afterImageData, event.dirtyRect, 7, 12)).toEqual(BLUE);
    // The draw stays inside the dirty rect.
    expect(h.pixel(4, 4)).toEqual(patternAt(4, 4));
    expect(h.pixel(15, 15)).toEqual(patternAt(15, 15));
    expect(h.record.admitted).toBe(10 * 10 * 8);
  });

  it('leaves only the last of several draws, with the original pixels as the step before', () => {
    const h = harness();
    const edit = h.open();
    const first: Rect = { height: 30, width: 30, x: 0, y: 0 };
    const shrunk: Rect = { height: 5, width: 5, x: 10, y: 10 };
    // Overlaps the first draw and reaches past the cache.
    const last: Rect = { height: 20, width: 20, x: 25, y: 25 };

    edit.apply(first, fill('#0000ff'));
    edit.apply(shrunk, fill('#00ff00'));
    expect(h.pixel(5, 5)).toEqual(patternAt(5, 5));
    expect(h.pixel(12, 12)).toEqual(GREEN);
    edit.apply(last, fill('#00ff00'));
    expect(edit.commit({ tool: 'paintBucket' })).toBe(true);

    expect(h.layers.peek(LAYER)?.rect).toEqual({ height: 45, width: 45, x: 0, y: 0 });
    for (const [x, y] of [
      [5, 5],
      [12, 12],
      [29, 20],
      [25, 25],
      [39, 39],
      [44, 44],
      [44, 0],
    ] as const) {
      expect(h.pixel(x, y), `(${x}, ${y})`).toEqual(inside(last, x, y) ? GREEN : originalAt(x, y));
    }
    const event = h.record.commits[0]!;
    expect(event.dirtyRect).toEqual(last);
    for (const [x, y] of [
      [25, 25],
      [29, 29],
      [39, 39],
      [44, 44],
    ] as const) {
      expect(imagePixel(event.beforeImageData, last, x, y), `before (${x}, ${y})`).toEqual(originalAt(x, y));
      expect(imagePixel(event.afterImageData, last, x, y)).toEqual(GREEN);
    }
    // Admission covers before and after over everything drawn: 0..45 square.
    expect(h.record.admitted).toBe(45 * 45 * 8);
  });

  it('puts the original pixels and extent back when cancelled after several draws', () => {
    const h = harness();
    const before = h.layers.peek(LAYER)!.surface.ctx.getImageData(0, 0, ORIGINAL.width, ORIGINAL.height).data.slice();
    const edit = h.open();

    edit.apply({ height: 20, width: 20, x: 10, y: 10 }, fill('#0000ff'));
    edit.apply({ height: 30, width: 30, x: 30, y: -5 }, fill('#00ff00'));
    edit.cancel();

    const entry = h.layers.peek(LAYER)!;
    expect(entry.rect).toEqual(ORIGINAL);
    expect(entry.surface.ctx.getImageData(0, 0, ORIGINAL.width, ORIGINAL.height).data).toEqual(before);
    expect(h.record.cancelled).toBe(1);
    expect(h.record.commits).toHaveLength(0);
    expect(h.ctx.notifyLayerPainted).toHaveBeenCalledWith(LAYER);
    expect(edit.commit({ tool: 'paintBucket' })).toBe(false);
  });

  it('removes a cache the edit created when cancelled', () => {
    const h = harness({ cached: false });
    const edit = h.open();

    edit.apply({ height: 10, width: 10, x: 0, y: 0 }, fill('#0000ff'));
    expect(h.pixel(5, 5)).toEqual(BLUE);
    edit.cancel();

    expect(h.layers.peek(LAYER)).toBeUndefined();
    expect(h.record.cancelled).toBe(1);
  });

  it('cancels instead of recording when the last draw covers no pixel', () => {
    const h = harness();
    const edit = h.open();

    edit.apply({ height: 10, width: 10, x: 0, y: 0 }, fill('#0000ff'));
    expect(edit.apply({ height: 0, width: 10, x: 0, y: 0 }, fill('#0000ff'))).toBe(true);
    expect(h.pixel(5, 5)).toEqual(patternAt(5, 5));
    expect(edit.commit({ tool: 'paintBucket' })).toBe(false);

    expect(h.record.commits).toHaveLength(0);
    expect(h.record.cancelled).toBe(1);
    expect(h.layers.peek(LAYER)?.rect).toEqual(ORIGINAL);
  });

  it('restores the layer and cancels when the admission cannot grow', () => {
    const h = harness({ admitBytes: 20 * 20 * 8 });
    const edit = h.open();

    expect(edit.apply({ height: 20, width: 20, x: 0, y: 0 }, fill('#0000ff'))).toBe(true);
    expect(edit.apply({ height: 30, width: 30, x: 20, y: 20 }, fill('#0000ff'))).toBe(false);

    expect(h.layers.peek(LAYER)?.rect).toEqual(ORIGINAL);
    expect(h.pixel(5, 5)).toEqual(patternAt(5, 5));
    expect(h.pixel(25, 25)).toEqual(patternAt(25, 25));
    expect(h.record.cancelled).toBe(1);
    expect(edit.apply({ height: 5, width: 5, x: 0, y: 0 }, fill('#0000ff'))).toBe(false);
    expect(edit.commit({ tool: 'paintBucket' })).toBe(false);
    expect(h.record.commits).toHaveLength(0);
  });

  it('erases the coverage with destination-out', () => {
    const h = harness();
    const edit = h.open('destination-out');

    edit.apply({ height: 10, width: 10, x: 5, y: 5 }, fill('#000000'));
    edit.commit({ tool: 'paintBucketErase' });

    expect(h.pixel(10, 10)).toEqual([0, 0, 0, 0]);
    expect(h.pixel(4, 10)).toEqual(patternAt(4, 10));
    expect(h.record.commits[0]?.tool).toBe('paintBucketErase');
  });

  it('paints only over existing alpha with source-atop', () => {
    const h = harness();
    const edit = h.open('source-atop');

    // Half inside the cached pixels, half over empty layer.
    edit.apply({ height: 10, width: 20, x: 30, y: 0 }, fill('#0000ff'));
    edit.commit({ tool: 'paintBucket' });

    expect(h.pixel(35, 5)).toEqual(BLUE);
    expect(h.pixel(45, 5)).toEqual([0, 0, 0, 0]);
  });

  it('keeps only the selected part of the coverage', () => {
    const backend = createDomRasterBackend();
    const selection = backend.createSurface(20, 20);
    selection.ctx.fillStyle = '#fff';
    selection.ctx.fillRect(0, 0, 10, 20);
    const h = harness({ clipMask: { rect: { height: 20, width: 20, x: 0, y: 0 }, surface: selection } });
    const edit = h.open();

    edit.apply({ height: 20, width: 20, x: 0, y: 0 }, fill('#0000ff'));

    expect(h.pixel(5, 10)).toEqual(BLUE);
    expect(h.pixel(15, 10)).toEqual(patternAt(15, 10));
  });

  it('clips a rotated layer to the exact document rect, not its local bounds', () => {
    const clipRect: Rect = { height: 20, width: 20, x: 0, y: 0 };
    const layerMatrix = rotate(identity(), Math.PI / 4);
    const toLocal = invert(layerMatrix)!;
    const h = harness({ cached: false, clipRect });
    const edit = h.open('source-over', toLocal);

    edit.apply(clipRect, fill('#0000ff'));
    edit.commit({ tool: 'paintBucket' });

    const dirty = h.record.commits[0]!.dirtyRect;
    const corner = { x: dirty.x + 0.5, y: dirty.y + 0.5 };
    const cornerInDocument = applyToPoint(layerMatrix, corner);
    expect(inside(clipRect, cornerInDocument.x, cornerInDocument.y)).toBe(false);
    expect(h.pixel(dirty.x, dirty.y)[3]).toBe(0);
    const centre = applyToPoint(toLocal, { x: 10, y: 10 });
    expect(h.pixel(Math.floor(centre.x), Math.floor(centre.y))).toEqual(BLUE);
  });
});
