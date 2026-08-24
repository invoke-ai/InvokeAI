import type { AlphaPixels } from '@workbench/canvas-engine/render/alphaBounds';

import { isEmpty } from '@workbench/canvas-engine/math/rect';
import { alphaBounds, hasVisiblePixels } from '@workbench/canvas-engine/render/alphaBounds';
import { describe, expect, it } from 'vitest';

/** A transparent RGBA buffer of the given size. */
const buffer = (width: number, height: number): AlphaPixels & { data: Uint8ClampedArray } => ({
  data: new Uint8ClampedArray(Math.max(0, width * height * 4)),
  height,
  width,
});

/** Sets one pixel's alpha (and, unless told otherwise, leaves RGB at zero). */
const setAlpha = (pixels: AlphaPixels, x: number, y: number, alpha: number): void => {
  pixels.data[(y * pixels.width + x) * 4 + 3] = alpha;
};

describe('alphaBounds', () => {
  it('returns an empty rect for a fully transparent buffer', () => {
    expect(isEmpty(alphaBounds(buffer(10, 10)))).toBe(true);
  });

  it('returns a 1x1 rect at a single opaque pixel', () => {
    const pixels = buffer(10, 10);
    setAlpha(pixels, 3, 5, 255);
    expect(alphaBounds(pixels)).toEqual({ height: 1, width: 1, x: 3, y: 5 });
  });

  it('returns the bounding box of a ring, not the ring itself', () => {
    const pixels = buffer(8, 8);
    for (let i = 2; i <= 5; i += 1) {
      setAlpha(pixels, i, 2, 255);
      setAlpha(pixels, i, 5, 255);
      setAlpha(pixels, 2, i, 255);
      setAlpha(pixels, 5, i, 255);
    }
    expect(alphaBounds(pixels)).toEqual({ height: 4, width: 4, x: 2, y: 2 });
  });

  it('returns the bounding box of an L-shape spanning both extremes', () => {
    const pixels = buffer(6, 6);
    setAlpha(pixels, 0, 5, 255);
    setAlpha(pixels, 5, 5, 255);
    setAlpha(pixels, 0, 0, 255);
    expect(alphaBounds(pixels)).toEqual({ height: 6, width: 6, x: 0, y: 0 });
  });

  it('treats alpha 1 as content', () => {
    const pixels = buffer(4, 4);
    setAlpha(pixels, 1, 2, 1);
    expect(alphaBounds(pixels)).toEqual({ height: 1, width: 1, x: 1, y: 2 });
  });

  it('reads only the alpha channel — opaque RGB with alpha 0 is empty', () => {
    const pixels = buffer(4, 4);
    for (let index = 0; index < pixels.data.length; index += 4) {
      pixels.data[index] = 255;
      pixels.data[index + 1] = 255;
      pixels.data[index + 2] = 255;
    }
    expect(isEmpty(alphaBounds(pixels))).toBe(true);
  });

  it('returns the whole buffer when every pixel is opaque', () => {
    const pixels = buffer(5, 3);
    pixels.data.fill(255);
    expect(alphaBounds(pixels)).toEqual({ height: 3, width: 5, x: 0, y: 0 });
  });

  it('finds the rightmost pixel even when a later row is narrower', () => {
    // Exercises the inward-from-the-right scan: row 0 sets the running maxX, and
    // row 1 must not be able to lower it, while row 2 must be able to raise it.
    const pixels = buffer(10, 3);
    setAlpha(pixels, 1, 0, 255);
    setAlpha(pixels, 6, 0, 255);
    setAlpha(pixels, 4, 1, 255);
    setAlpha(pixels, 9, 2, 255);
    expect(alphaBounds(pixels)).toEqual({ height: 3, width: 9, x: 1, y: 0 });
  });

  it('finds a row whose only pixel sits exactly at the running maxX', () => {
    const pixels = buffer(8, 2);
    setAlpha(pixels, 5, 0, 255);
    setAlpha(pixels, 5, 1, 255);
    expect(alphaBounds(pixels)).toEqual({ height: 2, width: 1, x: 5, y: 0 });
  });

  it('returns an empty rect for non-positive dimensions', () => {
    expect(isEmpty(alphaBounds(buffer(0, 0)))).toBe(true);
    expect(isEmpty(alphaBounds({ data: new Uint8ClampedArray(0), height: 4, width: -1 }))).toBe(true);
  });

  it('rejects a buffer too short to describe its dimensions', () => {
    expect(() => alphaBounds({ data: new Uint8ClampedArray(4 * 4 * 4 - 1), height: 4, width: 4 })).toThrow(
      'RGBA pixel buffer is shorter than its dimensions require.'
    );
  });

  it('ignores bytes beyond width * height * 4', () => {
    const pixels: AlphaPixels = { data: new Uint8ClampedArray(4 * 4 * 4 + 64), height: 4, width: 4 };
    pixels.data.fill(255, 4 * 4 * 4);
    expect(isEmpty(alphaBounds(pixels))).toBe(true);
  });
});

describe('hasVisiblePixels', () => {
  it('agrees with alphaBounds emptiness across representative buffers', () => {
    const single = buffer(6, 6);
    setAlpha(single, 5, 0, 1);
    const opaque = buffer(3, 3);
    opaque.data.fill(255);
    const cases: AlphaPixels[] = [buffer(6, 6), single, opaque, buffer(0, 0)];
    for (const pixels of cases) {
      expect(hasVisiblePixels(pixels)).toBe(!isEmpty(alphaBounds(pixels)));
    }
  });

  it('finds alpha in the final pixel', () => {
    const pixels = buffer(4, 4);
    setAlpha(pixels, 3, 3, 1);
    expect(hasVisiblePixels(pixels)).toBe(true);
  });

  it('rejects a buffer too short to prove that it has no visible pixels', () => {
    expect(() => hasVisiblePixels({ data: new Uint8ClampedArray(4 * 4 * 4 - 1), height: 4, width: 4 })).toThrow(
      'RGBA pixel buffer is shorter than its dimensions require.'
    );
  });
});
