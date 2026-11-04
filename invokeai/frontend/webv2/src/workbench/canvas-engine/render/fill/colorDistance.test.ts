import { createSeedColorDistance } from '@workbench/canvas-engine/render/fill/colorDistance';
import { seeded, type Rgba } from '@workbench/canvas-engine/render/fill/fillImages.testing';
import { describe, expect, it } from 'vitest';

const distance = (seed: Rgba, other: Rgba): number =>
  createSeedColorDistance(Uint8ClampedArray.from(seed), 0)(Uint8ClampedArray.from(other), 0);

describe('createSeedColorDistance', () => {
  it('is 0 for identical pixels and at least 1 for any one-byte difference', () => {
    const colors: Rgba[] = [
      [0, 0, 0, 255],
      [255, 255, 255, 255],
      [12, 200, 77, 128],
      [128, 128, 128, 1],
    ];
    for (const color of colors) {
      expect(distance(color, color)).toBe(0);
      for (let channel = 0; channel < 4; channel++) {
        const nudged = [...color] as [number, number, number, number];
        nudged[channel] += nudged[channel] === 255 ? -1 : 1;
        expect(distance(color, nudged)).toBeGreaterThanOrEqual(1);
      }
    }
  });

  it('compares fully transparent pixels by alpha alone', () => {
    expect(distance([0, 0, 0, 0], [255, 0, 0, 255])).toBe(255);
    expect(distance([255, 255, 255, 255], [0, 0, 0, 0])).toBe(255);
    expect(distance([10, 20, 30, 0], [200, 100, 50, 0])).toBe(0);
    expect(distance([0, 0, 0, 0], [90, 10, 200, 37])).toBe(37);
  });

  it('measures an alpha-only difference exactly in bytes', () => {
    expect(distance([40, 90, 160, 255], [40, 90, 160, 200])).toBe(55);
  });

  it('is symmetric', () => {
    const random = seeded(7);
    const color = (): Rgba => [random() * 256, random() * 256, random() * 256, random() * 256].map(Math.floor) as never;
    for (let i = 0; i < 200; i++) {
      const a = color();
      const b = color();
      expect(distance(a, b)).toBe(distance(b, a));
    }
  });

  it('grows monotonically with a channel delta and saturates at 255', () => {
    const seed: Rgba = [0, 120, 60, 255];
    const measure = createSeedColorDistance(Uint8ClampedArray.from(seed), 0);
    let previous = 0;
    for (let red = 0; red < 256; red++) {
      const value = measure(Uint8ClampedArray.of(red, 120, 60, 255), 0);
      expect(value).toBeGreaterThanOrEqual(previous);
      previous = value;
    }
    expect(previous).toBeGreaterThan(100);
    expect(distance([0, 0, 0, 255], [255, 255, 255, 255])).toBe(255);
  });

  it('never lets one color answer for another through its memo', () => {
    const random = seeded(11);
    const seed: Rgba = [30, 60, 90, 255];
    const pixels = new Uint8ClampedArray(20_000 * 4).map(() => Math.floor(random() * 256));
    const shared = createSeedColorDistance(Uint8ClampedArray.from(seed), 0);
    let mismatches = 0;
    for (let pass = 0; pass < 2; pass++) {
      for (let offset = 0; offset < pixels.length; offset += 4) {
        const own = distance(seed, [pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]]);
        mismatches += shared(pixels, offset) === own ? 0 : 1;
      }
    }
    expect(mismatches).toBe(0);
  });
});
