import type { FillSample } from '@workbench/canvas-engine/render/fill/fillLevelMap';

import { createSeedColorDistance } from '@workbench/canvas-engine/render/fill/colorDistance';
import {
  CLEAR,
  createGradientSample,
  createSample,
  indicesWhere,
  paintRect,
  type Rgba,
} from '@workbench/canvas-engine/render/fill/fillImages.testing';
import { buildFillLevelMap } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import { describe, expect, it } from 'vitest';

const RED: Rgba = [220, 30, 40, 255];
const BLUE: Rgba = [30, 60, 210, 255];
const MIX: Rgba = [125, 45, 125, 255];

/**
 * 24 × 12: red left of x = 11, an anti-aliased mixed column at x = 11, blue from x = 12, and a red island at
 * (17..19, 4..6) inside the blue.
 */
const createSplitSample = (): FillSample => {
  const sample = createSample(24, 12, BLUE);
  paintRect(sample, 0, 0, 11, 12, RED);
  paintRect(sample, 11, 0, 1, 12, MIX);
  paintRect(sample, 17, 4, 3, 3, RED);
  return sample;
};

const sameColor = (sample: FillSample, index: number, color: Rgba) =>
  color.every((value, channel) => sample.data[index * 4 + channel] === value);

/** Reference contiguous region: breadth-first flood over pixels within `threshold` of the seed color. */
const floodReference = (sample: FillSample, seedX: number, seedY: number, threshold: number): number[] => {
  const { height, width } = sample;
  const measure = createSeedColorDistance(sample.data, (seedY * width + seedX) * 4);
  const seen = new Uint8Array(width * height);
  const queue = [seedY * width + seedX];
  seen[queue[0]] = 1;
  for (let head = 0; head < queue.length; head++) {
    const index = queue[head];
    const x = index % width;
    const neighbors = [x > 0 ? index - 1 : -1, x < width - 1 ? index + 1 : -1, index - width, index + width];
    for (const next of neighbors) {
      if (next >= 0 && next < width * height && !seen[next] && measure(sample.data, next * 4) <= threshold) {
        seen[next] = 1;
        queue.push(next);
      }
    }
  }
  return queue.sort((a, b) => a - b);
};

describe('buildFillLevelMap', () => {
  it('ranks an anti-aliased edge between the two flat colors', () => {
    const sample = createSplitSample();
    const map = buildFillLevelMap(sample, 2, 2, true);
    const at = (x: number, y: number) => map.levels[y * sample.width + x];

    expect(at(2, 2)).toBe(0);
    expect(at(10, 7)).toBe(0);
    expect(at(11, 7)).toBeGreaterThan(0);
    expect(at(11, 7)).toBeLessThan(at(12, 7));
    // The island is only reachable through blue, so it joins at blue's level, not its own 0.
    expect(at(18, 5)).toBe(at(12, 7));
  });

  it('takes exactly the connected seed-colored pixels at level 0', () => {
    const sample = createSplitSample();
    const map = buildFillLevelMap(sample, 2, 2, true);
    const pixels = sample.width * sample.height;

    expect(indicesWhere(pixels, (index) => map.levels[index] === 0)).toEqual(
      indicesWhere(pixels, (index) => index % sample.width < 11)
    );
  });

  it('matches a flood of the thresholded distances at every tolerance on a noisy gradient', () => {
    const sample = createGradientSample(48, 32);
    const map = buildFillLevelMap(sample, 20, 14, true);
    const pixels = sample.width * sample.height;
    const sizes: number[] = [];
    for (const threshold of [0, 3, 8, 15, 30, 60, 120, 255]) {
      const region = indicesWhere(pixels, (index) => map.levels[index] <= threshold);
      expect(region).toEqual(floodReference(sample, 20, 14, threshold));
      sizes.push(region.length);
    }
    expect(sizes[0]).toBeGreaterThanOrEqual(1);
    expect(sizes.at(-1)).toBe(pixels);
    expect(new Set(sizes).size).toBeGreaterThan(4);
  });

  it('records per-level counts and bounds matching the map', () => {
    const sample = createGradientSample(40, 30);
    const map = buildFillLevelMap(sample, 5, 25, true);
    for (const level of [0, 4, 10, 40]) {
      const xs: number[] = [];
      const ys: number[] = [];
      map.levels.forEach((value, index) => {
        if (value === level) {
          xs.push(index % sample.width);
          ys.push(Math.floor(index / sample.width));
        }
      });
      expect(map.levelCounts[level]).toBe(xs.length);
      if (xs.length > 0) {
        expect([...map.levelBounds.subarray(level * 4, level * 4 + 4)]).toEqual([
          Math.min(...xs),
          Math.min(...ys),
          Math.max(...xs),
          Math.max(...ys),
        ]);
      }
    }
  });

  it('uses each pixel’s own distance when not contiguous', () => {
    const sample = createSplitSample();
    const map = buildFillLevelMap(sample, 2, 2, false);
    const measure = createSeedColorDistance(sample.data, (2 * sample.width + 2) * 4);
    const pixels = sample.width * sample.height;

    expect([...map.levels]).toEqual(Array.from({ length: pixels }, (_, index) => measure(sample.data, index * 4)));
    expect(indicesWhere(pixels, (index) => map.levels[index] === 0)).toEqual(
      indicesWhere(pixels, (index) => sameColor(sample, index, RED))
    );
  });

  it('stops a fill of a transparent area at any paint, however faint', () => {
    const sample = createSample(20, 20, CLEAR);
    // A closed ring of faint paint around (5..14, 5..14).
    paintRect(sample, 4, 4, 12, 1, [255, 0, 0, 12]);
    paintRect(sample, 4, 15, 12, 1, [255, 0, 0, 12]);
    paintRect(sample, 4, 4, 1, 12, [255, 0, 0, 12]);
    paintRect(sample, 15, 4, 1, 12, [255, 0, 0, 12]);
    const map = buildFillLevelMap(sample, 9, 9, true);
    const pixels = sample.width * sample.height;
    const inside = (index: number) => {
      const x = index % 20;
      const y = Math.floor(index / 20);
      return x >= 5 && x <= 14 && y >= 5 && y <= 14;
    };

    expect(indicesWhere(pixels, (index) => map.levels[index] < 12)).toEqual(indicesWhere(pixels, inside));
    expect(map.levels[0]).toBe(12);
  });

  it('refuses a seed outside the sample', () => {
    const sample = createSample(4, 4, CLEAR);
    expect(() => buildFillLevelMap(sample, 4, 0, true)).toThrow(RangeError);
    expect(() => buildFillLevelMap(sample, 0, -1, false)).toThrow(RangeError);
    expect(() => buildFillLevelMap(sample, 1.5, 0, true)).toThrow(RangeError);
  });
});
