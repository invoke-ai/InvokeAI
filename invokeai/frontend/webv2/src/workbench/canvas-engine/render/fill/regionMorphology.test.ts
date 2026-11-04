import type { FillSample } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import type { FillRegion, FillRegionParams } from '@workbench/canvas-engine/render/fill/regionMorphology';

import {
  createGradientSample,
  createLineArtSample,
  createSample,
  INK,
  LINE_ART,
  paintRect,
  WHITE,
} from '@workbench/canvas-engine/render/fill/fillImages.testing';
import { buildFillLevelMap } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import { computeFillRegion, createRegionScratch } from '@workbench/canvas-engine/render/fill/regionMorphology';
import { describe, expect, it } from 'vitest';

const HARD: FillRegionParams = { antiAlias: false, closeGaps: 0, expand: 0, tolerance: 20 };

const region = (sample: FillSample, seed: { x: number; y: number }, params: Partial<FillRegionParams> = {}) =>
  computeFillRegion(buildFillLevelMap(sample, seed.x, seed.y, true), { ...HARD, ...params });

/** Coverage at a sample pixel, 0 outside the region's bounds. */
const coverageAt = (fill: FillRegion, x: number, y: number): number => {
  if (!fill.bounds) {
    return 0;
  }
  const { bounds, coverage } = fill;
  const localX = x - bounds.x;
  const localY = y - bounds.y;
  return localX < 0 || localY < 0 || localX >= bounds.width || localY >= bounds.height
    ? 0
    : coverage[localY * bounds.width + localX];
};

const coverageGrid = (fill: FillRegion, width: number, height: number): number[][] =>
  Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) => coverageAt(fill, x, y)));

/** 30 × 30 white with an ink square at (10, 10), 10 × 10. */
const squareSample = () => {
  const sample = createSample(30, 30, WHITE);
  paintRect(sample, 10, 10, 10, 10, INK);
  return sample;
};

describe('computeFillRegion', () => {
  it('returns the thresholded region with tight bounds', () => {
    const fill = region(squareSample(), { x: 14, y: 14 });
    expect(fill.bounds).toEqual({ height: 10, width: 10, x: 10, y: 10 });
    expect(fill.pixelCount).toBe(100);
    expect(fill.coverage && [...new Set(fill.coverage)]).toEqual([255]);
  });

  it('takes the whole sample at tolerance 100', () => {
    const fill = region(createGradientSample(40, 24), { x: 3, y: 3 }, { tolerance: 100 });
    expect(fill.bounds).toEqual({ height: 24, width: 40, x: 0, y: 0 });
    expect(fill.pixelCount).toBe(40 * 24);
  });

  it('nests regions in tolerance', () => {
    const map = buildFillLevelMap(createGradientSample(40, 24), 12, 12, true);
    let previous: number[][] | null = null;
    for (const tolerance of [0, 2, 5, 10, 20, 50, 100]) {
      const grid = coverageGrid(computeFillRegion(map, { ...HARD, tolerance }), 40, 24);
      if (previous) {
        const before = previous;
        expect(grid.every((row, y) => row.every((value, x) => value >= before[y][x]))).toBe(true);
      }
      previous = grid;
    }
  });

  it('expands an axis-aligned edge by exactly the requested pixels, rounding only the corners', () => {
    const fill = region(squareSample(), { x: 14, y: 14 }, { expand: 2 });
    expect(fill.bounds).toEqual({ height: 14, width: 14, x: 8, y: 8 });
    for (let y = 10; y < 20; y++) {
      for (const x of [8, 9, 20, 21]) {
        expect(coverageAt(fill, x, y)).toBe(255);
        expect(coverageAt(fill, y, x)).toBe(255);
      }
    }
    // Corner pixels beyond 2.5 px of the square's pixel centers stay out: (8, 8) lies √8 away.
    expect(coverageAt(fill, 8, 8)).toBe(0);
    expect(coverageAt(fill, 9, 8)).toBe(255);
    expect(fill.pixelCount).toBe(14 * 14 - 4);
  });

  it('shrinks an axis-aligned square by exactly the requested pixels', () => {
    const fill = region(squareSample(), { x: 14, y: 14 }, { expand: -2 });
    expect(fill.bounds).toEqual({ height: 6, width: 6, x: 12, y: 12 });
    expect(fill.pixelCount).toBe(36);
  });

  it('treats the sample edge as neither inside nor outside', () => {
    const sample = createSample(30, 20, WHITE);
    paintRect(sample, 15, 0, 15, 20, INK);
    const fill = region(sample, { x: 3, y: 3 }, { expand: -2 });
    expect(fill.bounds).toEqual({ height: 20, width: 13, x: 0, y: 0 });
  });

  it('returns an empty region when shrinking removes everything', () => {
    const fill = region(squareSample(), { x: 14, y: 14 }, { expand: -5 });
    expect(fill).toEqual({ bounds: null, coverage: null, pixelCount: 0 });
  });

  it('closes the line art’s 2 px gap but not its 6 px gap at closeGaps 2', () => {
    const sample = createLineArtSample();
    const { narrow, outside, wide } = LINE_ART;
    const inInterior = (x: number, y: number) =>
      x >= narrow.interior.x &&
      x < narrow.interior.x + narrow.interior.width &&
      y >= narrow.interior.y &&
      y < narrow.interior.y + narrow.interior.height;

    expect(coverageAt(region(sample, narrow.seed), outside.x, outside.y)).toBe(255);

    const closed = region(sample, narrow.seed, { closeGaps: 2 });
    expect(coverageAt(closed, outside.x, outside.y)).toBe(0);
    expect(closed.bounds && closed.bounds.y).toBeGreaterThanOrEqual(narrow.interior.y - 2);
    for (let y = 0; y < LINE_ART.height; y++) {
      for (let x = 0; x < LINE_ART.width; x++) {
        if (inInterior(x, y)) {
          expect(coverageAt(closed, x, y)).toBe(255);
        }
      }
    }

    const open = region(sample, wide.seed, { closeGaps: 2 });
    expect(coverageAt(open, outside.x, outside.y)).toBe(255);
    // The flood that leaks out of the wide box still cannot enter the narrow one.
    expect(coverageAt(open, narrow.seed.x, narrow.seed.y)).toBe(0);
  });

  it('closes exactly the gaps up to closeGaps px wide, odd widths included', () => {
    /** 30 × 30 white, an ink box (5..24) with walls `thickness` px and a `gap` px opening centered in its top wall. */
    const boxSample = (thickness: number, gap: number) => {
      const sample = createSample(30, 30, WHITE);
      paintRect(sample, 5, 5, 20, thickness, INK);
      paintRect(sample, 5, 25 - thickness, 20, thickness, INK);
      paintRect(sample, 5, 5, thickness, 20, INK);
      paintRect(sample, 25 - thickness, 5, thickness, 20, INK);
      paintRect(sample, 15 - Math.floor(gap / 2), 5, gap, thickness, WHITE);
      return sample;
    };
    const leaks: string[] = [];
    for (let gap = 1; gap <= 6; gap++) {
      for (let closeGaps = 0; closeGaps <= 7; closeGaps++) {
        const fill = region(boxSample(2, gap), { x: 15, y: 15 }, { closeGaps });
        expect(coverageAt(fill, 15, 15)).toBe(255);
        if (coverageAt(fill, 0, 0) === 255) {
          leaks.push(`${gap}/${closeGaps}`);
        } else {
          // A closed gap fills at most to the wall's outer edge: nothing spills above the box.
          expect(fill.bounds && fill.bounds.y).toBeGreaterThanOrEqual(5);
        }
      }
    }
    const expected: string[] = [];
    for (let gap = 1; gap <= 6; gap++) {
      for (let closeGaps = 0; closeGaps < gap; closeGaps++) {
        expected.push(`${gap}/${closeGaps}`);
      }
    }
    expect(leaks).toEqual(expected);

    // A closed 1 px wall never leaks, and the dilation gives the interior back up to the wall.
    const walled = region(boxSample(1, 0), { x: 15, y: 15 }, { closeGaps: 8 });
    expect(walled.bounds).toEqual({ height: 18, width: 18, x: 6, y: 6 });
    expect(walled.pixelCount).toBe(18 * 18);
  });

  it('does not grow back through a closed gap into a narrow band behind the line', () => {
    // A 1 px line at x = 10 with a 2 px gap (y = 9, 10); a solid line at x = 14 seals the 3 px band x = 11..13.
    const sample = createSample(30, 20, WHITE);
    paintRect(sample, 10, 0, 1, 20, INK);
    paintRect(sample, 10, 9, 1, 2, WHITE);
    paintRect(sample, 14, 0, 1, 20, INK);
    const seed = { x: 3, y: 10 };

    expect(coverageAt(region(sample, seed), 12, 0)).toBe(255);
    const fill = region(sample, seed, { closeGaps: 4 });
    for (let y = 0; y < 20; y++) {
      for (let x = 0; x < 30; x++) {
        expect(coverageAt(fill, x, y)).toBe(x < 10 ? 255 : x === 10 && (y === 9 || y === 10) ? 255 : 0);
      }
    }
  });

  it('anti-aliases only the boundary ring and keeps a straight edge’s area', () => {
    const sample = createSample(32, 32, WHITE);
    // A disc of ink, radius 9 around (15.5, 15.5).
    for (let y = 0; y < 32; y++) {
      for (let x = 0; x < 32; x++) {
        if ((x - 15.5) ** 2 + (y - 15.5) ** 2 <= 81) {
          paintRect(sample, x, y, 1, 1, INK);
        }
      }
    }
    const hard = coverageGrid(region(sample, { x: 15, y: 15 }), 32, 32);
    for (const expand of [0, 2, -2]) {
      const hardExpanded = coverageGrid(region(sample, { x: 15, y: 15 }, { expand }), 32, 32);
      const soft = coverageGrid(region(sample, { x: 15, y: 15 }, { antiAlias: true, expand }), 32, 32);
      let intermediate = 0;
      for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 32; x++) {
          const inside = hardExpanded[y][x] === 255;
          const onRing = [-1, 0, 1].some((dy) =>
            [-1, 0, 1].some((dx) => ((hardExpanded[y + dy]?.[x + dx] ?? hardExpanded[y][x]) === 255) !== inside)
          );
          const value = soft[y][x];
          if (value !== 0 && value !== 255) {
            intermediate++;
            expect(onRing).toBe(true);
          } else if (!onRing) {
            expect(value).toBe(hardExpanded[y][x]);
          }
        }
      }
      expect(intermediate).toBeGreaterThan(0);
    }
    expect(hard.flat().every((value) => value === 0 || value === 255)).toBe(true);

    const edge = region(squareSample(), { x: 14, y: 14 }, { antiAlias: true });
    expect(coverageAt(edge, 9, 14) + coverageAt(edge, 10, 14)).toBe(255);
    expect(coverageAt(edge, 11, 14)).toBe(255);
  });

  it('gives the same answers when one scratch serves a sequence of requests', () => {
    const sample = createLineArtSample();
    const map = buildFillLevelMap(sample, LINE_ART.narrow.seed.x, LINE_ART.narrow.seed.y, true);
    const scratch = createRegionScratch();
    const requests: FillRegionParams[] = [
      { antiAlias: true, closeGaps: 0, expand: 4, tolerance: 100 },
      { antiAlias: false, closeGaps: 3, expand: 0, tolerance: 20 },
      { antiAlias: true, closeGaps: 0, expand: -1, tolerance: 20 },
    ];
    for (const params of requests) {
      expect(computeFillRegion(map, params, scratch)).toEqual(computeFillRegion(map, params));
    }
  });

  it('thresholds globally when not contiguous', () => {
    const sample = squareSample();
    paintRect(sample, 0, 0, 3, 3, INK);
    const fill = computeFillRegion(buildFillLevelMap(sample, 14, 14, false), HARD);
    expect(fill.bounds).toEqual({ height: 20, width: 20, x: 0, y: 0 });
    expect(fill.pixelCount).toBe(109);
  });
});
