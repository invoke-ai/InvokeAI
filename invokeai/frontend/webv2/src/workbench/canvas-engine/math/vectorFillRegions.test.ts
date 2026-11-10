import { describe, expect, it } from 'vitest';

import type { Vec2 } from '../types';

import { findVectorFillRegions, findVectorFillRegionsAsync, VectorFillLimitError } from './vectorFillRegions';

const area = (points: Vec2[]) => points.reduce((sum, point, i) => {
  const next = points[(i + 1) % points.length]!;
  return sum + point.x * next.y - next.x * point.y;
}, 0) / 2;

describe('vector fill regions', () => {
  it('finds a loop in an open self-crossing path without filling its tails', () => {
    const regions = findVectorFillRegions([[{ x: -2, y: -2 }, { x: 2, y: 2 },
      { x: -2, y: 2 }, { x: 2, y: -2 }]]);
    expect(regions).toHaveLength(1);
    expect(area(regions[0]!)).toBeCloseTo(4);
    expect(regions[0]!.every((point) => point.y >= 0)).toBe(true);
  });

  it('finds a region bounded by separate open paths and preserves the input', () => {
    const paths = [[{ x: -1, y: 0 }, { x: 3, y: 0 }], [{ x: 2, y: -1 }, { x: 2, y: 3 }],
      [{ x: 3, y: 2 }, { x: -1, y: 2 }], [{ x: 0, y: 3 }, { x: 0, y: -1 }]];
    const before = structuredClone(paths);
    const regions = findVectorFillRegions(paths);
    expect(regions).toHaveLength(1);
    expect(area(regions[0]!)).toBeCloseTo(4);
    expect(paths).toEqual(before);
  });

  it('does not fill an isolated crossing or bridge a gap', () => {
    expect(findVectorFillRegions([[{ x: -1, y: 0 }, { x: 1, y: 0 }],
      [{ x: 0, y: -1 }, { x: 0, y: 1 }]])).toEqual([]);
    expect(findVectorFillRegions([[{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 2, y: 2 },
      { x: 0, y: 2 }, { x: 0, y: 0.01 }]])).toEqual([]);
  });

  it('keeps both bounded faces of a divided closed contour', () => {
    const regions = findVectorFillRegions([[{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 2, y: 2 },
      { x: 0, y: 2 }, { x: 0, y: 0 }], [{ x: 1, y: -1 }, { x: 1, y: 3 }]]);
    expect(regions).toHaveLength(2);
    expect(regions.map(area)).toEqual([2, 2]);
  });
});


it('yields for cancellation and enforces work budgets', async () => {
  const paths = [[{ x: 0, y: 0 }, { x: 10, y: 10 }], [{ x: 0, y: 10 }, { x: 10, y: 0 }]];
  const abort = new AbortController();
  await expect(findVectorFillRegionsAsync(paths, { signal: abort.signal,
    yieldControl: async () => { abort.abort(); } })).rejects.toMatchObject({ name: 'AbortError' });
  await expect(findVectorFillRegionsAsync(paths, { yieldControl: async () => {},
    limits: { segments: 100, comparisons: 0, edges: 100 } })).rejects.toBeInstanceOf(VectorFillLimitError);
  expect(await findVectorFillRegionsAsync(paths, { yieldControl: async () => {} })).toEqual(findVectorFillRegions(paths));
});
