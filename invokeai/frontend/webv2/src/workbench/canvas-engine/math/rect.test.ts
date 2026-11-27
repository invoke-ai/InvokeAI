import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import { identity, rotate, translate } from './mat2d';
import { containsPoint, expand, intersect, isEmpty, mergeDirtyRects, roundOut, transformBounds, union } from './rect';

describe('isEmpty', () => {
  it('is true for zero width or height', () => {
    expect(isEmpty({ x: 0, y: 0, width: 0, height: 10 })).toBe(true);
    expect(isEmpty({ x: 0, y: 0, width: 10, height: 0 })).toBe(true);
  });
});

describe('intersect', () => {
  it('returns the overlapping region for overlapping rects', () => {
    const a: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const b: Rect = { x: 5, y: 5, width: 10, height: 10 };
    expect(intersect(a, b)).toEqual({ x: 5, y: 5, width: 5, height: 5 });
  });

  it('returns null for disjoint rects', () => {
    const a: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const b: Rect = { x: 20, y: 20, width: 10, height: 10 };
    expect(intersect(a, b)).toBeNull();
  });
});

describe('union', () => {
  it('returns the bounding box of two rects', () => {
    const a: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const b: Rect = { x: 5, y: -5, width: 10, height: 10 };
    expect(union(a, b)).toEqual({ x: 0, y: -5, width: 15, height: 15 });
  });

  it('returns the non-empty rect when the other is empty', () => {
    const a: Rect = { x: 0, y: 0, width: 0, height: 0 };
    const b: Rect = { x: 5, y: 5, width: 10, height: 10 };
    expect(union(a, b)).toEqual(b);
    expect(union(b, a)).toEqual(b);
  });
});

describe('containsPoint', () => {
  const r: Rect = { x: 0, y: 0, width: 10, height: 10 };

  it('is true for points inside the rect', () => {
    expect(containsPoint(r, { x: 5, y: 5 })).toBe(true);
    expect(containsPoint(r, { x: 0, y: 0 })).toBe(true);
  });

  it('is false for points on the max edge (exclusive)', () => {
    expect(containsPoint(r, { x: 10, y: 5 })).toBe(false);
    expect(containsPoint(r, { x: 5, y: 10 })).toBe(false);
  });
});

describe('expand', () => {
  it('grows the rect on all sides by margin', () => {
    const r: Rect = { x: 10, y: 10, width: 10, height: 10 };
    expect(expand(r, 5)).toEqual({ x: 5, y: 5, width: 20, height: 20 });
  });
});

describe('roundOut', () => {
  it('floors the min edges and ceils the max edges', () => {
    const r: Rect = { x: 1.2, y: 1.8, width: 3.5, height: 2.1 };
    // right edge = 1.2 + 3.5 = 4.7 -> ceil 5; bottom = 1.8 + 2.1 = 3.9 -> ceil 4
    expect(roundOut(r)).toEqual({ x: 1, y: 1, width: 4, height: 3 });
  });
});

describe('transformBounds', () => {
  it('translates the rect for a translation matrix', () => {
    const r: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const m = translate(identity(), { x: 5, y: -5 });
    expect(transformBounds(m, r)).toEqual({ x: 5, y: -5, width: 10, height: 10 });
  });

  it('returns the axis-aligned bounding box of a rotated rect', () => {
    const r: Rect = { x: -5, y: -5, width: 10, height: 10 };
    const m = rotate(identity(), Math.PI / 4);
    const bounds = transformBounds(m, r);
    const diagonal = Math.sqrt(2) * 10;
    expect(bounds.width).toBeCloseTo(diagonal, 8);
    expect(bounds.height).toBeCloseTo(diagonal, 8);
    expect(bounds.x).toBeCloseTo(-diagonal / 2, 8);
    expect(bounds.y).toBeCloseTo(-diagonal / 2, 8);
  });
});

describe('mergeDirtyRects', () => {
  it('filters out empty rects', () => {
    const r: Rect = { x: 0, y: 0, width: 5, height: 5 };
    const empty: Rect = { x: 0, y: 0, width: 0, height: 0 };
    expect(mergeDirtyRects([r, empty])).toEqual([r]);
  });

  it('merges overlapping rects into one', () => {
    const a: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const b: Rect = { x: 5, y: 5, width: 10, height: 10 };
    const result = mergeDirtyRects([a, b]);
    expect(result).toEqual([{ x: 0, y: 0, width: 15, height: 15 }]);
  });

  it('falls back to a single merged bounding rect beyond maxRegions', () => {
    const rects: Rect[] = [
      { x: 0, y: 0, width: 5, height: 5 },
      { x: 100, y: 0, width: 5, height: 5 },
      { x: 0, y: 100, width: 5, height: 5 },
      { x: 100, y: 100, width: 5, height: 5 },
      { x: 200, y: 200, width: 5, height: 5 },
    ];
    const result = mergeDirtyRects(rects, 4);
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ x: 0, y: 0, width: 205, height: 205 });
  });
});
