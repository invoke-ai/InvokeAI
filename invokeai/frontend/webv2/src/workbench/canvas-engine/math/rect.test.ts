import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import { identity, rotate, scale, translate } from './mat2d';
import { containsPoint, expand, intersect, isEmpty, mergeDirtyRects, roundOut, transformBounds, union } from './rect';

describe('isEmpty', () => {
  it('is true for zero width or height', () => {
    expect(isEmpty({ x: 0, y: 0, width: 0, height: 10 })).toBe(true);
    expect(isEmpty({ x: 0, y: 0, width: 10, height: 0 })).toBe(true);
  });

  it('is true for negative width or height', () => {
    expect(isEmpty({ x: 0, y: 0, width: -1, height: 10 })).toBe(true);
  });

  it('is false for a positive-area rect', () => {
    expect(isEmpty({ x: 0, y: 0, width: 1, height: 1 })).toBe(false);
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

  it('returns null for rects that only touch at an edge', () => {
    const a: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const b: Rect = { x: 10, y: 0, width: 10, height: 10 };
    expect(intersect(a, b)).toBeNull();
  });

  it('returns null when either rect is empty', () => {
    const a: Rect = { x: 0, y: 0, width: 0, height: 10 };
    const b: Rect = { x: 0, y: 0, width: 10, height: 10 };
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

  it('returns the first rect when both are empty', () => {
    const a: Rect = { x: 1, y: 1, width: 0, height: 0 };
    const b: Rect = { x: 2, y: 2, width: 0, height: 0 };
    expect(union(a, b)).toEqual(a);
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

  it('is false for points outside the rect', () => {
    expect(containsPoint(r, { x: -1, y: 5 })).toBe(false);
    expect(containsPoint(r, { x: 5, y: 20 })).toBe(false);
  });
});

describe('expand', () => {
  it('grows the rect on all sides by margin', () => {
    const r: Rect = { x: 10, y: 10, width: 10, height: 10 };
    expect(expand(r, 5)).toEqual({ x: 5, y: 5, width: 20, height: 20 });
  });

  it('shrinks the rect for negative margin', () => {
    const r: Rect = { x: 10, y: 10, width: 10, height: 10 };
    expect(expand(r, -2)).toEqual({ x: 12, y: 12, width: 6, height: 6 });
  });
});

describe('roundOut', () => {
  it('floors the min edges and ceils the max edges', () => {
    const r: Rect = { x: 1.2, y: 1.8, width: 3.5, height: 2.1 };
    // right edge = 1.2 + 3.5 = 4.7 -> ceil 5; bottom = 1.8 + 2.1 = 3.9 -> ceil 4
    expect(roundOut(r)).toEqual({ x: 1, y: 1, width: 4, height: 3 });
  });

  it('is a no-op for already-integer rects', () => {
    const r: Rect = { x: 2, y: 3, width: 4, height: 5 };
    expect(roundOut(r)).toEqual(r);
  });

  it('handles negative coordinates', () => {
    const r: Rect = { x: -1.5, y: -2.1, width: 3, height: 3 };
    // right = 1.5 -> ceil 2; bottom = 0.9 -> ceil 1
    expect(roundOut(r)).toEqual({ x: -2, y: -3, width: 4, height: 4 });
  });
});

describe('transformBounds', () => {
  it('returns the same rect for the identity matrix', () => {
    const r: Rect = { x: 1, y: 2, width: 10, height: 20 };
    expect(transformBounds(identity(), r)).toEqual(r);
  });

  it('translates the rect for a translation matrix', () => {
    const r: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const m = translate(identity(), { x: 5, y: -5 });
    expect(transformBounds(m, r)).toEqual({ x: 5, y: -5, width: 10, height: 10 });
  });

  it('scales the rect for a scale matrix', () => {
    const r: Rect = { x: 0, y: 0, width: 10, height: 10 };
    const m = scale(identity(), 2);
    expect(transformBounds(m, r)).toEqual({ x: 0, y: 0, width: 20, height: 20 });
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
  it('returns an empty array for no rects', () => {
    expect(mergeDirtyRects([])).toEqual([]);
  });

  it('returns the single rect unchanged (filtering empties)', () => {
    const r: Rect = { x: 0, y: 0, width: 5, height: 5 };
    expect(mergeDirtyRects([r])).toEqual([r]);
  });

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

  it('keeps far-apart rects separate when under the region cap', () => {
    const a: Rect = { x: 0, y: 0, width: 5, height: 5 };
    const b: Rect = { x: 1000, y: 1000, width: 5, height: 5 };
    const result = mergeDirtyRects([a, b]);
    expect(result).toHaveLength(2);
    expect(result).toEqual(expect.arrayContaining([a, b]));
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

  it('respects a custom maxRegions', () => {
    const rects: Rect[] = [
      { x: 0, y: 0, width: 5, height: 5 },
      { x: 100, y: 0, width: 5, height: 5 },
    ];
    const result = mergeDirtyRects(rects, 1);
    expect(result).toHaveLength(1);
  });
});
