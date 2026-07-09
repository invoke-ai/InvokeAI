import type { CanvasLayerBaseContract } from '@workbench/types';

import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { describe, expect, it } from 'vitest';

import { mergeDownMatrix } from './mergeDown';

type Transform = CanvasLayerBaseContract['transform'];

const transform = (patch: Partial<Transform> = {}): Transform => ({
  rotation: 0,
  scaleX: 1,
  scaleY: 1,
  x: 0,
  y: 0,
  ...patch,
});

const toMat = (t: Transform) => fromTRS({ x: t.x, y: t.y }, t.rotation, t.scaleX, t.scaleY);

const closeTo = (a: number, b: number) => Math.abs(a - b) < 1e-9;

describe('mergeDownMatrix', () => {
  it('is the identity when both transforms are identity', () => {
    expect(mergeDownMatrix(transform(), transform())).toEqual({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  });

  it('translates the upper cache into the below layer local space (offset below cancels)', () => {
    // Below sits at (10, 20); the upper layer sits at the document origin. In
    // below-local space the origin is therefore at (-10, -20).
    const m = mergeDownMatrix(transform({ x: 10, y: 20 }), transform());
    expect(m).not.toBeNull();
    expect(m!.e).toBeCloseTo(-10);
    expect(m!.f).toBeCloseTo(-20);
  });

  it('satisfies below · M = above, so merged pixels land at their original document position', () => {
    const below = transform({ scaleX: 2, scaleY: 2, x: 5, y: 7 });
    const above = transform({ scaleX: 3, x: 40, y: 12 });
    const m = mergeDownMatrix(below, above);
    expect(m).not.toBeNull();

    const recomposed = multiply(toMat(below), m!);
    const expected = toMat(above);
    for (const key of ['a', 'b', 'c', 'd', 'e', 'f'] as const) {
      expect(closeTo(recomposed[key], expected[key])).toBe(true);
    }
  });

  it('returns null when the below transform is non-invertible (zero scale)', () => {
    expect(mergeDownMatrix(transform({ scaleX: 0 }), transform())).toBeNull();
  });
});
