import { describe, expect, it } from 'vitest';
import { anchorsToBezierPoints, getBezierPathHitSamplesPerSegment } from './bezierPath';
import { identity } from './mat2d';
import { VectorHitTestCache } from './vectorHitTest';

describe('cached vector hit testing', () => {
  it('preserves segment indices and parameters, rejects distant curves and refreshes projection/geometry', () => {
    const cache = new VectorHitTestCache();
    const path = { id: 'p', name: null, isClosed: false,
      points: anchorsToBezierPoints([{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }]) };
    expect(cache.hitAnchor(path, identity(), { x: 100, y: 0 }, 1)).toBe(true);
    expect(cache.hitAnchor(path, identity(), { x: 1000, y: 0 }, 1)).toBe(false);
    const hit = cache.find(path, identity(), { x: 100, y: 50 }, 1)!;
    expect(hit.segmentIndex).toBe(1); expect(hit.t).toBeCloseTo(0.5); expect(hit.distance).toBe(0);
    expect(cache.find(path, identity(), { x: 100, y: 50 }, 1)).toEqual(hit);
    expect(cache.find(path, identity(), { x: 1000, y: 1000 }, 1)).toBeNull();
    const moved = { ...identity(), e: 200 };
    expect(cache.find(path, moved, { x: 100, y: 50 }, 1)).toBeNull();
    expect(cache.find(path, moved, { x: 300, y: 50 }, 1)?.segmentIndex).toBe(1);
    const changed = { ...path, points: anchorsToBezierPoints([{ x: 0, y: 0 }, { x: 20, y: 0 }]) };
    expect(cache.find(changed, identity(), { x: 100, y: 50 }, 1)).toBeNull();
    cache.clear();
    expect(cache.find(path, identity(), { x: 100, y: 50 }, 1)).toEqual(hit);
  });
  it('caps sampling at extreme zoom/scale', () => {
    expect(getBezierPathHitSamplesPerSegment(1e20)).toBe(128);
    expect(getBezierPathHitSamplesPerSegment(1)).toBe(24);
  });
});
