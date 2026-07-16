import type { StrokeSamplePoint } from '@workbench/canvas-engine/freehand';
import type { Vec2 } from '@workbench/canvas-engine/types';

import { polygonBounds, polygonToSvgPath, strokeOutlinePolygon, strokeToPath } from '@workbench/canvas-engine/freehand';
import { describe, expect, it } from 'vitest';

/** Signed polygon area (shoelace); sign indicates winding, magnitude the area. */
const signedArea = (polygon: readonly Vec2[]): number => {
  let sum = 0;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i]!;
    const b = polygon[(i + 1) % polygon.length]!;
    sum += a.x * b.y - b.x * a.y;
  }
  return sum / 2;
};

const horizontalLine = (pressure: number): StrokeSamplePoint[] =>
  Array.from({ length: 9 }, (_, i) => ({ pressure, x: 10 + i * 10, y: 50 }));

describe('strokeOutlinePolygon', () => {
  it('returns an empty polygon for no points', () => {
    expect(strokeOutlinePolygon([], { size: 20 })).toEqual([]);
  });

  it('produces a closed, non-degenerate outline around a straight line', () => {
    const polygon = strokeOutlinePolygon(horizontalLine(0.5), { size: 20, thinning: 0 });
    expect(polygon.length).toBeGreaterThan(2);
    // A real 2D band, not a collinear degenerate: non-zero winding area.
    expect(Math.abs(signedArea(polygon))).toBeGreaterThan(0);
  });

  it('brackets the input points with a width near the base size (no thinning)', () => {
    const size = 20;
    const bounds = polygonBounds(strokeOutlinePolygon(horizontalLine(0.5), { size, thinning: 0 }));
    // The band spans the drawn length (~80px) plus the round caps.
    expect(bounds.width).toBeGreaterThan(80);
    // With thinning disabled the perpendicular extent tracks the base diameter.
    expect(bounds.height).toBeGreaterThan(size * 0.5);
    expect(bounds.height).toBeLessThan(size * 2);
    // Centered on the y=50 line the points sit on.
    expect(bounds.y).toBeLessThan(50);
    expect(bounds.y + bounds.height).toBeGreaterThan(50);
  });

  it('widens the stroke as pressure increases when thinning is enabled', () => {
    const opts = { size: 40, thinning: 0.8 } as const;
    const light = polygonBounds(strokeOutlinePolygon(horizontalLine(0.1), opts));
    const heavy = polygonBounds(strokeOutlinePolygon(horizontalLine(0.95), opts));
    expect(heavy.height).toBeGreaterThan(light.height);
  });
});

describe('polygonToSvgPath', () => {
  it('serializes a polygon to a closed move/line path', () => {
    const path = polygonToSvgPath([
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 5 },
    ]);
    expect(path).toBe('M 0 0 L 10 0 L 10 5 Z');
  });

  it('returns an empty string for an empty polygon', () => {
    expect(polygonToSvgPath([])).toBe('');
  });
});

describe('strokeToPath', () => {
  it('builds the path via the injected factory and returns the polygon bounds', () => {
    const seen: string[] = [];
    const marker = { __brand: 'path' } as unknown as Path2D;
    const createPath2D = (d?: string): Path2D => {
      seen.push(d ?? '');
      return marker;
    };
    const result = strokeToPath(horizontalLine(0.5), { size: 20, thinning: 0 }, createPath2D);

    expect(result.path).toBe(marker);
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatch(/^M /);
    expect(result.polygon.length).toBeGreaterThan(2);
    expect(result.bounds).toEqual(polygonBounds(result.polygon));
  });
});
