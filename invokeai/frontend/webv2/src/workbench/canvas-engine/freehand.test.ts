import type { StrokeSamplePoint } from '@workbench/canvas-engine/freehand';
import type { Vec2 } from '@workbench/canvas-engine/types';

import {
  decimateSamples,
  MAX_SAMPLE_SPACING,
  outlineSmoothing,
  polygonBounds,
  polygonToSvgPath,
  sampleSpacing,
  strokeOutlinePolygon,
  strokeToPath,
  traceSmoothPolygon,
  TARGET_VERTEX_SPACING,
} from '@workbench/canvas-engine/freehand';
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

/** Deterministic LCG in [-0.5, 0.5), so jitter fixtures are reproducible. */
const noise = (seed: number): (() => number) => {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff - 0.5;
  };
};

/**
 * A slow, near-straight drag advancing `speed` document units per sample. Below
 * ~0.5 units/sample the jitter dominates the direction vector — the regime that
 * used to fill the outline with spurious full-radius corner caps.
 */
const slowDrag = (speed: number, jitter: number): StrokeSamplePoint[] => {
  const rand = noise(7);
  return Array.from({ length: 400 }, (_, i) => ({
    pressure: 0.5,
    x: i * speed + rand() * jitter,
    y: rand() * jitter,
  }));
};

/** An arc of `sweep` radians at `radius`, sampled every ~4 document units. */
const arc = (radius: number, sweep: number): StrokeSamplePoint[] => {
  const step = 4 / radius;
  const points: StrokeSamplePoint[] = [];
  for (let a = 0; a <= sweep; a += step) {
    points.push({ pressure: 0.5, x: radius * Math.cos(a), y: radius * Math.sin(a) });
  }
  return points;
};

/**
 * The edge lengths along the body of a stroke outline — how coarsely it
 * approximates the offset curve, which under any serialization is the visible
 * faceting. Edges within `size` of either end are excluded: the caps are drawn
 * with a fixed segment count rather than by the vertex cull, so they say nothing
 * about body density.
 */
const bodyEdgeLengths = (polygon: readonly Vec2[], points: readonly StrokeSamplePoint[], size: number): number[] => {
  const ends = [points[0]!, points[points.length - 1]!];
  const lengths: number[] = [];
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i]!;
    const b = polygon[(i + 1) % polygon.length]!;
    const inCap = [a, b].some((q) => ends.some((e) => Math.hypot(q.x - e.x, q.y - e.y) < size));
    if (!inCap) {
      lengths.push(Math.hypot(b.x - a.x, b.y - a.y));
    }
  }
  return lengths;
};

/** Total vertical extent of an outline where it crosses the vertical line at `x`. */
const widthAt = (polygon: readonly Vec2[], x: number): number => {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i]!;
    const b = polygon[(i + 1) % polygon.length]!;
    if ((a.x - x) * (b.x - x) > 0) {
      continue;
    }
    const y = a.x === b.x ? a.y : a.y + ((x - a.x) / (b.x - a.x)) * (b.y - a.y);
    lo = Math.min(lo, y);
    hi = Math.max(hi, y);
  }
  return hi - lo;
};

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

describe('decimateSamples', () => {
  const dense = (): StrokeSamplePoint[] => Array.from({ length: 50 }, (_, i) => ({ pressure: 0.5, x: i * 0.25, y: 0 }));

  it('always keeps the first sample', () => {
    expect(decimateSamples(dense(), 4, false)[0]).toEqual({ pressure: 0.5, x: 0, y: 0 });
    expect(decimateSamples([], 4, true)).toEqual([]);
  });

  it('keeps retained samples at least `spacing` apart', () => {
    const kept = decimateSamples(dense(), 4, false);
    expect(kept.length).toBeGreaterThan(1);
    expect(kept.length).toBeLessThan(50);
    for (let i = 1; i < kept.length; i++) {
      expect(Math.hypot(kept[i]!.x - kept[i - 1]!.x, kept[i]!.y - kept[i - 1]!.y)).toBeGreaterThanOrEqual(4);
    }
  });

  it('lands on the final sample only once the stroke is complete', () => {
    const points = dense();
    const final = points[points.length - 1]!;
    expect(decimateSamples(points, 4, false)).not.toContain(final);
    expect(decimateSamples(points, 4, true)).toContain(final);
  });

  it('retains a prefix of the full result for any prefix of the input', () => {
    // The greedy scan is what keeps already-drawn geometry from shifting as new
    // samples arrive mid-gesture.
    const points = dense();
    const full = decimateSamples(points, 4, false);
    for (const n of [5, 17, 33]) {
      const partial = decimateSamples(points.slice(0, n), 4, false);
      expect(full.slice(0, partial.length)).toEqual(partial);
    }
  });
});

describe('sampleSpacing', () => {
  it('scales with the brush but stays within its floor and ceiling', () => {
    expect(sampleSpacing(10)).toBe(1);
    expect(sampleSpacing(100)).toBe(2);
    expect(sampleSpacing(2000)).toBe(MAX_SAMPLE_SPACING);
  });

  it('never gates more coarsely than the outline it feeds', () => {
    for (const size of [1, 20, 200, 2000]) {
      expect(sampleSpacing(size)).toBeLessThan(TARGET_VERTEX_SPACING);
    }
  });
});

describe('outlineSmoothing', () => {
  it('leaves small brushes on the library default', () => {
    expect(outlineSmoothing(1)).toBe(0.5);
    expect(outlineSmoothing(16)).toBe(0.5);
  });

  it('holds the size × smoothing product — the vertex cull — near the target', () => {
    for (const size of [100, 400, 2000]) {
      expect(size * outlineSmoothing(size)).toBeCloseTo(TARGET_VERTEX_SPACING);
    }
  });
});

describe('stroke quality at large brush sizes', () => {
  it('does not let pointer jitter inflate the outline (spurious corner caps)', () => {
    // Each >90° direction reversal splices a cap of the CURRENT radius into the
    // outline, so jitter used to add hundreds of full-size lumps to a big brush.
    const jittery = slowDrag(0.2, 1);
    const clean = slowDrag(0.2, 0);
    for (const size of [20, 100, 400, 1000, 2000]) {
      const withJitter = strokeOutlinePolygon(jittery, { size, thinning: 0 }).length;
      const without = strokeOutlinePolygon(clean, { size, thinning: 0 }).length;
      expect(withJitter).toBeLessThanOrEqual(without * 2);
    }
  });

  it('keeps outline vertex spacing bounded independently of brush size', () => {
    // The library culls outline vertices closer than (size × smoothing), so a
    // constant smoothing lets spacing — and with it the faceting — grow with the
    // brush. On this arc a 400px brush used to reach 200-unit edges.
    const curve = arc(1000, Math.PI / 2);
    for (const size of [50, 200, 400]) {
      const polygon = strokeOutlinePolygon(curve, { size, thinning: 0 });
      const edges = bodyEdgeLengths(polygon, curve, size);
      expect(edges.length).toBeGreaterThan(10);
      expect(Math.max(...edges)).toBeLessThanOrEqual(TARGET_VERTEX_SPACING * 2);
    }
  });

  it('tracks pressure from the start of a large-brush stroke', () => {
    // The library's start-of-stroke noise gate drops every intermediate point —
    // and so every pressure reading — until the running length reaches `size`.
    // Sized in document units instead of brush diameters, it no longer flattens
    // the pressure ramp across the first 1000 units of a 1000px brush.
    const size = 1000;
    const rampLength = 300;
    const stroke: StrokeSamplePoint[] = [];
    for (let x = 0; x <= 1200; x += 3) {
      stroke.push({ pressure: Math.min(1, 0.05 + x / rampLength), x, y: 0 });
    }
    const polygon = strokeOutlinePolygon(stroke, { last: true, size, thinning: 0.5 });
    // At full pressure `getStrokeRadius` gives size × 0.75, so width = 1.5 × size.
    expect(widthAt(polygon, rampLength)).toBeGreaterThan(size * 1.5 * 0.95);
  });

  it('leaves small brushes on the library defaults', () => {
    const line = horizontalLine(0.5);
    expect(strokeOutlinePolygon(line, { size: 20, smoothing: 0.5, thinning: 0 })).toEqual(
      strokeOutlinePolygon(line, { size: 20, thinning: 0 })
    );
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

/**
 * A node-safe `Path2D` double that records the commands traced onto it, so the
 * curve can be asserted without a DOM.
 */
const recordingPath = () => {
  const commands: string[] = [];
  const points: Vec2[] = [];
  const path = {
    closePath: () => commands.push('Z'),
    lineTo: (x: number, y: number) => {
      commands.push(`L ${x} ${y}`);
      points.push({ x, y });
    },
    moveTo: (x: number, y: number) => {
      commands.push(`M ${x} ${y}`);
      points.push({ x, y });
    },
    quadraticCurveTo: (cx: number, cy: number, x: number, y: number) => {
      commands.push(`Q ${cx} ${cy} ${x} ${y}`);
      points.push({ x: cx, y: cy }, { x, y });
    },
  };
  return { commands, path: path as unknown as Path2D, points };
};

describe('traceSmoothPolygon', () => {
  it('runs a quadratic through each vertex, between its edge midpoints', () => {
    const { commands, path } = recordingPath();
    traceSmoothPolygon(path, [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ]);
    // Starts on the seam midpoint (between the last and first vertices) so the
    // ring closes as smoothly as it turns.
    expect(commands.join(' ')).toBe('M 0 5 Q 0 0 5 0 Q 10 0 10 5 Q 10 10 5 10 Q 0 10 0 5 Z');
  });

  it('falls back to straight segments when there is no curvature to describe', () => {
    const empty = recordingPath();
    traceSmoothPolygon(empty.path, []);
    expect(empty.commands).toEqual([]);

    const dot = recordingPath();
    traceSmoothPolygon(dot.path, [{ x: 1, y: 2 }]);
    expect(dot.commands.join(' ')).toBe('M 1 2 Z');

    const line = recordingPath();
    traceSmoothPolygon(line.path, [
      { x: 0, y: 0 },
      { x: 4, y: 0 },
    ]);
    expect(line.commands.join(' ')).toBe('M 0 0 L 4 0 Z');
  });

  it('stays within the polygon bounds, so the dirty rect still encloses it', () => {
    // Every control point is a polygon vertex and every on-curve point an edge
    // midpoint, so the curve cannot leave the polygon's convex hull — which is
    // what lets `strokeToPath` keep reporting the polygon's bounds.
    const polygon = strokeOutlinePolygon(arc(1000, Math.PI / 2), { last: true, size: 400, thinning: 0 });
    const bounds = polygonBounds(polygon);
    const { path, points } = recordingPath();
    traceSmoothPolygon(path, polygon);
    expect(points.length).toBeGreaterThan(2);
    for (const point of points) {
      expect(point.x).toBeGreaterThanOrEqual(bounds.x);
      expect(point.x).toBeLessThanOrEqual(bounds.x + bounds.width);
      expect(point.y).toBeGreaterThanOrEqual(bounds.y);
      expect(point.y).toBeLessThanOrEqual(bounds.y + bounds.height);
    }
  });
});

describe('strokeToPath', () => {
  it('traces onto a path from the injected factory and returns the polygon bounds', () => {
    const recorder = recordingPath();
    let calls = 0;
    const createPath2D = (): Path2D => {
      calls++;
      return recorder.path;
    };
    const result = strokeToPath(horizontalLine(0.5), { size: 20, thinning: 0 }, createPath2D);

    expect(result.path).toBe(recorder.path);
    expect(calls).toBe(1);
    expect(recorder.commands[0]).toMatch(/^M /);
    // The smooth form — straight joins would show as facets at size.
    expect(recorder.commands.some((command) => command.startsWith('Q '))).toBe(true);
    expect(result.polygon.length).toBeGreaterThan(2);
    expect(result.bounds).toEqual(polygonBounds(result.polygon));
  });
});
