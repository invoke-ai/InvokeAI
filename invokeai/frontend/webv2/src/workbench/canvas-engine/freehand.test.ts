import type { StrokeSamplePoint } from '@workbench/canvas-engine/freehand';
import type { Vec2 } from '@workbench/canvas-engine/types';

import {
  createSampleDecimator,
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

const noise = (seed: number): (() => number) => {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff - 0.5;
  };
};

const slowDrag = (speed: number, jitter: number): StrokeSamplePoint[] => {
  const rand = noise(7);
  return Array.from({ length: 400 }, (_, i) => ({
    pressure: 0.5,
    x: i * speed + rand() * jitter,
    y: rand() * jitter,
  }));
};

describe('strokeOutlinePolygon', () => {
  it('keeps a sub-pixel tap near the requested diameter', () => {
    const bounds = polygonBounds(strokeOutlinePolygon([{ pressure: 0.5, x: 10, y: 10 }], { last: true, size: 0.1 }));
    expect(bounds.width).toBeGreaterThan(0);
    expect(bounds.width).toBeLessThan(0.2);
    expect(bounds.height).toBeGreaterThan(0);
    expect(bounds.height).toBeLessThan(0.2);
  });

  it('produces a closed, non-degenerate outline around a straight line', () => {
    const polygon = strokeOutlinePolygon(horizontalLine(0.5), { size: 20, thinning: 0 });
    expect(polygon.length).toBeGreaterThan(2);
    expect(Math.abs(signedArea(polygon))).toBeGreaterThan(0);
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

  it('lands on the final sample only once the stroke is complete', () => {
    const points = dense();
    const final = points[points.length - 1]!;
    expect(decimateSamples(points, 4, false)).not.toContain(final);
    expect(decimateSamples(points, 4, true)).toContain(final);
  });

  it('retains a prefix of the full result for any prefix of the input', () => {
    const points = dense();
    const full = decimateSamples(points, 4, false);
    for (const n of [5, 17, 33]) {
      const partial = decimateSamples(points.slice(0, n), 4, false);
      expect(full.slice(0, partial.length)).toEqual(partial);
    }
  });
});

describe('createSampleDecimator', () => {
  it('matches decimateSamples for every prefix, live and completed', () => {
    const jittery: StrokeSamplePoint[] = Array.from({ length: 120 }, (_, i) => ({
      pressure: 0.3 + (i % 7) / 10,
      x: i * 0.7 + Math.sin(i) * 0.9,
      y: Math.cos(i / 3) * 6,
    }));
    const decimator = createSampleDecimator(4);
    jittery.forEach((point, index) => {
      decimator.push(point);
      const prefix = jittery.slice(0, index + 1);
      expect(decimator.samples(false)).toEqual(decimateSamples(prefix, 4, false));
      expect(decimator.samples(true)).toEqual(decimateSamples(prefix, 4, true));
    });
  });
});

describe('sampleSpacing', () => {
  it('scales with the brush but stays within its floor and ceiling', () => {
    expect(sampleSpacing(10)).toBe(1);
    expect(sampleSpacing(100)).toBe(2);
    expect(sampleSpacing(2000)).toBe(MAX_SAMPLE_SPACING);
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
    const jittery = slowDrag(0.2, 1);
    const clean = slowDrag(0.2, 0);
    for (const size of [20, 100, 400, 1000, 2000]) {
      const withJitter = strokeOutlinePolygon(jittery, { size, thinning: 0 }).length;
      const without = strokeOutlinePolygon(clean, { size, thinning: 0 }).length;
      expect(withJitter).toBeLessThanOrEqual(without * 2);
    }
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
});

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
    expect(commands.join(' ')).toBe('M 0 5 Q 0 0 5 0 Q 10 0 10 5 Q 10 10 5 10 Q 0 10 0 5 Z');
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
    expect(recorder.commands.some((command) => command.startsWith('Q '))).toBe(true);
    expect(result.polygon.length).toBeGreaterThan(2);
    expect(result.bounds).toEqual(polygonBounds(result.polygon));
  });
});
