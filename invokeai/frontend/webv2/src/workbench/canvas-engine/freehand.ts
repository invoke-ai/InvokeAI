/**
 * Pressure-sampled strokes use pure outline geometry and an injected Path2D factory. Large brushes need three
 * corrections to perfect-freehand: decimate subpixel direction jitter that creates radius-sized corner caps; keep
 * size*smoothing near {@link TARGET_VERTEX_SPACING}; and use {@link START_GATE_SIZE} independently of brush
 * diameter to retain early pressure samples. {@link traceSmoothPolygon} uses quadratic curves for remaining
 * outline facets.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { getStrokeOutlinePoints, getStrokePoints } from 'perfect-freehand';

/** A single pressure-sampled input point for a stroke. */
export interface StrokeSamplePoint {
  x: number;
  y: number;
  /** Pressure in [0, 1]. */
  pressure: number;
}

/** Tuning for the freehand outline. `size` is the base stroke diameter in document units. */
export interface FreehandOptions {
  /** Base stroke diameter (document units). */
  size: number;
  /** Effect of pressure on width, [0, 1]. 0 disables pressure sensitivity. */
  thinning?: number;
  /**
   * Outline vertex density, [0, 1]. Defaults to {@link outlineSmoothing} of
   * `size`; only set this to override the size-relative default.
   */
  smoothing?: number;
  /** Input smoothing / lag, [0, 1]. */
  streamline?: number;
  /** Whether the stroke is complete (adds the end cap). */
  last?: boolean;
}

/** Factory for a `Path2D`; injected so the polygon math stays node-testable. */
export type CreatePath2D = (d?: string) => Path2D;

const DEFAULT_SMOOTHING = 0.5;
const DEFAULT_STREAMLINE = 0.5;
const DEFAULT_THINNING = 0.5;

/**
 * Size-independent outline spacing via {@link outlineSmoothing}, chosen for quadratic tracing. Finer spacing
 * increases per-pointer-batch tessellation and fill work.
 */
export const TARGET_VERTEX_SPACING = 24;

/** Smallest spacing (document units) the input decimator will gate on. */
export const MIN_SAMPLE_SPACING = 1;

/**
 * Largest spacing (document units) the input decimator will gate on. Kept well
 * under {@link TARGET_VERTEX_SPACING} so decimation can never drop detail the
 * outline it feeds could have represented.
 */
export const MAX_SAMPLE_SPACING = 8;

/** Fraction of the brush diameter used as the input decimation spacing. */
const SAMPLE_SPACING_RATIO = 0.02;

/**
 * Diameter (document units) handed to `getStrokePoints` purely to size its
 * start-of-stroke noise gate — see the module docs. Never the real brush size.
 */
export const START_GATE_SIZE = 24;

/** Squared distance between two sample points. */
const dist2 = (a: StrokeSamplePoint, b: StrokeSamplePoint): number => {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
};

/** Minimum spacing (document units) between retained input samples for `size`. */
export const sampleSpacing = (size: number): number =>
  Math.min(MAX_SAMPLE_SPACING, Math.max(MIN_SAMPLE_SPACING, size * SAMPLE_SPACING_RATIO));

/** Holds vertex spacing near {@link TARGET_VERTEX_SPACING}; small brushes retain {@link DEFAULT_SMOOTHING}. */
export const outlineSmoothing = (size: number): number =>
  Math.min(DEFAULT_SMOOTHING, TARGET_VERTEX_SPACING / Math.max(size, 1));

/**
 * Distance-gates jitter with a greedy, prefix-stable scan so existing geometry stays fixed. Retain the exact final
 * sample only when `last`; live heads trail by at most spacing, bounded by {@link MAX_SAMPLE_SPACING}.
 */
export const decimateSamples = (
  points: readonly StrokeSamplePoint[],
  spacing: number,
  last: boolean
): StrokeSamplePoint[] => {
  const first = points[0];
  if (!first) {
    return [];
  }
  const minDistance = spacing * spacing;
  const kept: StrokeSamplePoint[] = [first];
  let anchor = first;
  for (let i = 1; i < points.length; i++) {
    const p = points[i]!;
    if (dist2(anchor, p) >= minDistance) {
      kept.push(p);
      anchor = p;
    }
  }
  // Land the completed stroke exactly on the last sample the pointer produced.
  const final = points[points.length - 1]!;
  if (last && anchor !== final) {
    kept.push(final);
  }
  return kept;
};

/**
 * Appends samples one at a time while keeping {@link decimateSamples}' output for the prefix seen so far, so a
 * growing stroke never re-scans the samples it has already gated.
 */
export interface SampleDecimator {
  push(point: StrokeSamplePoint): void;
  /** Exactly `decimateSamples(pushed, spacing, last)`; the returned array must not be mutated. */
  samples(last: boolean): readonly StrokeSamplePoint[];
}

export const createSampleDecimator = (spacing: number): SampleDecimator => {
  const minDistance = spacing * spacing;
  const kept: StrokeSamplePoint[] = [];
  let final: StrokeSamplePoint | null = null;
  return {
    push: (point) => {
      final = point;
      const anchor = kept[kept.length - 1];
      if (!anchor || dist2(anchor, point) >= minDistance) {
        kept.push(point);
      }
    },
    samples: (last) => (last && final && kept[kept.length - 1] !== final ? [...kept, final] : kept),
  };
};

/**
 * Document-space outline of already-decimated samples. Separate perfect-freehand stages keep the start gate small
 * while using the real diameter for outline width.
 */
export const strokeOutlineFromSamples = (samples: readonly StrokeSamplePoint[], opts: FreehandOptions): Vec2[] => {
  const first = samples[0];
  if (!first) {
    return [];
  }
  const last = opts.last ?? false;
  // perfect-freehand gives a lone sample a pen-oriented fallback dot whose
  // diameter bottoms out around 2.5px. Repeating the tap makes it use the real
  // outline size, which matters now that brushes can be smaller than one pixel.
  const input = samples.length === 1 ? [first, first] : samples;
  const strokePoints = getStrokePoints(
    input.map((p) => [p.x, p.y, p.pressure]),
    {
      last,
      size: Math.min(opts.size, START_GATE_SIZE),
      streamline: opts.streamline ?? DEFAULT_STREAMLINE,
    }
  );
  const outline = getStrokeOutlinePoints(strokePoints, {
    last,
    simulatePressure: false,
    size: opts.size,
    smoothing: opts.smoothing ?? outlineSmoothing(opts.size),
    thinning: opts.thinning ?? DEFAULT_THINNING,
  });
  return outline.map(([x, y]) => ({ x, y }));
};

/** Pure document-space stroke outline of raw samples, decimated for `opts.size`. */
export const strokeOutlinePolygon = (points: readonly StrokeSamplePoint[], opts: FreehandOptions): Vec2[] =>
  strokeOutlineFromSamples(decimateSamples(points, sampleSpacing(opts.size), opts.last ?? false), opts);

/** Serializes an outline polygon to an SVG path string (`M … L … Z`). */
export const polygonToSvgPath = (polygon: readonly Vec2[]): string => {
  if (polygon.length === 0) {
    return '';
  }
  const [first, ...rest] = polygon;
  let d = `M ${first.x} ${first.y}`;
  for (const p of rest) {
    d += ` L ${p.x} ${p.y}`;
  }
  return `${d} Z`;
};

/**
 * Trace quadratic segments between edge midpoints using vertices as controls. The curve stays inside the polygon
 * hull, preserving dirty bounds and collinear runs. Direct path calls avoid SVG parsing on each batch; fewer than
 * three vertices use lines.
 */
export const traceSmoothPolygon = (path: Path2D, polygon: readonly Vec2[]): void => {
  const n = polygon.length;
  if (n === 0) {
    return;
  }
  if (n < 3) {
    const [first, ...rest] = polygon;
    path.moveTo(first!.x, first!.y);
    for (const point of rest) {
      path.lineTo(point.x, point.y);
    }
    path.closePath();
    return;
  }
  // Start on the seam midpoint so the ring closes as smoothly as it turns.
  const seamA = polygon[n - 1]!;
  const seamB = polygon[0]!;
  path.moveTo((seamA.x + seamB.x) / 2, (seamA.y + seamB.y) / 2);
  for (let i = 0; i < n; i++) {
    const control = polygon[i]!;
    const next = polygon[(i + 1) % n]!;
    path.quadraticCurveTo(control.x, control.y, (control.x + next.x) / 2, (control.y + next.y) / 2);
  }
  path.closePath();
};

/** Axis-aligned bounds of an outline polygon; an empty polygon yields a zero-size rect. */
export const polygonBounds = (polygon: readonly Vec2[]): Rect => {
  if (polygon.length === 0) {
    return { height: 0, width: 0, x: 0, y: 0 };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of polygon) {
    if (p.x < minX) {
      minX = p.x;
    }
    if (p.y < minY) {
      minY = p.y;
    }
    if (p.x > maxX) {
      maxX = p.x;
    }
    if (p.y > maxY) {
      maxY = p.y;
    }
  }
  return { height: maxY - minY, width: maxX - minX, x: minX, y: minY };
};

/** Traces an outline polygon into a fresh path and reports its bounds. */
export const polygonToPath = (
  polygon: Vec2[],
  createPath2D: CreatePath2D
): { path: Path2D; polygon: Vec2[]; bounds: Rect } => {
  const path = createPath2D();
  traceSmoothPolygon(path, polygon);
  return { bounds: polygonBounds(polygon), path, polygon };
};

/**
 * Builds the smooth stroke path through an injected Path2D factory and returns enclosing polygon bounds, avoiding
 * duplicate outline computation.
 */
export const strokeToPath = (
  points: readonly StrokeSamplePoint[],
  opts: FreehandOptions,
  createPath2D: CreatePath2D
): { path: Path2D; polygon: Vec2[]; bounds: Rect } => polygonToPath(strokeOutlinePolygon(points, opts), createPath2D);
