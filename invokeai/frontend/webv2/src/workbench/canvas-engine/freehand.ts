/**
 * A thin wrapper over `perfect-freehand`, turning a pressure-sampled point path
 * into the filled outline polygon of a variable-width stroke.
 *
 * The polygon math ({@link strokeOutlinePolygon}) is pure and DOM-free, so it is
 * fully unit-testable in node. `Path2D` does not exist in node, so building the
 * fillable path is split out into {@link strokeToPath}, which takes an injected
 * `createPath2D` factory (the engine passes `() => new Path2D()`; tests pass a
 * recording stub). This keeps the interesting geometry testable without a DOM.
 *
 * ## Why this isn't a bare `getStroke` call
 *
 * `perfect-freehand` is tuned for pen-sized strokes; driven naively it degrades
 * badly as the brush diameter grows, in three independent ways. Each is
 * corrected here, and each correction is a no-op at small sizes:
 *
 * 1. **Input decimation** ({@link decimateSamples}). The library flags any turn
 *    sharper than 90° between consecutive point vectors as a corner and splices
 *    a cap of the CURRENT RADIUS into the outline. Nothing upstream decimates
 *    pointer input, so once the pointer advances less than ~0.5 document units
 *    per sample — a slow drag, a high-rate pen, or any zoom above ~2×, since
 *    document units per screen pixel shrink with zoom — sub-pixel jitter
 *    dominates the direction vector and fires those reversals constantly. At a
 *    10px brush the spurious caps are invisible; at a 1000px brush each one is a
 *    500px-radius lump. Distance-gating the samples removes them outright.
 *
 * 2. **Size-relative smoothing** ({@link outlineSmoothing}). `smoothing` feeds
 *    exactly one thing in the library: the `(size * smoothing)²` outline-vertex
 *    cull. Left at a constant, vertex spacing scales with the brush, so a large
 *    brush traces a smooth arc with edges hundreds of units long. Pinning the
 *    PRODUCT instead holds spacing near {@link TARGET_VERTEX_SPACING}.
 *
 * 3. **A decoupled start gate** ({@link START_GATE_SIZE}). `getStrokePoints`
 *    drops every intermediate point until the running length reaches `size`, to
 *    suppress start-of-stroke noise. Since `size` is the brush diameter, a 400px
 *    brush renders the first 400 document units as one straight capsule. Running
 *    the two library stages separately lets that gate keep its small absolute
 *    value while the outline stage gets the real diameter.
 *
 * Finally, the outline is traced as quadratic curves
 * ({@link traceSmoothPolygon}) rather than the straight segments the lasso
 * wants, so whatever facets remain read as curves.
 *
 * Zero React, zero import-time side effects.
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
 * Target spacing (document units) between adjacent outline vertices, held
 * roughly constant across brush sizes by {@link outlineSmoothing}.
 *
 * Chosen against the QUADRATIC curve, not straight segments: at this spacing the
 * rendered curve strays from the true offset curve by under 0.2 document pixels
 * even where the stroke turns on a 60px radius. Going finer only multiplies
 * vertices — the outline is re-tessellated and re-filled on every pointer batch,
 * so vertex count is squarely on the hot path.
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

/**
 * The `smoothing` value that holds outline vertex spacing near
 * {@link TARGET_VERTEX_SPACING} for a stroke of diameter `size`. Falls back to
 * the library's own {@link DEFAULT_SMOOTHING} for brushes small enough that it
 * already yields tight enough spacing.
 */
export const outlineSmoothing = (size: number): number =>
  Math.min(DEFAULT_SMOOTHING, TARGET_VERTEX_SPACING / Math.max(size, 1));

/**
 * Drops input samples closer together than `spacing` (document units) so
 * sub-pixel pointer jitter can't masquerade as a direction reversal — see the
 * module docs.
 *
 * The scan is greedy left-to-right, so the samples retained for a prefix of
 * `points` are themselves a prefix of those retained for the whole array: as new
 * samples arrive mid-gesture, already-drawn geometry never shifts. The one
 * exception is the final sample, which is retained only once `last` is set, so
 * the completed stroke ends exactly under the pointer. Mid-gesture the head
 * therefore trails by at most `spacing` (≤ {@link MAX_SAMPLE_SPACING}).
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
 * Computes the filled outline polygon (a closed ring of document-space points)
 * of a variable-width stroke through `points`. Pure and DOM-free.
 *
 * The two `perfect-freehand` stages are driven separately so the start-of-stroke
 * noise gate (which `getStrokePoints` derives from `size`) can keep a small
 * absolute value while the outline is built at the real brush diameter. Passing
 * the same `size` to both is exactly equivalent to calling `getStroke`.
 */
export const strokeOutlinePolygon = (points: readonly StrokeSamplePoint[], opts: FreehandOptions): Vec2[] => {
  if (points.length === 0) {
    return [];
  }
  const last = opts.last ?? false;
  const samples = decimateSamples(points, sampleSpacing(opts.size), last);
  const strokePoints = getStrokePoints(
    samples.map((p) => [p.x, p.y, p.pressure]),
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
 * Traces an outline polygon onto `path` as a smooth closed curve, treating each
 * vertex as the control point of a quadratic segment running between the
 * midpoints of its two edges.
 *
 * The curve stays inside the convex hull of the polygon, so
 * {@link polygonBounds} of the polygon bounds the rendered shape — which is what
 * the dirty-rect accounting in the stroke session relies on. Collinear runs are
 * reproduced exactly, so this only differs from {@link polygonToSvgPath} where
 * the outline actually turns.
 *
 * Drawn with direct path calls rather than by building an SVG string for the
 * browser to re-parse. A stroke's outline is re-traced on every pointer batch
 * and runs to thousands of vertices, and at that size the round trip through a
 * string costs an order of magnitude more than the drawing does — 3.31ms versus
 * 0.18ms at 6000 samples — while describing exactly the same curve.
 *
 * Polygons with fewer than three vertices have no meaningful curvature and fall
 * back to straight segments.
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

/**
 * Builds a fillable `Path2D` for the stroke through `points`, using the injected
 * `createPath2D` factory (so callers in node can stub it). Returns the built
 * path and the polygon's document-space {@link Rect} bounds so the caller can
 * derive the dirty region without recomputing the outline.
 *
 * The path is the smooth ({@link traceSmoothPolygon}) form — the outline is a
 * coarse sampling of a curve, so joining its vertices with straight lines would
 * show as facets at large brush sizes. The returned bounds still come from the
 * polygon, which encloses the curve.
 */
export const strokeToPath = (
  points: readonly StrokeSamplePoint[],
  opts: FreehandOptions,
  createPath2D: CreatePath2D
): { path: Path2D; polygon: Vec2[]; bounds: Rect } => {
  const polygon = strokeOutlinePolygon(points, opts);
  const path = createPath2D();
  traceSmoothPolygon(path, polygon);
  return { bounds: polygonBounds(polygon), path, polygon };
};
