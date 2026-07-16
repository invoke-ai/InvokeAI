/**
 * A thin wrapper over `perfect-freehand`, turning a pressure-sampled point path
 * into the filled outline polygon of a variable-width stroke.
 *
 * The polygon math ({@link strokeOutlinePolygon}) is pure and DOM-free, so it is
 * fully unit-testable in node. `Path2D` does not exist in node, so building the
 * fillable path is split out into {@link strokeToPath}, which takes an injected
 * `createPath2D` factory (the engine passes `(d) => new Path2D(d)`; tests pass a
 * stub). This keeps the interesting geometry testable without a DOM.
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { getStroke } from 'perfect-freehand';

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
  /** Edge softening, [0, 1]. */
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
 * Computes the filled outline polygon (a closed ring of document-space points)
 * of a variable-width stroke through `points`. Pure and DOM-free.
 */
export const strokeOutlinePolygon = (points: readonly StrokeSamplePoint[], opts: FreehandOptions): Vec2[] => {
  if (points.length === 0) {
    return [];
  }
  const outline = getStroke(
    points.map((p) => [p.x, p.y, p.pressure]),
    {
      last: opts.last ?? false,
      simulatePressure: false,
      size: opts.size,
      smoothing: opts.smoothing ?? DEFAULT_SMOOTHING,
      streamline: opts.streamline ?? DEFAULT_STREAMLINE,
      thinning: opts.thinning ?? DEFAULT_THINNING,
    }
  );
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
 */
export const strokeToPath = (
  points: readonly StrokeSamplePoint[],
  opts: FreehandOptions,
  createPath2D: CreatePath2D
): { path: Path2D; polygon: Vec2[]; bounds: Rect } => {
  const polygon = strokeOutlinePolygon(points, opts);
  const path = createPath2D(polygonToSvgPath(polygon));
  return { bounds: polygonBounds(polygon), path, polygon };
};
