import type { ImageMapPoint } from './api';

/**
 * Pure viewport math for the image map: initial axis ranges, focal-point
 * zoom, and keep-the-marker-in-view recentering. Kept free of plotly/DOM so
 * every invariant is unit-testable.
 */

export interface AxisRanges {
  x: [number, number];
  y: [number, number];
}

const RANGE_PAD_FRACTION = 0.05;

const quantile = (sorted: number[], q: number): number => {
  if (sorted.length === 0) {
    return 0;
  }

  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;

  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
};

/**
 * Initial view: the 1st..99th percentile of each axis (PhotoMapAI's choice —
 * a handful of outlier points must not dwarf the whole map), padded slightly.
 */
export const computePercentileRanges = (
  points: ImageMapPoint[],
  lowerQ: number = 0.01,
  upperQ: number = 0.99
): AxisRanges | null => {
  if (points.length === 0) {
    return null;
  }

  const xs = points.map((point) => point.x).sort((a, b) => a - b);
  const ys = points.map((point) => point.y).sort((a, b) => a - b);
  const build = (values: number[]): [number, number] => {
    const low = quantile(values, lowerQ);
    const high = quantile(values, upperQ);
    // Degenerate spans (single point, tiny gallery) still need a visible box.
    const pad = Math.max((high - low) * RANGE_PAD_FRACTION, 1e-3);

    return [low - pad, high + pad];
  };

  return { x: build(xs), y: build(ys) };
};

/**
 * Grow ranges minimally so `point` sits at least `padFraction` of the span
 * inside every edge. Used before the initial fit so the current-image marker
 * starts comfortably in view and the auto-recenter has no reason to shift
 * the freshly fitted map.
 */
export const expandRangesToInclude = (
  ranges: AxisRanges,
  point: { x: number; y: number },
  padFraction: number = 0.15
): AxisRanges => {
  const expandAxis = (range: [number, number], value: number): [number, number] => {
    const pad = (range[1] - range[0]) * padFraction;

    return [Math.min(range[0], value - pad), Math.max(range[1], value + pad)];
  };

  return { x: expandAxis(ranges.x, point.x), y: expandAxis(ranges.y, point.y) };
};

/**
 * Expand ranges so the whole box stays visible in a plot whose axes are
 * constrained to equal unit scale (xaxis.scaleanchor = 'y'). Handing plotly
 * an over-constrained range pair makes it re-solve against the container's
 * aspect ratio, which can CROP one axis — silently zooming the first render
 * into a sliver of the map. Pre-expanding the narrower constraint instead
 * guarantees everything inside the box remains in view. Only ever expands.
 */
export const fitRangesToAspect = (ranges: AxisRanges, aspectRatio: number): AxisRanges => {
  if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) {
    return ranges;
  }

  const spanX = ranges.x[1] - ranges.x[0];
  const spanY = ranges.y[1] - ranges.y[0];

  if (spanX <= 0 || spanY <= 0) {
    return ranges;
  }

  const centerX = (ranges.x[0] + ranges.x[1]) / 2;
  const centerY = (ranges.y[0] + ranges.y[1]) / 2;

  if (spanX / spanY < aspectRatio) {
    const nextSpanX = spanY * aspectRatio;

    return { x: [centerX - nextSpanX / 2, centerX + nextSpanX / 2], y: ranges.y };
  }

  const nextSpanY = spanX / aspectRatio;

  return { x: ranges.x, y: [centerY - nextSpanY / 2, centerY + nextSpanY / 2] };
};

/** exp() gain per wheel tick; ctrl+wheel is trackpad pinch and moves faster. */
export const zoomFactorFromWheel = (deltaY: number, isPinch: boolean): number =>
  Math.exp(deltaY * (isPinch ? 0.01 : 0.001));

/**
 * Scale both ranges by `factor` while keeping the point at the given
 * fractional position (0..1 from each range's start) fixed on screen.
 */
export const zoomRangesAroundFraction = (
  ranges: AxisRanges,
  fractionX: number,
  fractionY: number,
  factor: number
): AxisRanges => {
  // A non-positive factor would collapse or invert the ranges.
  factor = Math.max(factor, 1e-6);

  const zoomAxis = (range: [number, number], fraction: number): [number, number] => {
    const width = range[1] - range[0];
    const focal = range[0] + width * fraction;
    const nextWidth = width * factor;

    return [focal - nextWidth * fraction, focal + nextWidth * (1 - fraction)];
  };

  return { x: zoomAxis(ranges.x, fractionX), y: zoomAxis(ranges.y, fractionY) };
};

/**
 * If the point sits within `padFraction` of (or beyond) a viewport edge,
 * translate the ranges to center it — preserving the zoom width. Returns
 * null when the point is comfortably in view.
 */
export const rangesToKeepMarkerInView = (
  ranges: AxisRanges,
  point: { x: number; y: number },
  padFraction: number = 0.1
): AxisRanges | null => {
  const isComfortable = (range: [number, number], value: number): boolean => {
    const pad = (range[1] - range[0]) * padFraction;

    return value >= range[0] + pad && value <= range[1] - pad;
  };

  if (isComfortable(ranges.x, point.x) && isComfortable(ranges.y, point.y)) {
    return null;
  }

  const center = (range: [number, number], value: number): [number, number] => {
    const half = (range[1] - range[0]) / 2;

    return [value - half, value + half];
  };

  return { x: center(ranges.x, point.x), y: center(ranges.y, point.y) };
};
