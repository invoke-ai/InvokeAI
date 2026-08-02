import { describe, expect, it } from 'vitest';

import type { ImageMapPoint } from './api';

import {
  computePercentileRanges,
  expandRangesToInclude,
  fitRangesToAspect,
  rangesToKeepMarkerInView,
  zoomFactorFromWheel,
  zoomRangesAroundFraction,
  type AxisRanges,
} from './imageMapViewport';

const point = (x: number, y: number): ImageMapPoint => ({ cluster: 0, imageName: `${x},${y}`, x, y });

describe('computePercentileRanges', () => {
  it('returns null for no points and a non-degenerate box for one point', () => {
    expect(computePercentileRanges([])).toBeNull();

    const single = computePercentileRanges([point(3, 4)]);
    expect(single).not.toBeNull();
    expect(single!.x[0]).toBeLessThan(3);
    expect(single!.x[1]).toBeGreaterThan(3);
    expect(single!.y[0]).toBeLessThan(4);
    expect(single!.y[1]).toBeGreaterThan(4);
  });

  it('excludes extreme outliers from the initial view', () => {
    const cloud = Array.from({ length: 999 }, (_, i) => point(i % 10, Math.floor(i / 100)));
    const withOutlier = [...cloud, point(10_000, 10_000)];

    const ranges = computePercentileRanges(withOutlier);

    expect(ranges!.x[1]).toBeLessThan(100);
    expect(ranges!.y[1]).toBeLessThan(100);
  });
});

describe('zoomRangesAroundFraction', () => {
  const ranges: AxisRanges = { x: [0, 10], y: [100, 200] };

  it('keeps the focal point fixed while scaling the span', () => {
    const zoomed = zoomRangesAroundFraction(ranges, 0.25, 0.5, 0.5);

    // Focal x = 2.5 sits at 25% of the new range; focal y = 150 at 50%.
    expect(zoomed.x[0] + (zoomed.x[1] - zoomed.x[0]) * 0.25).toBeCloseTo(2.5);
    expect(zoomed.y[0] + (zoomed.y[1] - zoomed.y[0]) * 0.5).toBeCloseTo(150);
    expect(zoomed.x[1] - zoomed.x[0]).toBeCloseTo(5);
    expect(zoomed.y[1] - zoomed.y[0]).toBeCloseTo(50);
  });

  it('never collapses or inverts ranges for non-positive factors', () => {
    const collapsed = zoomRangesAroundFraction(ranges, 0.5, 0.5, 0);
    expect(collapsed.x[1]).toBeGreaterThan(collapsed.x[0]);
    expect(collapsed.y[1]).toBeGreaterThan(collapsed.y[0]);
  });

  it('is the inverse of itself with the reciprocal factor', () => {
    const there = zoomRangesAroundFraction(ranges, 0.3, 0.7, 2);
    const back = zoomRangesAroundFraction(there, 0.3, 0.7, 0.5);

    expect(back.x[0]).toBeCloseTo(ranges.x[0]);
    expect(back.x[1]).toBeCloseTo(ranges.x[1]);
    expect(back.y[0]).toBeCloseTo(ranges.y[0]);
    expect(back.y[1]).toBeCloseTo(ranges.y[1]);
  });
});

describe('zoomFactorFromWheel', () => {
  it('zooms out on positive deltas, in on negative, faster for pinch', () => {
    expect(zoomFactorFromWheel(100, false)).toBeCloseTo(Math.exp(0.1), 10);
    expect(zoomFactorFromWheel(-100, false)).toBeCloseTo(Math.exp(-0.1), 10);
    expect(zoomFactorFromWheel(100, true)).toBeCloseTo(Math.exp(1), 10);
  });
});

describe('rangesToKeepMarkerInView', () => {
  const ranges: AxisRanges = { x: [0, 10], y: [0, 10] };

  it('returns null when the marker is comfortably inside', () => {
    expect(rangesToKeepMarkerInView(ranges, { x: 5, y: 5 })).toBeNull();
    expect(rangesToKeepMarkerInView(ranges, { x: 1.5, y: 8.5 })).toBeNull();
  });

  it('recenters on the marker while preserving the zoom width', () => {
    const recentered = rangesToKeepMarkerInView(ranges, { x: 25, y: 0.2 });

    expect(recentered).not.toBeNull();
    expect(recentered!.x[1] - recentered!.x[0]).toBeCloseTo(10);
    expect(recentered!.y[1] - recentered!.y[0]).toBeCloseTo(10);
    expect((recentered!.x[0] + recentered!.x[1]) / 2).toBeCloseTo(25);
    expect((recentered!.y[0] + recentered!.y[1]) / 2).toBeCloseTo(0.2);
  });

  it('treats the pad band as out of view', () => {
    // Within 10% of the edge -> recenter.
    expect(rangesToKeepMarkerInView(ranges, { x: 0.5, y: 5 })).not.toBeNull();
    expect(rangesToKeepMarkerInView(ranges, { x: 5, y: 9.8 })).not.toBeNull();
  });
});

describe('fitRangesToAspect', () => {
  const box = { x: [0, 10] as [number, number], y: [0, 20] as [number, number] };

  const containsBox = (outer: { x: [number, number]; y: [number, number] }) =>
    outer.x[0] <= box.x[0] && outer.x[1] >= box.x[1] && outer.y[0] <= box.y[0] && outer.y[1] >= box.y[1];

  it('only ever expands, and matches the requested aspect ratio', () => {
    for (const aspect of [0.25, 0.5, 1, 2, 4]) {
      const fitted = fitRangesToAspect(box, aspect);

      expect(containsBox(fitted)).toBe(true);
      expect((fitted.x[1] - fitted.x[0]) / (fitted.y[1] - fitted.y[0])).toBeCloseTo(aspect);
    }
  });

  it('returns the input unchanged for degenerate aspect ratios', () => {
    expect(fitRangesToAspect(box, 0)).toEqual(box);
    expect(fitRangesToAspect(box, Number.NaN)).toEqual(box);
  });

  it('keeps at least 90% of points in view for any container shape', () => {
    // The user-facing contract for the first render: the fitted view must
    // show (nearly) the whole map, however skewed the container.
    const points = Array.from({ length: 100 }, (_, index) => ({
      cluster: 0,
      imageName: `p${index}.png`,
      x: Math.sin(index * 2.399) * 13,
      y: Math.cos(index * 1.618) * 7 + (index % 10),
    }));
    const percentile = computePercentileRanges(points)!;

    for (const aspect of [0.3, 1, 3.5]) {
      const fitted = fitRangesToAspect(percentile, aspect);
      const visible = points.filter(
        (point) => point.x >= fitted.x[0] && point.x <= fitted.x[1] && point.y >= fitted.y[0] && point.y <= fitted.y[1]
      );

      expect(visible.length).toBeGreaterThanOrEqual(90);
    }
  });
});

describe('expandRangesToInclude', () => {
  const box = { x: [0, 10] as [number, number], y: [0, 10] as [number, number] };

  it('leaves ranges unchanged when the point is comfortably inside', () => {
    expect(expandRangesToInclude(box, { x: 5, y: 5 })).toEqual(box);
  });

  it('grows the box so an outside point sits inside with margin', () => {
    const grown = expandRangesToInclude(box, { x: 14, y: -3 });

    expect(grown.x[1]).toBeGreaterThan(14);
    expect(grown.y[0]).toBeLessThan(-3);
    // Untouched edges stay put.
    expect(grown.x[0]).toBe(0);
    expect(grown.y[1]).toBe(10);
  });
});
