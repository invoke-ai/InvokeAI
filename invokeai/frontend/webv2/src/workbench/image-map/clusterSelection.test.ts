import { describe, expect, it } from 'vitest';

import type { ImageMapPoint } from './api';

import { collectClusterSelection } from './clusterSelection';
import { buildClusterAnnotations, buildHighlightedPointsTrace, HIGHLIGHTED_POINTS_TRACE } from './imageMapTraces';

const point = (imageName: string, x: number, y: number, cluster: number): ImageMapPoint => ({
  cluster,
  imageName,
  x,
  y,
});

const POINTS = [
  point('a.png', 0, 0, 0),
  point('b.png', 3, 0, 0),
  point('c.png', 1, 0, 0),
  point('other.png', 50, 50, 1),
  point('noise.png', -50, 50, -1),
];

describe('collectClusterSelection', () => {
  it('returns the clicked cluster ordered by distance from the click', () => {
    expect(collectClusterSelection(POINTS, 'b.png')).toEqual(['b.png', 'c.png', 'a.png']);
  });

  it('returns null for noise points and unknown names', () => {
    expect(collectClusterSelection(POINTS, 'noise.png')).toBeNull();
    expect(collectClusterSelection(POINTS, 'missing.png')).toBeNull();
  });

  it('caps oversized clusters, keeping the nearest members', () => {
    const capped = collectClusterSelection(POINTS, 'a.png', 2);
    expect(capped).toEqual(['a.png', 'c.png']);
  });
});

describe('buildHighlightedPointsTrace', () => {
  it('draws only multi-selections, larger and outlined', () => {
    const single = buildHighlightedPointsTrace(POINTS, new Set(['a.png']));
    expect(single.x).toEqual([]);

    const multi = buildHighlightedPointsTrace(POINTS, new Set(['a.png', 'c.png']));
    expect(multi.name).toBe(HIGHLIGHTED_POINTS_TRACE);
    expect(multi.customdata).toEqual(['a.png', 'c.png']);
    expect(multi.marker.size).toBe(8);
    expect(multi.marker.line).toEqual({ color: '#FFFFFF', width: 1 });
  });
});

describe('buildClusterAnnotations', () => {
  it('places one annotation per labeled cluster, centered above its topmost point', () => {
    const varied = [point('a.png', 0, 1, 0), point('b.png', 3, 5, 0), point('c.png', 1, 3, 0)];
    const annotations = buildClusterAnnotations(varied, { '0': 'landscapes' });

    expect(annotations).toHaveLength(1);
    const landscapes = annotations[0];
    // Centered on the cluster's x centroid, anchored above its topmost point
    // with a pixel lift so the label never covers the points it names.
    expect(landscapes?.x).toBeCloseTo(4 / 3);
    expect(landscapes?.y).toBeCloseTo(5);
    expect(landscapes?.yanchor).toBe('bottom');
    expect(landscapes?.yshift).toBeGreaterThan(0);
    // Readable on any theme: white text on a dark pill.
    expect(landscapes?.font.color).toBe('#FFFFFF');
  });

  it('skips noise, unlabeled clusters, and null label maps', () => {
    expect(buildClusterAnnotations(POINTS, null)).toEqual([]);
    const onlyOne = buildClusterAnnotations(POINTS, { '1': 'portraits' });
    expect(onlyOne.map((annotation) => annotation.text)).toEqual(['portraits']);
  });
});
