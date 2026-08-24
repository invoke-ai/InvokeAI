import { describe, expect, it } from 'vitest';

import type { ImageMapPoint } from './api';

import { collectClusterSelection } from './clusterSelection';
import {
  buildClusterAnnotations,
  buildHighlightedPointsTrace,
  declutterAnnotations,
  HIGHLIGHTED_POINTS_TRACE,
} from './imageMapTraces';

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

  it('orders annotations by cluster size so declutter keeps the biggest labels', () => {
    // The smaller cluster's point comes FIRST in the points array: without the
    // explicit sort, Map insertion order would put 'portraits' first.
    const smallFirst = [point('solo.png', 10, 0, 1), point('a.png', 0, 0, 0), point('b.png', 1, 0, 0)];
    const annotations = buildClusterAnnotations(smallFirst, { '0': 'landscapes', '1': 'portraits' });
    expect(annotations.map((annotation) => annotation.text)).toEqual(['landscapes', 'portraits']);
  });

  it('breaks size ties by cluster id, ascending', () => {
    const tied = [point('b.png', 10, 0, 1), point('a.png', 0, 0, 0)];
    const annotations = buildClusterAnnotations(tied, { '0': 'landscapes', '1': 'portraits' });
    expect(annotations.map((annotation) => annotation.text)).toEqual(['landscapes', 'portraits']);
  });
});

describe('declutterAnnotations', () => {
  // Cluster 0: three points near the origin; cluster 1: one point at (10, 0).
  // The small cluster's point deliberately comes first, so these tests fail
  // if the size-priority ordering in buildClusterAnnotations regresses.
  const clusters = [
    point('far.png', 10, 0, 1),
    point('a.png', 0, 0, 0),
    point('b.png', 1, 0, 0),
    point('c.png', 2, 0, 0),
  ];
  const annotations = buildClusterAnnotations(clusters, { '0': 'landscapes', '1': 'portraits' });
  const view = { widthPx: 800, heightPx: 600 };

  it('keeps every label when the view gives them room', () => {
    // Zoomed in: 11 data units across 800px puts the anchors ~650px apart.
    const ranges = { x: [-0.5, 10.5] as [number, number], y: [-4, 4] as [number, number] };
    const kept = declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx);
    expect(kept.map((annotation) => annotation.text)).toEqual(['landscapes', 'portraits']);
  });

  it('drops the smaller cluster label when zooming out collapses the gap', () => {
    // Zoomed way out: 1000 data units across 800px squeezes the anchors to
    // ~7px apart — well inside either label's pixel footprint.
    const ranges = { x: [-500, 500] as [number, number], y: [-375, 375] as [number, number] };
    const kept = declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx);
    expect(kept.map((annotation) => annotation.text)).toEqual(['landscapes']);
  });

  it('keeps labels that overlap horizontally but are far apart vertically', () => {
    // Both clusters share x≈0 but sit 2 data units apart in y; at 75px per
    // unit that is 150px of vertical separation — no collision, both kept.
    // Collapsing the y term in the collision math would fail this.
    const stacked = [point('top.png', 0, 2, 1), point('a.png', 0, 0, 0), point('b.png', 1, 0, 0)];
    const stackedAnnotations = buildClusterAnnotations(stacked, { '0': 'landscapes', '1': 'portraits' });
    const ranges = { x: [-4, 4] as [number, number], y: [-3, 5] as [number, number] };
    const kept = declutterAnnotations(stackedAnnotations, ranges, 800, 600);
    expect(kept.map((annotation) => annotation.text)).toEqual(['landscapes', 'portraits']);
  });

  it('returns the full set unfiltered when the viewport is degenerate', () => {
    const ranges = { x: [0, 0] as [number, number], y: [-375, 375] as [number, number] };
    expect(declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx)).toEqual(annotations);
    const sane = { x: [-500, 500] as [number, number], y: [-375, 375] as [number, number] };
    expect(declutterAnnotations(annotations, sane, 0, view.heightPx)).toEqual(annotations);
  });
});
