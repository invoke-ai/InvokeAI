import type { GalleryItemRef } from '@features/gallery/contracts';

import { toGalleryItemKey } from '@features/gallery/contracts';
import { describe, expect, it } from 'vitest';

import type { ImageMapPoint } from './api';

import { getClusterColor } from './clusterPalette';
import { collectClusterSelection } from './clusterSelection';
import {
  buildAllPointsTraces,
  buildClusterAnnotations,
  buildClusterSelectionTraces,
  buildHighlightedPointsTrace,
  getTraceAppearance,
  toBaseAppearanceRestyle,
  toClusterSelectionRestyle,
  declutterAnnotations,
  HIGHLIGHTED_POINTS_TRACE,
} from './imageMapTraces';

const point = (name: string, x: number, y: number, cluster: number): ImageMapPoint => {
  const item: GalleryItemRef = { kind: name.endsWith('.mp4') ? 'video' : 'image', name };

  return { cluster, item, key: toGalleryItemKey(item), x, y };
};

const POINTS = [
  point('a.png', 0, 0, 0),
  point('b.png', 3, 0, 0),
  point('c.png', 1, 0, 0),
  // A clip that landed in the same cluster as the images it looks like.
  point('clip.mp4', 2, 0, 0),
  point('other.png', 50, 50, 1),
  point('noise.png', -50, 50, -1),
];

describe('collectClusterSelection', () => {
  it('returns the clicked cluster ordered by distance from the click', () => {
    // Members are kind-tagged keys resolved through their kind's endpoint, so a video cannot be read as an image.
    expect(collectClusterSelection(POINTS, 'image:b.png')).toEqual([
      'image:b.png',
      'video:clip.mp4',
      'image:c.png',
      'image:a.png',
    ]);
  });

  it('selects every member of a cluster far larger than the old 5,000 cap', () => {
    const members = Array.from({ length: 12_000 }, (_, index) => ({
      cluster: 3,
      item: { kind: 'image' as const, name: `m${index}.png` },
      key: `image:m${index}.png` as const,
      x: index,
      y: 0,
    }));
    const selection = collectClusterSelection([...members, ...POINTS], 'image:m6000.png');

    expect(selection).toHaveLength(12_000);
    // Still walks outward from the click.
    expect(selection?.slice(0, 3)).toEqual(['image:m6000.png', 'image:m5999.png', 'image:m6001.png']);
  });
});

describe('buildHighlightedPointsTrace', () => {
  it('draws only multi-selections, larger and outlined', () => {
    const single = buildHighlightedPointsTrace(POINTS, new Set(['image:a.png' as const]));
    expect(single.x).toEqual([]);

    const multi = buildHighlightedPointsTrace(POINTS, new Set(['image:a.png' as const, 'video:clip.mp4' as const]));
    expect(multi.name).toBe(HIGHLIGHTED_POINTS_TRACE);
    expect(multi.customdata).toEqual(['image:a.png', 'video:clip.mp4']);
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
    // Anchor at cluster x-centroid above its top point with fixed-pixel clearance.
    expect(landscapes?.x).toBeCloseTo(4 / 3);
    expect(landscapes?.y).toBeCloseTo(5);
    expect(landscapes?.yanchor).toBe('bottom');
    expect(landscapes?.yshift).toBeGreaterThan(0);
    // Readable on any theme: white text on a dark pill.
    expect(landscapes?.font.color).toBe('#FFFFFF');
  });

  it('orders annotations by cluster size so declutter keeps the biggest labels', () => {
    // The smaller cluster's point comes FIRST: without the sort, Map insertion order would put 'portraits' first.
    const smallFirst = [point('solo.png', 10, 0, 1), point('a.png', 0, 0, 0), point('b.png', 1, 0, 0)];
    const annotations = buildClusterAnnotations(smallFirst, { '0': 'landscapes', '1': 'portraits' });
    expect(annotations.map((annotation) => annotation.text)).toEqual(['landscapes', 'portraits']);
  });
});

describe('declutterAnnotations', () => {
  // Place the smaller cluster first to prove size priority overrides input order.
  const clusters = [
    point('far.png', 10, 0, 1),
    point('a.png', 0, 0, 0),
    point('b.png', 1, 0, 0),
    point('c.png', 2, 0, 0),
  ];
  const annotations = buildClusterAnnotations(clusters, { '0': 'landscapes', '1': 'portraits' });
  const view = { widthPx: 800, heightPx: 600 };

  it('drops the smaller cluster label when zooming out collapses the gap', () => {
    // Zoomed way out: 1000 data units across 800px squeeze the anchors ~7px apart, inside either label's footprint.
    const ranges = { x: [-500, 500] as [number, number], y: [-375, 375] as [number, number] };
    const kept = declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx);
    expect(kept.map((annotation) => annotation.text)).toEqual(['landscapes']);
  });

  it('drops labels covering the current-image marker, which draws beneath them', () => {
    // Hide the label over the gold target: SVG annotations sit above WebGL markers and cannot be reordered below them.
    const ranges = { x: [-0.5, 10.5] as [number, number], y: [-4, 4] as [number, number] };
    const kept = declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx, { x: 10, y: 0 });
    expect(kept.map((annotation) => annotation.text)).toEqual(['landscapes']);
  });
});

describe('cluster selection drawing', () => {
  const members = new Set(['image:a.png', 'image:b.png', 'image:c.png', 'video:clip.mp4'] as const);

  it('redraws the members at their cluster colour, one scalar-styled trace per kind', () => {
    const [images, videos] = buildClusterSelectionTraces(POINTS, members);

    expect(images.customdata).toEqual(['image:a.png', 'image:b.png', 'image:c.png']);
    expect(videos.customdata).toEqual(['video:clip.mp4']);
    expect([images.marker.symbol, videos.marker.symbol]).toEqual(['circle', 'diamond']);
    for (const trace of [images, videos]) {
      // Scalar, never per point: that is what keeps a 100k-member cluster zooming like the base.
      expect(trace.marker.color).toBe(getClusterColor(0));
      expect(typeof trace.marker.opacity).toBe('number');
      // Clicks and hovers fall through to the base point underneath.
      expect(trace.hoverinfo).toBe('skip');
    }
    expect(toClusterSelectionRestyle([images, videos])).toEqual({
      customdata: [images.customdata, videos.customdata],
      'marker.color': [getClusterColor(0), getClusterColor(0)],
      x: [images.x, videos.x],
      y: [images.y, videos.y],
    });
  });

  it('dims every base trace with one greyer, fainter scalar each, and restores them exactly', () => {
    const appearances = buildAllPointsTraces(POINTS).map(getTraceAppearance);
    const dimmed = toBaseAppearanceRestyle(appearances, true);
    const restored = toBaseAppearanceRestyle(appearances, false);

    expect(dimmed['marker.color']).toHaveLength(appearances.length);
    appearances.forEach((appearance, index) => {
      const color = dimmed['marker.color']![index] as string;
      const opacity = dimmed['marker.opacity']![index] as number;

      expect(color).toMatch(/^#[0-9a-f]{6}$/);
      expect(color).not.toBe(appearance.color.toLowerCase());
      expect(opacity).toBeLessThan(appearance.opacity);
    });
    expect(restored).toEqual({
      'marker.color': appearances.map((appearance) => appearance.color),
      'marker.opacity': appearances.map((appearance) => appearance.opacity),
    });
  });
});
