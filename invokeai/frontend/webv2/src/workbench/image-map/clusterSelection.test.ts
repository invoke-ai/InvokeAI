import type { GalleryItemRef } from '@features/gallery/contracts';

import { toGalleryItemKey } from '@features/gallery/contracts';
import { describe, expect, it } from 'vitest';

import type { ImageMapPoint } from './api';

import { getClusterColor } from './clusterPalette';
import { collectClusterSelection, formatClusterSize } from './clusterSelection';
import {
  buildAllPointsTraces,
  buildClusterAnnotations,
  buildClusterSelectionTraces,
  buildHighlightedPointsTrace,
  dimAppearance,
  getDominantCluster,
  getTraceAppearance,
  toBaseAppearanceRestyle,
  toClusterSelectionRestyle,
  declutterAnnotations,
  HIGHLIGHTED_POINTS_TRACE,
  toHighlightRestyle,
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
    // Members are kind-tagged keys: the gallery resolves each through the
    // endpoint its kind names, so a video member cannot be read as an image.
    expect(collectClusterSelection(POINTS, 'image:b.png')).toEqual([
      'image:b.png',
      'video:clip.mp4',
      'image:c.png',
      'image:a.png',
    ]);
  });

  it('selects the cluster around a clicked video just as it does around an image', () => {
    // b and c are equidistant from the clip, so input order breaks the tie.
    expect(collectClusterSelection(POINTS, 'video:clip.mp4')).toEqual([
      'video:clip.mp4',
      'image:b.png',
      'image:c.png',
      'image:a.png',
    ]);
  });

  it('returns null for noise points and unknown items', () => {
    expect(collectClusterSelection(POINTS, 'image:noise.png')).toBeNull();
    expect(collectClusterSelection(POINTS, 'image:missing.png')).toBeNull();
    // The same name in the other namespace is a different item.
    expect(collectClusterSelection(POINTS, 'image:clip.mp4')).toBeNull();
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
  it('keeps each kind\u2019s shape under the selection overlay', () => {
    // The overlay is drawn over the base point, so a selected video that lost
    // its diamond here would read as an image for as long as it is selected.
    const multi = buildHighlightedPointsTrace(POINTS, new Set(['image:a.png' as const, 'video:clip.mp4' as const]));

    expect(multi.marker.symbol).toEqual(['circle', 'diamond']);
  });

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

describe('toHighlightRestyle', () => {
  it('carries every per-point array the highlight trace holds', () => {
    // Plotly retains omitted restyle arrays; stale lengths crash scattergl when cluster clicks resize the trace.
    const trace = buildHighlightedPointsTrace(POINTS, new Set(['image:a.png' as const, 'video:clip.mp4' as const]));
    const payload = toHighlightRestyle(trace);

    const perPointKeys = Object.entries(trace.marker)
      .filter(([, value]) => Array.isArray(value))
      .map(([name]) => `marker.${name}`);

    expect(perPointKeys.length).toBeGreaterThan(0);
    for (const key of [...perPointKeys, 'customdata', 'x', 'y']) {
      expect(Object.keys(payload)).toContain(key);
    }
    // Wrap every property for one trace and keep point arrays aligned.
    for (const value of Object.values(payload)) {
      expect(value).toHaveLength(1);
      expect(value[0]).toHaveLength(trace.x.length);
    }
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
  // Place the smaller cluster first to prove size priority overrides input order.
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

  it('drops labels covering the current-image marker, which draws beneath them', () => {
    // Hide the label overlapping the gold target: SVG annotations sit above WebGL markers and cannot be reordered
    // below them.
    const ranges = { x: [-0.5, 10.5] as [number, number], y: [-4, 4] as [number, number] };
    const kept = declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx, { x: 10, y: 0 });
    expect(kept.map((annotation) => annotation.text)).toEqual(['landscapes']);
  });

  it('ignores a marker that is nowhere near a label', () => {
    const ranges = { x: [-0.5, 10.5] as [number, number], y: [-4, 4] as [number, number] };
    const kept = declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx, { x: 5, y: -3.5 });
    expect(kept.map((annotation) => annotation.text)).toEqual(['landscapes', 'portraits']);
    expect(declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx, null)).toEqual(
      declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx)
    );
  });

  it('does not hand the space it clears to a lower-priority label', () => {
    // A marker hiding the larger winning label must not promote a smaller overlapping label into the cleared
    // region.
    const crowded = [
      point('a.png', 0, 0, 0),
      point('b.png', 1, 0, 0),
      point('c.png', 0.5, 0, 0),
      point('solo.png', 1.256, 0, 1),
    ];
    const crowdedAnnotations = buildClusterAnnotations(crowded, { '0': 'landscapes', '1': 'portraits' });
    const ranges = { x: [-0.5, 10.5] as [number, number], y: [-4, 4] as [number, number] };

    expect(
      declutterAnnotations(crowdedAnnotations, ranges, view.widthPx, view.heightPx).map((annotation) => annotation.text)
    ).toEqual(['landscapes']);
    expect(
      declutterAnnotations(crowdedAnnotations, ranges, view.widthPx, view.heightPx, { x: 0.5, y: 0 }).map(
        (annotation) => annotation.text
      )
    ).toEqual([]);
  });

  it('leaves labels the marker does not cover exactly as they were', () => {
    // An already-suppressed label reserves no space; placing a marker there must not evict neighboring labels.
    const row = [
      point('a1.png', 0, 0, 0),
      point('a2.png', 0, 0, 0),
      point('a3.png', 0, 0, 0),
      point('b1.png', 0.6875, 0, 1),
      point('b2.png', 0.6875, 0, 1),
      point('c1.png', 1.375, 0, 2),
    ];
    const rowAnnotations = buildClusterAnnotations(row, { '0': 'aaaaaaaaaa', '1': 'bbbbbbbbbb', '2': 'cccccccccc' });
    const ranges = { x: [-0.5, 10.5] as [number, number], y: [-4, 4] as [number, number] };
    const withoutMarker = declutterAnnotations(rowAnnotations, ranges, view.widthPx, view.heightPx);

    expect(withoutMarker.map((annotation) => annotation.text)).toEqual(['aaaaaaaaaa', 'cccccccccc']);
    expect(declutterAnnotations(rowAnnotations, ranges, view.widthPx, view.heightPx, { x: 0.6875, y: 0 })).toEqual(
      withoutMarker
    );
  });

  it('ignores a marker panned out of view', () => {
    // Nothing off screen can be covered, so an off-view marker must not evict
    // an off-view label — that would churn the applied set on every pan.
    const offscreen = [point('a.png', -900, 0, 0), point('b.png', -900.1, 0, 0)];
    const offscreenAnnotations = buildClusterAnnotations(offscreen, { '0': 'landscapes' });
    const ranges = { x: [100, 200] as [number, number], y: [-50, 50] as [number, number] };

    expect(
      declutterAnnotations(offscreenAnnotations, ranges, view.widthPx, view.heightPx, { x: -900, y: 0 }).map(
        (annotation) => annotation.text
      )
    ).toEqual(['landscapes']);
  });

  it('treats a marker with non-finite coordinates as absent', () => {
    const ranges = { x: [-0.5, 10.5] as [number, number], y: [-4, 4] as [number, number] };
    expect(
      declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx, { x: Number.NaN, y: 0 }).map(
        (annotation) => annotation.text
      )
    ).toEqual(['landscapes', 'portraits']);
  });

  it('returns the full set unfiltered when the viewport is degenerate', () => {
    const ranges = { x: [0, 0] as [number, number], y: [-375, 375] as [number, number] };
    expect(declutterAnnotations(annotations, ranges, view.widthPx, view.heightPx)).toEqual(annotations);
    const sane = { x: [-500, 500] as [number, number], y: [-375, 375] as [number, number] };
    expect(declutterAnnotations(annotations, sane, 0, view.heightPx)).toEqual(annotations);
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

  it('takes the colour of the cluster most members are in now, after a refresh renumbered them', () => {
    const renumbered = POINTS.map((entry) => (entry.cluster === 0 ? { ...entry, cluster: 7 } : entry));
    const moved = renumbered.map((entry) => (entry.key === 'image:c.png' ? { ...entry, cluster: 1 } : entry));

    expect(getDominantCluster(moved, members)).toBe(7);
    expect(buildClusterSelectionTraces(moved, members)[0].marker.color).toBe(getClusterColor(7));
  });

  it('draws nothing for an empty selection or members no longer on the map', () => {
    for (const keys of [new Set<never>(), new Set(['image:gone.png'] as const)]) {
      const traces = buildClusterSelectionTraces(POINTS, keys);

      expect(traces.map((trace) => trace.x.length)).toEqual([0, 0]);
      expect(getDominantCluster(POINTS, keys)).toBeNull();
    }
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

  it('mutes colour toward grey rather than only fading it', () => {
    // Pure red keeps a trace of its hue but loses most of its saturation.
    const { color } = dimAppearance({ color: '#FF0000', opacity: 0.85 });
    const [r, g, b] = [1, 3, 5].map((start) => parseInt(color.slice(start, start + 2), 16));

    expect(r).toBeGreaterThan(g!);
    expect(r! - g!).toBeLessThan(60);
    expect(Math.abs(g! - b!)).toBeLessThan(10);
  });

  it('keeps noise fainter than clustered points once dimmed', () => {
    expect(dimAppearance({ color: '#8A8A8A', opacity: 0.25 }).opacity).toBeLessThan(
      dimAppearance({ color: '#4E79A7', opacity: 0.85 }).opacity
    );
  });

  it('words the size with grouping and number', () => {
    expect(formatClusterSize(1)).toBe('1 item');
    expect(formatClusterSize(38112)).toBe('38,112 items');
  });
});
