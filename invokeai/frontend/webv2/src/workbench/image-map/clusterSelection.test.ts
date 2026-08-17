import { describe, expect, it } from 'vitest';

import type { ImageMapPoint } from './api';

import { collectClusterSelection } from './clusterSelection';
import { buildHighlightedPointsTrace, HIGHLIGHTED_POINTS_TRACE } from './imageMapTraces';

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
