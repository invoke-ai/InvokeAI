import { describe, expect, it } from 'vitest';

import type { ImageMapPoint } from './api';

import { describeClusters, summarizeClusters } from './clusterStats';

const points = (...clusters: number[]): ImageMapPoint[] =>
  clusters.map((cluster, index) => ({
    cluster,
    item: { kind: 'image', name: `${index}.png` },
    key: `image:${index}.png` as ImageMapPoint['key'],
    x: index,
    y: index,
  }));

describe('summarizeClusters', () => {
  it('sizes each cluster and counts noise apart from them', () => {
    expect(summarizeClusters(points(0, 0, 0, 1, 1, 2, -1, -1))).toEqual({
      clusterCount: 3,
      largestCluster: 3,
      unclustered: 2,
    });
  });

  it('reports no clusters rather than one when DBSCAN labelled everything noise', () => {
    // The 170k-gallery symptom: the map draws, and every point is noise.
    expect(summarizeClusters(points(-1, -1, -1))).toEqual({
      clusterCount: 0,
      largestCluster: 0,
      unclustered: 3,
    });
  });

  it('does not assume cluster ids are dense or start at zero', () => {
    // DBSCAN renumbers between runs and the map serves only the points the
    // user can still see, so served ids can skip values entirely.
    expect(summarizeClusters(points(7, 7, 42))).toEqual({
      clusterCount: 2,
      largestCluster: 2,
      unclustered: 0,
    });
  });

  it('handles an empty map', () => {
    expect(summarizeClusters([])).toEqual({ clusterCount: 0, largestCluster: 0, unclustered: 0 });
  });
});

describe('describeClusters', () => {
  it('agrees with the noun it is counting', () => {
    expect(describeClusters(summarizeClusters(points(0, 0, 1, -1)))).toBe(
      'Cluster count: 2, Largest cluster: 2 media points, Unclustered: 1 media point'
    );
  });

  it('groups digits, so it reads consistently beside the index progress in the same row', () => {
    const noise: number[] = Array.from({ length: 1500 }, () => -1);

    expect(describeClusters(summarizeClusters(points(...noise)))).toContain('Unclustered: 1,500 media points');
  });
});
