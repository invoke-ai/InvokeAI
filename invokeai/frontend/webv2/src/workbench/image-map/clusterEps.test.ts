import { describe, expect, it } from 'vitest';

import { getImageMapClusterEps, MAX_CLUSTER_EPS, MIN_CLUSTER_EPS } from './imageMapSettings';

describe('getImageMapClusterEps', () => {
  it('reads a chosen strength', () => {
    expect(getImageMapClusterEps({ clusterEps: 0.25 })).toBe(0.25);
  });

  it('means "derive it" when unset', () => {
    // The default, and the state the control returns to when cleared.
    expect(getImageMapClusterEps({})).toBeNull();
    expect(getImageMapClusterEps({ clusterEps: null })).toBeNull();
  });

  it('rejects stored values the endpoint would refuse', () => {
    // Persisted widget values survive upgrades and hand-edited project files,
    // so a bad one must fall back to the heuristic rather than 422 every
    // points request and leave the map permanently empty.
    expect(getImageMapClusterEps({ clusterEps: 0 })).toBeNull();
    expect(getImageMapClusterEps({ clusterEps: -1 })).toBeNull();
    expect(getImageMapClusterEps({ clusterEps: Number.NaN })).toBeNull();
    expect(getImageMapClusterEps({ clusterEps: Infinity })).toBeNull();
    expect(getImageMapClusterEps({ clusterEps: '0.2' })).toBeNull();
    expect(getImageMapClusterEps({ clusterEps: MIN_CLUSTER_EPS / 2 })).toBeNull();
    // And the high side: the endpoint caps eps at 2.0, so a larger stored
    // value 422s every refresh and strands the map on an error.
    expect(getImageMapClusterEps({ clusterEps: MAX_CLUSTER_EPS + 1 })).toBeNull();
    expect(getImageMapClusterEps({ clusterEps: 1000 })).toBeNull();
  });

  it('keeps the bounds the control and the endpoint agree on', () => {
    expect(getImageMapClusterEps({ clusterEps: MIN_CLUSTER_EPS })).toBe(MIN_CLUSTER_EPS);
    expect(getImageMapClusterEps({ clusterEps: MAX_CLUSTER_EPS })).toBe(MAX_CLUSTER_EPS);
  });
});
