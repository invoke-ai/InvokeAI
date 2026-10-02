import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  ApiError: class ApiError extends Error {
    readonly status: number;

    constructor(message: string, status: number) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
    }
  },
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  ApiError: mocks.ApiError,
  apiFetchJson: mocks.apiFetchJson,
  getApiErrorMessage: (_error: unknown, fallback: string) => fallback,
}));

import type { ImageMapPoint } from './api';

import { fetchImageMapPoints, fetchImageMapStatus, requestImageMapRefresh } from './api';
import { CLUSTER_PALETTE, getClusterColor, NOISE_COLOR } from './clusterPalette';
import {
  ensureImageMapLoaded,
  imageMapStore,
  recordImageIndexStatus,
  refreshImageIndexStatus,
  refreshImageMapPoints,
  setClusterEps,
} from './imageMapStore';
import {
  ALL_POINTS_TRACE,
  buildAllPointsTraces,
  buildCurrentImageTrace,
  buildMapLayout,
  CURRENT_IMAGE_TRACE,
} from './imageMapTraces';

const BACKEND_RESPONSE = {
  cluster_eps: 0.42,
  point_count: 3,
  points: [
    { cluster: 0, image_name: 'a.png', x: 1.5, y: -2 },
    { cluster: -1, image_name: 'b.png', x: 0, y: 3 },
    { cluster: 0, image_name: 'clip.mp4', kind: 'video', x: 2, y: -1 },
  ],
  stale: false,
  state: 'ready',
  updated_at: '2026-08-02 12:00:00',
  visible_hash: 'hash-1',
};

// Return valid labels stamped for another projection to exercise discard. Malformed fixtures throw TypeError and
// arm network retries beyond the test.
const FOREIGN_LABELS_RESPONSE = { labels: {}, updated_at: 'another projection', visible_hash: 'another set' };

const mockPointsWithForeignLabels = (): void => {
  mocks.apiFetchJson.mockImplementation((url: string) =>
    url.startsWith('/api/v1/image_map/cluster_labels')
      ? Promise.resolve(FOREIGN_LABELS_RESPONSE)
      : Promise.resolve(BACKEND_RESPONSE)
  );
};

describe('image map api', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('maps snake_case points to camelCase', async () => {
    mocks.apiFetchJson.mockResolvedValue(BACKEND_RESPONSE);

    const result = await fetchImageMapPoints();

    // include_videos is what opts this client into video points; without it the
    // backend serves images only, however much of the gallery is indexed.
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/points?include_videos=true');
    expect(result.state).toBe('ready');
    expect(result.pointCount).toBe(3);
    expect(result.clusterEps).toBe(0.42);
    expect(result.visibleHash).toBe('hash-1');
    expect(result.points[0]).toEqual({
      cluster: 0,
      item: { kind: 'image', name: 'a.png' },
      key: 'image:a.png',
      x: 1.5,
      y: -2,
    });
    // A point without `kind` predates indexed videos and reads as an image.
    expect(result.points[1]?.item).toEqual({ kind: 'image', name: 'b.png' });
    expect(result.points[2]?.item).toEqual({ kind: 'video', name: 'clip.mp4' });
  });

  it('maps the model_missing state with the configured model name', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      model_name: 'clip-vit-large-patch14',
      point_count: 0,
      points: [],
      stale: false,
      state: 'model_missing',
      updated_at: null,
    });

    const result = await fetchImageMapPoints();

    expect(result.state).toBe('model_missing');
    expect(result.modelName).toBe('clip-vit-large-patch14');
  });

  it('passes eps and min_samples as query params', async () => {
    mocks.apiFetchJson.mockResolvedValue({ ...BACKEND_RESPONSE, points: [] });

    await fetchImageMapPoints({ eps: 0.4, minSamples: 5 });

    expect(mocks.apiFetchJson).toHaveBeenCalledWith(
      '/api/v1/image_map/points?include_videos=true&eps=0.4&min_samples=5'
    );
  });

  it('posts refresh requests', async () => {
    mocks.apiFetchJson.mockResolvedValue({ enqueued: true });

    await expect(requestImageMapRefresh()).resolves.toBe(true);
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/refresh', { method: 'POST' });
  });
});

describe('image map store', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    imageMapStore.setSnapshot({
      clusterLabels: null,
      clusterLabelsEps: null,
      clusterLabelsHash: null,
      data: null,
      error: null,
      indexCounts: null,
      indexUpdatedAt: null,
      loadState: 'idle',
      renderError: null,
    });
  });

  // Retry-schedule tests fake the clock; restore it whatever their outcome.
  afterEach(() => {
    vi.useRealTimers();
  });

  it('clusters at the chosen strength on every later refresh', async () => {
    // The strength is module state rather than a call argument precisely so
    // that socket-driven refreshes carry it too; nothing else would.
    mockPointsWithForeignLabels();
    await refreshImageMapPoints();
    setClusterEps(0.25);
    await Promise.resolve();
    await refreshImageMapPoints();

    const pointsCalls = mocks.apiFetchJson.mock.calls
      .map((call) => String(call[0]))
      .filter((url) => url.startsWith('/api/v1/image_map/points'));

    expect(pointsCalls[0]).not.toContain('eps=');
    expect(pointsCalls.at(-1)).toContain('eps=0.25');

    setClusterEps(null);
    await Promise.resolve();
    await refreshImageMapPoints();

    const afterClearing = mocks.apiFetchJson.mock.calls
      .map((call) => String(call[0]))
      .filter((url) => url.startsWith('/api/v1/image_map/points'))
      .at(-1);

    expect(afterClearing).not.toContain('eps=');
    setClusterEps(null);
  });

  it('refetches when the strength changes, and retires the old labels', async () => {
    // Changing eps renumbers every cluster while the projection and the
    // visible set stay put, so `visibleHash` cannot detect it: labels left in
    // the store would be shown against a clustering they do not describe.
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.startsWith('/api/v1/image_map/cluster_labels')
        ? Promise.resolve({
            labels: { '0': { label: 'boats' } },
            updated_at: '2026-08-02 12:00:00',
            visible_hash: 'hash-1',
          })
        : Promise.resolve(BACKEND_RESPONSE)
    );

    await refreshImageMapPoints();
    await Promise.resolve();
    await Promise.resolve();
    expect(imageMapStore.getSnapshot().clusterLabels).not.toBeNull();
    expect(imageMapStore.getSnapshot().clusterLabelsEps).toBe(0.42);

    const before = mocks.apiFetchJson.mock.calls.length;
    setClusterEps(0.3);

    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();
    expect(imageMapStore.getSnapshot().clusterLabelsEps).toBeNull();
    expect(mocks.apiFetchJson.mock.calls.length).toBeGreaterThan(before);
    setClusterEps(null);
  });

  it('does not refetch for a strength that is already in force', async () => {
    mockPointsWithForeignLabels();
    await refreshImageMapPoints();
    const before = mocks.apiFetchJson.mock.calls.length;

    setClusterEps(null);

    expect(mocks.apiFetchJson.mock.calls.length).toBe(before);
  });

  it('loads points into the snapshot', async () => {
    mockPointsWithForeignLabels();

    await refreshImageMapPoints();

    const snapshot = imageMapStore.getSnapshot();
    expect(snapshot.loadState).toBe('loaded');
    expect(snapshot.data?.points).toHaveLength(3);
    expect(snapshot.error).toBeNull();
  });

  it('passes the points response eps through to the cluster labels fetch', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { alternates: ['kittens', 'pets'], label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();

    // The labels endpoint must receive the exact eps the map was clustered
    // with; the adaptive default could resolve differently on a drifted set.
    await vi.waitFor(() => {
      expect(mocks.apiFetchJson).toHaveBeenCalledWith(
        // Must match /points: the client discards labels whose visible hash
        // disagrees, which a different item set would guarantee.
        '/api/v1/image_map/cluster_labels?include_videos=true&eps=0.42'
      );
      expect(imageMapStore.getSnapshot().clusterLabels).toEqual({
        '0': { alternates: ['kittens', 'pets'], label: 'cats' },
      });
    });
    // Fingerprint labels with their clustered visible set.
    expect(imageMapStore.getSnapshot().clusterLabelsHash).toBe(BACKEND_RESPONSE.visible_hash);
  });

  it('marks labels stale as soon as a refresh renumbers the clusters', async () => {
    // L1: points and matching labels land together.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { alternates: [], label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });
    await refreshImageMapPoints();
    await vi.waitFor(() => {
      expect(imageMapStore.getSnapshot().clusterLabelsHash).toBe(BACKEND_RESPONSE.visible_hash);
    });

    // While new labels lag refreshed points, retain the old fingerprint so consumers reject potentially renumbered
    // cluster labels.
    const drifted = { ...BACKEND_RESPONSE, visible_hash: 'hash-2' };
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.startsWith('/api/v1/image_map/cluster_labels') ? new Promise(() => {}) : Promise.resolve(drifted)
    );
    await refreshImageMapPoints();

    const snapshot = imageMapStore.getSnapshot();
    expect(snapshot.clusterLabels).not.toBeNull();
    expect(snapshot.clusterLabelsHash).not.toBe(snapshot.data?.visibleHash);
  });

  it('discards label responses clustered over a drifted visible set', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: 'hash-2',
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();

    await vi.waitFor(() => {
      expect(mocks.apiFetchJson).toHaveBeenCalledWith(
        // Must match /points: the client discards labels whose visible hash
        // disagrees, which a different item set would guarantee.
        '/api/v1/image_map/cluster_labels?include_videos=true&eps=0.42'
      );
    });
    // Settle the handler and reject cluster ids from a different visible set.
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();
  });

  it('ignores a stale labels failure after a newer request already set labels', async () => {
    // L1: points land, but the labels request hangs and will fail late.
    let rejectFirstLabels: (reason: unknown) => void = () => {};
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return new Promise((_resolve, reject) => {
          rejectFirstLabels = reject;
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });
    await refreshImageMapPoints();
    await vi.waitFor(() => {
      expect(mocks.apiFetchJson).toHaveBeenCalledWith(
        // Must match /points: the client discards labels whose visible hash
        // disagrees, which a different item set would guarantee.
        '/api/v1/image_map/cluster_labels?include_videos=true&eps=0.42'
      );
    });

    // L2: a newer refresh whose labels resolve first.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });
    await refreshImageMapPoints();
    await vi.waitFor(() => {
      expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': { alternates: [], label: 'cats' } });
    });

    // L1's late failure is stale: it must not wipe the labels L2 set.
    rejectFirstLabels(new Error('slow failure'));
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': { alternates: [], label: 'cats' } });
  });

  it('retries a labels request that raced the vocabulary build', async () => {
    vi.useFakeTimers();
    // After a backend restart the vocabulary embeddings are not in memory
    // yet: the first labels request 409s while the index worker builds them.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.reject(new mocks.ApiError('Cluster labels are still being prepared; try again shortly', 409));
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();
    await vi.advanceTimersByTimeAsync(0);
    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();

    // The build lands; the retry succeeds and the labels appear.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await vi.advanceTimersByTimeAsync(1_000);

    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(3);
    expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': { alternates: [], label: 'cats' } });
  });

  it('retires a pending retry when a newer labels request takes over', async () => {
    vi.useFakeTimers();
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.reject(new mocks.ApiError('Cluster labels are still being prepared; try again shortly', 409));
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();
    await vi.advanceTimersByTimeAsync(0);

    // A newer refresh supersedes the failing chain and resolves its own labels.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();
    await vi.advanceTimersByTimeAsync(0);
    expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': { alternates: [], label: 'cats' } });

    // The first chain's retry timer fires: it must neither re-request nor
    // disturb the newer chain's labels.
    const callsBeforeRetry = mocks.apiFetchJson.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(mocks.apiFetchJson.mock.calls.length).toBe(callsBeforeRetry);
    expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': { alternates: [], label: 'cats' } });
  });

  it('does not retry a labels failure the client cannot outwait', async () => {
    vi.useFakeTimers();
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.reject(new mocks.ApiError('Forbidden', 403));
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();
    await vi.advanceTimersByTimeAsync(300_000);

    // One points fetch, one labels attempt, nothing after.
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();
  });

  it('gives up on a persistently unavailable vocabulary after the schedule', async () => {
    vi.useFakeTimers();
    const labelsCalls = (): number =>
      mocks.apiFetchJson.mock.calls.filter((call: unknown[]) =>
        String(call[0]).startsWith('/api/v1/image_map/cluster_labels')
      ).length;
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.reject(new mocks.ApiError('Cluster labels are still being prepared; try again shortly', 409));
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();
    // The whole backoff schedule: 9 retries after the first attempt.
    await vi.advanceTimersByTimeAsync(300_000);

    expect(labelsCalls()).toBe(10);
    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();
  });

  it('retries a labels request that failed at the network layer', async () => {
    vi.useFakeTimers();
    // fetch() rejects with a bare TypeError when the request never reaches a
    // server — a backend restarting under an open map, say.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.reject(new TypeError('Failed to fetch'));
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();
    await vi.advanceTimersByTimeAsync(0);
    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();

    // The backend comes back; the retry succeeds and the labels appear.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await vi.advanceTimersByTimeAsync(1_000);

    expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': { alternates: [], label: 'cats' } });
  });

  it('clears labels when the newest request fails outright', async () => {
    // Current-request failure clears older labels instead of annotating new points with stale clusters; plain
    // Error avoids retry timers.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve({
          labels: { '0': { label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });
    await refreshImageMapPoints();
    await vi.waitFor(() => {
      expect(imageMapStore.getSnapshot().clusterLabels).not.toBeNull();
    });

    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.reject(new Error('boom'));
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });
    await refreshImageMapPoints();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();
  });

  it('records errors and keeps prior data', async () => {
    mockPointsWithForeignLabels();
    await refreshImageMapPoints();

    mocks.apiFetchJson.mockRejectedValue(new Error('boom'));
    await refreshImageMapPoints();

    const snapshot = imageMapStore.getSnapshot();
    expect(snapshot.loadState).toBe('error');
    expect(snapshot.error).toBe('Failed to load the image map');
    expect(snapshot.data?.points).toHaveLength(3);
  });

  it('enters loading while retrying a failed points request', async () => {
    const response = {
      model_name: null,
      point_count: 0,
      points: [],
      stale: false,
      state: 'disabled',
      updated_at: null,
    };
    let resolveRequest: (value: typeof response) => void = () => {};
    const request = new Promise<typeof response>((resolve) => {
      resolveRequest = resolve;
    });

    imageMapStore.patchSnapshot({ error: 'previous failure', loadState: 'error' });
    mocks.apiFetchJson.mockReturnValueOnce(request);

    const refresh = refreshImageMapPoints();

    try {
      expect(imageMapStore.getSnapshot().loadState).toBe('loading');
    } finally {
      resolveRequest(response);
      await refresh;
    }
  });

  it('collapses mid-flight refresh requests into one rerun', async () => {
    // Concurrent refreshes join one in-flight fetch and coalesce into exactly one follow-up, including socket
    // events during initial load.
    const pointsResolvers: Array<(value: typeof BACKEND_RESPONSE) => void> = [];
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/points')) {
        return new Promise((resolve) => {
          pointsResolvers.push(resolve);
        });
      }

      return Promise.resolve({
        labels: { '0': { label: 'cats' } },
        updated_at: BACKEND_RESPONSE.updated_at,
        visible_hash: BACKEND_RESPONSE.visible_hash,
      });
    });

    const calls = [refreshImageMapPoints(), refreshImageMapPoints(), refreshImageMapPoints()];
    // All three joined the one in-flight fetch; no parallel point set.
    expect(pointsResolvers).toHaveLength(1);

    pointsResolvers[0]?.(BACKEND_RESPONSE);
    await Promise.all(calls);

    // Exactly one rerun follows the settle.
    await vi.waitFor(() => expect(pointsResolvers).toHaveLength(2));
    pointsResolvers[1]?.(BACKEND_RESPONSE);
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    // No runaway: the rerun found nothing to queue behind it.
    expect(pointsResolvers).toHaveLength(2);
    expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
  });

  it('still runs the queued rerun when the in-flight fetch fails', async () => {
    // A refresh admitted during 'loading' must not be lost to the first fetch
    // failing: the rerun runs anyway and can still recover the map.
    let pointsCall = 0;
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (!url.startsWith('/api/v1/image_map/points')) {
        return Promise.resolve({
          labels: { '0': { label: 'cats' } },
          updated_at: BACKEND_RESPONSE.updated_at,
          visible_hash: BACKEND_RESPONSE.visible_hash,
        });
      }

      pointsCall += 1;
      return pointsCall === 1 ? Promise.reject(new Error('backend down')) : Promise.resolve(BACKEND_RESPONSE);
    });

    const first = refreshImageMapPoints();
    refreshImageMapPoints();
    await first;

    await vi.waitFor(() => expect(pointsCall).toBe(2));
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
  });
});

describe('cluster palette', () => {
  it('cycles the palette by cluster id and dims noise', () => {
    expect(getClusterColor(0)).toBe(CLUSTER_PALETTE[0]);
    expect(getClusterColor(CLUSTER_PALETTE.length)).toBe(CLUSTER_PALETTE[0]);
    expect(getClusterColor(3)).toBe(CLUSTER_PALETTE[3]);
    expect(getClusterColor(-1)).toBe(NOISE_COLOR);
  });
});

/** Cluster and kind vary on different cycles, so neither can stand in for the other. */
const manyPoints = (): ImageMapPoint[] =>
  Array.from({ length: 500 }, (_, index) => {
    const kind = index % 11 === 0 ? ('video' as const) : ('image' as const);
    const name = `${index}.${kind === 'video' ? 'mp4' : 'png'}`;

    return {
      cluster: index % 7 === 0 ? -1 : index % 37,
      item: { kind, name },
      key: `${kind}:${name}` as ImageMapPoint['key'],
      x: index,
      y: -index,
    };
  });

describe('trace builders', () => {
  const points: ImageMapPoint[] = [
    { cluster: 0, item: { kind: 'image', name: 'a.png' }, key: 'image:a.png', x: 1, y: 2 },
    { cluster: -1, item: { kind: 'video', name: 'clip.mp4' }, key: 'video:clip.mp4', x: 3, y: 4 },
  ];

  it('splits the base points into one scattergl trace per appearance', () => {
    const traces = buildAllPointsTraces(points);

    // Every marker property is scalar. Per-point arrays are what plotly
    // reprocesses on each relayout, and a zoom is a relayout per frame.
    for (const trace of traces) {
      expect(trace.type).toBe('scattergl');
      expect(trace.name).toBe(ALL_POINTS_TRACE);
      expect(typeof trace.marker.color).toBe('string');
      expect(typeof trace.marker.opacity).toBe('number');
      expect(typeof trace.marker.symbol).toBe('string');
    }

    const clustered = traces.find((trace) => trace.marker.color === getClusterColor(0));
    const noise = traces.find((trace) => trace.marker.color === getClusterColor(-1));

    expect(clustered?.x).toEqual([1]);
    expect(clustered?.y).toEqual([2]);
    // Keys, not bare names: a click has to know which namespace to resolve in.
    expect(clustered?.customdata).toEqual(['image:a.png']);
    // Kind gets the one channel colour and size do not already carry, so a
    // clip is findable on the map without hovering every point.
    expect(clustered?.marker.symbol).toBe('circle');
    expect(noise?.marker.symbol).toBe('diamond');
    expect(noise?.customdata).toEqual(['video:clip.mp4']);
    // Noise points are dimmed relative to clustered points.
    expect(noise?.marker.opacity as number).toBeLessThan(clustered?.marker.opacity as number);
    // And dimmed points are drawn first, so they sit under the clustered ones.
    expect(traces.indexOf(noise!)).toBeLessThan(traces.indexOf(clustered!));
  });

  it('gives every point the appearance it would have had, and keeps it exactly once', () => {
    // The split is the only thing standing between a point and the map. A
    // bucketing slip drops or duplicates part of the gallery; a subtler one
    // keys on the wrong field and every clustered point comes out the same
    // colour, or a video comes out a circle. Cluster and kind vary
    // independently here so that keying on either alone fails.
    const many = manyPoints();
    const byKey = new Map(many.map((point) => [point.key, point]));
    const traces = buildAllPointsTraces(many);
    const seen: string[] = [];

    for (const trace of traces) {
      for (const [index, key] of trace.customdata.entries()) {
        const point = byKey.get(key as ImageMapPoint['key']);

        expect(point).toBeDefined();
        seen.push(key);
        // Index alignment: plotly reads x, y and customdata positionally, so
        // a shuffle inside one trace plots a point at another's coordinates
        // and resolves a click to the wrong image.
        expect(trace.x[index]).toBe(point?.x);
        expect(trace.y[index]).toBe(point?.y);
        // And the scalar appearance has to be the one this point earned.
        expect(trace.marker.color).toBe(getClusterColor(point!.cluster));
        expect(trace.marker.opacity).toBe(point!.cluster < 0 ? 0.25 : 0.85);
        expect(trace.marker.symbol).toBe(point!.item.kind === 'video' ? 'diamond' : 'circle');
      }
    }

    expect(seen).toHaveLength(many.length);
    expect(new Set(seen).size).toBe(many.length);
    // Bounded by the palette, not by the gallery: 15 colours + noise, each
    // able to carry images and videos.
    expect(traces.length).toBeLessThanOrEqual((CLUSTER_PALETTE.length + 1) * 2);
  });

  it('draws a point whose cluster is not an integer rather than losing it', () => {
    // The bucket slot is an array index. A fractional label would collide
    // with another cluster's slot and steal its colour; a non-finite one
    // would write a string property the emit loop never visits, and those
    // points would vanish from the map with nothing to show for it. The
    // endpoint declares `int` and the client does not validate, so only this
    // stands between a contract slip and missing images.
    const odd: ImageMapPoint[] = [2.5, Number.NaN, Infinity, -0.5].map((cluster, index) => ({
      cluster,
      item: { kind: 'image', name: `${index}.png` },
      key: `image:${index}.png`,
      x: index,
      y: index,
    }));

    const traces = buildAllPointsTraces(odd);

    expect(traces.flatMap((trace) => trace.customdata)).toHaveLength(odd.length);
    // Treated as unclustered: nothing sensible names their cluster.
    for (const trace of traces) {
      expect(trace.marker.color).toBe(getClusterColor(-1));
      expect(trace.marker.opacity).toBe(0.25);
    }
  });

  it('yields no base traces for an empty map', () => {
    // The view never mounts the plot for an empty point set, and plotly
    // handles the trace count changing in either direction, so there is
    // nothing to stand in for.
    expect(buildAllPointsTraces([])).toEqual([]);
  });

  it('builds an empty gold current-image trace that stays last in z-order', () => {
    const trace = buildCurrentImageTrace();

    expect(trace.name).toBe(CURRENT_IMAGE_TRACE);
    expect(trace.x).toEqual([]);
    expect(trace.marker.color).toBe('#FFD700');
    expect(trace.marker.symbol).toBe('circle-dot');
    expect(trace.marker.size).toBe(18);
  });

  it('builds an isotropic pannable layout', () => {
    const layout = buildMapLayout();

    expect(layout.dragmode).toBe('pan');
    expect(layout.xaxis?.scaleanchor).toBe('y');
    expect(layout.plot_bgcolor).toBe('rgba(0,0,0,0)');
    // Orientation grid: lines on, labels off (no units), on both axes.
    expect(layout.xaxis?.showgrid).toBe(true);
    expect(layout.yaxis?.showgrid).toBe(true);
    expect(layout.xaxis?.showticklabels).toBe(false);
    expect(layout.yaxis?.showticklabels).toBe(false);
  });
});

describe('map layout stability', () => {
  it('pins uirevision to a constant so pan/zoom survives a data refresh', () => {
    // Stable uirevision preserves viewport across data refreshes; data-derived revisions would silently reset it.
    const first = buildMapLayout();
    const second = buildMapLayout();

    expect(first.uirevision).toBeTruthy();
    expect(first.uirevision).toBe(second.uirevision);
    expect(typeof first.uirevision).toBe('string');
  });
});

describe('snapshot transitions', () => {
  it('clears a previous error once a refresh succeeds', async () => {
    // Asserting `error === null` after a success is vacuous when the fixture
    // starts at null; seed a real error first so the clearing is what is tested.
    imageMapStore.setSnapshot({
      clusterLabels: null,
      clusterLabelsEps: null,
      clusterLabelsHash: null,
      data: null,
      error: 'boom',
      indexCounts: null,
      indexUpdatedAt: null,
      loadState: 'error',
      renderError: null,
    });
    mocks.apiFetchJson.mockResolvedValueOnce({
      cluster_eps: null,
      model_name: null,
      point_count: 0,
      points: [],
      stale: false,
      state: 'ready',
      updated_at: null,
    });
    // The labels request this refresh fires must not hit the exhausted
    // once-queue (an undefined body arms a retry timer — see above).
    mocks.apiFetchJson.mockResolvedValueOnce(FOREIGN_LABELS_RESPONSE);

    await refreshImageMapPoints();

    expect(imageMapStore.getSnapshot().error).toBeNull();
    expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
  });

  it('clears a render failure on a successful refresh so the plot can retry', () => {
    // Retry must clear renderError so a transient WebGL failure can remount the plot.
    imageMapStore.setSnapshot({
      clusterLabels: null,
      clusterLabelsEps: null,
      clusterLabelsHash: null,
      data: null,
      error: null,
      indexCounts: null,
      indexUpdatedAt: null,
      loadState: 'loaded',
      renderError: 'The map failed to render (WebGL unavailable).',
    });

    expect(imageMapStore.getSnapshot().renderError).not.toBeNull();
  });
});

/** Lets everything a just-resolved request queued behind it run. */
const drainMacrotask = (): Promise<void> =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

const EMPTY_SNAPSHOT = {
  clusterLabels: null,
  clusterLabelsEps: null,
  clusterLabelsHash: null,
  data: null,
  error: null,
  indexCounts: null,
  indexUpdatedAt: null,
  loadState: 'idle',
  renderError: null,
} as const;

describe('image map status', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('derives the pending count the backend computes but does not serialize', async () => {
    mocks.apiFetchJson.mockResolvedValue({ enabled: true, index: { embedded: 30, failed: 2, total: 100 } });

    const status = await fetchImageMapStatus();

    // The same opt-in as /points: the projection counts in this response are
    // filtered by it, so the two must agree about what the map contains.
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/status?include_videos=true');
    // Failures are excluded, exactly as `ImageIndexStatus.pending` does it, so
    // the queue can still drain to zero with images given up on.
    expect(status.index).toEqual({ embedded: 30, failed: 2, pending: 68, total: 100 });
  });

  it('has no counts for a non-admin, who is not told the aggregate totals', async () => {
    mocks.apiFetchJson.mockResolvedValue({ enabled: true, index: null });

    expect((await fetchImageMapStatus()).index).toBeNull();
  });
});

describe('image index progress', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    imageMapStore.setSnapshot({ ...EMPTY_SNAPSHOT });
  });

  it('stamps when the index last moved so the UI can say how long it has stood still', () => {
    recordImageIndexStatus({ embedded: 10, failed: 0, pending: 90, total: 100 }, 4_000);

    expect(imageMapStore.getSnapshot().indexUpdatedAt).toBe(4_000);
  });

  it('does not treat a growing gallery as the index making progress', () => {
    // `total` moves as the generation the indexer is waiting out saves its
    // images. Counting that would reset the clock on every generation.
    recordImageIndexStatus({ embedded: 10, failed: 0, pending: 90, total: 100 }, 1_000);
    recordImageIndexStatus({ embedded: 10, failed: 0, pending: 95, total: 105 }, 60_000);

    expect(imageMapStore.getSnapshot().indexUpdatedAt).toBe(1_000);
    expect(imageMapStore.getSnapshot().indexCounts?.total).toBe(105);
  });

  it('restarts the clock as soon as the index does move', () => {
    recordImageIndexStatus({ embedded: 10, failed: 0, pending: 90, total: 100 }, 1_000);
    recordImageIndexStatus({ embedded: 18, failed: 0, pending: 82, total: 100 }, 60_000);

    expect(imageMapStore.getSnapshot().indexUpdatedAt).toBe(60_000);
  });

  it('seeds the counts from the status endpoint when the map first loads', async () => {
    // Status events only fire as batches complete, so a panel opened while the
    // worker is parked behind a generation would otherwise show no progress.
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/status')) {
        return Promise.resolve({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });
      }

      return url.startsWith('/api/v1/image_map/cluster_labels')
        ? Promise.resolve(FOREIGN_LABELS_RESPONSE)
        : Promise.resolve(BACKEND_RESPONSE);
    });

    ensureImageMapLoaded();

    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts).not.toBeNull());
    expect(imageMapStore.getSnapshot().indexCounts).toEqual({ embedded: 40, failed: 0, pending: 60, total: 100 });
  });

  it('does not let a slow seed rewind counts a status event already delivered', async () => {
    let resolveStatus: (value: unknown) => void = () => {};
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/status')) {
        return new Promise((resolve) => {
          resolveStatus = resolve;
        });
      }

      return url.startsWith('/api/v1/image_map/cluster_labels')
        ? Promise.resolve(FOREIGN_LABELS_RESPONSE)
        : Promise.resolve(BACKEND_RESPONSE);
    });

    ensureImageMapLoaded();
    recordImageIndexStatus({ embedded: 90, failed: 0, pending: 10, total: 100 }, 1_000);
    resolveStatus({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });

    // A macrotask drains everything the resolved seed queued behind it.
    await drainMacrotask();

    expect(imageMapStore.getSnapshot().indexCounts?.embedded).toBe(90);
  });

  it('re-reads the counts on every mount, not just the first load of the map', async () => {
    // Reopening mid-backfill must fetch counts even if the worker is paused and no event is due.
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.startsWith('/api/v1/image_map/status')
        ? Promise.resolve({ enabled: true, index: { embedded: 70, failed: 0, total: 100 } })
        : Promise.resolve(BACKEND_RESPONSE)
    );

    imageMapStore.setSnapshot({ ...EMPTY_SNAPSHOT, loadState: 'loaded' });
    ensureImageMapLoaded();

    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts).not.toBeNull());
    expect(imageMapStore.getSnapshot().indexCounts).toEqual({ embedded: 70, failed: 0, pending: 30, total: 100 });
  });

  it('lets a later status fetch correct counts an event left stale', async () => {
    // The run's final `pending: 0` report is lost while the socket is down, so
    // without this the progress UI claims a finished backfill is still running
    // until the page is reloaded.
    recordImageIndexStatus({ embedded: 40, failed: 0, pending: 60, total: 100 }, 1_000);
    mocks.apiFetchJson.mockResolvedValue({ enabled: true, index: { embedded: 100, failed: 0, total: 100 } });

    refreshImageIndexStatus();

    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts?.pending).toBe(0));
  });

  it('ages the counts from when they last moved, not from when they were re-read', async () => {
    // Otherwise pressing "Check again" — the one thing a user watching a
    // frozen bar will do — pushes the note out by another interval, forever.
    recordImageIndexStatus({ embedded: 40, failed: 0, pending: 60, total: 100 }, 1_000);
    mocks.apiFetchJson.mockResolvedValue({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });

    refreshImageIndexStatus();

    await vi.waitFor(() => expect(mocks.apiFetchJson).toHaveBeenCalled());
    await drainMacrotask();

    expect(imageMapStore.getSnapshot().indexUpdatedAt).toBe(1_000);
  });

  it('runs one status request at a time so an older response cannot land last', async () => {
    // Mount, retry and reconnect requests must coalesce despite unordered completions.
    const resolvers: Array<(value: unknown) => void> = [];
    mocks.apiFetchJson.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvers.push(resolve);
        })
    );

    refreshImageIndexStatus();
    refreshImageIndexStatus();
    refreshImageIndexStatus();

    expect(resolvers).toHaveLength(1);

    resolvers[0]?.({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });
    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts).not.toBeNull());

    refreshImageIndexStatus();
    expect(resolvers).toHaveLength(2);
    resolvers[1]?.({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });
    await drainMacrotask();
  });
});
