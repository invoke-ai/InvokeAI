import { accountLifecycle } from '@platform/state/accountLifecycle';
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

import type { ImageMapPoint, ImageMapPoints } from './api';

import { fetchImageMapPoints, fetchImageMapStatus } from './api';
import { getClusterColor } from './clusterPalette';
import { getImageLabels } from './imageLabelCache';
import {
  attachImageMapStatusPolling,
  ensureImageMapLoaded,
  imageMapStore,
  recordImageIndexStatus,
  refreshImageIndexStatus,
  refreshImageMapPoints,
  requestImageMapRecompute,
  setClusterEps,
  attachImageMapViewer,
  setImageMapMediaKind,
} from './imageMapStore';
import { ALL_POINTS_TRACE, buildAllPointsTraces, buildClusterAnnotations, buildMapLayout } from './imageMapTraces';

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

// Malformed label fixtures throw TypeError and arm network retries beyond the test.
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

    // Without include_videos the backend serves images only.
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
    expect(result.points[1]?.item).toEqual({ kind: 'image', name: 'b.png' });
    expect(result.points[2]?.item).toEqual({ kind: 'video', name: 'clip.mp4' });
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

  afterEach(() => {
    vi.useRealTimers();
  });

  it('clusters at the chosen strength on every later refresh', async () => {
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

  it('drops points fetched for the previous media kind and labels the new set with the same kind', async () => {
    const VIDEO_RESPONSE = {
      ...BACKEND_RESPONSE,
      point_count: 1,
      points: [{ cluster: 0, image_name: 'clip.mp4', kind: 'video', x: 2, y: -1 }],
      visible_hash: 'hash-videos',
    };
    let resolveStale: (value: unknown) => void = () => {};
    const detachViewer = attachImageMapViewer();
    mockPointsWithForeignLabels();
    await refreshImageMapPoints();
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.resolve(FOREIGN_LABELS_RESPONSE);
      }

      return url.includes('media_kind=video')
        ? Promise.resolve(VIDEO_RESPONSE)
        : new Promise((resolve) => {
            resolveStale = resolve;
          });
    });

    const staleRefresh = refreshImageMapPoints();
    setImageMapMediaKind('video');
    resolveStale(BACKEND_RESPONSE);
    await staleRefresh;
    await vi.waitFor(() => expect(imageMapStore.getSnapshot().data?.visibleHash).toBe('hash-videos'));

    const urls = mocks.apiFetchJson.mock.calls.map((call) => String(call[0]));
    expect(urls.filter((url) => url.startsWith('/api/v1/image_map/points'))).toEqual([
      '/api/v1/image_map/points?include_videos=true',
      '/api/v1/image_map/points?include_videos=true&media_kind=video',
    ]);
    expect(urls.filter((url) => url.startsWith('/api/v1/image_map/cluster_labels'))).toEqual([
      '/api/v1/image_map/cluster_labels?include_videos=true&media_kind=video&eps=0.42',
    ]);
    expect(imageMapStore.getSnapshot().data?.points.map((point) => point.key)).toEqual(['video:clip.mp4']);

    detachViewer();
    imageMapStore.patchSnapshot({ data: null, loadState: 'idle' });
    setImageMapMediaKind(undefined);
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
        // Must match /points: labels whose visible hash disagrees are discarded, as any other item set would cause.
        '/api/v1/image_map/cluster_labels?include_videos=true&eps=0.42'
      );
    });
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();
  });

  it('ignores a stale labels failure after a newer request already set labels', async () => {
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
        // Must match /points: labels whose visible hash disagrees are discarded, as any other item set would cause.
        '/api/v1/image_map/cluster_labels?include_videos=true&eps=0.42'
      );
    });

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

    rejectFirstLabels(new Error('slow failure'));
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': { alternates: [], label: 'cats' } });
  });

  it('retries a labels request that raced the vocabulary build', async () => {
    vi.useFakeTimers();
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/cluster_labels')) {
        return Promise.reject(new mocks.ApiError('Cluster labels are still being prepared; try again shortly', 409));
      }

      return Promise.resolve(BACKEND_RESPONSE);
    });

    await refreshImageMapPoints();
    await vi.advanceTimersByTimeAsync(0);
    expect(imageMapStore.getSnapshot().clusterLabels).toBeNull();

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

  it('collapses mid-flight refresh requests into one rerun', async () => {
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
    expect(pointsResolvers).toHaveLength(1);

    pointsResolvers[0]?.(BACKEND_RESPONSE);
    await Promise.all(calls);

    await vi.waitFor(() => expect(pointsResolvers).toHaveLength(2));
    pointsResolvers[1]?.(BACKEND_RESPONSE);
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(pointsResolvers).toHaveLength(2);
    expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
  });

  it('asks for a refit and fetches the points when a person requests a fresh map', async () => {
    mockPointsWithForeignLabels();

    await requestImageMapRecompute();

    const calls = mocks.apiFetchJson.mock.calls;

    expect(calls).toContainEqual(['/api/v1/image_map/refresh', { method: 'POST' }]);
    expect(calls.some((call) => String(call[0]).startsWith('/api/v1/image_map/points'))).toBe(true);
    expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
  });
});

describe('trace builders', () => {
  const points: ImageMapPoint[] = [
    { cluster: 0, item: { kind: 'image', name: 'a.png' }, key: 'image:a.png', x: 1, y: 2 },
    { cluster: -1, item: { kind: 'video', name: 'clip.mp4' }, key: 'video:clip.mp4', x: 3, y: 4 },
  ];

  it('splits the base points into one scattergl trace per appearance', () => {
    const traces = buildAllPointsTraces(points);

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
    expect(clustered?.customdata).toEqual(['image:a.png']);
    expect(clustered?.marker.symbol).toBe('circle');
    expect(noise?.marker.symbol).toBe('diamond');
    expect(noise?.customdata).toEqual(['video:clip.mp4']);
    expect(noise?.marker.opacity as number).toBeLessThan(clustered?.marker.opacity as number);
    expect(traces.indexOf(noise!)).toBeLessThan(traces.indexOf(clustered!));
  });
});

describe('map layout stability', () => {
  it('pins uirevision to a constant so pan/zoom survives a data refresh', () => {
    const annotations = buildClusterAnnotations(
      [{ cluster: 0, item: { kind: 'image', name: 'a.png' }, key: 'image:a.png', x: 1, y: 2 }],
      { '0': 'Boats' }
    );
    const first = buildMapLayout();
    const second = buildMapLayout({ x: [0, 1], y: [0, 1] }, annotations);

    expect(annotations).toHaveLength(1);
    expect(typeof first.uirevision).toBe('string');
    expect(first.uirevision).toBeTruthy();
    expect(second.uirevision).toBe(first.uirevision);
  });
});

describe('snapshot transitions', () => {
  it('clears a previous error once a refresh succeeds', async () => {
    // Seed a real error first, or asserting `error === null` would be vacuous.
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
    // Must miss the exhausted once-queue: an undefined body arms a retry.
    mocks.apiFetchJson.mockResolvedValueOnce(FOREIGN_LABELS_RESPONSE);

    await refreshImageMapPoints();

    expect(imageMapStore.getSnapshot().error).toBeNull();
    expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
  });
});

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

const LOADED_POINTS: ImageMapPoints = {
  clusterEps: null,
  modelId: null,
  modelName: null,
  pointCount: 0,
  points: [],
  stale: false,
  state: 'ready',
  updatedAt: null,
  visibleHash: null,
};

describe('image map status', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('derives the pending count the backend computes but does not serialize', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      enabled: true,
      projection: { state: 'ready' },
      index: { embedded: 30, failed: 2, total: 100 },
    });

    const status = await fetchImageMapStatus();

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/status?include_videos=true');
    // Failures are excluded, as `ImageIndexStatus.pending` does.
    expect(status.index).toEqual({ embedded: 30, failed: 2, pending: 68, total: 100 });
  });

  it('has no counts for a non-admin, who is not told the aggregate totals', async () => {
    mocks.apiFetchJson.mockResolvedValue({ enabled: true, projection: { state: 'ready' }, index: null });

    expect((await fetchImageMapStatus()).index).toBeNull();
  });
});

describe('image index progress', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    accountLifecycle.invalidate();
    imageMapStore.setSnapshot({ ...EMPTY_SNAPSHOT });
  });

  it('polls only for the mounted view lifetime, without overlapping requests or surviving account changes', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('document', Object.assign(new EventTarget(), { visibilityState: 'visible' }));
    try {
      let resolveStatus: (value: unknown) => void = () => {};
      mocks.apiFetchJson.mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveStatus = resolve;
          })
      );
      const detach = attachImageMapStatusPolling();
      await vi.advanceTimersByTimeAsync(5_000);
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(10_000);
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
      resolveStatus({ enabled: true, index: null, projection: { state: 'computing' } });
      await vi.advanceTimersByTimeAsync(0);
      detach();
      await vi.advanceTimersByTimeAsync(5_000);
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

      const detachNext = attachImageMapStatusPolling();
      accountLifecycle.activate('next-account');
      await vi.advanceTimersByTimeAsync(5_000);
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
      detachNext();
    } finally {
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });

  it('replaces stalled progress with the missing encoder reported by a status poll, including for non-admins', async () => {
    const item = { kind: 'image', name: 'before-removal.png' } as const;
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'old encoder label', score: 0.7 });
    expect((await getImageLabels(item))?.label).toBe('old encoder label');
    imageMapStore.patchSnapshot({ data: LOADED_POINTS, loadState: 'loaded' });
    recordImageIndexStatus({ embedded: 40, failed: 0, pending: 60, total: 100 }, 1_000);
    mocks.apiFetchJson.mockResolvedValue({
      enabled: false,
      index: null,
      model_name: 'removed-encoder',
      projection: { point_count: 0, stale: false, state: 'model_missing' },
    });

    refreshImageIndexStatus();
    await drainMacrotask();

    expect(imageMapStore.getSnapshot().data).toMatchObject({ modelName: 'removed-encoder', state: 'model_missing' });
    expect(imageMapStore.getSnapshot().indexCounts).toBeNull();
    expect(imageMapStore.getSnapshot().indexUpdatedAt).toBeNull();
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'reinstalled encoder label', score: 0.8 });
    expect((await getImageLabels(item))?.label).toBe('reinstalled encoder label');
  });

  it('discards old in-flight points when status discovers a replacement encoder', async () => {
    imageMapStore.patchSnapshot({ data: { ...LOADED_POINTS, modelId: 'encoder-a' }, loadState: 'loaded' });
    let resolveOldPoints: (value: unknown) => void = () => {};
    mocks.apiFetchJson.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveOldPoints = resolve;
        })
    );
    const oldPoints = refreshImageMapPoints();
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/status')) {
        return Promise.resolve({ enabled: true, model_id: 'encoder-b', projection: { state: 'ready' } });
      }
      return Promise.resolve(
        url.startsWith('/api/v1/image_map/cluster_labels')
          ? FOREIGN_LABELS_RESPONSE
          : { ...BACKEND_RESPONSE, model_id: 'encoder-b' }
      );
    });

    refreshImageIndexStatus();
    await drainMacrotask();
    expect(imageMapStore.getSnapshot().data).toMatchObject({ modelId: 'encoder-b', points: [] });
    resolveOldPoints({ ...BACKEND_RESPONSE, model_id: 'encoder-a' });
    await oldPoints;
    await drainMacrotask();

    expect(imageMapStore.getSnapshot().data).toMatchObject({ modelId: 'encoder-b', pointCount: 3, state: 'ready' });
  });

  it('ignores an old account status response without clearing its newer request', async () => {
    const resolvers: Array<(value: unknown) => void> = [];
    mocks.apiFetchJson.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvers.push(resolve);
        })
    );
    accountLifecycle.activate('first');
    refreshImageIndexStatus();
    accountLifecycle.activate('second');
    refreshImageIndexStatus();
    resolvers[0]?.({ enabled: false, index: null, model_name: 'foreign', projection: { state: 'model_missing' } });
    await drainMacrotask();
    refreshImageIndexStatus();
    expect(resolvers).toHaveLength(2);
    expect(imageMapStore.getSnapshot().data).toBeNull();
    resolvers[1]?.({ enabled: false, index: null, model_name: 'current', projection: { state: 'model_missing' } });
    await drainMacrotask();
    expect(imageMapStore.getSnapshot().data?.modelName).toBe('current');
  });

  it('seeds the counts from the status endpoint when the map first loads', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url.startsWith('/api/v1/image_map/status')) {
        return Promise.resolve({
          enabled: true,
          projection: { state: 'ready' },
          index: { embedded: 40, failed: 0, total: 100 },
        });
      }

      return url.startsWith('/api/v1/image_map/cluster_labels')
        ? Promise.resolve(FOREIGN_LABELS_RESPONSE)
        : Promise.resolve(BACKEND_RESPONSE);
    });

    ensureImageMapLoaded();

    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts).not.toBeNull());
    expect(imageMapStore.getSnapshot().indexCounts).toEqual({ embedded: 40, failed: 0, pending: 60, total: 100 });
  });

  it('runs one status request at a time so an older response cannot land last', async () => {
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

    resolvers[0]?.({ enabled: true, projection: { state: 'ready' }, index: { embedded: 40, failed: 0, total: 100 } });
    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts).not.toBeNull());

    refreshImageIndexStatus();
    expect(resolvers).toHaveLength(2);
    resolvers[1]?.({ enabled: true, projection: { state: 'ready' }, index: { embedded: 40, failed: 0, total: 100 } });
    await drainMacrotask();
  });
});
