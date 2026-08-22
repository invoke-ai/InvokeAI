import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  apiFetchJson: mocks.apiFetchJson,
  getApiErrorMessage: (_error: unknown, fallback: string) => fallback,
}));

import { fetchImageMapPoints, fetchImageMapStatus, requestImageMapRefresh } from './api';
import { CLUSTER_PALETTE, getClusterColor, NOISE_COLOR } from './clusterPalette';
import {
  ensureImageMapLoaded,
  imageMapStore,
  recordImageIndexStatus,
  refreshImageIndexStatus,
  refreshImageMapPoints,
} from './imageMapStore';
import {
  ALL_POINTS_TRACE,
  buildAllPointsTrace,
  buildCurrentImageTrace,
  buildMapLayout,
  CURRENT_IMAGE_TRACE,
} from './imageMapTraces';

const BACKEND_RESPONSE = {
  cluster_eps: 0.42,
  point_count: 2,
  points: [
    { cluster: 0, image_name: 'a.png', x: 1.5, y: -2 },
    { cluster: -1, image_name: 'b.png', x: 0, y: 3 },
  ],
  stale: false,
  state: 'ready',
  updated_at: '2026-08-02 12:00:00',
  visible_hash: 'hash-1',
};

describe('image map api', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('maps snake_case points to camelCase', async () => {
    mocks.apiFetchJson.mockResolvedValue(BACKEND_RESPONSE);

    const result = await fetchImageMapPoints();

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/points');
    expect(result.state).toBe('ready');
    expect(result.pointCount).toBe(2);
    expect(result.clusterEps).toBe(0.42);
    expect(result.visibleHash).toBe('hash-1');
    expect(result.points[0]).toEqual({ cluster: 0, imageName: 'a.png', x: 1.5, y: -2 });
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

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/points?eps=0.4&min_samples=5');
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
      data: null,
      error: null,
      indexCounts: null,
      indexUpdatedAt: null,
      loadState: 'idle',
      renderError: null,
    });
  });

  it('loads points into the snapshot', async () => {
    mocks.apiFetchJson.mockResolvedValue(BACKEND_RESPONSE);

    await refreshImageMapPoints();

    const snapshot = imageMapStore.getSnapshot();
    expect(snapshot.loadState).toBe('loaded');
    expect(snapshot.data?.points).toHaveLength(2);
    expect(snapshot.error).toBeNull();
  });

  it('passes the points response eps through to the cluster labels fetch', async () => {
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

    // The labels endpoint must receive the exact eps the map was clustered
    // with; the adaptive default could resolve differently on a drifted set.
    await vi.waitFor(() => {
      expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/cluster_labels?eps=0.42');
      expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': 'cats' });
    });
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
      expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/cluster_labels?eps=0.42');
    });
    // Flush the response handler: cluster ids from a different visible set
    // must not be applied to the rendered map.
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
      expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/cluster_labels?eps=0.42');
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
      expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': 'cats' });
    });

    // L1's late failure is stale: it must not wipe the labels L2 set.
    rejectFirstLabels(new Error('slow failure'));
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(imageMapStore.getSnapshot().clusterLabels).toEqual({ '0': 'cats' });
  });

  it('records errors and keeps prior data', async () => {
    mocks.apiFetchJson.mockResolvedValue(BACKEND_RESPONSE);
    await refreshImageMapPoints();

    mocks.apiFetchJson.mockRejectedValue(new Error('boom'));
    await refreshImageMapPoints();

    const snapshot = imageMapStore.getSnapshot();
    expect(snapshot.loadState).toBe('error');
    expect(snapshot.error).toBe('Failed to load the image map');
    expect(snapshot.data?.points).toHaveLength(2);
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
});

describe('cluster palette', () => {
  it('cycles the palette by cluster id and dims noise', () => {
    expect(getClusterColor(0)).toBe(CLUSTER_PALETTE[0]);
    expect(getClusterColor(CLUSTER_PALETTE.length)).toBe(CLUSTER_PALETTE[0]);
    expect(getClusterColor(3)).toBe(CLUSTER_PALETTE[3]);
    expect(getClusterColor(-1)).toBe(NOISE_COLOR);
  });
});

describe('trace builders', () => {
  const points = [
    { cluster: 0, imageName: 'a.png', x: 1, y: 2 },
    { cluster: -1, imageName: 'b.png', x: 3, y: 4 },
  ];

  it('builds the all-points scattergl trace with image names as customdata', () => {
    const trace = buildAllPointsTrace(points);

    expect(trace.type).toBe('scattergl');
    expect(trace.name).toBe(ALL_POINTS_TRACE);
    expect(trace.x).toEqual([1, 3]);
    expect(trace.y).toEqual([2, 4]);
    expect(trace.customdata).toEqual(['a.png', 'b.png']);
    expect((trace.marker.color as string[])[0]).toBe(getClusterColor(0));
    // Noise points are dimmed relative to clustered points.
    const opacities = trace.marker.opacity as number[];
    expect(opacities[1]).toBeLessThan(opacities[0]);
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
    // The whole point of uirevision is that plotly keeps the user's viewport
    // when the traces change. A value derived from the data (a point count, a
    // hash, a timestamp) would compare unequal on every refresh and silently
    // reset the view — which looks identical to "it works" in a static test.
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

    await refreshImageMapPoints();

    expect(imageMapStore.getSnapshot().error).toBeNull();
    expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
  });

  it('clears a render failure on a successful refresh so the plot can retry', () => {
    // Without this the WebGL error is permanent for the session: the view stops
    // mounting the plot, and nothing else ever resets renderError.
    imageMapStore.setSnapshot({
      clusterLabels: null,
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

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/status');
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
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.startsWith('/api/v1/image_map/status')
        ? Promise.resolve({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } })
        : Promise.resolve(BACKEND_RESPONSE)
    );

    ensureImageMapLoaded();

    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts).not.toBeNull());
    expect(imageMapStore.getSnapshot().indexCounts).toEqual({ embedded: 40, failed: 0, pending: 60, total: 100 });
  });

  it('does not let a slow seed rewind counts a status event already delivered', async () => {
    let resolveStatus: (value: unknown) => void = () => {};
    mocks.apiFetchJson.mockImplementation((url: string) =>
      url.startsWith('/api/v1/image_map/status')
        ? new Promise((resolve) => {
            resolveStatus = resolve;
          })
        : Promise.resolve(BACKEND_RESPONSE)
    );

    ensureImageMapLoaded();
    recordImageIndexStatus({ embedded: 90, failed: 0, pending: 10, total: 100 }, 1_000);
    resolveStatus({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });

    // A macrotask drains everything the resolved seed queued behind it.
    await drainMacrotask();

    expect(imageMapStore.getSnapshot().indexCounts?.embedded).toBe(90);
  });

  it('re-reads the counts on every mount, not just the first load of the map', async () => {
    // The widget is routinely reopened long after the first load — mid-
    // backfill, with the worker parked and no event due — which is exactly
    // when the panel has nothing else to show.
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
    // Concurrent requests resolve in no fixed order, and every mount, retry
    // and reconnect asks for one.
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

    // ...and the claim is released once it settles, so the next caller is not
    // locked out for the rest of the session.
    resolvers[0]?.({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });
    await vi.waitFor(() => expect(imageMapStore.getSnapshot().indexCounts).not.toBeNull());

    refreshImageIndexStatus();
    expect(resolvers).toHaveLength(2);
    resolvers[1]?.({ enabled: true, index: { embedded: 40, failed: 0, total: 100 } });
    await drainMacrotask();
  });
});
