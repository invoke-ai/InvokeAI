import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  apiFetchJson: mocks.apiFetchJson,
  getApiErrorMessage: (_error: unknown, fallback: string) => fallback,
}));

import { fetchImageMapPoints, requestImageMapRefresh } from './api';
import { CLUSTER_PALETTE, getClusterColor, NOISE_COLOR } from './clusterPalette';
import { imageMapStore, refreshImageMapPoints } from './imageMapStore';
import {
  ALL_POINTS_TRACE,
  buildAllPointsTrace,
  buildCurrentImageTrace,
  buildMapLayout,
  CURRENT_IMAGE_TRACE,
} from './imageMapTraces';

const BACKEND_RESPONSE = {
  point_count: 2,
  points: [
    { cluster: 0, image_name: 'a.png', x: 1.5, y: -2 },
    { cluster: -1, image_name: 'b.png', x: 0, y: 3 },
  ],
  stale: false,
  state: 'ready',
  updated_at: '2026-08-02 12:00:00',
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
    imageMapStore.setSnapshot({ data: null, error: null, loadState: 'idle' });
  });

  it('loads points into the snapshot', async () => {
    mocks.apiFetchJson.mockResolvedValue(BACKEND_RESPONSE);

    await refreshImageMapPoints();

    const snapshot = imageMapStore.getSnapshot();
    expect(snapshot.loadState).toBe('loaded');
    expect(snapshot.data?.points).toHaveLength(2);
    expect(snapshot.error).toBeNull();
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
