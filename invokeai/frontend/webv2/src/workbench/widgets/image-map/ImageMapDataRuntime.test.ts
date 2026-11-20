import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  fetchImageMapPoints: vi.fn(),
  getAuthSession: vi.fn(() => ({ user: null }) as { user: { user_id: string } | null }),
  onConnectionChange: vi.fn(),
  refreshImageMapPoints: vi.fn(),
  socketOn: vi.fn(),
}));

vi.mock('@features/identity', () => ({ getAuthSession: mocks.getAuthSession }));

vi.mock('@platform/transport/socketHub', () => ({
  socketHub: {
    on: mocks.socketOn,
    onConnectionChange: mocks.onConnectionChange,
  },
}));

vi.mock('@workbench/image-map/api', () => ({
  fetchImageMapClusterLabels: vi.fn(),
  fetchImageMapPoints: mocks.fetchImageMapPoints,
  fetchImageMapStatus: vi.fn(() => new Promise(() => {})),
}));

vi.mock('@workbench/image-map/imageMapStore', async (importOriginal) => {
  const original = (await importOriginal()) as object;

  return { ...original, refreshImageMapPoints: mocks.refreshImageMapPoints };
});

import type { GalleryMediaFilter } from '@features/gallery/contracts';

import { attachImageMapViewer, imageMapStore, setImageMapMediaKind } from '@workbench/image-map/imageMapStore';

import { attachImageMapDataRuntime } from './ImageMapDataRuntime';

const BOTH_KINDS = { getMediaFilter: (): GalleryMediaFilter => 'all', subscribe: () => () => {} };

const getHandler = (event: string): ((payload?: unknown) => void) => {
  const call = mocks.socketOn.mock.calls.find(([name]) => name === event);
  expect(call, `no listener for ${event}`).toBeDefined();

  return call![1];
};

describe('attachImageMapDataRuntime', () => {
  afterEach(() => {
    imageMapStore.patchSnapshot({ loadState: 'idle' });
    setImageMapMediaKind(undefined);
  });

  beforeEach(() => {
    mocks.fetchImageMapPoints.mockReset().mockResolvedValue({
      clusterEps: null,
      modelId: null,
      modelName: null,
      pointCount: 0,
      points: [],
      stale: false,
      state: 'empty',
      updatedAt: null,
      visibleHash: null,
    });
    mocks.socketOn.mockReset().mockReturnValue(() => {});
    mocks.onConnectionChange.mockReset().mockReturnValue(() => {});
    mocks.refreshImageMapPoints.mockReset();
    mocks.getAuthSession.mockReset().mockReturnValue({ user: null });
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

  it('refreshes on projection-ready only after the map has been loaded once', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onReady = getHandler('image_map_projection_ready');

    onReady();
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    imageMapStore.patchSnapshot({ loadState: 'loaded' });
    onReady();
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });

  it('stores index progress counts from status events', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);

    getHandler('image_index_status')({ embedded: 3, failed: 1, pending: 2, total: 5 });

    expect(imageMapStore.getSnapshot().indexCounts).toEqual({ embedded: 3, failed: 1, pending: 2, total: 5 });
    detach();
  });

  it('defaults the given-up count for a server that predates the field', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);

    getHandler('image_index_status')({ embedded: 3, pending: 2, total: 5 });

    expect(imageMapStore.getSnapshot().indexCounts?.failed).toBe(0);
    detach();
  });

  it('leaves a failed canvas alone instead of remounting it once per event', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);

    imageMapStore.patchSnapshot({ loadState: 'loaded', renderError: 'The map failed to render.' });
    getHandler('image_map_projection_ready')({ user_id: 'u1' });
    getHandler('image_index_updated')({ user_id: 'u1' });
    getHandler('image_index_status')({ embedded: 5, pending: 0, total: 5 });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    imageMapStore.patchSnapshot({ renderError: null });
    getHandler('image_index_updated')({ user_id: 'u1' });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });

  it('does not queue a duplicate fetch for an event during the first load', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);

    imageMapStore.patchSnapshot({ loadState: 'loading' });
    getHandler('image_index_updated')({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();
    detach();
  });

  it('refetches for a projection ready during the first load', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onReady = getHandler('image_map_projection_ready');

    imageMapStore.patchSnapshot({ loadState: 'loading' });
    onReady({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);
    detach();
  });

  it('refetches for a projection ready after a failed load', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onReady = getHandler('image_map_projection_ready');

    imageMapStore.patchSnapshot({ loadState: 'error' });
    onReady({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);
    detach();
  });

  it('still ignores a foreign projection ready during the first load', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onReady = getHandler('image_map_projection_ready');

    mocks.getAuthSession.mockReturnValue({ user: { user_id: 'me' } });
    imageMapStore.patchSnapshot({ loadState: 'loading' });

    onReady({ user_id: 'someone-else' });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    detach();
  });

  it('leaves a failed canvas alone for a projection ready during the first load', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onReady = getHandler('image_map_projection_ready');

    imageMapStore.patchSnapshot({ loadState: 'loading', renderError: 'The map failed to render.' });
    onReady({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();
    detach();
  });

  it('ignores a projection recomputed for a different user', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onReady = getHandler('image_map_projection_ready');

    mocks.getAuthSession.mockReturnValue({ user: { user_id: 'me' } });
    imageMapStore.patchSnapshot({ loadState: 'loaded' });

    onReady({ user_id: 'someone-else' });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    onReady({ user_id: 'me' });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    mocks.getAuthSession.mockReturnValue({ user: null });
    onReady({ user_id: 'whoever' });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(2);

    detach();
  });

  it('refreshes a loaded map when a status event reports the index quiescent', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onStatus = getHandler('image_index_status');

    onStatus({ embedded: 5, pending: 0, total: 5 });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    imageMapStore.patchSnapshot({ loadState: 'loaded' });
    onStatus({ embedded: 3, pending: 2, total: 5 });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    onStatus({ embedded: 5, pending: 0, total: 5 });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });

  it('refreshes a loaded map on the per-user index-updated poke', () => {
    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onUpdated = getHandler('image_index_updated');

    onUpdated({ user_id: 'u1' });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    imageMapStore.patchSnapshot({ loadState: 'loaded' });
    onUpdated({ user_id: 'u1' });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });

  it('refreshes a loaded map when the socket reconnects, and detaches cleanly', () => {
    const detachSocket = vi.fn();
    const detachConnection = vi.fn();
    mocks.socketOn.mockReturnValue(detachSocket);
    mocks.onConnectionChange.mockReturnValue(detachConnection);

    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onConnection = mocks.onConnectionChange.mock.calls[0][0];

    imageMapStore.patchSnapshot({ loadState: 'error' });
    onConnection('disconnected');
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    onConnection('connected');
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);
    onConnection('disconnected');
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
    expect(detachSocket).toHaveBeenCalledTimes(3);
    expect(detachConnection).toHaveBeenCalledTimes(1);
  });

  it('does not refetch when it attaches to an already-connected socket', () => {
    imageMapStore.patchSnapshot({ loadState: 'loaded' });

    const detach = attachImageMapDataRuntime(BOTH_KINDS);
    const onConnection = mocks.onConnectionChange.mock.calls[0][0];

    onConnection('connected');
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    onConnection('disconnected');
    onConnection('connected');
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });
  it('refetches a loaded map exactly once per change of the shared media filter', async () => {
    const listeners = new Set<() => void>();
    let mediaFilter: GalleryMediaFilter = 'all';
    const notify = () => listeners.forEach((listener) => listener());
    const detachViewer = attachImageMapViewer();
    imageMapStore.patchSnapshot({ loadState: 'loaded' });

    const detach = attachImageMapDataRuntime({
      getMediaFilter: () => mediaFilter,
      subscribe: (listener) => {
        listeners.add(listener);

        return () => listeners.delete(listener);
      },
    });

    notify();
    expect(mocks.fetchImageMapPoints).not.toHaveBeenCalled();

    mediaFilter = 'video';
    notify();
    notify();
    await vi.waitFor(() => expect(mocks.fetchImageMapPoints).toHaveBeenCalledTimes(1));
    expect(mocks.fetchImageMapPoints).toHaveBeenLastCalledWith({ mediaKind: 'video' });

    mediaFilter = 'all';
    notify();
    await vi.waitFor(() => expect(mocks.fetchImageMapPoints).toHaveBeenCalledTimes(2));
    expect(mocks.fetchImageMapPoints).toHaveBeenLastCalledWith({ mediaKind: undefined });

    detach();
    mediaFilter = 'image';
    notify();
    expect(listeners.size).toBe(0);
    expect(mocks.fetchImageMapPoints).toHaveBeenCalledTimes(2);
    detachViewer();
  });
});
