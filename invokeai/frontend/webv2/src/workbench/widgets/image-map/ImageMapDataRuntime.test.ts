import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
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

vi.mock('@workbench/image-map/imageMapStore', async (importOriginal) => {
  const original = (await importOriginal()) as object;

  return { ...original, refreshImageMapPoints: mocks.refreshImageMapPoints };
});

import { imageMapStore } from '@workbench/image-map/imageMapStore';

import { attachImageMapDataRuntime } from './ImageMapDataRuntime';

const getHandler = (event: string): ((payload?: unknown) => void) => {
  const call = mocks.socketOn.mock.calls.find(([name]) => name === event);
  expect(call, `no listener for ${event}`).toBeDefined();

  return call![1];
};

describe('attachImageMapDataRuntime', () => {
  beforeEach(() => {
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
    const detach = attachImageMapDataRuntime();
    const onReady = getHandler('image_map_projection_ready');

    onReady();
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    imageMapStore.patchSnapshot({ loadState: 'loaded' });
    onReady();
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });

  it('stores index progress counts from status events', () => {
    const detach = attachImageMapDataRuntime();

    getHandler('image_index_status')({ embedded: 3, failed: 1, pending: 2, total: 5 });

    expect(imageMapStore.getSnapshot().indexCounts).toEqual({ embedded: 3, failed: 1, pending: 2, total: 5 });
    detach();
  });

  it('defaults the given-up count for a server that predates the field', () => {
    const detach = attachImageMapDataRuntime();

    getHandler('image_index_status')({ embedded: 3, pending: 2, total: 5 });

    expect(imageMapStore.getSnapshot().indexCounts?.failed).toBe(0);
    detach();
  });

  it('leaves a failed canvas alone instead of remounting it once per event', () => {
    const detach = attachImageMapDataRuntime();

    // Refresh must not repeatedly clear a WebGL failure and remount the broken plot during backfill.
    imageMapStore.patchSnapshot({ loadState: 'loaded', renderError: 'The map failed to render.' });
    getHandler('image_map_projection_ready')({ user_id: 'u1' });
    getHandler('image_index_updated')({ user_id: 'u1' });
    getHandler('image_index_status')({ embedded: 5, pending: 0, total: 5 });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    // The user's own retry still clears it, and refreshes resume.
    imageMapStore.patchSnapshot({ renderError: null });
    getHandler('image_index_updated')({ user_id: 'u1' });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });

  it('does not queue a duplicate fetch for an event during the first load', () => {
    const detach = attachImageMapDataRuntime();

    // Ignore ordinary pokes during first loading to avoid duplicate full fetches; newly embedded items may wait
    // for a later event/manual refresh.
    imageMapStore.patchSnapshot({ loadState: 'loading' });
    getHandler('image_index_updated')({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();
    detach();
  });

  it('refetches for a projection ready during the first load', () => {
    const detach = attachImageMapDataRuntime();
    const onReady = getHandler('image_map_projection_ready');

    // Queue one rerun when recompute overtakes its triggering fetch so stale points and mismatched labels do not
    // persist.
    imageMapStore.patchSnapshot({ loadState: 'loading' });
    onReady({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);
    detach();
  });

  it('refetches for a projection ready after a failed load', () => {
    const detach = attachImageMapDataRuntime();
    const onReady = getHandler('image_map_projection_ready');

    // loadState 'error' with no fetch in flight (the first load failed): the
    // event starts a fresh fetch rather than being dropped — the failing
    // first load may have missed the projection it announced.
    imageMapStore.patchSnapshot({ loadState: 'error' });
    onReady({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);
    detach();
  });

  it('still ignores a foreign projection ready during the first load', () => {
    const detach = attachImageMapDataRuntime();
    const onReady = getHandler('image_map_projection_ready');

    mocks.getAuthSession.mockReturnValue({ user: { user_id: 'me' } });
    imageMapStore.patchSnapshot({ loadState: 'loading' });

    onReady({ user_id: 'someone-else' });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    detach();
  });

  it('leaves a failed canvas alone for a projection ready during the first load', () => {
    const detach = attachImageMapDataRuntime();
    const onReady = getHandler('image_map_projection_ready');

    // A retry in flight after a WebGL failure: the canvas stays failed until
    // that retry's own success clears the error, and a mid-flight event must
    // not start another refresh that would remount into the same failure.
    imageMapStore.patchSnapshot({ loadState: 'loading', renderError: 'The map failed to render.' });
    onReady({ user_id: 'u1' });

    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();
    detach();
  });

  it('ignores a projection recomputed for a different user', () => {
    const detach = attachImageMapDataRuntime();
    const onReady = getHandler('image_map_projection_ready');

    mocks.getAuthSession.mockReturnValue({ user: { user_id: 'me' } });
    imageMapStore.patchSnapshot({ loadState: 'loaded' });

    // Ignore other users' projection events; admin refetches can otherwise trigger repeated recomputes.
    onReady({ user_id: 'someone-else' });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    onReady({ user_id: 'me' });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    // Single-user mode has no session user, so nothing is ever foreign.
    mocks.getAuthSession.mockReturnValue({ user: null });
    onReady({ user_id: 'whoever' });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(2);

    detach();
  });

  it('refreshes a loaded map when a status event reports the index quiescent', () => {
    const detach = attachImageMapDataRuntime();
    const onStatus = getHandler('image_index_status');

    // Not yet loaded: counts are stored but nothing refetches.
    onStatus({ embedded: 5, pending: 0, total: 5 });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    imageMapStore.patchSnapshot({ loadState: 'loaded' });
    // Mid-backfill events keep the map quiet; only quiescence pokes /points.
    onStatus({ embedded: 3, pending: 2, total: 5 });
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    onStatus({ embedded: 5, pending: 0, total: 5 });
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });

  it('refreshes a loaded map on the per-user index-updated poke', () => {
    const detach = attachImageMapDataRuntime();
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

    const detach = attachImageMapDataRuntime();
    const onConnection = mocks.onConnectionChange.mock.calls[0][0];

    imageMapStore.patchSnapshot({ loadState: 'error' });
    // The subscribe-time replay establishes the baseline, nothing more.
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

    // Treat synchronous connection replay as baseline, not reconnect, on editor remount.
    const detach = attachImageMapDataRuntime();
    const onConnection = mocks.onConnectionChange.mock.calls[0][0];

    onConnection('connected');
    expect(mocks.refreshImageMapPoints).not.toHaveBeenCalled();

    // A genuine drop and recovery still refreshes.
    onConnection('disconnected');
    onConnection('connected');
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
  });
});
