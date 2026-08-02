import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  onConnectionChange: vi.fn(),
  refreshImageMapPoints: vi.fn(),
  socketOn: vi.fn(),
}));

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
    imageMapStore.setSnapshot({ data: null, error: null, indexCounts: null, loadState: 'idle', renderError: null });
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

    getHandler('image_index_status')({ embedded: 3, pending: 2, total: 5 });

    expect(imageMapStore.getSnapshot().indexCounts).toEqual({ embedded: 3, pending: 2, total: 5 });
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
    onConnection('connected');
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);
    onConnection('disconnected');
    expect(mocks.refreshImageMapPoints).toHaveBeenCalledTimes(1);

    detach();
    expect(detachSocket).toHaveBeenCalledTimes(3);
    expect(detachConnection).toHaveBeenCalledTimes(1);
  });
});
