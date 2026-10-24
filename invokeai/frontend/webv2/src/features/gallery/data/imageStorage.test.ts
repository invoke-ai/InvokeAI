import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
  invalidateGallery: vi.fn(() => Promise.resolve()),
  refreshGalleryThumbnails: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({ apiFetchJson: mocks.apiFetchJson }));
vi.mock('./queryCache', () => ({
  invalidateGallery: mocks.invalidateGallery,
  refreshGalleryThumbnails: mocks.refreshGalleryThumbnails,
}));

import {
  imageMoveStatusQueryOptions,
  imageStorageKeys,
  imageStorageLayoutQueryOptions,
  startImageMove,
  startImageMoveRecovery,
  updateImageStorageLayout,
} from './imageStorage';

const createDeferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const runningStatus = {
  active_job_id: 3,
  is_running: true,
  last_error: null,
  latest_job: { error_message: null, id: 3, state: 'moving' },
  needs_move_count: 12,
  operation: 'move_all',
};

const committedStatus = {
  active_job_id: null,
  is_running: false,
  last_error: null,
  latest_job: { error_message: null, id: 3, state: 'committed' },
  needs_move_count: 0,
  operation: null,
};

beforeEach(() => {
  vi.clearAllMocks();
  accountLifecycle.activate('admin');
});

describe('image storage layout', () => {
  it('reads the layout from the nested runtime config and keeps unknown server values', async () => {
    mocks.apiFetchJson.mockResolvedValueOnce({ config: { image_subfolder_strategy: 'date' }, set_fields: [] });
    await expect(new QueryClient().fetchQuery(imageStorageLayoutQueryOptions())).resolves.toBe('date');
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/app/runtime_config', expect.anything());

    mocks.apiFetchJson.mockResolvedValueOnce({ config: { image_subfolder_strategy: 'by_user' } });
    await expect(new QueryClient().fetchQuery(imageStorageLayoutQueryOptions())).resolves.toBe('by_user');

    mocks.apiFetchJson.mockResolvedValueOnce({ config: {} });
    await expect(new QueryClient().fetchQuery(imageStorageLayoutQueryOptions())).resolves.toBe('flat');
  });

  it('patches only the layout field, caches the answer, and refreshes the move count', async () => {
    const client = new QueryClient();
    const owner = accountLifecycle.capture();
    client.setQueryData(imageStorageKeys.moveStatus(owner), { needsMoveCount: 0 });
    mocks.apiFetchJson.mockResolvedValueOnce({ config: { image_subfolder_strategy: 'hash' } });

    await updateImageStorageLayout(client, 'hash', owner);

    expect(mocks.apiFetchJson).toHaveBeenCalledExactlyOnceWith(
      '/api/v1/app/runtime_config',
      expect.objectContaining({ body: JSON.stringify({ image_subfolder_strategy: 'hash' }), method: 'PATCH' })
    );
    expect(client.getQueryData(imageStorageKeys.layout(owner))).toBe('hash');
    expect(client.getQueryState(imageStorageKeys.moveStatus(owner))?.isInvalidated).toBe(true);
  });
});

describe('image move status', () => {
  it('maps the wire status to the read model', async () => {
    mocks.apiFetchJson.mockResolvedValueOnce(runningStatus);

    await expect(new QueryClient().fetchQuery(imageMoveStatusQueryOptions())).resolves.toEqual({
      activeJobId: 3,
      isRunning: true,
      lastError: null,
      latestJob: { errorMessage: null, id: 3, state: 'moving' },
      needsMoveCount: 12,
      operation: 'move_all',
    });
  });

  it('polls while running and refreshes the gallery on the read that sees the job stop', async () => {
    const client = new QueryClient();
    const owner = accountLifecycle.capture();
    const options = imageMoveStatusQueryOptions(owner);

    mocks.apiFetchJson.mockResolvedValueOnce(runningStatus);
    const running = await client.fetchQuery(options);
    expect(running.isRunning).toBe(true);
    expect(options.refetchInterval).toBeTypeOf('function');
    const interval = options.refetchInterval as (query: { state: { data?: unknown } }) => number | false;
    expect(interval({ state: { data: running } })).toBe(2_000);
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();

    mocks.apiFetchJson.mockResolvedValueOnce(committedStatus);
    const committed = await client.fetchQuery({ ...options, staleTime: 0 });
    expect(committed.isRunning).toBe(false);
    expect(interval({ state: { data: committed } })).toBe(false);
    expect(mocks.invalidateGallery).toHaveBeenCalledExactlyOnceWith(client, owner);
    // Thumbnails the move answered with 409 are asked for again as well.
    expect(mocks.refreshGalleryThumbnails).toHaveBeenCalledExactlyOnceWith(owner);

    // An idle read after an idle read is not a job ending.
    mocks.apiFetchJson.mockResolvedValueOnce(committedStatus);
    await client.fetchQuery({ ...options, staleTime: 0 });
    expect(mocks.invalidateGallery).toHaveBeenCalledTimes(1);
  });

  it('seeds the status query with the accepted job so polling starts without a second read', async () => {
    const client = new QueryClient();
    const owner = accountLifecycle.capture();
    mocks.apiFetchJson.mockResolvedValueOnce(runningStatus);

    await expect(startImageMove(client, owner)).resolves.toMatchObject({ isRunning: true, operation: 'move_all' });

    expect(mocks.apiFetchJson).toHaveBeenCalledExactlyOnceWith(
      '/api/v1/image_moves/start',
      expect.objectContaining({ method: 'POST' })
    );
    expect(client.getQueryData(imageStorageKeys.moveStatus(owner))).toMatchObject({ activeJobId: 3, isRunning: true });
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });

  it('keeps the accepted status when a stale status read lands after the POST', async () => {
    const client = new QueryClient();
    const owner = accountLifecycle.capture();
    const options = imageMoveStatusQueryOptions(owner);
    const staleRead = createDeferred<unknown>();
    mocks.apiFetchJson.mockImplementation((path: string, init?: RequestInit) =>
      init?.method === 'POST' ? Promise.resolve(runningStatus) : staleRead.promise
    );

    // A poll started before the POST is still in flight when the POST is accepted.
    const pendingRead = client.fetchQuery(options).catch((error: unknown) => error);
    await expect(startImageMove(client, owner)).resolves.toMatchObject({ isRunning: true });
    expect(client.getQueryData(imageStorageKeys.moveStatus(owner))).toMatchObject({ isRunning: true });

    staleRead.resolve(committedStatus);
    await pendingRead;
    await Promise.resolve();

    // The idle answer the server gave before the job existed must not stop the polling or free the controls.
    expect(client.getQueryData(imageStorageKeys.moveStatus(owner))).toMatchObject({ activeJobId: 3, isRunning: true });
    expect(client.getQueryState(imageStorageKeys.moveStatus(owner))?.fetchStatus).toBe('idle');
  });

  it('refreshes the gallery at once when the accepted job already finished', async () => {
    const client = new QueryClient();
    const owner = accountLifecycle.capture();
    mocks.apiFetchJson.mockResolvedValueOnce(committedStatus);

    await expect(startImageMoveRecovery(client, owner)).resolves.toMatchObject({ isRunning: false });

    expect(mocks.apiFetchJson).toHaveBeenCalledExactlyOnceWith(
      '/api/v1/image_moves/recover',
      expect.objectContaining({ method: 'POST' })
    );
    expect(client.getQueryData(imageStorageKeys.moveStatus(owner))).toMatchObject({ isRunning: false });
    expect(mocks.invalidateGallery).toHaveBeenCalledExactlyOnceWith(client, owner);
    expect(mocks.refreshGalleryThumbnails).toHaveBeenCalledExactlyOnceWith(owner);
  });

  it('re-reads the status when the server refuses to start, since the cached status is what misled the caller', async () => {
    const client = new QueryClient();
    const owner = accountLifecycle.capture();
    client.setQueryData(imageStorageKeys.moveStatus(owner), { isRunning: false });
    const refusal = new Error('{"detail":"An image move job is already running"}');
    mocks.apiFetchJson.mockRejectedValueOnce(refusal);

    await expect(startImageMove(client, owner)).rejects.toBe(refusal);

    expect(client.getQueryState(imageStorageKeys.moveStatus(owner))?.isInvalidated).toBe(true);
    expect(mocks.invalidateGallery).not.toHaveBeenCalled();
  });

  it('drops a response that lands after the account changed', async () => {
    const client = new QueryClient();
    const owner = accountLifecycle.capture();
    mocks.apiFetchJson.mockImplementationOnce(() => {
      accountLifecycle.activate('someone-else');
      return Promise.resolve(runningStatus);
    });

    await expect(startImageMove(client, owner)).rejects.toThrow();
    expect(client.getQueryData(imageStorageKeys.moveStatus(owner))).toBeUndefined();
  });
});
