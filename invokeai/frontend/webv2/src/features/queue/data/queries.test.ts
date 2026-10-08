import type { QueueBackendPort } from '@features/queue/core/types';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { queueBackend } from './httpRealtimeQueueBackend';
import {
  invalidateQueueReadModels,
  QUEUE_RECENT_WINDOW,
  queueKeys,
  queueReadModelOptions,
  queueStatusOptions,
} from './queries';
import { createQueueItemDTO, createQueueServer } from './queueServer.testing';

const createBackend = (): QueueBackendPort => ({
  cancelCurrentItem: vi.fn(),
  cancelQueueItems: vi.fn(),
  cancelQueueItemsByBatchIds: vi.fn(),
  cancelItem: vi.fn(),
  cancelScopedItems: vi.fn(),
  clearFailedItems: vi.fn(),
  clearItems: vi.fn(),
  emit: vi.fn(),
  enqueueGenerate: vi.fn(),
  enqueueWorkflow: vi.fn(),
  getItem: vi.fn(),
  getResultImages: vi.fn(),
  getResultVideoNames: vi.fn().mockResolvedValue([]),
  getResultVideos: vi.fn().mockResolvedValue([]),
  listItems: vi.fn(),
  on: vi.fn(),
  onConnectionChange: vi.fn(),
  pauseProcessor: vi.fn(),
  readCurrent: vi.fn().mockResolvedValue(null),
  readItemIds: vi.fn().mockResolvedValue({ itemIds: [2, 1] }),
  readItemsById: vi.fn().mockResolvedValue([
    { batchId: 'batch', createdAt: '', id: 2, resultImageNames: [], sessionId: 's2', status: 'pending', updatedAt: '' },
    {
      batchId: 'batch',
      createdAt: '',
      id: 1,
      resultImageNames: [],
      sessionId: 's1',
      status: 'completed',
      updatedAt: '',
    },
  ]),
  readNext: vi.fn().mockResolvedValue(null),
  readStatus: vi.fn().mockResolvedValue({
    processor: { isProcessing: false, isStarted: true },
    queue: { canceled: 0, completed: 1, failed: 0, inProgress: 0, pending: 1, queueId: 'default', total: 2 },
  }),
  retryItems: vi.fn(),
  resumeProcessor: vi.fn(),
});

describe('queue queries', () => {
  it('deduplicates concurrent reads and serves fresh cached data', async () => {
    const backend = createBackend();
    const client = new QueryClient();
    const options = queueReadModelOptions(backend, {});

    const [first, second] = await Promise.all([client.fetchQuery(options), client.fetchQuery(options)]);
    const cached = await client.fetchQuery(options);

    expect(first).toBe(second);
    expect(cached).toBe(first);
    expect(backend.readStatus).toHaveBeenCalledTimes(1);
    expect(backend.readItemsById).toHaveBeenCalledWith([2, 1], expect.any(AbortSignal));
  });

  it('keeps project-scoped queue reads in distinct cache entries', async () => {
    const backend = createBackend();
    const client = new QueryClient();

    await client.fetchQuery(queueReadModelOptions(backend, { originPrefix: 'webv2:p:one:q:' }));
    await client.fetchQuery(queueReadModelOptions(backend, { originPrefix: 'webv2:p:two:q:' }));

    expect(client.getQueryData(queueKeys.readModel({ originPrefix: 'webv2:p:one:q:' }))).toBeDefined();
    expect(client.getQueryData(queueKeys.readModel({ originPrefix: 'webv2:p:two:q:' }))).toBeDefined();
    expect(backend.readStatus).toHaveBeenCalledTimes(2);
  });

  it('invalidates every scoped read model and status read through the feature key', async () => {
    const backend = createBackend();
    const client = new QueryClient();
    const scope = { originPrefix: 'webv2:p:one:q:' };

    await client.fetchQuery(queueReadModelOptions(backend, scope));
    await client.fetchQuery(queueStatusOptions(backend, {}));
    await invalidateQueueReadModels(client);

    expect(client.getQueryState(queueKeys.readModel(scope))?.isInvalidated).toBe(true);
    expect(client.getQueryState(queueKeys.status({}))?.isInvalidated).toBe(true);
  });

  it('does not continue or publish a read after its account epoch expires', async () => {
    accountLifecycle.activate('user-a');
    const backend = createBackend();
    const onRead = vi.fn();
    let resolveStatus: ((value: Awaited<ReturnType<QueueBackendPort['readStatus']>>) => void) | undefined;

    vi.mocked(backend.readStatus).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveStatus = resolve;
      })
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const oldRead = client.fetchQuery(queueReadModelOptions(backend, {}, onRead));

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    resolveStatus?.({
      processor: { isProcessing: false, isStarted: true },
      queue: {
        canceled: 0,
        completed: 0,
        failed: 0,
        inProgress: 0,
        pending: 0,
        queueId: 'default',
        total: 0,
        waiting: 0,
      },
    });

    await expect(oldRead).rejects.toThrow('no longer active');
    expect(backend.readItemsById).not.toHaveBeenCalled();
    expect(onRead).not.toHaveBeenCalled();
  });

  it('does not publish a status read whose account epoch expired', async () => {
    accountLifecycle.activate('user-a');
    const backend = createBackend();
    let resolveStatus: ((value: Awaited<ReturnType<QueueBackendPort['readStatus']>>) => void) | undefined;

    vi.mocked(backend.readStatus).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveStatus = resolve;
      })
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const oldRead = client.fetchQuery(queueStatusOptions(backend, {}));
    const signal = vi.mocked(backend.readStatus).mock.calls[0]?.[1];

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    resolveStatus?.(await createBackend().readStatus());

    await expect(oldRead).rejects.toThrow('no longer active');
    expect(signal?.aborted).toBe(true);
    expect(client.getQueryData(queueKeys.status({}))).toBeUndefined();
  });
});

describe('queue reads at the transport boundary', () => {
  // Ids ascend with age; the newest item is the one running.
  const queueOf = (size: number) =>
    Array.from({ length: size }, (_, index) =>
      createQueueItemDTO(index + 1, { status: index === size - 1 ? 'in_progress' : 'completed' })
    );

  afterEach(() => {
    vi.unstubAllGlobals();
    accountLifecycle.invalidate();
  });

  it.each([
    { predatesItemIdsLimit: false, size: 60 },
    { predatesItemIdsLimit: false, size: 1_000 },
    // An older server ignores the limit and answers with every id.
    { predatesItemIdsLimit: true, size: 1_000 },
  ])(
    'refreshes a $size-item read model with one window-sized id read and one hydration of at most the window (server predates the limit: $predatesItemIdsLimit)',
    async ({ predatesItemIdsLimit, size }) => {
      accountLifecycle.activate('queue-transport-reads');
      const server = createQueueServer(queueOf(size), { predatesItemIdsLimit });
      vi.stubGlobal('fetch', server.fetch);
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      const window = Array.from({ length: QUEUE_RECENT_WINDOW }, (_, index) => size - index);

      const model = await client.fetchQuery(queueReadModelOptions(queueBackend, {}));

      expect([...server.requests].sort()).toEqual([
        'GET current',
        'GET item_ids?limit=50&order_dir=DESC',
        'GET next',
        'GET status',
        'POST items_by_ids',
      ]);
      expect(server.hydratedIds).toEqual([window]);
      expect(model.items.map((item) => item.id)).toEqual(window);
      expect(model.current?.id).toBe(size);
    }
  );

  it('reads Home counts with a single status request and no item reads', async () => {
    accountLifecycle.activate('queue-transport-status');
    const server = createQueueServer(queueOf(1_000));
    vi.stubGlobal('fetch', server.fetch);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    const status = await client.fetchQuery(queueStatusOptions(queueBackend, {}));

    expect(server.requests).toEqual(['GET status']);
    expect(status.queue).toMatchObject({ completed: 999, inProgress: 1, total: 1_000 });
  });
});
