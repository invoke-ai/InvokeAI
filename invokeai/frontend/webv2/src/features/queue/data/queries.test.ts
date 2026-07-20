import type { QueueBackendPort } from '@features/queue/core/types';

import { QueryClient } from '@tanstack/react-query';
import { describe, expect, it, vi } from 'vitest';

import { invalidateQueueReadModels, queueKeys, queueReadModelOptions } from './queries';

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
  listItems: vi.fn(),
  on: vi.fn(),
  onConnectionChange: vi.fn(),
  pauseProcessor: vi.fn(),
  readCurrent: vi.fn().mockResolvedValue(null),
  readItemIds: vi.fn().mockResolvedValue({ itemIds: [2, 1], totalCount: 2 }),
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
    expect(backend.readItemsById).toHaveBeenCalledWith([2, 1]);
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

  it('invalidates every scoped read model through the feature key', async () => {
    const backend = createBackend();
    const client = new QueryClient();
    const key = queueKeys.readModel({ originPrefix: 'webv2:p:one:q:' });

    await client.fetchQuery(queueReadModelOptions(backend, { originPrefix: 'webv2:p:one:q:' }));
    await invalidateQueueReadModels(client);

    expect(client.getQueryState(key)?.isInvalidated).toBe(true);
  });
});
