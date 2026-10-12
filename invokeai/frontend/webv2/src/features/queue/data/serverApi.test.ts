import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => transport);

import {
  cancelCurrentQueueItem,
  cancelQueueItems,
  cancelScopedQueueItems,
  clearFailedQueueItems,
  clearFinishedQueueItems,
  clearScopedQueue,
  getQueueItemsByIds,
} from './serverApi';

const requestedUrls = (): string[] => transport.apiFetchJson.mock.calls.map((call) => String(call[0]));

describe('queue command account ownership', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    transport.apiFetchJson.mockReset();
  });

  it.each([
    {
      name: 'clear failed',
      resolveFirst: { item_ids: [11], limit: 1, offset: 0 },
      run: () => clearFailedQueueItems({ originPrefix: 'webv2:p:a:' }),
    },
    {
      name: 'clear scoped',
      resolveFirst: { item_ids: [11], limit: 1, offset: 0 },
      run: () => clearScopedQueue({ originPrefix: 'webv2:p:a:' }),
    },
    {
      name: 'cancel current',
      resolveFirst: { item_id: 11, status: 'in_progress' },
      run: () => cancelCurrentQueueItem(),
    },
    {
      name: 'cancel scoped',
      resolveFirst: { item_ids: [11], limit: 1, offset: 0 },
      run: () => cancelScopedQueueItems({ originPrefix: 'webv2:p:a:' }),
    },
  ])('does not start the second $name request after an account switch', async ({ resolveFirst, run }) => {
    let resolveRequest: ((value: unknown) => void) | undefined;
    transport.apiFetchJson.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve;
        })
    );

    const oldCommand = run();
    const signal = (transport.apiFetchJson.mock.calls[0]?.[1] as RequestInit | undefined)?.signal;

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    resolveRequest?.(resolveFirst);

    await expect(oldCommand).rejects.toThrow('no longer active');
    expect(signal?.aborted).toBe(true);
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('bounds fallback item cancellation concurrency', async () => {
    let active = 0;
    let maximum = 0;
    transport.apiFetchJson.mockImplementation(async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await new Promise((resolve) => {
        setTimeout(resolve, 1);
      });
      active -= 1;
    });

    await cancelQueueItems(Array.from({ length: 40 }, (_, index) => index + 1));

    expect(transport.apiFetchJson).toHaveBeenCalledTimes(40);
    expect(maximum).toBeLessThanOrEqual(8);
  });
});

describe('cancelScopedQueueItems', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    transport.apiFetchJson.mockReset();
    transport.apiFetchJson.mockResolvedValue({ canceled: 4581 });
  });

  it('sweeps the scope, in-progress items included, in one server-side request', async () => {
    await cancelScopedQueueItems({ originPrefix: 'webv2:p:a:' });

    expect(requestedUrls()).toEqual(['/api/v1/queue/default/cancel_all?origin_prefix=webv2%3Ap%3Aa%3A']);
    expect(transport.apiFetchJson.mock.calls[0]?.[1]).toMatchObject({ method: 'PUT' });
  });

  it('spares whatever is in progress when asked to keep the current item', async () => {
    await cancelScopedQueueItems({}, { keepCurrent: true });

    expect(requestedUrls()).toEqual(['/api/v1/queue/default/cancel_all_except_current']);
  });
});

describe('scoped clears', () => {
  // More than the read model's recent window: a clear must handle the whole scope, not its newest page.
  const scopeIds = Array.from({ length: 120 }, (_, index) => 120 - index);

  beforeEach(() => {
    accountLifecycle.activate('user-a');
    transport.apiFetchJson.mockReset();
    transport.apiFetchJson.mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/item_ids')) {
        return Promise.resolve({ item_ids: scopeIds, total_count: scopeIds.length });
      }
      if (url.endsWith('/item_summaries_by_ids')) {
        const { item_ids: itemIds } = JSON.parse(String(init?.body)) as { item_ids: number[] };
        return Promise.resolve(itemIds.map((itemId) => ({ item_id: itemId, status: 'failed' })));
      }
      return Promise.resolve({});
    });
  });

  it.each([
    { clear: clearFailedQueueItems, name: 'clear failed' },
    { clear: clearScopedQueue, name: 'clear scoped' },
  ])('$name reads every id in the scope, without a limit, and deletes each one', async ({ clear }) => {
    await clear({ originPrefix: 'webv2:p:a:' });

    const urls = requestedUrls();
    expect(urls.filter((url) => url.includes('/item_ids'))).toEqual([
      '/api/v1/queue/default/item_ids?order_dir=DESC&origin_prefix=webv2%3Ap%3Aa%3A',
    ]);
    expect(urls.filter((url) => /\/i\/\d+$/.test(url)).sort()).toEqual(
      scopeIds.map((itemId) => `/api/v1/queue/default/i/${String(itemId)}`).sort()
    );
  });
});

describe('getQueueItemsByIds', () => {
  beforeEach(() => {
    transport.apiFetchJson.mockReset();
  });

  it('pages ids at the backend cap, keeping id order', async () => {
    transport.apiFetchJson.mockImplementation((_url: string, init?: RequestInit) => {
      const { item_ids: itemIds } = JSON.parse(String(init?.body)) as { item_ids: number[] };
      return Promise.resolve(itemIds.map((itemId) => ({ item_id: itemId })));
    });
    const itemIds = Array.from({ length: 2345 }, (_, index) => index + 1);

    const items = await getQueueItemsByIds(itemIds);

    expect(transport.apiFetchJson).toHaveBeenCalledTimes(3);
    expect(
      transport.apiFetchJson.mock.calls.map((call) => JSON.parse(String((call[1] as RequestInit).body)).item_ids.length)
    ).toEqual([1000, 1000, 345]);
    expect(items.map((item) => item.item_id)).toEqual(itemIds);
  });

  it('issues no request for an empty id list', async () => {
    await expect(getQueueItemsByIds([])).resolves.toEqual([]);
    expect(transport.apiFetchJson).not.toHaveBeenCalled();
  });
});

describe('clearFinishedQueueItems', () => {
  beforeEach(() => {
    accountLifecycle.activate('user-a');
    transport.apiFetchJson.mockReset();
  });

  it('prunes the whole queue in one server sweep when no scope is given', async () => {
    transport.apiFetchJson.mockResolvedValue({ deleted: 3 });

    await clearFinishedQueueItems();

    expect(requestedUrls()).toEqual(['/api/v1/queue/default/prune']);
    expect(transport.apiFetchJson.mock.calls[0]?.[1]).toMatchObject({ method: 'PUT' });
  });

  it('deletes only the terminal items of a scope and leaves active work alone', async () => {
    transport.apiFetchJson.mockImplementation((url: string) => {
      if (url.includes('item_ids')) {
        return Promise.resolve({ item_ids: [5, 4, 3, 2, 1] });
      }
      if (url.endsWith('item_summaries_by_ids')) {
        return Promise.resolve([
          { item_id: 7, parent_item_id: 5, root_item_id: 5, status: 'completed' },
          { item_id: 6, parent_item_id: 3, root_item_id: 3, status: 'completed' },
          { item_id: 5, status: 'waiting' },
          { item_id: 4, status: 'pending' },
          { item_id: 3, status: 'completed' },
          { item_id: 2, status: 'canceled' },
          { item_id: 1, status: 'failed' },
        ]);
      }
      return Promise.resolve({});
    });

    await clearFinishedQueueItems({ originPrefix: 'webv2:p:a:' });

    const deletions = transport.apiFetchJson.mock.calls
      .filter((call) => (call[1] as RequestInit | undefined)?.method === 'DELETE')
      .map((call) => String(call[0]));

    expect(deletions.sort()).toEqual([
      '/api/v1/queue/default/i/1',
      '/api/v1/queue/default/i/2',
      '/api/v1/queue/default/i/3',
    ]);
    expect(requestedUrls()).not.toContain('/api/v1/queue/default/prune');
    // Choosing what to delete needs status and lineage alone, never the items' sessions.
    expect(requestedUrls()).not.toContain('/api/v1/queue/default/items_by_ids');
  });

  it('clears a failed workflow-call chain through its root alone', async () => {
    transport.apiFetchJson.mockImplementation((url: string) => {
      if (url.includes('item_ids')) {
        return Promise.resolve({ item_ids: [3, 2, 1] });
      }
      if (url.endsWith('item_summaries_by_ids')) {
        return Promise.resolve([
          { item_id: 3, parent_item_id: 1, status: 'failed' },
          { item_id: 2, status: 'completed' },
          { item_id: 1, status: 'failed' },
        ]);
      }
      return Promise.resolve({});
    });

    await clearFailedQueueItems({ originPrefix: 'webv2:p:a:' });

    expect(
      transport.apiFetchJson.mock.calls
        .filter((call) => (call[1] as RequestInit | undefined)?.method === 'DELETE')
        .map((call) => String(call[0]))
    ).toEqual(['/api/v1/queue/default/i/1']);
  });
});
