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
