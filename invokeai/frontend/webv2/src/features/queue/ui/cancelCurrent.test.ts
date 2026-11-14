import type { QueueItemReadModel, QueueReadModel } from '@features/queue/core/types';

import { applyQueueItemStatusToReadModels } from '@features/queue/data/queries';
import { accountLifecycle, captureAccountScope, registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const server = vi.hoisted(() => ({
  cancelItem: vi.fn<(id: number) => Promise<{ id: number; status: string }>>(),
  /** What a fresh read of the queue finds. */
  model: null as unknown,
  reads: 0,
}));

vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({
    queryFn: () => {
      server.reads += 1;
      return server.model;
    },
    queryKey: ['queue', 'read-model', 'all'],
    staleTime: Number.POSITIVE_INFINITY,
  }),
  queueCommands: { cancelItem: (id: number) => server.cancelItem(id) },
}));

const { cancelCurrentQueueItem, getCancelCurrentTarget } = await import('./cancelCurrent');

const item = (
  id: number,
  status: QueueItemReadModel['status'],
  overrides: Partial<QueueItemReadModel> = {}
): QueueItemReadModel => ({
  batchId: 'batch-1',
  createdAt: '2026-11-14T00:00:00.000Z',
  id,
  resultImageNames: [],
  sessionId: `session-${id}`,
  status,
  updatedAt: '2026-11-14T00:00:00.000Z',
  userId: 'me',
  ...overrides,
});

const readModel = (items: QueueItemReadModel[]): QueueReadModel => {
  const pending = items.filter((candidate) => candidate.status === 'pending').sort((left, right) => left.id - right.id);
  return {
    current: items.find((candidate) => candidate.status === 'in_progress') ?? null,
    items,
    next: pending[0] ?? null,
    scope: {},
    status: {
      processor: { isProcessing: true, isStarted: true },
      queue: {
        canceled: 0,
        completed: 0,
        failed: 0,
        inProgress: 1,
        pending: 2,
        queueId: 'default',
        total: 3,
        waiting: 0,
      },
    },
  };
};

const mine = (candidate: QueueItemReadModel) => candidate.userId === 'me';
const batch = () => readModel([item(9, 'pending'), item(8, 'pending'), item(7, 'in_progress')]);

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, reject, resolve };
};

let client: QueryClient;
// As the app does with its query client: a sign-out clears the cache before the next account arrives.
registerAccountOwnedResource({ clear: () => client.clear(), name: 'cancel-current-test-query-client' });
/** A cancel the server has not answered, which an account change aborts, as the production adapter's does. */
const unansweredCancel = () => {
  const signal = captureAccountScope().signal;
  return new Promise<{ id: number; status: string }>((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new Error('Account is no longer active')), { once: true });
  });
};
const cached = () => client.getQueryData<QueueReadModel>(['queue', 'read-model', 'all'])!;
const cancelCurrent = () => cancelCurrentQueueItem({ canManageItem: mine, client, scope: {} });

beforeEach(() => {
  accountLifecycle.activate('user-a');
  client = new QueryClient();
  client.setQueryData(['queue', 'read-model', 'all'], batch());
  server.cancelItem.mockReset();
  server.cancelItem.mockImplementation((id) => Promise.resolve({ id, status: 'canceled' }));
  server.model = batch();
  server.reads = 0;
});

describe('cancel-current target', () => {
  const targetId = (model: Parameters<typeof getCancelCurrentTarget>[0], skippedItemId?: number) =>
    getCancelCurrentTarget(model, mine, skippedItemId)?.id ?? null;

  it('stops the running item, else between items the next one waiting', () => {
    expect(targetId(batch())).toBe(7);
    // The hand-off between items, or a canceled item's worker still finishing: nothing runs.
    expect(targetId(readModel([item(9, 'pending'), item(8, 'pending'), item(7, 'canceled')]))).toBe(8);
    expect(targetId(readModel([item(6, 'completed')]))).toBeNull();
    expect(targetId(null)).toBeNull();
  });

  it('reaches past items the viewer cannot manage, in the order the server runs them', () => {
    const theirs = item(20, 'in_progress', { userId: 'someone' });

    // Another user's item runs; of the viewer's own, the one queued at the front runs first.
    expect(
      targetId({
        current: theirs,
        items: [item(23, 'pending', { priority: 10 }), item(22, 'pending'), theirs],
        next: item(21, 'pending', { userId: 'someone' }),
      })
    ).toBe(23);
    expect(targetId(batch(), 7)).toBe(8);
  });
});

describe('cancelCurrentQueueItem', () => {
  it('stops one item for a held key or a double click, and the next on a later press', async () => {
    const first = deferred<{ id: number; status: string }>();
    server.cancelItem.mockReturnValueOnce(first.promise);

    const pressed = cancelCurrent();
    const repeats = await Promise.all([cancelCurrent(), cancelCurrent()]);
    first.resolve({ id: 7, status: 'canceled' });

    expect(repeats).toEqual([{ kind: 'busy' }, { kind: 'busy' }]);
    expect(await pressed).toMatchObject({ item: { id: 7 }, kind: 'canceled' });
    // Shown canceled at once, before any read: the next press stops the next item.
    expect(cached().current).toBeNull();
    await expect(cancelCurrent()).resolves.toMatchObject({ item: { id: 8 }, kind: 'canceled' });
    expect(server.cancelItem.mock.calls).toEqual([[7], [8]]);
  });

  it('puts the item back as running when the cancel fails, so the next press tries it again', async () => {
    const failure = new Error('Server unavailable');
    server.cancelItem.mockRejectedValueOnce(failure);

    await expect(cancelCurrent()).resolves.toEqual({ error: failure, kind: 'failed' });
    expect(cached().current).toMatchObject({ id: 7, status: 'in_progress' });
    expect(cached().items.find((candidate) => candidate.id === 7)?.status).toBe('in_progress');

    await expect(cancelCurrent()).resolves.toMatchObject({ item: { id: 7 }, kind: 'canceled' });
  });

  it('stops what is current now when the item it read had already finished, once', async () => {
    server.cancelItem.mockImplementation((id) => Promise.resolve({ id, status: id === 7 ? 'completed' : 'canceled' }));
    server.model = readModel([item(9, 'pending'), item(8, 'in_progress'), item(7, 'completed')]);

    await expect(cancelCurrent()).resolves.toMatchObject({ item: { id: 8 }, kind: 'canceled' });
    expect(server.cancelItem.mock.calls).toEqual([[7], [8]]);
  });

  it('says there is nothing to cancel when the fresh read finds nothing either', async () => {
    server.cancelItem.mockImplementation((id) => Promise.resolve({ id, status: 'completed' }));
    server.model = readModel([item(7, 'completed')]);

    await expect(cancelCurrent()).resolves.toEqual({ kind: 'nothing' });
    expect(server.cancelItem.mock.calls).toEqual([[7]]);
  });

  it('puts nothing back, reads nothing and retargets nothing once the account has signed out', async () => {
    server.cancelItem.mockImplementationOnce(unansweredCancel);

    const pressed = cancelCurrent();
    // The sign-out clears the cache, then the request it aborted fails.
    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');

    await expect(pressed).resolves.toEqual({ kind: 'superseded' });
    // The next account must not see this one's queue.
    expect(client.getQueryData(['queue', 'read-model', 'all'])).toBeUndefined();
    expect(server.reads).toBe(0);
    expect(server.cancelItem.mock.calls).toEqual([[7]]);
  });

  it('keeps what an event changed while a cancel was waiting, when that cancel then fails', async () => {
    const failure = new Error('Server unavailable');
    let fail!: () => void;
    server.cancelItem.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          fail = () => reject(failure);
        })
    );

    const pressed = cancelCurrent();
    // Item 8 starts meanwhile.
    applyQueueItemStatusToReadModels(client, {
      item: item(8, 'in_progress', { statusSequence: 2 }),
      queueCounts: null,
    });
    fail();

    await expect(pressed).resolves.toEqual({ error: failure, kind: 'failed' });
    expect(cached().items.find((candidate) => candidate.id === 8)?.status).toBe('in_progress');
  });
});
