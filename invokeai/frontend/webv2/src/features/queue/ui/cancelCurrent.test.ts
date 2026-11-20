import type { QueueItemReadModel, QueueReadModel, QueueStatusReadModel } from '@features/queue/core/types';

import { applyQueueItemStatusToReadModels } from '@features/queue/data/queries';
import { accountLifecycle, captureAccountScope, registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const server = vi.hoisted(() => ({
  cancelItem: vi.fn<(id: number) => Promise<unknown>>(),
  /** What a fresh read of the queue finds; a function gives each read its own answer. */
  model: null as unknown,
  reads: 0,
}));

vi.mock('@features/queue/publicApi', () => ({
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({
    queryFn: () => {
      server.reads += 1;
      return typeof server.model === 'function' ? (server.model as () => unknown)() : server.model;
    },
    queryKey: ['queue', 'read-model', 'all'],
    staleTime: Number.POSITIVE_INFINITY,
  }),
  queueCommands: { cancelItem: (id: number) => server.cancelItem(id) },
}));

const { CANCEL_CURRENT_SETTLE_MS, cancelCurrentQueueItem, getCancelCurrentTarget, getInProgressAnywhere } =
  await import('./cancelCurrent');

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
  statusSequence: 1,
  updatedAt: '2026-11-14T00:00:00.000Z',
  userId: 'me',
  ...overrides,
});

/** A reading of the scope: its counts are the items', unless `pending` says more wait than the window shows. */
const readModel = (
  items: QueueItemReadModel[],
  { isStarted = true, pending }: { isStarted?: boolean; pending?: number } = {}
): QueueReadModel => {
  const waiting = items.filter((candidate) => candidate.status === 'pending').sort((left, right) => left.id - right.id);
  const running = items.filter((candidate) => candidate.status === 'in_progress');
  return {
    current: running[0] ?? null,
    items,
    next: waiting[0] ?? null,
    scope: {},
    status: {
      processor: { isProcessing: running.length > 0, isStarted },
      queue: {
        canceled: 0,
        completed: 0,
        failed: 0,
        inProgress: running.length,
        pending: pending ?? waiting.length,
        queueId: 'default',
        total: items.length,
        waiting: 0,
      },
    },
  };
};

const mine = (candidate: QueueItemReadModel) => candidate.userId === 'me';
const batch = () => readModel([item(9, 'pending'), item(8, 'pending'), item(7, 'in_progress')]);
/** The server's answer to a cancel: the item canceled, or unchanged when it had already finished. */
const answer = (id: number, status: QueueItemReadModel['status'] = 'canceled') =>
  Promise.resolve(item(id, status, { statusSequence: 5 }));

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
  return new Promise<never>((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new Error('Account is no longer active')), { once: true });
  });
};
const cached = () => client.getQueryData<QueueReadModel>(['queue', 'read-model', 'all'])!;
/** The whole queue's counts, from the status read the Launchpad shares. */
const wholeQueue = () => client.getQueryData<QueueStatusReadModel>(['queue', 'status', 'all'])?.queue;
const cancelCurrent = () => cancelCurrentQueueItem({ canManageItem: mine, client, scope: {} });

beforeEach(() => {
  vi.useFakeTimers();
  accountLifecycle.activate('user-a');
  client = new QueryClient();
  client.setQueryData(['queue', 'read-model', 'all'], batch());
  server.cancelItem.mockReset();
  server.cancelItem.mockImplementation((id) => answer(id));
  server.model = batch();
  server.reads = 0;
});

afterEach(async () => {
  // Every test leaves the command free: its settle window ends with its own timers.
  await vi.advanceTimersByTimeAsync(10_000);
  vi.useRealTimers();
});

describe('cancel-current target', () => {
  /** In an unscoped view, whose own counts are the whole queue's. */
  const targetId = (model: Parameters<typeof getCancelCurrentTarget>[0], skippedItemId?: number) =>
    getCancelCurrentTarget(model, {
      canManageItem: mine,
      inProgressAnywhere: getInProgressAnywhere({}, model, null),
      skippedItemId,
    })?.id ?? null;

  it('stops the running item, else between items the next one waiting', () => {
    expect(targetId(batch())).toBe(7);
    // The hand-off between items, or a canceled item's worker still finishing: nothing is in progress.
    expect(targetId(readModel([item(9, 'pending'), item(8, 'pending'), item(7, 'canceled')]))).toBe(8);
    expect(targetId(readModel([item(6, 'completed')]))).toBeNull();
    expect(targetId(null)).toBeNull();
    expect(targetId(batch(), 7)).toBeNull();
  });

  it("stops nothing waiting while someone else's item runs, but the viewer's own item running beside it", () => {
    const theirs = item(20, 'in_progress', { userId: 'someone' });

    expect(targetId(readModel([item(22, 'pending'), item(21, 'pending'), theirs]))).toBeNull();
    // Another GPU runs one of the viewer's items.
    expect(targetId(readModel([item(22, 'pending'), item(21, 'in_progress'), theirs]))).toBe(21);
  });
});

describe('cancelCurrentQueueItem', () => {
  it('stops one item for a double click, a double tap or a held key, and the next on a later press', async () => {
    const first = deferred<QueueItemReadModel>();
    server.cancelItem.mockReturnValueOnce(first.promise);

    const pressed = cancelCurrent();
    // A held key's repeats while the request is out.
    const repeats = await Promise.all([cancelCurrent(), cancelCurrent()]);
    first.resolve(item(7, 'canceled', { statusSequence: 5 }));
    expect(repeats).toEqual([{ kind: 'busy' }, { kind: 'busy' }]);
    expect(await pressed).toMatchObject({ item: { id: 7 }, kind: 'canceled' });

    // The second click of a double click, 150 ms after the first settled.
    await vi.advanceTimersByTimeAsync(150);
    await expect(cancelCurrent()).resolves.toEqual({ kind: 'busy' });

    // The read that reconciles the queue lands (the app's queue views keep the read model observed); then a deliberate
    // press after the settle window stops the next item.
    client.setQueryData(
      ['queue', 'read-model', 'all'],
      readModel([item(9, 'pending'), item(8, 'pending'), item(7, 'canceled', { statusSequence: 5 })])
    );
    await vi.advanceTimersByTimeAsync(CANCEL_CURRENT_SETTLE_MS);
    await expect(cancelCurrent()).resolves.toMatchObject({ item: { id: 8 }, kind: 'canceled' });
    expect(server.cancelItem.mock.calls).toEqual([[7], [8]]);
  });

  it('puts the item back as running when the cancel fails, so the next press tries it again at once', async () => {
    const failure = new Error('Server unavailable');
    server.cancelItem.mockRejectedValueOnce(failure);

    await expect(cancelCurrent()).resolves.toEqual({ error: failure, kind: 'failed' });
    expect(cached().current).toMatchObject({ id: 7, status: 'in_progress' });
    expect(cached().items.find((candidate) => candidate.id === 7)?.status).toBe('in_progress');

    await expect(cancelCurrent()).resolves.toMatchObject({ item: { id: 7 }, kind: 'canceled' });
  });

  it('cancels nothing it cannot know is next: a long batch whose oldest waiting items are outside the window', async () => {
    // Items 101 to 200 wait; the window shows the newest 50, and the server names 101 next.
    const window = Array.from({ length: 50 }, (_, index) => item(200 - index, 'pending'));
    client.setQueryData(['queue', 'read-model', 'all'], {
      ...readModel(window, { pending: 100 }),
      next: item(101, 'pending'),
    });

    await expect(cancelCurrent()).resolves.toMatchObject({ item: { id: 101 }, kind: 'canceled' });
    await vi.advanceTimersByTimeAsync(CANCEL_CURRENT_SETTLE_MS);
    // Before a read names 102, the next is unknown: nothing is stopped, rather than 151.
    expect(cached().next).toBeNull();
    await expect(cancelCurrent()).resolves.toEqual({ kind: 'nothing' });
    expect(server.cancelItem.mock.calls).toEqual([[101]]);
  });

  it('stops what is current now when the item it read had already finished, once', async () => {
    server.cancelItem.mockImplementation((id) => answer(id, id === 7 ? 'completed' : 'canceled'));
    server.model = readModel([item(9, 'pending'), item(8, 'in_progress'), item(7, 'completed')]);

    await expect(cancelCurrent()).resolves.toMatchObject({ item: { id: 8 }, kind: 'canceled' });
    expect(server.cancelItem.mock.calls).toEqual([[7], [8]]);
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

  it("keeps a fresh read that lands after a sign-out out of the next account's cache", async () => {
    server.cancelItem.mockImplementation((id) => answer(id, 'completed'));
    const read = deferred<QueueReadModel>();
    server.model = () => read.promise;

    const pressed = cancelCurrent();
    await vi.advanceTimersByTimeAsync(0);
    expect(server.reads).toBe(1);
    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    await expect(pressed).resolves.toEqual({ kind: 'superseded' });

    // The old account's server answers only now.
    read.resolve(batch());
    await vi.advanceTimersByTimeAsync(0);
    expect(client.getQueryData(['queue', 'read-model', 'all'])).toBeUndefined();
  });

  it("keeps the item canceled when the server's event confirmed it before the request failed", async () => {
    const request = deferred<QueueItemReadModel>();
    server.cancelItem.mockReturnValueOnce(request.promise);
    client.setQueryData(['queue', 'status', 'all'], batch().status);
    const guessed = { ...batch().status.queue, canceled: 1, inProgress: 0 };

    const pressed = cancelCurrent();
    // The server cancels item 7 and says so, exactly as guessed; then its answer to the request is lost.
    applyQueueItemStatusToReadModels(client, {
      item: item(7, 'canceled', { statusSequence: 5 }),
      queueCounts: guessed,
    });
    request.reject(new Error('Connection lost'));

    await expect(pressed).resolves.toMatchObject({ item: { id: 7 }, kind: 'canceled' });
    expect(cached().items.find((candidate) => candidate.id === 7)?.status).toBe('canceled');
    expect(wholeQueue()).toEqual(guessed);
  });
});
