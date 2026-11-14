import type {
  QueueBackendPort,
  QueueCounts,
  QueueItemDetailReadModel,
  QueueItemReadModel,
  QueueReadModel,
} from '@features/queue/core/types';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { queueBackend } from './httpRealtimeQueueBackend';
import {
  applyQueueItemStatusChange,
  invalidateQueueQueries,
  queueItemDetailOptions,
  QUEUE_RECENT_WINDOW,
  queueKeys,
  queueReadModelOptions,
  queueStatusOptions,
} from './queries';
import { createQueueItemDTO, createQueueServer } from './queueServer.testing';

const createBackend = (): QueueBackendPort => ({
  cancelQueueItems: vi.fn(),
  cancelQueueItemsByBatchIds: vi.fn(),
  cancelBatch: vi.fn(),
  clearFinishedItems: vi.fn(),
  readItemDetail: vi.fn(),
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
  claimItemDelivery: vi.fn(),
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
    await invalidateQueueQueries(client);

    expect(client.getQueryState(queueKeys.readModel(scope))?.isInvalidated).toBe(true);
    expect(client.getQueryState(queueKeys.status({}))?.isInvalidated).toBe(true);
  });

  it("re-reads an open item's details only while the item can still change", async () => {
    const backend = createBackend();
    const client = new QueryClient();
    vi.mocked(backend.readItemDetail).mockImplementation((itemId) =>
      Promise.resolve({ status: itemId === 1 ? 'completed' : 'in_progress' } as QueueItemDetailReadModel)
    );

    await client.fetchQuery(queueItemDetailOptions(backend, 1));
    await client.fetchQuery(queueItemDetailOptions(backend, 2));
    await invalidateQueueQueries(client);

    expect(client.getQueryState(queueKeys.itemDetail(1))?.isInvalidated).toBe(false);
    expect(client.getQueryState(queueKeys.itemDetail(2))?.isInvalidated).toBe(true);
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

describe('applyQueueItemStatusChange', () => {
  const PROJECT = 'webv2:p:project-1:q:';
  const item = (
    id: number,
    status: QueueItemReadModel['status'],
    overrides: Partial<QueueItemReadModel> = {}
  ): QueueItemReadModel => ({
    batchId: 'batch-1',
    createdAt: '2026-11-14T00:00:00.000Z',
    id,
    origin: `${PROJECT}run-1`,
    resultImageNames: [],
    sessionId: `session-${id}`,
    status,
    statusSequence: 1,
    updatedAt: '2026-11-14T00:00:00.000Z',
    ...overrides,
  });
  const counts = (overrides: Partial<QueueCounts>): QueueCounts => ({
    canceled: 0,
    completed: 0,
    failed: 0,
    inProgress: 0,
    pending: 0,
    queueId: 'default',
    total: 0,
    waiting: 0,
    ...overrides,
  });
  /** A reading whose counts are its window's unless `queue` says otherwise (more waiting than it lists). */
  const model = (
    overrides: Partial<Omit<QueueReadModel, 'status'>>,
    queue: Partial<QueueCounts> = {}
  ): QueueReadModel => {
    const items = overrides.items ?? [];
    return {
      current: null,
      items,
      next: null,
      scope: { originPrefix: PROJECT },
      ...overrides,
      status: {
        processor: { isProcessing: true, isStarted: true },
        queue: counts({
          inProgress: items.filter((candidate) => candidate.status === 'in_progress').length,
          pending: items.filter((candidate) => candidate.status === 'pending').length,
          ...queue,
        }),
      },
    };
  };

  it("takes only the whole queue's counts, never the owner's own counts an administrator also receives", () => {
    const before = model({ scope: {} }, { userInProgress: 0, userPending: 4 });

    // Another user's item started; the event carries that user's own counts.
    const after = applyQueueItemStatusChange(before, {
      item: item(7, 'in_progress', { origin: 'webv2:p:other:q:run-9', statusSequence: 2, userId: 'someone' }),
      queueCounts: counts({ inProgress: 1, pending: 5, total: 6, userInProgress: 1, userPending: 2 }),
    });

    // An item the model did not list still becomes the current one.
    expect(after.current?.id).toBe(7);
    expect(after.status.queue).toMatchObject({ inProgress: 1, pending: 5, userInProgress: 0, userPending: 4 });
  });

  it("moves a scoped model's known item between its buckets, never taking the whole queue's counts", () => {
    const running = item(1, 'in_progress');
    const before = model({ current: running, items: [running] });
    const wholeQueue = counts({ canceled: 1, inProgress: 3, pending: 9, total: 13 });

    const after = applyQueueItemStatusChange(before, {
      item: { ...running, status: 'canceled', statusSequence: 2 },
      queueCounts: wholeQueue,
    });

    expect(after.current).toBeNull();
    expect(after.status.queue).toMatchObject({ canceled: 1, inProgress: 0, pending: 0 });
    expect(
      applyQueueItemStatusChange(before, {
        item: item(9, 'in_progress', { origin: 'webv2:p:other:q:run-9' }),
        queueCounts: null,
      })
    ).toBe(before);
  });

  it("takes an event's whole-queue counts even when it already shows the item so, as after the viewer's own cancel", () => {
    const running = item(1, 'in_progress', { origin: 'webv2:p:any:q:run' });
    // This tab assumed the cancel; then the server's event says the same.
    const assumed = applyQueueItemStatusChange(model({ current: running, items: [running], scope: {} }), {
      item: { ...running, status: 'canceled', statusSequence: null },
      queueCounts: null,
    });
    expect(assumed.status.queue).toMatchObject({ canceled: 1, inProgress: 0 });

    const confirmed = applyQueueItemStatusChange(assumed, {
      item: { ...running, status: 'canceled', statusSequence: 2 },
      queueCounts: counts({ canceled: 4, inProgress: 2, pending: 6, total: 12 }),
    });

    expect(confirmed.status.queue).toMatchObject({ canceled: 4, inProgress: 2, pending: 6 });
  });

  describe('the window standing in for the next item', () => {
    // Item 2 is next; it starts, and the window's next pending item in run order would stand in.
    const window = () => [item(4, 'pending'), item(3, 'pending'), item(2, 'pending')];
    const start2 = (before: QueueReadModel) =>
      applyQueueItemStatusChange(before, { item: item(2, 'in_progress', { statusSequence: 2 }), queueCounts: null });

    it('stands in when the window holds every pending item the counts report', () => {
      expect(start2(model({ items: window(), next: item(2, 'pending') })).next?.id).toBe(3);
    });

    it('leaves the next item unknown when more wait than the window lists', () => {
      // A long batch: the true next is older than every item the window shows.
      expect(start2(model({ items: window(), next: item(2, 'pending') }, { pending: 40 })).next).toBeNull();
    });

    it('leaves the next item unknown when the counts disagree with the window', () => {
      expect(start2(model({ items: window(), next: item(2, 'pending') }, { pending: 0 })).next).toBeNull();
      expect(start2(model({ items: window(), next: item(2, 'pending') }, { pending: -3 })).next).toBeNull();
    });

    it('leaves it unknown when an item the window does not list starts waiting', () => {
      // The window holds every pending item; then an older item (a retry) waits again, which it does not list.
      const before = model({ items: [item(9, 'pending'), item(8, 'pending')], next: null });

      expect(
        applyQueueItemStatusChange(before, { item: item(5, 'pending', { statusSequence: 3 }), queueCounts: null }).next
      ).toBeNull();
    });
  });

  it('names the next item in the order the server runs them: highest priority first, then oldest', () => {
    // Item 4 was queued at the front of the queue.
    const before = model({
      items: [item(4, 'pending', { priority: 10 }), item(3, 'pending'), item(2, 'pending')],
      next: item(2, 'pending'),
    });

    expect(
      applyQueueItemStatusChange(before, { item: item(2, 'in_progress', { statusSequence: 2 }), queueCounts: null })
        .next?.id
    ).toBe(4);
  });

  it('keeps a read newer than the event, and follows sequences again once a read brings in restarted ones', () => {
    const fetched = model({ current: item(5, 'in_progress', { statusSequence: 3 }), items: [] });

    // A late event from before the read.
    expect(
      applyQueueItemStatusChange(fetched, { item: item(5, 'pending', { statusSequence: 2 }), queueCounts: null })
    ).toBe(fetched);
    // A change this tab assumes, with no sequence, never revives a finished item.
    const finished = model({ items: [item(6, 'completed', { statusSequence: 4 })] });
    expect(
      applyQueueItemStatusChange(finished, { item: item(6, 'canceled', { statusSequence: null }), queueCounts: null })
    ).toBe(finished);
    // An event with a higher sequence does correct it: the item had finished, not been canceled.
    const assumedCanceled = model({ items: [item(6, 'canceled', { statusSequence: 4 })] });
    expect(
      applyQueueItemStatusChange(assumedCanceled, {
        item: item(6, 'completed', { statusSequence: 5 }),
        queueCounts: null,
      }).items[0]?.status
    ).toBe('completed');
  });

  it('keeps the current item while another starts beside it, and hands over to it when the first leaves', () => {
    const before = model({
      current: item(1, 'in_progress'),
      items: [item(3, 'pending'), item(2, 'pending'), item(1, 'in_progress')],
      next: item(2, 'pending'),
    });

    const after = applyQueueItemStatusChange(before, {
      item: item(2, 'in_progress', { statusSequence: 2 }),
      queueCounts: null,
    });

    expect(after.current?.id).toBe(1);
    expect(after.next?.id).toBe(3);
    expect(
      applyQueueItemStatusChange(after, { item: item(1, 'completed', { statusSequence: 2 }), queueCounts: null })
        .current?.id
    ).toBe(2);
  });
});
