import type { QueueBackendPort, QueueCounts, QueueItemReadModel, QueueReadModel } from '@features/queue/core/types';

import { getCurrentBatchItems } from '@features/queue/ui/currentBatchItems';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { queueBackend } from './httpRealtimeQueueBackend';
import {
  applyQueueItemStatusChange,
  invalidateQueueQueries,
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
  readActiveItemIds: vi.fn().mockResolvedValue({ itemIds: [] }),
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
        'GET active_ids',
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

  it('preserves enqueue order within recent batches and appends older executing items', async () => {
    accountLifecycle.activate('queue-batch-order');
    const items = [
      createQueueItemDTO(1, { status: 'in_progress' }),
      ...[2, 3, 4].map((id) => createQueueItemDTO(id, { created_at: '2026-01-02T00:00:00Z' })),
      ...[5, 6, 7].map((id) => createQueueItemDTO(id, { created_at: '2026-01-03T00:00:00Z' })),
      ...Array.from({ length: 44 }, (_, index) =>
        createQueueItemDTO(index + 8, { created_at: '2026-01-04T00:00:00Z' })
      ),
    ];
    const server = createQueueServer(items);
    vi.stubGlobal('fetch', server.fetch);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const model = await client.fetchQuery(queueReadModelOptions(queueBackend, {}));
    const expected = [...Array.from({ length: 44 }, (_, index) => index + 8), 5, 6, 7, 2, 3, 4, 1];
    expect(server.hydratedIds).toEqual([expected]);
    expect(model.items.map((item) => item.id)).toEqual(expected);
  });

  it('bounds waiting-parent hydration without hiding older executing items', async () => {
    accountLifecycle.activate('queue-many-waiting');
    const items = [
      createQueueItemDTO(1, { status: 'in_progress' }),
      createQueueItemDTO(2, { status: 'in_progress' }),
      ...Array.from({ length: 2_500 }, (_, index) => createQueueItemDTO(index + 3, { status: 'waiting' })),
      ...Array.from({ length: 50 }, (_, index) => createQueueItemDTO(index + 2_503, { status: 'pending' })),
    ];
    const server = createQueueServer(items);
    vi.stubGlobal('fetch', server.fetch);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const model = await client.fetchQuery(queueReadModelOptions(queueBackend, {}));
    expect(server.hydratedIds).toHaveLength(1);
    expect(model.items).toHaveLength(102);
    expect(model.items.filter((item) => item.status === 'in_progress').map((item) => item.id)).toEqual([2, 1]);
    expect(model.items.filter((item) => item.status === 'waiting')).toHaveLength(50);
  });

  it.each([1, 55])('retains %s required waiting roots only in scoped active IDs', async (requiredCount) => {
    accountLifecycle.activate('queue-active-root-retention');
    const originPrefix = 'webv2:p:project-a:q:';
    const parents = Array.from({ length: 60 }, (_, index) => index + 1);
    const roots = parents.slice(0, requiredCount);
    const children = roots.map((id) => id + 60);
    const server = createQueueServer([
      ...parents.map((id) => createQueueItemDTO(id, { origin: `${originPrefix}run`, status: 'waiting' })),
      ...roots.map((id) =>
        createQueueItemDTO(id + 60, {
          origin: `${originPrefix}run`,
          parent_item_id: id,
          root_item_id: id,
          status: 'in_progress',
        })
      ),
    ]);
    vi.stubGlobal('fetch', server.fetch);

    const scoped = await queueBackend.readActiveItemIds({ originPrefix });
    const ordinary = parents
      .slice(requiredCount)
      .reverse()
      .slice(0, Math.max(0, 50 - requiredCount));
    expect(scoped.itemIds).toEqual([
      ...[...children].reverse(),
      ...[...roots, ...ordinary].sort((left, right) => right - left),
    ]);
    expect(scoped.executingRootItemIds).toEqual(roots);

    const unscoped = await queueBackend.readActiveItemIds({});
    expect(unscoped.itemIds).toEqual([...children.reverse(), ...parents.slice(-50).reverse()]);
    expect(unscoped.executingRootItemIds).toBeNull();
  });

  it('recovers older running GPU jobs and waiting parents outside recent history', async () => {
    accountLifecycle.activate('queue-active-ids');
    const projectA = 'webv2:p:project-a:q:';
    const items = [
      createQueueItemDTO(1, { batch_id: 'batch-a', origin: `${projectA}run-a`, status: 'in_progress' }),
      createQueueItemDTO(2, { batch_id: 'batch-b', origin: `${projectA}run-b`, status: 'in_progress' }),
      createQueueItemDTO(3, { batch_id: 'batch-parent', origin: `${projectA}parent`, status: 'waiting' }),
      ...Array.from({ length: QUEUE_RECENT_WINDOW }, (_, index) =>
        createQueueItemDTO(index + 4, { batch_id: 'batch-new', origin: `${projectA}queued`, status: 'pending' })
      ),
      createQueueItemDTO(54, { batch_id: 'batch-foreign', origin: 'webv2:p:project-b:q:run', status: 'in_progress' }),
    ];
    const server = createQueueServer(items);
    vi.stubGlobal('fetch', server.fetch);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

    const scoped = await client.fetchQuery(queueReadModelOptions(queueBackend, { originPrefix: projectA }));
    const running = getCurrentBatchItems(scoped).filter((item) => item.status === 'in_progress');

    expect(running.map((item) => item.id)).toEqual([1, 2]);
    expect(scoped.items.find((item) => item.id === 3)?.status).toBe('waiting');
    expect(scoped.items.some((item) => item.id === 54)).toBe(false);
    expect(server.requests).toContain(`GET active_ids?origin_prefix=${encodeURIComponent(projectA)}`);
    const expectedIds = [...Array.from({ length: 50 }, (_, index) => 53 - index), 2, 1, 3];
    expect(server.hydratedIds).toEqual([expectedIds]);
    expect(scoped.items.map((item) => item.id)).toEqual(expectedIds);
  });

  it.each([true, 'html'] as const)(
    'keeps the 50-item hydration bound when an older backend has no active_ids endpoint (%s)',
    async (legacyResponse) => {
      accountLifecycle.activate('queue-legacy-active-ids');
      const server = createQueueServer(queueOf(1_000), {
        predatesItemIdsLimit: true,
        predatesActiveIdsEndpoint: legacyResponse,
      });
      vi.stubGlobal('fetch', server.fetch);
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

      const model = await client.fetchQuery(queueReadModelOptions(queueBackend, {}));

      expect(server.requests).toContain('GET active_ids');
      expect(server.hydratedIds).toHaveLength(1);
      expect(server.hydratedIds[0]).toHaveLength(QUEUE_RECENT_WINDOW);
      expect(model.items).toHaveLength(QUEUE_RECENT_WINDOW);
      expect(model.current?.id).toBe(1_000);

      // Every read may retry the optional endpoint; a transient failure must not disable it.
      await client.invalidateQueries({ queryKey: queueKeys.all });
      await client.fetchQuery(queueReadModelOptions(queueBackend, {}));

      const scoped = await queueBackend.readActiveItemIds({ originPrefix: 'webv2:p:another-project:q:' }, undefined);
      expect(scoped.itemIds).toEqual([]);
      expect(server.requests.filter((request) => request.startsWith('GET active_ids'))).toHaveLength(3);
      expect(server.hydratedIds).toHaveLength(2);
      expect(server.hydratedIds[1]).toHaveLength(QUEUE_RECENT_WINDOW);

      // Fallback must still respect cancellation.
      const controller = new AbortController();
      controller.abort();
      await expect(queueBackend.readActiveItemIds({}, controller.signal)).rejects.toMatchObject({
        name: 'AbortError',
      });

      // Recover without requiring a logout or account change.
      const upgradedServer = createQueueServer(queueOf(1_000));
      vi.stubGlobal('fetch', upgradedServer.fetch);

      const upgraded = await queueBackend.readActiveItemIds({}, undefined);
      expect(upgraded.itemIds).toEqual([1_000]);
      expect(upgradedServer.requests).toContain('GET active_ids');
    }
  );

  it.each([403, 500])('does not hide HTTP %s errors from the active IDs endpoint', async (status) => {
    accountLifecycle.activate('queue-active-ids-error');
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          new Response('{"detail":"Request failed"}', {
            headers: { 'content-type': 'application/json' },
            status,
          })
        )
      )
    );
    await expect(queueBackend.readActiveItemIds({}, undefined)).rejects.toMatchObject({ status });
    await expect(queueBackend.readActiveItemIds({}, undefined)).rejects.toMatchObject({ status });
  });

  it('does not treat malformed JSON from the active IDs endpoint as an unsupported route', async () => {
    accountLifecycle.activate('queue-active-ids-malformed');
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          new Response('{', {
            headers: { 'content-type': 'application/json' },
            status: 200,
          })
        )
      )
    );
    await expect(queueBackend.readActiveItemIds({}, undefined)).rejects.toBeInstanceOf(SyntaxError);
    await expect(queueBackend.readActiveItemIds({}, undefined)).rejects.toBeInstanceOf(SyntaxError);
  });

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

  describe('the window standing in for the next item', () => {
    // Item 2 is next; it starts, and the window's next pending item in run order would stand in.
    const window = () => [item(4, 'pending'), item(3, 'pending'), item(2, 'pending')];
    const start2 = (before: QueueReadModel) =>
      applyQueueItemStatusChange(before, { item: item(2, 'in_progress', { statusSequence: 2 }), queueCounts: null });

    it('stands in when the window holds every pending item the counts report', () => {
      expect(start2(model({ items: window(), next: item(2, 'pending') })).next?.id).toBe(3);
    });
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
});
