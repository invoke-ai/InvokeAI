import type { QueueItem } from '@features/queue/core/historyTypes';
import type {
  QueueBackendInvocation,
  QueueBackendPort,
  QueueResultImage,
  QueueWorkflowRunSink,
} from '@features/queue/core/types';
import type { BackendConnectionStatus } from '@platform/transport/types';

import { buildQueueItemOrigin } from '@features/queue/data/events';
import { ApiError } from '@platform/transport/http';
import { describe, expect, it, vi } from 'vitest';

import {
  createQueueItemBackendSubmission,
  createQueueRuntime,
  type QueueHistoryCommands,
  type QueueRunJournalPort,
  type QueueRunLock,
  type QueueRunLockPort,
} from './runtime';

const createPendingQueueItem = (): QueueItem => ({
  cancellable: true,
  id: 'local-queue-item',
  snapshot: {
    backendSubmission: {
      batchCount: 1,
      graph: { edges: [], id: 'backend-graph', nodes: {} },
      kind: 'generate',
      negativePrompt: '',
      negativePromptNodeId: 'negative_prompt',
      positivePrompt: '',
      positivePromptNodeId: 'positive_prompt',
      seed: 0,
      seedNodeId: 'seed',
      seedStep: 0,
    },
    destination: 'canvas',
    filterIntermediateResults: false,
    galleryBoardId: null,
    graph: { id: 'graph', label: 'Generate' },
    presentation: { batchCount: 1, height: 1024, width: 1024 },
    resultNodeIds: ['canvas_output'],
    sourceId: 'generate',
    submittedAt: '2026-07-17T00:00:00.000Z',
  },
  status: 'pending',
});

const createTestBackend = (overrides: Partial<QueueBackendPort> = {}): QueueBackendPort => ({
  cancelCurrentItem: vi.fn(),
  cancelItem: vi.fn(),
  cancelQueueItems: vi.fn(),
  cancelQueueItemsByBatchIds: vi.fn(),
  cancelScopedItems: vi.fn(),
  clearFailedItems: vi.fn(),
  clearItems: vi.fn(),
  emit: vi.fn(),
  enqueueGenerate: vi.fn(),
  enqueueWorkflow: vi.fn(),
  getItem: vi.fn(),
  getResultImages: vi.fn().mockResolvedValue([]),
  getResultVideoNames: vi.fn().mockResolvedValue([]),
  listItems: vi.fn().mockResolvedValue([]),
  on: vi.fn(() => vi.fn()),
  onConnectionChange: vi.fn((listener) => {
    listener('connected');
    return vi.fn();
  }),
  pauseProcessor: vi.fn(),
  readCurrent: vi.fn(),
  readItemIds: vi.fn(),
  readItemsById: vi.fn(),
  readNext: vi.fn(),
  readStatus: vi.fn(),
  resumeProcessor: vi.fn(),
  retryItems: vi.fn(),
  ...overrides,
});

const createTestCommands = (overrides: Partial<QueueHistoryCommands> = {}): QueueHistoryCommands => ({
  markBackendCancelled: vi.fn(),
  markBackendSubmitted: vi.fn(),
  setCancellationPending: vi.fn(),
  setLocalRecoveryState: vi.fn(),
  recordError: vi.fn(),
  refreshBackendData: vi.fn(),
  restoreFromJournal: vi.fn(),
  routePartialResults: vi.fn(),
  routeResults: vi.fn(),
  setConnectionStatus: vi.fn(),
  setStatus: vi.fn(),
  ...overrides,
});

const createTestJournal = (overrides: Partial<QueueRunJournalPort> = {}): QueueRunJournalPort => ({
  listForProject: vi.fn().mockResolvedValue({ entries: [], kind: 'available' }),
  record: vi.fn().mockResolvedValue({ kind: 'stored' }),
  settle: vi.fn().mockResolvedValue({ kind: 'removed' }),
  ...overrides,
});

const runtimeServices = {
  destinations: { addImagesToGalleryBoard: vi.fn(), addVideosToGalleryBoard: vi.fn() },
  ensureTemplatesLoaded: vi.fn(),
  modelLoads: { completed: vi.fn(), reset: vi.fn(), started: vi.fn() },
  nodeExecution: {
    clearAll: vi.fn(),
    completed: vi.fn(),
    failed: vi.fn(),
    progress: vi.fn(),
    settleRunning: vi.fn(),
    started: vi.fn(),
  },
};

describe('queue runtime', () => {
  it('turns a malformed legacy submission into a failed request instead of throwing', () => {
    const queueItem = createPendingQueueItem();
    delete (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, queueItem)).toEqual({
      error: 'Queue item is missing a compiled backend submission.',
      kind: 'invalid',
    });
  });

  it('replays items queued before seed modes with the step their random toggle implied', () => {
    const asLegacy = (shouldRandomizeSeed: boolean) => {
      const queueItem = createPendingQueueItem();
      const submission = queueItem.snapshot.backendSubmission as Record<string, unknown>;
      delete submission.seedStep;
      submission.shouldRandomizeSeed = shouldRandomizeSeed;
      return queueItem;
    };

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asLegacy(true))).toMatchObject({
      kind: 'generate',
      request: { legacySeedPlan: true, seedStep: 1 },
    });
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asLegacy(false))).toMatchObject({
      kind: 'generate',
      request: { legacySeedPlan: true, seedStep: 0 },
    });
    // An item that already carries a step never takes the legacy expansion, even if a
    // stale toggle rides along beside it.
    const withBoth = createPendingQueueItem();
    (withBoth.snapshot.backendSubmission as Record<string, unknown>).shouldRandomizeSeed = true;

    for (const queueItem of [createPendingQueueItem(), withBoth]) {
      const { request } = createQueueItemBackendSubmission({ id: 'project-1' }, queueItem) as { request: object };

      expect(request).not.toHaveProperty('legacySeedPlan');
      expect(request).toMatchObject({ seedStep: 0 });
    }
    expect(
      (createQueueItemBackendSubmission({ id: 'project-1' }, asLegacy(false)) as { request: object }).request
    ).not.toHaveProperty('shouldRandomizeSeed');
  });

  it('replays workflow seeds as recorded and rejects records the graph cannot honour', () => {
    const asWorkflow = (seeds: unknown) => {
      const queueItem = createPendingQueueItem();
      queueItem.snapshot.backendSubmission = {
        batchCount: 3,
        graph: { edges: [], id: 'backend-graph', nodes: { noise: { id: 'noise', seed: 7, type: 'noise' } } },
        kind: 'workflow',
        workflow: { edges: [], name: 'Call-only workflow', nodes: [], version: '1.0.0' },
        ...(seeds === undefined ? {} : { seeds: seeds as never }),
      };
      queueItem.snapshot.sourceId = 'workflow';
      return queueItem;
    };
    const seed = { fieldName: 'seed', nodeId: 'noise', seed: 7, seedStep: 1 };
    const invalid = { error: 'Queue item has malformed workflow seed metadata.', kind: 'invalid' };

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([seed]))).toMatchObject({
      kind: 'workflow',
      request: {
        batchCount: 3,
        seeds: [seed],
        workflow: { name: 'Call-only workflow' },
      },
    });
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow(undefined))).toMatchObject({
      kind: 'workflow',
    });
    // Out of range, a step the plan never emits, a node the graph lost, and the same field twice.
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([{ ...seed, seed: -1 }]))).toEqual(invalid);
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([{ ...seed, seedStep: 0 }]))).toEqual(
      invalid
    );
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([{ ...seed, nodeId: 'gone' }]))).toEqual(
      invalid
    );
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([seed, seed]))).toEqual(invalid);
  });

  it('replays persisted batch groups and rejects ones the graph or the backend would refuse', () => {
    const asWorkflow = (batchData: unknown, seeds?: unknown) => {
      const queueItem = createPendingQueueItem();
      queueItem.snapshot.backendSubmission = {
        batchCount: 2,
        graph: {
          edges: [],
          id: 'backend-graph',
          nodes: { denoise: { id: 'denoise', type: 'denoise' }, noise: { id: 'noise', seed: 7, type: 'noise' } },
        },
        kind: 'workflow',
        ...(seeds === undefined ? {} : { seeds: seeds as never }),
        batchData: batchData as never,
      };
      queueItem.snapshot.sourceId = 'workflow';
      return queueItem;
    };
    const cfg = { fieldName: 'cfg', items: [1, 2], nodeId: 'denoise' };
    const invalid = { error: 'Queue item has malformed workflow batch metadata.', kind: 'invalid' };

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([[cfg]]))).toMatchObject({
      kind: 'workflow',
      request: { batchData: [[cfg]] },
    });
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([]))).toEqual(invalid);
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([[]]))).toEqual(invalid);
    expect(
      createQueueItemBackendSubmission(
        { id: 'project-1' },
        asWorkflow([[cfg, { ...cfg, fieldName: 'steps', items: [1] }]])
      )
    ).toEqual(invalid);
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([[{ ...cfg, nodeId: 'gone' }]]))).toEqual(
      invalid
    );
    expect(createQueueItemBackendSubmission({ id: 'project-1' }, asWorkflow([[cfg], [cfg]]))).toEqual(invalid);
    // The product must fit the queue together with the runs, as the planner enforced when it was queued.
    expect(
      createQueueItemBackendSubmission(
        { id: 'project-1' },
        asWorkflow([[{ ...cfg, items: Array.from({ length: 5_001 }, (_, i) => i) }]])
      )
    ).toEqual(invalid);
    // A field a seed already walks cannot also be batched.
    expect(
      createQueueItemBackendSubmission(
        { id: 'project-1' },
        asWorkflow(
          [[{ fieldName: 'seed', items: [1, 2], nodeId: 'noise' }]],
          [{ fieldName: 'seed', nodeId: 'noise', seed: 7, seedStep: 1 }]
        )
      )
    ).toEqual(invalid);
  });

  it('rejects a generate item that records neither a seed step nor the legacy toggle', () => {
    const queueItem = createPendingQueueItem();
    delete (queueItem.snapshot.backendSubmission as Record<string, unknown>).seedStep;

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, queueItem)).toEqual({
      error: 'Queue item has malformed generate submission metadata.',
      kind: 'invalid',
    });
  });

  it('adopts a persisted backend run before submission so reload cannot duplicate it', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const enqueueGenerate = vi.fn();
    const listItems = vi.fn().mockResolvedValue([
      {
        batchId: 'backend-batch',
        id: 77,
        origin: buildQueueItemOrigin(queueItem.id, project.id),
        status: 'in_progress',
      },
    ]);
    const backend: QueueBackendPort = {
      cancelCurrentItem: vi.fn(),
      cancelItem: vi.fn(),
      cancelQueueItems: vi.fn(),
      cancelQueueItemsByBatchIds: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate,
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: vi.fn(),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
      listItems,
      on: vi.fn(() => vi.fn()),
      onConnectionChange: vi.fn((listener) => {
        listener('connected');
        return vi.fn();
      }),
      pauseProcessor: vi.fn(),
      readCurrent: vi.fn(),
      readItemIds: vi.fn(),
      readItemsById: vi.fn(),
      readNext: vi.fn(),
      readStatus: vi.fn(),
      resumeProcessor: vi.fn(),
      retryItems: vi.fn(),
    };
    const commands: QueueHistoryCommands = {
      markBackendCancelled: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
      },
      setLocalRecoveryState: vi.fn(),
      recordError: vi.fn(),
      refreshBackendData: vi.fn(),
      restoreFromJournal: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
    };
    const runtime = createQueueRuntime({
      backend,
      destinations: { addImagesToGalleryBoard: vi.fn(), addVideosToGalleryBoard: vi.fn() },
      ensureTemplatesLoaded: vi.fn(),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      modelLoads: {
        completed: vi.fn(),
        reset: vi.fn(),
        started: vi.fn(),
      },
      nodeExecution: {
        clearAll: vi.fn(),
        completed: vi.fn(),
        failed: vi.fn(),
        progress: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(queueItem).toMatchObject({
        backendBatchId: 'backend-batch',
        backendItemIds: [77],
        status: 'running',
      });
    });
    expect(listItems).toHaveBeenCalledTimes(1);
    expect(enqueueGenerate).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('durably records a new run before enqueueing it', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    let finishRecord: (result: { kind: 'stored' }) => void = () => undefined;
    const record = vi.fn(
      () =>
        new Promise<{ kind: 'stored' }>((resolve) => {
          finishRecord = resolve;
        })
    );
    const journal = createTestJournal({ record });
    const enqueueGenerate = vi.fn().mockResolvedValue({ enqueued: 1, itemIds: [77], requested: 1 });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate }),
      history: {
        commands: createTestCommands(),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal,
    });

    runtime.start();

    await vi.waitFor(() => expect(record).toHaveBeenCalledWith(project.id, queueItem));
    expect(enqueueGenerate).not.toHaveBeenCalled();

    finishRecord({ kind: 'stored' });
    await vi.waitFor(() => expect(enqueueGenerate).toHaveBeenCalledTimes(1));

    runtime.dispose();
  });

  it('checks that a project is durable after journaling and before enqueueing', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const order: string[] = [];
    const journal = createTestJournal({
      record: vi.fn(() => {
        order.push('journal');
        return Promise.resolve({ kind: 'stored' as const });
      }),
    });
    const ensureProjectPersisted = vi.fn(() => {
      order.push('project');
      return Promise.resolve('ready' as const);
    });
    const enqueueGenerate = vi.fn(() => {
      order.push('enqueue');
      return Promise.resolve({ enqueued: 1, itemIds: [77], requested: 1 });
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate }),
      ensureProjectPersisted,
      history: {
        commands: createTestCommands(),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal,
    });

    runtime.start();

    await vi.waitFor(() => expect(enqueueGenerate).toHaveBeenCalledTimes(1));
    expect(ensureProjectPersisted).toHaveBeenCalledWith(project.id);
    expect(order.slice(0, 3)).toEqual(['journal', 'project', 'enqueue']);
    await runtime.dispose();
  });

  it.each([
    ['retry', false],
    ['refused', true],
  ] as const)('does not enqueue when project persistence returns %s', async (outcome, shouldFail) => {
    vi.useFakeTimers();
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const enqueueGenerate = vi.fn();
    const commands = createTestCommands();
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate }),
      ensureProjectPersisted: vi.fn().mockResolvedValue(outcome),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal(),
    });

    runtime.start();
    await vi.advanceTimersByTimeAsync(1);

    expect(enqueueGenerate).not.toHaveBeenCalled();
    if (shouldFail) {
      expect(commands.setStatus).toHaveBeenCalledWith(
        expect.objectContaining({ projectId: project.id, queueItemId: queueItem.id, status: 'failed' })
      );
    } else {
      expect(commands.setStatus).not.toHaveBeenCalled();
    }
    await runtime.dispose();
    vi.useRealTimers();
  });

  it('leaves durable enqueue receipt acknowledgement to the durable acknowledgement worker', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    let finishAcceptanceRecord: (result: { kind: 'stored' }) => void = () => undefined;
    const acceptanceRecord = new Promise<{ kind: 'stored' }>((resolve) => {
      finishAcceptanceRecord = resolve;
    });
    const record = vi
      .fn<QueueRunJournalPort['record']>()
      .mockResolvedValueOnce({ kind: 'stored' })
      .mockReturnValueOnce(acceptanceRecord);
    const acknowledgeEnqueue = vi.fn().mockResolvedValue(undefined);
    const commands = createTestCommands({
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        acknowledgeEnqueue,
        enqueueGenerate: vi.fn().mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [77], requested: 1 }),
      }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal({ record }),
    });

    runtime.start();

    await vi.waitFor(() => expect(record).toHaveBeenCalledTimes(2));
    expect(acknowledgeEnqueue).not.toHaveBeenCalled();
    finishAcceptanceRecord({ kind: 'stored' });
    await Promise.resolve();
    expect(acknowledgeEnqueue).not.toHaveBeenCalled();

    await runtime.dispose();
  });

  it('acknowledges a terminal receipt in memory when the run was proven never durable', async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), localRecoveryState: 'local-only' };
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    const acknowledgeEnqueue = vi.fn().mockResolvedValue(undefined);
    const commands = createTestCommands({
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      routeResults: () => {
        queueItem.status = 'completed';
        listeners.forEach((listener) => listener());
      },
      setLocalRecoveryState: ({ state }) => {
        queueItem.localRecoveryState = state;
        listeners.forEach((listener) => listener());
      },
      setStatus: ({ status }) => {
        queueItem.status = status;
        listeners.forEach((listener) => listener());
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        acknowledgeEnqueue,
        getEnqueueReceipt: vi.fn().mockResolvedValue({
          batchId: 'batch-1',
          enqueued: 1,
          itemIds: [77],
          requested: 1,
        }),
        getItem: vi.fn().mockRejectedValue(new ApiError('not found', 404)),
      }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({ record: vi.fn().mockResolvedValue({ kind: 'quota' }) }),
    });

    runtime.start();

    await vi.waitFor(() => expect(acknowledgeEnqueue).toHaveBeenCalledWith(project.id, queueItem.id));
    await runtime.dispose();
  });

  it.each(['durable', 'uncertain'] as const)(
    'never fallback-acknowledges a receipt when recovery state is %s',
    async (localRecoveryState) => {
      const queueItem: QueueItem = {
        ...createPendingQueueItem(),
        backendItemIds: [77],
        localRecoveryState,
        status: 'running',
      };
      const project = { id: 'project-1', queue: { items: [queueItem] } };
      const listeners = new Set<() => void>();
      const acknowledgeEnqueue = vi.fn().mockResolvedValue(undefined);
      const runtime = createQueueRuntime({
        ...runtimeServices,
        backend: createTestBackend({
          acknowledgeEnqueue,
          getItem: vi.fn().mockResolvedValue({
            id: 77,
            origin: buildQueueItemOrigin(queueItem.id, project.id),
            status: 'completed',
          }),
        }),
        history: {
          commands: createTestCommands({
            routeResults: () => {
              queueItem.status = 'completed';
              listeners.forEach((listener) => listener());
            },
            setLocalRecoveryState: ({ state }) => {
              queueItem.localRecoveryState = state;
            },
          }),
          getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
          subscribe: (listener) => {
            listeners.add(listener);
            return () => listeners.delete(listener);
          },
        },
        journal: createTestJournal({ record: vi.fn().mockResolvedValue({ kind: 'unavailable' }) }),
      });

      runtime.start();

      await vi.waitFor(() => expect(queueItem.status).toBe('completed'));
      expect(acknowledgeEnqueue).not.toHaveBeenCalled();
      await runtime.dispose();
    }
  );

  it('discovers a committed row after a runtime replacement before allowing fallback acknowledgement', async () => {
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [77],
      localRecoveryState: 'local-only',
      status: 'running',
    };
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    let finishRecord: (result: { kind: 'stored' }) => void = () => undefined;
    const record = vi.fn(
      () =>
        new Promise<{ kind: 'stored' }>((resolve) => {
          finishRecord = resolve;
        })
    );
    const firstRuntime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend(),
      history: {
        commands: createTestCommands({
          setLocalRecoveryState: ({ state }) => {
            queueItem.localRecoveryState = state;
          },
        }),
        getSnapshot: () => ({ connectionStatus: 'disconnected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal({ record }),
    });

    firstRuntime.start();
    await vi.waitFor(() => expect(queueItem.localRecoveryState).toBe('uncertain'));
    const disposeFirst = firstRuntime.dispose();
    finishRecord({ kind: 'stored' });
    await disposeFirst;
    expect(queueItem.localRecoveryState).toBe('uncertain');

    const listeners = new Set<() => void>();
    const acknowledgeEnqueue = vi.fn().mockResolvedValue(undefined);
    const secondRuntime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        acknowledgeEnqueue,
        getItem: vi.fn().mockResolvedValue({
          id: 77,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'completed',
        }),
      }),
      history: {
        commands: createTestCommands({
          restoreFromJournal: () => {
            queueItem.localRecoveryState = 'durable';
          },
          routeResults: () => {
            queueItem.status = 'completed';
            listeners.forEach((listener) => listener());
          },
          setLocalRecoveryState: ({ state }) => {
            queueItem.localRecoveryState = state;
          },
        }),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({
        listForProject: vi.fn().mockResolvedValue({
          entries: [{ item: structuredClone(queueItem), projectId: project.id, queueItemId: queueItem.id }],
          kind: 'available',
        }),
        record: vi.fn().mockResolvedValue({ kind: 'unavailable' }),
      }),
    });

    secondRuntime.start();
    await vi.waitFor(() => expect(queueItem.status).toBe('completed'));
    expect(queueItem.localRecoveryState).toBe('durable');
    expect(acknowledgeEnqueue).not.toHaveBeenCalled();
    await secondRuntime.dispose();
  });

  it('does not retry a fallback acknowledgement after account disposal', async () => {
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [77],
      localRecoveryState: 'local-only',
      status: 'running',
    };
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    let rejectAcknowledgement: (error: Error) => void = () => undefined;
    const acknowledgeEnqueue = vi.fn(
      () =>
        new Promise<void>((_resolve, reject) => {
          rejectAcknowledgement = reject;
        })
    );
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        acknowledgeEnqueue,
        getItem: vi.fn().mockResolvedValue({
          id: 77,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'completed',
        }),
      }),
      history: {
        commands: createTestCommands({
          routeResults: () => {
            queueItem.status = 'completed';
            listeners.forEach((listener) => listener());
          },
          setLocalRecoveryState: ({ state }) => {
            queueItem.localRecoveryState = state;
          },
        }),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({ record: vi.fn().mockResolvedValue({ kind: 'quota' }) }),
    });

    runtime.start();
    await vi.waitFor(() => expect(acknowledgeEnqueue).toHaveBeenCalledOnce());
    await runtime.dispose();
    rejectAcknowledgement(new Error('offline'));
    await Promise.resolve();
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 150);
    });

    expect(acknowledgeEnqueue).toHaveBeenCalledOnce();
  });

  it('continues the backend run and reports when local recovery cannot be written', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const journal = createTestJournal({ record: vi.fn().mockResolvedValue({ kind: 'quota' }) });
    const enqueueGenerate = vi.fn().mockResolvedValue({ enqueued: 1, itemIds: [77], requested: 1 });
    const commands = createTestCommands();
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal,
    });

    runtime.start();

    await vi.waitFor(() => expect(enqueueGenerate).toHaveBeenCalledTimes(1));
    expect(commands.recordError).toHaveBeenCalledWith(
      expect.objectContaining({ area: 'queue-recovery', namespace: 'queue', projectId: project.id })
    );

    runtime.dispose();
  });

  it('records but does not enqueue an item cancelled while run ownership is being acquired', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    let finishAcquire: (result: { kind: 'acquired'; release(): Promise<void> }) => void = () => undefined;
    const release = vi.fn().mockResolvedValue(undefined);
    const locks: QueueRunLockPort = {
      acquire: vi.fn(
        () =>
          new Promise<QueueRunLock>((resolve) => {
            finishAcquire = resolve;
          })
      ),
    };
    const recordedItems: QueueItem[] = [];
    const journal = createTestJournal({
      listForProject: vi.fn().mockResolvedValue({
        entries: [{ item: queueItem, projectId: 'project-1', queueItemId: queueItem.id }],
        kind: 'available',
      }),
      record: vi.fn((_projectId, item) => {
        recordedItems.push(structuredClone(item));
        return Promise.resolve({ kind: 'stored' as const });
      }),
    });
    const backend = createTestBackend();
    const commands = createTestCommands({
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
        listeners.forEach((listener) => listener());
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend,
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
      locks,
    });

    runtime.start();
    await vi.waitFor(() => expect(locks.acquire).toHaveBeenCalled());
    queueItem.status = 'cancelled';
    queueItem.cancellationPending = true;
    listeners.forEach((listener) => listener());
    finishAcquire({ kind: 'acquired', release });

    await vi.waitFor(() => expect(release).toHaveBeenCalledTimes(1));
    expect(recordedItems).toContainEqual(expect.objectContaining({ cancellationPending: true, status: 'cancelled' }));
    expect(journal.settle).toHaveBeenCalledWith(project.id, queueItem.id);
    expect(backend.enqueueGenerate).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('reconciles exactly once after a project closes and reopens while ownership is pending', async () => {
    const queueItem = { ...createPendingQueueItem(), backendItemIds: [88], status: 'running' as const };
    const projects: Array<{ id: string; queue: { items: QueueItem[] } }> = [
      { id: 'project-1', queue: { items: [queueItem] } },
    ];
    const listeners = new Set<() => void>();
    let finishAcquire: (result: QueueRunLock) => void = () => undefined;
    const release = vi.fn().mockResolvedValue(undefined);
    let acquisitionCount = 0;
    const locks: QueueRunLockPort = {
      acquire: vi.fn(() => {
        acquisitionCount += 1;
        if (acquisitionCount > 1) {
          return Promise.resolve<QueueRunLock>({ kind: 'acquired', release });
        }
        return new Promise<QueueRunLock>((resolve) => {
          finishAcquire = resolve;
        });
      }),
    };
    const journal = createTestJournal({
      listForProject: vi.fn().mockResolvedValue({
        entries: [{ item: queueItem, projectId: 'project-1', queueItemId: queueItem.id }],
        kind: 'available',
      }),
    });
    const getItem = vi.fn().mockResolvedValue({
      id: 88,
      origin: buildQueueItemOrigin(queueItem.id, 'project-1'),
      status: 'in_progress',
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ getItem }),
      history: {
        commands: createTestCommands({
          restoreFromJournal: ({ items, projectId }) => {
            const project = projects.find((candidate) => candidate.id === projectId);
            if (project) {
              project.queue.items = items as QueueItem[];
            }
          },
        }),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
      locks,
    });

    runtime.start();
    await vi.waitFor(() => expect(locks.acquire).toHaveBeenCalledTimes(1));
    projects.splice(0, 1);
    listeners.forEach((listener) => listener());
    projects.push({ id: 'project-1', queue: { items: [] } });
    listeners.forEach((listener) => listener());
    finishAcquire({ kind: 'acquired', release });

    await vi.waitFor(() => expect(getItem).toHaveBeenCalledTimes(1));
    expect(locks.acquire).toHaveBeenCalledTimes(2);
    runtime.dispose();
  });

  it('does not let result routing from a closed lifecycle mutate a reopened project', async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [88], status: 'running' };
    const projects: Array<{ id: string; queue: { items: QueueItem[] } }> = [
      { id: 'project-1', queue: { items: [queueItem] } },
    ];
    const listeners = new Set<() => void>();
    let finishResults: (images: QueueResultImage[]) => void = () => undefined;
    const results = new Promise<QueueResultImage[]>((resolve) => {
      finishResults = resolve;
    });
    const getResultImages = vi.fn(() => results);
    const getItem = vi.fn().mockResolvedValue({
      id: 88,
      origin: buildQueueItemOrigin(queueItem.id, 'project-1'),
      status: 'completed',
    });
    const commands = createTestCommands({
      restoreFromJournal: ({ items, projectId }) => {
        const project = projects.find((candidate) => candidate.id === projectId);
        if (project) {
          project.queue.items = items as QueueItem[];
        }
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ getItem, getResultImages }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({
        listForProject: vi.fn().mockResolvedValue({
          entries: [{ item: queueItem, projectId: 'project-1', queueItemId: queueItem.id }],
          kind: 'available',
        }),
      }),
    });

    runtime.start();
    await vi.waitFor(() => expect(getResultImages).toHaveBeenCalled());
    projects.splice(0, 1);
    listeners.forEach((listener) => listener());
    projects.push({ id: 'project-1', queue: { items: [] } });
    listeners.forEach((listener) => listener());
    await vi.waitFor(() => expect(getItem).toHaveBeenCalledTimes(2));

    finishResults([]);
    await vi.waitFor(() => expect(commands.routeResults).toHaveBeenCalledTimes(1));
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'cancelled' }));
    await runtime.dispose();
  });

  it('keeps ownership until an in-flight journal write drains on close', async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [88], status: 'running' };
    const projects: Array<{ id: string; queue: { items: QueueItem[] } }> = [
      { id: 'project-1', queue: { items: [queueItem] } },
    ];
    const listeners = new Set<() => void>();
    let finishRecord: (result: { kind: 'stored' }) => void = () => undefined;
    const record = vi.fn(
      () =>
        new Promise<{ kind: 'stored' }>((resolve) => {
          finishRecord = resolve;
        })
    );
    const release = vi.fn().mockResolvedValue(undefined);
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend(),
      history: {
        commands: createTestCommands(),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({ record }),
      locks: { acquire: vi.fn().mockResolvedValue({ kind: 'acquired', release }) },
    });

    runtime.start();
    await vi.waitFor(() => expect(record).toHaveBeenCalled());
    projects.splice(0, 1);
    listeners.forEach((listener) => listener());
    await Promise.resolve();
    expect(release).not.toHaveBeenCalled();

    finishRecord({ kind: 'stored' });
    await vi.waitFor(() => expect(release).toHaveBeenCalledTimes(1));
    runtime.dispose();
  });

  it('coalesces status bursts to one in-flight and one latest journal write', async () => {
    const initial: QueueItem = { ...createPendingQueueItem(), backendItemIds: [88, 89], status: 'running' };
    const project = { id: 'project-1', queue: { items: [initial] } };
    const listeners = new Set<() => void>();
    let finishFirst: (result: { kind: 'stored' }) => void = () => undefined;
    const record = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<{ kind: 'stored' }>((resolve) => {
            finishFirst = resolve;
          })
      )
      .mockResolvedValue({ kind: 'stored' });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        getItem: vi.fn().mockResolvedValue({
          id: 88,
          origin: buildQueueItemOrigin(initial.id, project.id),
          status: 'in_progress',
        }),
      }),
      history: {
        commands: createTestCommands(),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({ record }),
    });

    runtime.start();
    await vi.waitFor(() => expect(record).toHaveBeenCalledTimes(1));
    project.queue.items = [{ ...initial, completedBackendItemIds: [88] }];
    listeners.forEach((listener) => listener());
    project.queue.items = [{ ...initial, completedBackendItemIds: [88, 89] }];
    listeners.forEach((listener) => listener());

    finishFirst({ kind: 'stored' });
    await vi.waitFor(() => expect(record).toHaveBeenCalledTimes(2));
    expect(record.mock.calls[1]?.[1]).toMatchObject({ completedBackendItemIds: [88, 89] });
    await runtime.dispose();
  });

  it('retains recovery and ownership until backend cancellation is acknowledged', async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [88], status: 'running' };
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    let finishCancel: () => void = () => undefined;
    const cancelQueueItems = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishCancel = resolve;
        })
    );
    const journal = createTestJournal();
    const release = vi.fn().mockResolvedValue(undefined);
    const commands = createTestCommands({
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
        listeners.forEach((listener) => listener());
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ cancelQueueItems }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
      locks: { acquire: vi.fn().mockResolvedValue({ kind: 'acquired', release }) },
    });

    runtime.start();
    await vi.waitFor(() => expect(journal.record).toHaveBeenCalled());
    queueItem.status = 'cancelled';
    queueItem.cancellationPending = true;
    listeners.forEach((listener) => listener());
    await vi.waitFor(() => expect(cancelQueueItems).toHaveBeenCalledWith([88]));
    expect(cancelQueueItems).toHaveBeenCalledTimes(1);
    expect(journal.record).toHaveBeenLastCalledWith(
      project.id,
      expect.objectContaining({ cancellationPending: true, status: 'cancelled' })
    );
    expect(journal.settle).not.toHaveBeenCalled();
    expect(release).not.toHaveBeenCalled();

    finishCancel();
    await vi.waitFor(() => expect(journal.settle).toHaveBeenCalledWith(project.id, queueItem.id));
    await vi.waitFor(() => expect(release).toHaveBeenCalledTimes(1));
    listeners.forEach((listener) => listener());
    await Promise.resolve();
    expect(cancelQueueItems).toHaveBeenCalledTimes(1);
    expect(journal.settle).toHaveBeenCalledTimes(1);
    await runtime.dispose();
  });

  it('retains ownership and retries a failed terminal journal removal', async () => {
    vi.useFakeTimers();
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [88],
      cancellationPending: true,
      status: 'cancelled',
    };
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    const settle = vi
      .fn<QueueRunJournalPort['settle']>()
      .mockResolvedValueOnce({ kind: 'unavailable' })
      .mockResolvedValueOnce({ kind: 'removed' });
    const release = vi.fn().mockResolvedValue(undefined);
    const commands = createTestCommands({
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
        listeners.forEach((listener) => listener());
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ cancelQueueItems: vi.fn().mockResolvedValue(undefined) }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({ settle }),
      locks: { acquire: vi.fn().mockResolvedValue({ kind: 'acquired', release }) },
    });

    runtime.start();
    await vi.advanceTimersByTimeAsync(1);

    expect(settle).toHaveBeenCalledTimes(1);
    expect(release).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(100);
    expect(settle).toHaveBeenCalledTimes(2);
    await vi.waitFor(() => expect(release).toHaveBeenCalledTimes(1));
    await runtime.dispose();
    vi.useRealTimers();
  });

  it('restores a durable cancellation intent and retries cancellation before settling it', async () => {
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [88],
      cancellationPending: true,
      status: 'cancelled',
    };
    const project = { id: 'project-1', queue: { items: [] as QueueItem[] } };
    const listeners = new Set<() => void>();
    const cancelQueueItems = vi.fn().mockResolvedValue(undefined);
    const journal = createTestJournal({
      listForProject: vi.fn().mockResolvedValue({
        entries: [{ item: queueItem, projectId: project.id, queueItemId: queueItem.id }],
        kind: 'available',
      }),
    });
    const commands = createTestCommands({
      restoreFromJournal: ({ items }) => {
        project.queue.items = items as QueueItem[];
      },
      setCancellationPending: ({ pending }) => {
        const item = project.queue.items[0];
        if (item) {
          item.cancellationPending = pending || undefined;
        }
        listeners.forEach((listener) => listener());
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        cancelQueueItems,
        getItem: vi.fn().mockResolvedValue({
          id: 88,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'in_progress',
        }),
      }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
    });

    runtime.start();

    await vi.waitFor(() => expect(cancelQueueItems).toHaveBeenCalledWith([88]));
    await vi.waitFor(() => expect(journal.settle).toHaveBeenCalledWith(project.id, queueItem.id));
    expect(project.queue.items[0]?.cancellationPending).toBeUndefined();
    await runtime.dispose();
  });

  it('clears a restored cancellation intent when its backend run is already gone', async () => {
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [88],
      cancellationPending: true,
      status: 'cancelled',
    };
    const project = { id: 'project-1', queue: { items: [] as QueueItem[] } };
    const listeners = new Set<() => void>();
    const journal = createTestJournal({
      listForProject: vi.fn().mockResolvedValue({
        entries: [{ item: queueItem, projectId: project.id, queueItemId: queueItem.id }],
        kind: 'available',
      }),
    });
    const commands = createTestCommands({
      restoreFromJournal: ({ items }) => {
        project.queue.items = items as QueueItem[];
      },
      setCancellationPending: ({ pending }) => {
        const item = project.queue.items[0];
        if (item) {
          item.cancellationPending = pending || undefined;
        }
        listeners.forEach((listener) => listener());
      },
    });
    const cancelQueueItems = vi.fn().mockRejectedValue(new ApiError('not found', 404));
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        cancelQueueItems,
        getItem: vi.fn().mockRejectedValue(new ApiError('not found', 404)),
      }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
    });

    runtime.start();

    await vi.waitFor(() => expect(journal.settle).toHaveBeenCalledWith(project.id, queueItem.id));
    expect(project.queue.items[0]?.cancellationPending).toBeUndefined();
    expect(cancelQueueItems).toHaveBeenCalledTimes(1);
    listeners.forEach((listener) => listener());
    await Promise.resolve();
    expect(journal.settle).toHaveBeenCalledTimes(1);
    await runtime.dispose();
  });

  it('fallback-acknowledges a proven local-only cancelled run found only by its receipt', async () => {
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      cancellationPending: true,
      localRecoveryState: 'local-only',
      status: 'cancelled',
    };
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    const acknowledgeEnqueue = vi.fn().mockResolvedValue(undefined);
    const commands = createTestCommands({
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
        listeners.forEach((listener) => listener());
      },
      setLocalRecoveryState: ({ state }) => {
        queueItem.localRecoveryState = state;
        listeners.forEach((listener) => listener());
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        acknowledgeEnqueue,
        getEnqueueReceipt: vi.fn().mockResolvedValue({
          batchId: 'batch-1',
          enqueued: 1,
          itemIds: [77],
          requested: 1,
        }),
        getItem: vi.fn().mockRejectedValue(new ApiError('not found', 404)),
      }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal({ record: vi.fn().mockResolvedValue({ kind: 'quota' }) }),
    });

    runtime.start();

    await vi.waitFor(() => expect(acknowledgeEnqueue).toHaveBeenCalledWith(project.id, queueItem.id));
    expect(queueItem.cancellationPending).toBeUndefined();
    await runtime.dispose();
  });

  it('cancels an enqueue accepted after local cancellation before settling recovery', async () => {
    const queueItem: QueueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    let finishEnqueue: (result: {
      batchId: string;
      enqueued: number;
      itemIds: number[];
      requested: number;
    }) => void = () => undefined;
    const enqueueGenerate = vi.fn(
      () =>
        new Promise<{ batchId: string; enqueued: number; itemIds: number[]; requested: number }>((resolve) => {
          finishEnqueue = resolve;
        })
    );
    const cancelQueueItems = vi.fn().mockResolvedValue(undefined);
    const cancelQueueItemsByBatchIds = vi.fn().mockResolvedValue(undefined);
    const journal = createTestJournal();
    const commands = createTestCommands({
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
        listeners.forEach((listener) => listener());
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ cancelQueueItems, cancelQueueItemsByBatchIds, enqueueGenerate }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
    });

    runtime.start();
    await vi.waitFor(() => expect(enqueueGenerate).toHaveBeenCalledTimes(1));
    queueItem.status = 'cancelled';
    queueItem.cancellationPending = true;
    listeners.forEach((listener) => listener());
    await vi.waitFor(() =>
      expect(journal.record).toHaveBeenLastCalledWith(
        project.id,
        expect.objectContaining({ cancellationPending: true, status: 'cancelled' })
      )
    );
    expect(journal.settle).not.toHaveBeenCalled();

    finishEnqueue({ batchId: 'webv2:local-queue-item', enqueued: 1, itemIds: [88], requested: 1 });
    await vi.waitFor(() => expect(cancelQueueItemsByBatchIds).toHaveBeenCalledWith(['webv2:local-queue-item']));
    expect(cancelQueueItems).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(journal.settle).toHaveBeenCalledWith(project.id, queueItem.id));
    await runtime.dispose();
  });

  it('reconciles an ambiguously accepted enqueue instead of failing or submitting twice', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const commands = createTestCommands();
    const enqueueGenerate = vi.fn().mockRejectedValueOnce(new Error('response lost'));
    const listItems = vi
      .fn()
      .mockResolvedValueOnce([])
      .mockResolvedValue([
        {
          batchId: 'webv2:local-queue-item',
          id: 88,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'in_progress',
        },
      ]);
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate, listItems }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal(),
    });

    runtime.start();

    await vi.waitFor(() => expect(commands.markBackendSubmitted).toHaveBeenCalled(), { timeout: 2_000 });
    expect(enqueueGenerate).toHaveBeenCalledTimes(1);
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));
    runtime.dispose();
  });

  it('keeps reconciling after three consecutive lost enqueue responses', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const commands = createTestCommands();
    const enqueueGenerate = vi
      .fn()
      .mockRejectedValueOnce(new Error('response lost 1'))
      .mockRejectedValueOnce(new Error('response lost 2'))
      .mockRejectedValueOnce(new Error('response lost 3'))
      .mockResolvedValue({ batchId: 'backend-batch', enqueued: 1, itemIds: [88], requested: 1 });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate, listItems: vi.fn().mockResolvedValue([]) }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal(),
    });

    runtime.start();

    await vi.waitFor(() => expect(enqueueGenerate).toHaveBeenCalledTimes(4), { timeout: 4_000 });
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));
    await runtime.dispose();
  });

  it('owns and journals a fresh pending run while offline, then submits it after reconnect', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    let connectionStatus: BackendConnectionStatus = 'disconnected';
    let connectionListener: ((status: BackendConnectionStatus, error?: string) => void) | undefined;
    const enqueueGenerate = vi.fn().mockResolvedValue({ enqueued: 1, itemIds: [77], requested: 1 });
    const journal = createTestJournal();
    const acquire = vi.fn().mockResolvedValue({ kind: 'acquired', release: vi.fn().mockResolvedValue(undefined) });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        enqueueGenerate,
        onConnectionChange: (listener) => {
          connectionListener = listener;
          listener(connectionStatus);
          return vi.fn();
        },
      }),
      history: {
        commands: createTestCommands(),
        getSnapshot: () => ({ connectionStatus, isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
      locks: { acquire },
    });

    runtime.start();

    await vi.waitFor(() => expect(acquire).toHaveBeenCalledWith(project.id, queueItem.id));
    await vi.waitFor(() => expect(journal.record).toHaveBeenCalledWith(project.id, queueItem));
    expect(enqueueGenerate).not.toHaveBeenCalled();

    connectionStatus = 'connected';
    connectionListener?.('connected');
    listeners.forEach((listener) => listener());
    await vi.waitFor(() => expect(enqueueGenerate).toHaveBeenCalledTimes(1));
    await runtime.dispose();
  });

  it('bounds ownership acquisition and journal writes while preparing many offline runs', async () => {
    const projects = Array.from({ length: 40 }, (_, index) => ({
      id: `project-${index}`,
      queue: { items: [{ ...createPendingQueueItem(), id: `queue-item-${index}` }] },
    }));
    let activeAcquisitions = 0;
    let activeRecords = 0;
    let maxAcquisitions = 0;
    let maxRecords = 0;
    const acquire = vi.fn(async (): Promise<QueueRunLock> => {
      activeAcquisitions += 1;
      maxAcquisitions = Math.max(maxAcquisitions, activeAcquisitions);
      await Promise.resolve();
      activeAcquisitions -= 1;
      return { kind: 'acquired', release: vi.fn().mockResolvedValue(undefined) };
    });
    const record = vi.fn<QueueRunJournalPort['record']>(async () => {
      activeRecords += 1;
      maxRecords = Math.max(maxRecords, activeRecords);
      await Promise.resolve();
      activeRecords -= 1;
      return { kind: 'stored' };
    });
    const backend = createTestBackend();
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend,
      history: {
        commands: createTestCommands(),
        getSnapshot: () => ({ connectionStatus: 'disconnected', isHydrated: true, projects }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal({ record }),
      locks: { acquire },
    });

    runtime.start();

    await vi.waitFor(() => expect(record).toHaveBeenCalledTimes(projects.length));
    expect(maxAcquisitions).toBeLessThanOrEqual(16);
    expect(maxRecords).toBeLessThanOrEqual(16);
    expect(backend.enqueueGenerate).not.toHaveBeenCalled();
    await runtime.dispose();
  });

  it('settles a cancellation when a pending enqueue accepts no items and never retries it', async () => {
    vi.useFakeTimers();
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    let finishEnqueue: (result: { enqueued: number; itemIds: number[]; requested: number }) => void = () => undefined;
    const enqueueGenerate = vi.fn(
      () =>
        new Promise<{ enqueued: number; itemIds: number[]; requested: number }>((resolve) => {
          finishEnqueue = resolve;
        })
    );
    const journal = createTestJournal();
    const commands = createTestCommands({
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
      },
      setLocalRecoveryState: vi.fn(),
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
    });

    runtime.start();
    await vi.waitFor(() => expect(enqueueGenerate).toHaveBeenCalledTimes(1));
    queueItem.status = 'cancelled';
    queueItem.cancellationPending = true;
    listeners.forEach((listener) => listener());
    finishEnqueue({ enqueued: 0, itemIds: [], requested: 1 });

    await vi.waitFor(() => expect(journal.settle).toHaveBeenCalledWith(project.id, queueItem.id));
    await vi.advanceTimersByTimeAsync(10_000);
    expect(enqueueGenerate).toHaveBeenCalledTimes(1);
    expect(queueItem.cancellationPending).toBeUndefined();
    await runtime.dispose();
    vi.useRealTimers();
  });

  it('adopts a partially accepted enqueue instead of abandoning its backend items', async () => {
    const queueItem = createPendingQueueItem();
    queueItem.snapshot.presentation.batchCount = 2;
    if (queueItem.snapshot.backendSubmission.kind !== 'invalid') {
      queueItem.snapshot.backendSubmission.batchCount = 2;
    }
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const commands = createTestCommands();
    const enqueueGenerate = vi
      .fn()
      .mockResolvedValueOnce({ batchId: 'webv2:local-queue-item', enqueued: 1, itemIds: [88], requested: 2 });
    const listItems = vi
      .fn()
      .mockResolvedValueOnce([])
      .mockResolvedValue([
        {
          batchId: 'webv2:local-queue-item',
          id: 88,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'in_progress',
        },
      ]);
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate, listItems }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal(),
    });

    runtime.start();

    await vi.waitFor(() => expect(commands.markBackendSubmitted).toHaveBeenCalled(), { timeout: 2_000 });
    expect(enqueueGenerate).toHaveBeenCalledTimes(1);
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));
    expect(commands.recordError).toHaveBeenCalledWith(
      expect.objectContaining({
        area: 'queue-submission',
        message: expect.stringContaining('accepted 1 of 2'),
        projectId: project.id,
      })
    );
    await runtime.dispose();
  });

  it('bounds per-item result and video reads for a large restored run', async () => {
    const backendItemIds = Array.from({ length: 40 }, (_, index) => index + 1);
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds,
      status: 'running',
    };
    queueItem.snapshot.destination = 'gallery';
    queueItem.snapshot.galleryBoardId = 'board-1';
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    let activeImageReads = 0;
    let activeVideoReads = 0;
    let maxImageReads = 0;
    let maxVideoReads = 0;
    const getResultImages = vi.fn(async () => {
      activeImageReads += 1;
      maxImageReads = Math.max(maxImageReads, activeImageReads);
      await Promise.resolve();
      activeImageReads -= 1;
      return [];
    });
    const getResultVideoNames = vi.fn(async () => {
      activeVideoReads += 1;
      maxVideoReads = Math.max(maxVideoReads, activeVideoReads);
      await Promise.resolve();
      activeVideoReads -= 1;
      return [];
    });
    const commands = createTestCommands();
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        getItem: vi.fn((itemId: number) =>
          Promise.resolve({
            batchId: 'batch-1',
            id: itemId,
            origin: buildQueueItemOrigin(queueItem.id, project.id),
            status: 'completed' as const,
          })
        ),
        getResultImages,
        getResultVideoNames,
      }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal(),
    });

    runtime.start();

    await vi.waitFor(() => expect(commands.routeResults).toHaveBeenCalledTimes(1));
    expect(getResultImages.mock.calls.length).toBeGreaterThanOrEqual(backendItemIds.length);
    expect(getResultVideoNames.mock.calls.length).toBeGreaterThanOrEqual(backendItemIds.length);
    expect(maxImageReads).toBeLessThanOrEqual(16);
    expect(maxVideoReads).toBeLessThanOrEqual(16);
    await runtime.dispose();
  });

  it('retries transient reconciliation failures while the connection remains up', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listItems = vi
      .fn()
      .mockRejectedValueOnce(new Error('transient'))
      .mockResolvedValueOnce([
        {
          batchId: 'backend-batch',
          id: 88,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'in_progress',
        },
      ]);
    const commands = createTestCommands();
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ listItems }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal(),
    });

    runtime.start();

    await vi.waitFor(() => expect(listItems).toHaveBeenCalledTimes(2), { timeout: 2_000 });
    expect(commands.markBackendSubmitted).toHaveBeenCalled();
    runtime.dispose();
  });

  it('restores journal rows before reconciliation and leaves a contended run to its owning tab', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [] as QueueItem[] } };
    const listeners = new Set<() => void>();
    const restoreFromJournal = vi.fn(({ items }: { items: unknown[] }) => {
      project.queue.items = items as QueueItem[];
    });
    const journal = createTestJournal({
      listForProject: vi.fn().mockResolvedValue({
        entries: [{ item: queueItem, projectId: project.id, queueItemId: queueItem.id }],
        kind: 'available',
      }),
    });
    const acquire = vi.fn().mockResolvedValue({ kind: 'contended' });
    const locks: QueueRunLockPort = { acquire };
    const backend = createTestBackend();
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend,
      history: {
        commands: createTestCommands({ restoreFromJournal }),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
      locks,
    });

    runtime.start();

    await vi.waitFor(() => expect(acquire).toHaveBeenCalledWith(project.id, queueItem.id));
    expect(restoreFromJournal).toHaveBeenCalledWith({ items: [queueItem], projectId: project.id });
    expect(backend.listItems).not.toHaveBeenCalled();
    expect(backend.enqueueGenerate).not.toHaveBeenCalled();
    expect(journal.record).not.toHaveBeenCalled();

    queueItem.status = 'cancelled';
    listeners.forEach((listener) => listener());
    await Promise.resolve();
    expect(journal.settle).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('retries reconciliation after another tab releases run ownership', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [] as QueueItem[] } };
    const release = vi.fn().mockResolvedValue(undefined);
    const acquire = vi
      .fn<QueueRunLockPort['acquire']>()
      .mockResolvedValueOnce({ kind: 'contended' })
      .mockResolvedValue({ kind: 'acquired', release });
    const listItems = vi.fn().mockResolvedValue([
      {
        batchId: 'backend-batch',
        id: 77,
        origin: buildQueueItemOrigin(queueItem.id, project.id),
        status: 'in_progress',
      },
    ]);
    const commands = createTestCommands({
      restoreFromJournal: ({ items }) => {
        project.queue.items = items as QueueItem[];
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ listItems }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal({
        listForProject: vi.fn().mockResolvedValue({
          entries: [{ item: queueItem, projectId: project.id, queueItemId: queueItem.id }],
          kind: 'available',
        }),
      }),
      locks: { acquire },
    });

    runtime.start();

    await vi.waitFor(() => expect(acquire).toHaveBeenCalledTimes(2), { timeout: 2_000 });
    await vi.waitFor(() => expect(listItems).toHaveBeenCalledOnce());
    expect(commands.markBackendSubmitted).toHaveBeenCalled();

    await runtime.dispose();
  });

  it('settles the journal and releases ownership after a restored run completes', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const listeners = new Set<() => void>();
    const notify = () => listeners.forEach((listener) => listener());
    const journal = createTestJournal();
    const release = vi.fn().mockResolvedValue(undefined);
    const locks: QueueRunLockPort = {
      acquire: vi.fn().mockResolvedValue({ kind: 'acquired', release }),
    };
    const commands = createTestCommands({
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
        notify();
      },
      routeResults: () => {
        queueItem.status = 'completed';
        notify();
      },
    });
    const backend = createTestBackend({
      listItems: vi.fn().mockResolvedValue([
        {
          batchId: 'backend-batch',
          id: 77,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'completed',
        },
      ]),
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend,
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
      locks,
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(journal.settle).toHaveBeenCalledWith(project.id, queueItem.id);
      expect(release).toHaveBeenCalledTimes(1);
    });

    runtime.dispose();
  });

  it('loads and reconciles journal rows before processing a project opened mid-session', async () => {
    const restoredItem = {
      ...createPendingQueueItem(),
      backendItemIds: [88],
      status: 'running' as const,
    };
    const projects = [{ id: 'project-1', queue: { items: [] as QueueItem[] } }];
    const listeners = new Set<() => void>();
    const journal = createTestJournal({
      listForProject: vi.fn().mockImplementation((projectId: string) =>
        Promise.resolve({
          entries: projectId === 'project-2' ? [{ item: restoredItem, projectId, queueItemId: restoredItem.id }] : [],
          kind: 'available',
        })
      ),
    });
    const commands = createTestCommands({
      restoreFromJournal: ({ items, projectId }) => {
        const project = projects.find((candidate) => candidate.id === projectId);
        if (project) {
          project.queue.items = items as QueueItem[];
        }
        listeners.forEach((listener) => listener());
      },
    });
    const getItem = vi.fn().mockResolvedValue({
      id: 88,
      origin: buildQueueItemOrigin(restoredItem.id, 'project-2'),
      status: 'in_progress',
    });
    const backend = createTestBackend({ getItem });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend,
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
    });

    runtime.start();
    await vi.waitFor(() => expect(journal.listForProject).toHaveBeenCalledWith('project-1'));

    projects.push({ id: 'project-2', queue: { items: [] } });
    listeners.forEach((listener) => listener());

    await vi.waitFor(() => expect(getItem).toHaveBeenCalledWith(88));
    expect(journal.listForProject).toHaveBeenCalledWith('project-2');
    expect(backend.enqueueGenerate).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('retains a running journal row when its project tab closes', async () => {
    const queueItem = { ...createPendingQueueItem(), backendItemIds: [88], status: 'running' as const };
    const projects: Array<{ id: string; queue: { items: QueueItem[] } }> = [
      { id: 'project-1', queue: { items: [queueItem] } },
    ];
    const listeners = new Set<() => void>();
    const journal = createTestJournal({
      listForProject: vi.fn().mockResolvedValue({
        entries: [{ item: queueItem, projectId: 'project-1', queueItemId: queueItem.id }],
        kind: 'available',
      }),
    });
    const release = vi.fn().mockResolvedValue(undefined);
    const getItem = vi.fn().mockResolvedValue({
      id: 88,
      origin: buildQueueItemOrigin(queueItem.id, 'project-1'),
      status: 'in_progress',
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ getItem }),
      history: {
        commands: createTestCommands({
          restoreFromJournal: ({ items, projectId }) => {
            const project = projects.find((candidate) => candidate.id === projectId);
            if (project) {
              project.queue.items = items as QueueItem[];
            }
            listeners.forEach((listener) => listener());
          },
        }),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal,
      locks: { acquire: vi.fn().mockResolvedValue({ kind: 'acquired', release }) },
    });

    runtime.start();
    await vi.waitFor(() => expect(journal.record).toHaveBeenCalled());

    projects.splice(0, 1);
    listeners.forEach((listener) => listener());

    await vi.waitFor(() => expect(release).toHaveBeenCalledTimes(1));
    expect(journal.settle).not.toHaveBeenCalled();

    projects.push({ id: 'project-1', queue: { items: [] } });
    listeners.forEach((listener) => listener());

    await vi.waitFor(() => expect(getItem).toHaveBeenCalledTimes(2));
    expect(journal.listForProject).toHaveBeenCalledWith('project-1');

    runtime.dispose();
  });
});

describe('queue runtime video board routing', () => {
  const createHarness = (options: {
    getResultVideoNames: QueueBackendPort['getResultVideoNames'];
    galleryBoardId?: string | null;
    getResultImages?: QueueBackendPort['getResultImages'];
    /** Nodes of the compiled submission graph — media values here mark run INPUTS. */
    graphNodes?: Record<string, QueueBackendInvocation>;
  }) => {
    const queueItem = createPendingQueueItem();
    queueItem.snapshot.destination = 'gallery';
    queueItem.snapshot.galleryBoardId = options.galleryBoardId === undefined ? 'board-1' : options.galleryBoardId;
    queueItem.snapshot.filterIntermediateResults = true;
    delete (queueItem.snapshot as { resultNodeIds?: unknown }).resultNodeIds;
    if (options.graphNodes && queueItem.snapshot.backendSubmission.kind !== 'invalid') {
      queueItem.snapshot.backendSubmission.graph.nodes = options.graphNodes;
    }
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const backend: QueueBackendPort = {
      cancelCurrentItem: vi.fn(),
      cancelItem: vi.fn(),
      cancelQueueItems: vi.fn(),
      cancelQueueItemsByBatchIds: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate: vi.fn(),
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: options.getResultImages ?? vi.fn().mockResolvedValue([]),
      getResultVideoNames: options.getResultVideoNames,
      // Reconcile an already-completed backend run to exercise both settlement paths without sockets.
      listItems: vi.fn().mockResolvedValue([
        {
          batchId: 'backend-batch',
          id: 77,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'completed',
        },
      ]),
      on: vi.fn(() => vi.fn()),
      onConnectionChange: vi.fn(() => vi.fn()),
      pauseProcessor: vi.fn(),
      readCurrent: vi.fn().mockResolvedValue(null),
      readItemIds: vi.fn().mockResolvedValue({ itemIds: [], totalCount: 0 }),
      readItemsById: vi.fn().mockResolvedValue([]),
      readNext: vi.fn().mockResolvedValue(null),
      readStatus: vi.fn().mockResolvedValue({
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
      }),
      resumeProcessor: vi.fn(),
      retryItems: vi.fn(),
    };
    const commands: QueueHistoryCommands = {
      markBackendCancelled: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
      },
      setLocalRecoveryState: vi.fn(),
      recordError: vi.fn(),
      refreshBackendData: vi.fn(),
      restoreFromJournal: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
    };
    const destinations = { addImagesToGalleryBoard: vi.fn(), addVideosToGalleryBoard: vi.fn() };
    const runtime = createQueueRuntime({
      backend,
      destinations,
      ensureTemplatesLoaded: vi.fn(),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      modelLoads: { completed: vi.fn(), reset: vi.fn(), started: vi.fn() },
      nodeExecution: {
        clearAll: vi.fn(),
        completed: vi.fn(),
        failed: vi.fn(),
        progress: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
    });

    return { commands, destinations, runtime };
  };

  it('lands result videos on the enqueue-time board, excluding intermediates', async () => {
    const getResultVideoNames = vi.fn().mockResolvedValue(['clip-1.mp4']);
    const { destinations, runtime } = createHarness({ getResultVideoNames });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['clip-1.mp4']);
    });
    // filterIntermediateResults on the snapshot maps to the video-side intermediate filter.
    expect(getResultVideoNames).toHaveBeenCalledWith(77, expect.objectContaining({ excludeIntermediate: true }));

    runtime.dispose();
  });

  it('records a video-routing failure without failing the completed run', async () => {
    const getResultVideoNames = vi.fn().mockRejectedValue(new Error('transient 502'));
    const { commands, destinations, runtime } = createHarness({ getResultVideoNames });

    runtime.start();

    await vi.waitFor(() => {
      // The run still settles and records its results...
      expect(commands.routeResults).toHaveBeenCalled();
      // ...with the board-attach hiccup recorded, not fatal.
      expect(commands.recordError).toHaveBeenCalledWith(expect.objectContaining({ area: 'queue-results' }));
    });
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('skips the board attach entirely when no board was active at enqueue', async () => {
    const getResultVideoNames = vi.fn().mockResolvedValue(['clip-1.mp4']);
    const { commands, destinations, runtime } = createHarness({ galleryBoardId: null, getResultVideoNames });

    runtime.start();

    // Wait for full settlement first so the negative assertions below are meaningful.
    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(destinations.addImagesToGalleryBoard).not.toHaveBeenCalled();
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();
    expect(getResultVideoNames).not.toHaveBeenCalled();

    runtime.dispose();
  });

  const resultImage = (imageName: string): QueueResultImage => ({
    height: 512,
    imageName,
    imageUrl: `http://test/i/${imageName}`,
    isIntermediate: false,
    queuedAt: '2026-07-17T00:00:00.000Z',
    sourceQueueItemId: 'local-queue-item',
    thumbnailUrl: `http://test/t/${imageName}`,
    width: 512,
  });

  it('never routes an input image echoed into the results (first-frame keyframe)', async () => {
    // Exclude echoed i2v keyframes from result recording and board attachment.
    const { commands, destinations, runtime } = createHarness({
      getResultImages: vi.fn().mockResolvedValue([resultImage('keyframe.png'), resultImage('generated.png')]),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
      graphNodes: {
        first_frame: { id: 'first_frame', image: { image_name: 'keyframe.png' }, type: 'image' },
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addImagesToGalleryBoard).toHaveBeenCalledWith('board-1', ['generated.png']);
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({ images: [expect.objectContaining({ imageName: 'generated.png' })] })
      );
    });
    expect(destinations.addImagesToGalleryBoard).not.toHaveBeenCalledWith(
      'board-1',
      expect.arrayContaining(['keyframe.png'])
    );

    runtime.dispose();
  });

  it('never routes an input video echoed into the results (extend-video source clip)', async () => {
    const { destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['source.mp4', 'extended.mp4']),
      graphNodes: {
        source_clip: { id: 'source_clip', type: 'video', video: { video_name: 'source.mp4' } },
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['extended.mp4']);
    });
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalledWith(
      'board-1',
      expect.arrayContaining(['source.mp4'])
    );

    runtime.dispose();
  });

  it('leaves an image on the board its node saved it to and attaches only unassigned images', async () => {
    // A workflow node with an explicit board writes the image there server-side; re-attaching
    // it to the active board would move it, and the recents overlay must show its real board.
    const nodeBoardImage = { ...resultImage('node-board.png'), boardId: 'board-b' };
    const { commands, destinations, runtime } = createHarness({
      getResultImages: vi.fn().mockResolvedValue([nodeBoardImage, resultImage('unassigned.png')]),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addImagesToGalleryBoard).toHaveBeenCalledWith('board-1', ['unassigned.png']);
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({
          images: [
            expect.objectContaining({ boardId: 'board-b' }),
            expect.objectContaining({ imageName: 'unassigned.png' }),
          ],
        })
      );
    });
    expect(destinations.addImagesToGalleryBoard).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.arrayContaining(['node-board.png'])
    );

    runtime.dispose();
  });

  it('skips the board attach call when every image already has a board', async () => {
    const { commands, destinations, runtime } = createHarness({
      getResultImages: vi.fn().mockResolvedValue([{ ...resultImage('node-board.png'), boardId: 'board-b' }]),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(destinations.addImagesToGalleryBoard).not.toHaveBeenCalled();

    runtime.dispose();
  });
});

describe('queue runtime workflow run capture', () => {
  const resultImage = (imageName: string): QueueResultImage => ({
    height: 512,
    imageName,
    imageUrl: `http://test/i/${imageName}`,
    isIntermediate: false,
    queuedAt: '2026-07-17T00:00:00.000Z',
    sourceQueueItemId: 'local-queue-item',
    thumbnailUrl: `http://test/t/${imageName}`,
    width: 512,
  });

  const createHarness = (options: {
    images?: QueueResultImage[];
    /** Replaces the item's generate submission with a workflow one. */
    libraryWorkflowId?: string | null;
    onWorkflowRunCompleted?: QueueWorkflowRunSink['onWorkflowRunCompleted'];
  }) => {
    const queueItem = createPendingQueueItem();

    queueItem.snapshot.destination = 'gallery';
    queueItem.snapshot.galleryBoardId = null;
    if (options.libraryWorkflowId !== undefined) {
      queueItem.snapshot.backendSubmission = {
        batchCount: 1,
        graph: { edges: [], id: 'backend-graph', nodes: {} },
        kind: 'workflow',
        ...(options.libraryWorkflowId ? { libraryWorkflowId: options.libraryWorkflowId } : {}),
      };
      queueItem.snapshot.sourceId = 'workflow';
    }

    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const backend: QueueBackendPort = {
      cancelCurrentItem: vi.fn(),
      cancelItem: vi.fn(),
      cancelQueueItems: vi.fn(),
      cancelQueueItemsByBatchIds: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate: vi.fn(),
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: vi.fn().mockResolvedValue(options.images ?? [resultImage('generated.png')]),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
      listItems: vi.fn().mockResolvedValue([
        {
          batchId: 'backend-batch',
          id: 77,
          origin: buildQueueItemOrigin(queueItem.id, project.id),
          status: 'completed',
        },
      ]),
      on: vi.fn(() => vi.fn()),
      onConnectionChange: vi.fn(() => vi.fn()),
      pauseProcessor: vi.fn(),
      readCurrent: vi.fn().mockResolvedValue(null),
      readItemIds: vi.fn().mockResolvedValue({ itemIds: [], totalCount: 0 }),
      readItemsById: vi.fn().mockResolvedValue([]),
      readNext: vi.fn().mockResolvedValue(null),
      readStatus: vi.fn().mockResolvedValue({
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
      }),
      resumeProcessor: vi.fn(),
      retryItems: vi.fn(),
    };
    const commands: QueueHistoryCommands = {
      markBackendCancelled: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      setCancellationPending: ({ pending }) => {
        queueItem.cancellationPending = pending || undefined;
      },
      setLocalRecoveryState: vi.fn(),
      recordError: vi.fn(),
      refreshBackendData: vi.fn(),
      restoreFromJournal: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
    };
    const onWorkflowRunCompleted = vi.fn(options.onWorkflowRunCompleted);
    const runtime = createQueueRuntime({
      backend,
      destinations: { addImagesToGalleryBoard: vi.fn(), addVideosToGalleryBoard: vi.fn() },
      ensureTemplatesLoaded: vi.fn(),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      modelLoads: { completed: vi.fn(), reset: vi.fn(), started: vi.fn() },
      nodeExecution: {
        clearAll: vi.fn(),
        completed: vi.fn(),
        failed: vi.fn(),
        progress: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
      workflowRuns: { onWorkflowRunCompleted },
    });

    return { commands, onWorkflowRunCompleted, queueItem, runtime };
  };

  it('notifies the capture sink once with the completed run and its result image names', async () => {
    const { commands, onWorkflowRunCompleted, runtime } = createHarness({
      images: [resultImage('early.png'), resultImage('final.png')],
      libraryWorkflowId: 'library-workflow-1',
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(onWorkflowRunCompleted).toHaveBeenCalledTimes(1);
    expect(onWorkflowRunCompleted).toHaveBeenCalledWith({
      imageNames: ['early.png', 'final.png'],
      libraryWorkflowId: 'library-workflow-1',
      projectId: 'project-1',
      queueItemId: 'local-queue-item',
    });

    runtime.dispose();
  });

  it('never notifies for a workflow run that is not bound to a library record', async () => {
    const { commands, onWorkflowRunCompleted, runtime } = createHarness({ libraryWorkflowId: null });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(onWorkflowRunCompleted).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('never notifies for a generate run', async () => {
    const { commands, onWorkflowRunCompleted, runtime } = createHarness({});

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(onWorkflowRunCompleted).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('never notifies for a bound workflow run that produced no images', async () => {
    const { commands, onWorkflowRunCompleted, runtime } = createHarness({
      images: [],
      libraryWorkflowId: 'library-workflow-1',
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(onWorkflowRunCompleted).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('settles the run normally when the capture sink throws', async () => {
    const { commands, onWorkflowRunCompleted, runtime } = createHarness({
      libraryWorkflowId: 'library-workflow-1',
      onWorkflowRunCompleted: () => {
        throw new Error('capture exploded');
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(onWorkflowRunCompleted).toHaveBeenCalled();
    });
    expect(commands.routeResults).toHaveBeenCalled();
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));

    runtime.dispose();
  });
});
