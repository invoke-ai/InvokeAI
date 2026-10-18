import type { QueueItem } from '@features/queue/core/historyTypes';
import type {
  QueueBackendInvocation,
  QueueBackendPort,
  QueueResultImage,
  QueueResultVideo,
} from '@features/queue/core/types';
import type { BackendConnectionStatus } from '@platform/transport/types';
import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import { activeProgressTargetStore } from '@features/queue/data/activeProgressTargetStore';
import { buildQueueItemOrigin } from '@features/queue/data/events';
import { ApiError } from '@platform/transport/http';
import { createInitialWorkbenchState, type WorkbenchAction, workbenchReducer } from '@workbench/workbenchState.testing';
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
  cancelBatch: vi.fn(),
  clearFinishedItems: vi.fn(),
  readItemDetail: vi.fn(),
  cancelScopedItems: vi.fn(),
  claimItemDelivery: vi.fn().mockResolvedValue({ kind: 'claimed' }),
  clearFailedItems: vi.fn(),
  clearItems: vi.fn(),
  emit: vi.fn(),
  enqueueGenerate: vi.fn(),
  enqueueWorkflow: vi.fn(),
  getItem: vi.fn(),
  getResultImages: vi.fn().mockResolvedValue([]),
  getResultVideoNames: vi.fn().mockResolvedValue([]),
  getResultVideos: vi.fn().mockResolvedValue([]),
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
  refreshBoardDestinations: vi.fn(),
  reopenForRetry: vi.fn(),
  restoreFromJournal: vi.fn(),
  routePartialResults: vi.fn(),
  routeResults: vi.fn(),
  setConnectionStatus: vi.fn(),
  setStatus: vi.fn(),
  stageRetriedCanvasResults: vi.fn(),
  ...overrides,
});

const createTestJournal = (overrides: Partial<QueueRunJournalPort> = {}): QueueRunJournalPort => ({
  listForProject: vi.fn().mockResolvedValue({ entries: [], kind: 'available' }),
  record: vi.fn().mockResolvedValue({ kind: 'stored' }),
  settle: vi.fn().mockResolvedValue({ kind: 'removed' }),
  ...overrides,
});

/** Run locks that no other tab holds, around the given acquire. */
const createTestLocks = (
  acquire: QueueRunLockPort['acquire'],
  overrides: Partial<QueueRunLockPort> = {}
): QueueRunLockPort => ({
  acquire,
  isHeld: () => Promise.resolve(false),
  waitForRunRelease: () => Promise.resolve(),
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
    setOrigin: vi.fn(),
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

  it('carries an Invoke Front snapshot to the backend request and leaves ordinary runs appending', () => {
    const ordinary = createQueueItemBackendSubmission({ id: 'project-1' }, createPendingQueueItem()) as {
      request: object;
    };

    expect(ordinary.request).not.toHaveProperty('prepend');

    const front = createPendingQueueItem();
    front.snapshot.prepend = true;

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, front)).toMatchObject({
      kind: 'generate',
      request: { prepend: true },
    });

    const workflow = createPendingQueueItem();
    workflow.snapshot = {
      ...workflow.snapshot,
      backendSubmission: { batchCount: 1, graph: { edges: [], id: 'workflow-graph', nodes: {} }, kind: 'workflow' },
      prepend: true,
    };

    expect(createQueueItemBackendSubmission({ id: 'project-1' }, workflow)).toMatchObject({
      kind: 'workflow',
      request: { prepend: true },
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

  it('keeps workflow provenance off the backend request, including the binding older records carry', () => {
    const queueItem = createPendingQueueItem();
    queueItem.snapshot.backendSubmission = {
      batchCount: 1,
      graph: { edges: [], id: 'backend-graph', nodes: {} },
      kind: 'workflow',
      libraryWorkflowId: 'lib-1',
      projectWorkflowId: 'wf-1',
    } as never;
    queueItem.snapshot.sourceId = 'workflow';

    const submission = createQueueItemBackendSubmission({ id: 'project-1' }, queueItem);

    expect(submission).toMatchObject({ kind: 'workflow', request: { batchCount: 1, projectId: 'project-1' } });
    expect(submission).toHaveProperty('request');
    expect((submission as { request: object }).request).not.toHaveProperty('projectWorkflowId');
    expect((submission as { request: object }).request).not.toHaveProperty('libraryWorkflowId');
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
      cancelBatch: vi.fn(),
      clearFinishedItems: vi.fn(),
      readItemDetail: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate,
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: vi.fn(),
      getResultVideoNames: vi.fn().mockResolvedValue([]),
      getResultVideos: vi.fn().mockResolvedValue([]),
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
      claimItemDelivery: vi.fn().mockResolvedValue({ kind: 'claimed' }),
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
      refreshBoardDestinations: vi.fn(),
      reopenForRetry: vi.fn(),
      restoreFromJournal: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
      stageRetriedCanvasResults: vi.fn(),
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
        setOrigin: vi.fn(),
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
    ['generate', 'gallery', 1],
    ['generate', 'canvas', 0],
    // A workflow names boards on its nodes whatever its destination.
    ['workflow', 'canvas', 1],
  ] as const)(
    'reads the board listing again when a %s run bound for the %s fails, so a board that refused it leaves it',
    async (sourceId, destination, refreshes) => {
      vi.useFakeTimers();
      const queueItem = createPendingQueueItem();
      queueItem.snapshot.sourceId = sourceId;
      queueItem.snapshot.destination = destination;
      const project = { id: 'project-1', queue: { items: [queueItem] } };
      const commands = createTestCommands();
      const runtime = createQueueRuntime({
        ...runtimeServices,
        backend: createTestBackend(),
        ensureProjectPersisted: vi.fn().mockResolvedValue('refused'),
        history: {
          commands,
          getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
          subscribe: vi.fn(() => vi.fn()),
        },
        journal: createTestJournal(),
      });

      runtime.start();
      await vi.advanceTimersByTimeAsync(1);

      expect(commands.setStatus).toHaveBeenCalledWith(
        expect.objectContaining({ queueItemId: queueItem.id, status: 'failed' })
      );
      expect(commands.refreshBoardDestinations).toHaveBeenCalledTimes(refreshes);
      expect(commands.refreshBackendData).not.toHaveBeenCalled();
      await runtime.dispose();
      vi.useRealTimers();
    }
  );

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
    const locks = createTestLocks(
      vi.fn(
        () =>
          new Promise<QueueRunLock>((resolve) => {
            finishAcquire = resolve;
          })
      )
    );
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
    const locks = createTestLocks(
      vi.fn(() => {
        acquisitionCount += 1;
        if (acquisitionCount > 1) {
          return Promise.resolve<QueueRunLock>({ kind: 'acquired', release });
        }
        return new Promise<QueueRunLock>((resolve) => {
          finishAcquire = resolve;
        });
      })
    );
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
      locks: createTestLocks(vi.fn().mockResolvedValue({ kind: 'acquired', release })),
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
      locks: createTestLocks(vi.fn().mockResolvedValue({ kind: 'acquired', release })),
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
      locks: createTestLocks(vi.fn().mockResolvedValue({ kind: 'acquired', release })),
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

  /** Two running batches whose cancellations fail `failures[queueItemId]` times before the backend accepts. */
  const createFailingCancelHarness = (failures: Record<string, number>) => {
    const items = ['batch-eyes', 'batch-mouth'].map((id): QueueItem => ({ ...createPendingQueueItem(), id }));
    const project = { id: 'project-1', queue: { items } };
    const listeners = new Set<() => void>();
    const notify = () => listeners.forEach((listener) => listener());
    const byBackendId = new Map<number, QueueItem>();
    let nextBackendId = 88;
    const enqueueGenerate = vi.fn(() => {
      const backendId = nextBackendId++;
      return Promise.resolve({ batchId: `backend-${backendId}`, enqueued: 1, itemIds: [backendId], requested: 1 });
    });
    const cancelQueueItemsByBatchIds = vi.fn(([batchId]: string[]) => {
      const item = items.find((candidate) => candidate.backendBatchId === batchId)!;
      const left = failures[item.id] ?? 0;
      failures[item.id] = left - 1;
      return left > 0 ? Promise.reject(new Error(`${item.id} unavailable`)) : Promise.resolve();
    });
    const recordError = vi.fn();
    const commands = createTestCommands({
      markBackendSubmitted: ({ backendBatchId, backendItemIds, queueItemId }) => {
        const item = items.find((candidate) => candidate.id === queueItemId)!;
        Object.assign(item, { backendBatchId, backendItemIds, status: 'running' });
        backendItemIds.forEach((backendId) => byBackendId.set(backendId, item));
        notify();
      },
      recordError,
      setCancellationPending: ({ pending, queueItemId }) => {
        items.find((candidate) => candidate.id === queueItemId)!.cancellationPending = pending || undefined;
        notify();
      },
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({
        cancelQueueItemsByBatchIds,
        enqueueGenerate,
        // Retries reconcile against the backend, which still reports each batch running.
        getItem: vi.fn((backendId: number) =>
          Promise.resolve({
            id: backendId,
            origin: buildQueueItemOrigin(byBackendId.get(backendId)!.id, project.id),
            status: 'in_progress' as const,
          })
        ),
      }),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal(),
    });
    const cancel = (queueItemId: string) => {
      Object.assign(
        items.find((item) => item.id === queueItemId)!,
        { cancellationPending: true, status: 'cancelled' }
      );
      notify();
    };
    const settled = (queueItemId: string) =>
      vi.waitFor(() => expect(items.find((item) => item.id === queueItemId)!.cancellationPending).toBeUndefined(), {
        timeout: 3000,
      });
    const reportedFor = () =>
      recordError.mock.calls.map(
        ([payload]) => /queue item (\S+) yet/.exec(String((payload as { message: string }).message))?.[1]
      );
    return { cancel, cancelQueueItemsByBatchIds, items, project, recordError, reportedFor, runtime, settled };
  };

  it('reports a failing batch cancellation once and keeps retrying until the backend stops it', async () => {
    const h = createFailingCancelHarness({ 'batch-eyes': 2 });
    h.runtime.start();
    await vi.waitFor(() => expect(h.items.every((item) => item.backendBatchId)).toBe(true));

    h.cancel('batch-eyes');
    await h.settled('batch-eyes');

    expect(
      h.cancelQueueItemsByBatchIds.mock.calls.filter(([[batchId]]) => batchId === h.items[0]!.backendBatchId)
    ).toHaveLength(3);
    expect(h.recordError).toHaveBeenCalledOnce();
    expect(h.recordError).toHaveBeenCalledWith(
      expect.objectContaining({
        area: 'queue-cancel',
        message: 'Could not stop queue item batch-eyes yet; retrying automatically. batch-eyes unavailable',
        projectId: h.project.id,
      })
    );
    expect(h.items[1]!.status).toBe('running');
    await h.runtime.dispose();
  });

  it('reports each batch whose cancellation fails, whether at the same time or after another settled', async () => {
    const h = createFailingCancelHarness({ 'batch-eyes': 1, 'batch-mouth': 1 });
    h.runtime.start();
    await vi.waitFor(() => expect(h.items.every((item) => item.backendBatchId)).toBe(true));

    h.cancel('batch-eyes');
    await h.settled('batch-eyes');
    expect(h.reportedFor()).toEqual(['batch-eyes']);

    h.cancel('batch-mouth');
    await h.settled('batch-mouth');
    expect(h.reportedFor()).toEqual(['batch-eyes', 'batch-mouth']);
    await h.runtime.dispose();
  });

  it('reports a second batch that fails while the first is still retrying', async () => {
    const h = createFailingCancelHarness({ 'batch-eyes': 2, 'batch-mouth': 2 });
    h.runtime.start();
    await vi.waitFor(() => expect(h.items.every((item) => item.backendBatchId)).toBe(true));

    h.cancel('batch-eyes');
    h.cancel('batch-mouth');
    await h.settled('batch-eyes');
    await h.settled('batch-mouth');

    expect(h.reportedFor().sort()).toEqual(['batch-eyes', 'batch-mouth']);
    await h.runtime.dispose();
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
      locks: createTestLocks(acquire),
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
      locks: createTestLocks(acquire),
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
    const locks = createTestLocks(acquire);
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
      locks: createTestLocks(acquire),
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
    const locks = createTestLocks(vi.fn().mockResolvedValue({ kind: 'acquired', release }));
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

  it.each([
    { hasWorkflowCall: true, name: 'saved-workflow call', node: { id: 'call', type: 'call_saved_workflow' } },
    { hasWorkflowCall: false, name: 'ordinary workflow', node: { id: 'denoise', type: 'denoise' } },
  ])('recovery routes waiting preview correctly for $name', async ({ hasWorkflowCall, node }) => {
    const queueItem = createPendingQueueItem();
    queueItem.snapshot.backendSubmission = {
      batchCount: 1,
      graph: {
        edges: [],
        id: 'backend-graph',
        nodes: { [node.id]: node },
      },
      kind: 'workflow',
    };
    queueItem.snapshot.sourceId = 'workflow';
    queueItem.backendItemIds = [88];
    queueItem.status = 'running';
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const getItem = vi.fn().mockResolvedValue({
      id: 88,
      origin: buildQueueItemOrigin(queueItem.id, project.id),
      status: 'waiting',
    });
    const readProgressPreviews = vi.fn().mockResolvedValue([]);
    const backend = createTestBackend({ getItem, readProgressPreviews });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend,
      history: {
        commands: createTestCommands(),
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
    });
    activeProgressTargetStore.clear();
    const clear = vi.spyOn(activeProgressTargetStore, 'clear');
    const set = vi.spyOn(activeProgressTargetStore, 'set');

    try {
      runtime.start();
      await vi.waitFor(() => expect(readProgressPreviews).toHaveBeenCalled());

      if (hasWorkflowCall) {
        expect(clear).not.toHaveBeenCalledWith({ itemIndex: 1, queueItemId: queueItem.id });
        expect(set).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: queueItem.id });
      } else {
        expect(clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: queueItem.id });
        expect(set).not.toHaveBeenCalledWith({ itemIndex: 1, queueItemId: queueItem.id });
      }
    } finally {
      await runtime.dispose();
      clear.mockRestore();
      set.mockRestore();
    }
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
      locks: createTestLocks(vi.fn().mockResolvedValue({ kind: 'acquired', release })),
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
    getResultVideos?: QueueBackendPort['getResultVideos'];
    galleryBoardId?: string | null;
    getResultImages?: QueueBackendPort['getResultImages'];
    /** Nodes of the compiled submission graph — media values here mark run INPUTS. */
    graphNodes?: Record<string, QueueBackendInvocation>;
    /** Backend board of each video already on one; attaching records here and video reads report it. */
    videoBoards?: Record<string, string>;
  }) => {
    const videoBoards = new Map(Object.entries(options.videoBoards ?? {}));
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
      cancelBatch: vi.fn(),
      clearFinishedItems: vi.fn(),
      readItemDetail: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate: vi.fn(),
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: options.getResultImages ?? vi.fn().mockResolvedValue([]),
      getResultVideoNames: options.getResultVideoNames,
      getResultVideos:
        options.getResultVideos ??
        vi.fn((videoNames: string[], sourceQueueItemId: string, queuedAt: string) =>
          Promise.resolve(
            videoNames.map((videoName) => {
              const boardId = videoBoards.get(videoName);
              return { ...resultVideo(videoName, sourceQueueItemId, queuedAt), ...(boardId ? { boardId } : {}) };
            })
          )
        ),
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
      readItemIds: vi.fn().mockResolvedValue({ itemIds: [] }),
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
      claimItemDelivery: vi.fn().mockResolvedValue({ kind: 'claimed' }),
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
      refreshBoardDestinations: vi.fn(),
      reopenForRetry: vi.fn(),
      restoreFromJournal: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
      stageRetriedCanvasResults: vi.fn(),
    };
    const destinations = {
      addImagesToGalleryBoard: vi.fn(),
      addVideosToGalleryBoard: vi.fn((boardId: string, videoNames: string[]) => {
        for (const videoName of videoNames) {
          videoBoards.set(videoName, boardId);
        }
        return Promise.resolve();
      }),
    };
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
        setOrigin: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
    });

    return { backend, commands, destinations, runtime, videoBoards };
  };

  const resultVideo = (videoName: string, sourceQueueItemId: string, queuedAt: string): QueueResultVideo => ({
    category: 'general',
    durationSeconds: 5,
    height: 512,
    isIntermediate: false,
    queuedAt,
    sourceQueueItemId,
    thumbnailUrl: `http://test/v/${videoName}/thumbnail`,
    videoName,
    videoUrl: `http://test/v/${videoName}`,
    width: 512,
  });

  it('lands result videos on the enqueue-time board, excluding intermediates', async () => {
    const getResultVideoNames = vi.fn().mockResolvedValue(['clip-1.mp4']);
    const { commands, destinations, runtime } = createHarness({ getResultVideoNames });

    runtime.start();

    await vi.waitFor(() => {
      const routedClip = expect.objectContaining({
        videos: [expect.objectContaining({ boardId: 'board-1', videoName: 'clip-1.mp4' })],
      });
      // Both passes queue follow-up reads while earlier ones drain; none may be stranded.
      expect(commands.routePartialResults).toHaveBeenCalledWith(routedClip);
      expect(commands.routeResults).toHaveBeenCalledWith(routedClip);
    });
    // The final pass finds the clip already on its board and does not move it again.
    expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledExactlyOnceWith('board-1', ['clip-1.mp4']);
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

  it.each([null, 'none'])(
    "routes an uncategorized (%s) run's results for display without a board attach",
    async (galleryBoardId) => {
      const { commands, destinations, runtime } = createHarness({
        galleryBoardId,
        getResultImages: vi.fn().mockResolvedValue([resultImage('generated.png')]),
        getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4']),
      });
      const routedClip = expect.objectContaining({
        images: [expect.objectContaining({ imageName: 'generated.png' })],
        videos: [expect.objectContaining({ videoName: 'clip-1.mp4' })],
      });

      runtime.start();

      await vi.waitFor(() => {
        expect(commands.routePartialResults).toHaveBeenCalledWith(routedClip);
        expect(commands.routeResults).toHaveBeenCalledWith(routedClip);
      });
      expect(destinations.addImagesToGalleryBoard).not.toHaveBeenCalled();
      expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();

      runtime.dispose();
    }
  );

  it('still shows a finished video when its board attach fails', async () => {
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4']),
    });
    destinations.addVideosToGalleryBoard.mockRejectedValue(new Error('attach failed'));

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    // The clip is shown on both passes and, never having reached the board, does not claim one.
    const unassignedClip = resultVideo('clip-1.mp4', 'local-queue-item', '2026-07-17T00:00:00.000Z');
    for (const route of [commands.routePartialResults, commands.routeResults]) {
      expect(route).toHaveBeenCalledWith(expect.objectContaining({ videos: [unassignedClip] }));
    }
    expect(commands.recordError).toHaveBeenCalledWith(expect.objectContaining({ message: 'attach failed' }));

    runtime.dispose();
  });

  it('reports the board each video reached when an attach partly fails, then retries only the rest', async () => {
    const { commands, destinations, runtime, videoBoards } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4', 'clip-2.mp4']),
    });
    destinations.addVideosToGalleryBoard.mockImplementationOnce((boardId) => {
      videoBoards.set('clip-1.mp4', boardId);
      return Promise.reject(new Error('1 of 2 video(s) could not be added to the board.'));
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({
          videos: [
            expect.objectContaining({ boardId: 'board-1', videoName: 'clip-1.mp4' }),
            expect.objectContaining({ boardId: 'board-1', videoName: 'clip-2.mp4' }),
          ],
        })
      );
    });
    expect(commands.routePartialResults).toHaveBeenCalledWith(
      expect.objectContaining({
        videos: [
          expect.objectContaining({ boardId: 'board-1', videoName: 'clip-1.mp4' }),
          resultVideo('clip-2.mp4', 'local-queue-item', '2026-07-17T00:00:00.000Z'),
        ],
      })
    );
    expect(destinations.addVideosToGalleryBoard.mock.calls).toEqual([
      ['board-1', ['clip-1.mp4', 'clip-2.mp4']],
      ['board-1', ['clip-2.mp4']],
    ]);

    runtime.dispose();
  });

  it('reports no board when the destination port attaches nothing (virtual board)', async () => {
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4']),
    });
    destinations.addVideosToGalleryBoard.mockResolvedValue(undefined);

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({
          videos: [resultVideo('clip-1.mp4', 'local-queue-item', '2026-07-17T00:00:00.000Z')],
        })
      );
    });
    expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['clip-1.mp4']);

    runtime.dispose();
  });

  it('records unreadable result videos and attaches only the readable ones', async () => {
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4', 'gone.mp4']),
      getResultVideos: vi.fn((videoNames: string[], sourceQueueItemId: string, queuedAt: string) =>
        Promise.resolve(
          videoNames
            .filter((videoName) => videoName !== 'gone.mp4')
            .map((videoName) => resultVideo(videoName, sourceQueueItemId, queuedAt))
        )
      ),
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({ videos: [expect.objectContaining({ videoName: 'clip-1.mp4' })] })
      );
    });
    expect(commands.recordError).toHaveBeenCalledWith(
      expect.objectContaining({
        area: 'queue-results',
        message: '1 of 2 result video(s) could not be read; they were not added to the board.',
      })
    );
    expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['clip-1.mp4']);
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.arrayContaining(['gone.mp4'])
    );

    runtime.dispose();
  });

  it('records a failed video read without attaching or routing the videos', async () => {
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4']),
      getResultVideos: vi.fn().mockRejectedValue(new Error('video read failed')),
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(expect.objectContaining({ videos: [] }));
    });
    expect(commands.recordError).toHaveBeenCalledWith(expect.objectContaining({ message: 'video read failed' }));
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));

    runtime.dispose();
  });

  it('still shows a video when its post-attach re-read fails', async () => {
    const clip = resultVideo('clip-1.mp4', 'local-queue-item', '2026-07-17T00:00:00.000Z');
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4']),
      // Partial pass: first read, then the post-attach re-read; the final pass finds the clip on its board.
      getResultVideos: vi
        .fn()
        .mockResolvedValueOnce([clip])
        .mockRejectedValueOnce(new Error('re-read failed'))
        .mockResolvedValue([{ ...clip, boardId: 'board-1' }]),
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
    });
    expect(commands.routePartialResults).toHaveBeenCalledWith(expect.objectContaining({ videos: [clip] }));
    expect(commands.recordError).toHaveBeenCalledWith(expect.objectContaining({ message: 're-read failed' }));
    expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledExactlyOnceWith('board-1', ['clip-1.mp4']);

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
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['source.mp4', 'extended.mp4']),
      graphNodes: {
        source_clip: { id: 'source_clip', type: 'video', video: { video_name: 'source.mp4' } },
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['extended.mp4']);
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({ videos: [expect.objectContaining({ videoName: 'extended.mp4' })] })
      );
    });
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalledWith(
      'board-1',
      expect.arrayContaining(['source.mp4'])
    );

    runtime.dispose();
  });

  it('leaves a video on the board its node saved it to', async () => {
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['node-board.mp4']),
      videoBoards: { 'node-board.mp4': 'board-b' },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({
          videos: [expect.objectContaining({ boardId: 'board-b', videoName: 'node-board.mp4' })],
        })
      );
    });
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();

    runtime.dispose();
  });

  it('attaches only the unassigned videos of a mixed batch and reports each final board', async () => {
    const { commands, destinations, runtime } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['node-board.mp4', 'unassigned.mp4']),
      videoBoards: { 'node-board.mp4': 'board-b' },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({
          videos: [
            expect.objectContaining({ boardId: 'board-b', videoName: 'node-board.mp4' }),
            expect.objectContaining({ boardId: 'board-1', videoName: 'unassigned.mp4' }),
          ],
        })
      );
    });
    expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledExactlyOnceWith('board-1', ['unassigned.mp4']);

    runtime.dispose();
  });

  it('does not move a video back after the user moves it between partial and final settlement', async () => {
    const { commands, destinations, runtime, videoBoards } = createHarness({
      getResultVideoNames: vi.fn().mockResolvedValue(['clip-1.mp4']),
    });
    vi.mocked(commands.routePartialResults).mockImplementation(() => {
      videoBoards.set('clip-1.mp4', 'board-c');
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({ videos: [expect.objectContaining({ boardId: 'board-c', videoName: 'clip-1.mp4' })] })
      );
    });
    expect(commands.routePartialResults).toHaveBeenCalledBefore(vi.mocked(commands.routeResults));
    expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledExactlyOnceWith('board-1', ['clip-1.mp4']);

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

describe('queue runtime retries', () => {
  const createRetryHarness = (
    queueItem: QueueItem,
    backendOverrides: Partial<QueueBackendPort> = {},
    services: { journal?: QueueRunJournalPort; locks?: QueueRunLockPort; viewerUserId?: string | null } = {}
  ) => {
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const projects = [project];
    const listeners = new Set<() => void>();
    const notify = () => listeners.forEach((listener) => listener());
    const closeProject = () => {
      projects.splice(0);
      notify();
    };
    const handlers = new Map<string, Set<(payload: never) => void>>();
    const backend = createTestBackend({
      on: vi.fn((event: string, handler: (payload: never) => void) => {
        const eventHandlers = handlers.get(event) ?? new Set();
        eventHandlers.add(handler);
        handlers.set(event, eventHandlers);
        return () => eventHandlers.delete(handler);
      }),
      ...backendOverrides,
    });
    const commands = createTestCommands({
      routeResults: vi.fn(() => {
        queueItem.status = 'completed';
        notify();
      }),
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      ...services,
      backend,
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
    });
    const emit = (event: string, payload: unknown) =>
      handlers.get(event)?.forEach((handler) => handler(payload as never));

    return { backend, closeProject, commands, emit, notify, project, runtime };
  };

  const retriedEvent = (pairs: { item_id: number; source_item_id: number }[]) => ({
    queue_id: 'default',
    retried_item_ids: pairs.map((pair) => pair.source_item_id),
    retried_item_ids_by_user: {},
    retried_items: pairs,
    timestamp: 0,
    user_ids: [],
  });

  const completedEvent = (itemId: number, origin: string, destination: string) => ({
    batch_id: 'batch',
    completed_at: null,
    created_at: '2026-07-17T00:00:00.000Z',
    destination,
    error_message: null,
    error_traceback: null,
    error_type: null,
    item_id: itemId,
    origin,
    queue_id: 'default',
    session_id: `session-${itemId}`,
    started_at: null,
    status: 'completed',
    status_sequence: null,
    timestamp: 0,
    updated_at: '2026-07-17T00:00:00.000Z',
    user_id: 'user',
  });

  it('reopens the run that owns a retried item and routes its rerun to the run destination', async () => {
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendBatchId: 'batch',
      backendItemIds: [41],
      error: 'Out of memory',
      status: 'failed',
    };
    const rerunImage = { imageName: 'rerun.png' } as QueueResultImage;
    const getResultImages = vi.fn().mockResolvedValue([rerunImage]);
    const { commands, emit, notify, project, runtime } = createRetryHarness(queueItem, {
      getItem: vi.fn().mockResolvedValue({
        batchId: 'batch',
        id: 99,
        origin: buildQueueItemOrigin(queueItem.id, 'project-1'),
        status: 'completed',
      }),
      getResultImages,
    });
    vi.mocked(commands.reopenForRetry).mockImplementation(({ retriedItems }) => {
      queueItem.backendItemIds = retriedItems.map(({ itemId }) => itemId);
      queueItem.status = 'running';
      delete queueItem.error;
      notify();
    });

    runtime.start();
    emit('queue_items_retried', retriedEvent([{ item_id: 99, source_item_id: 41 }]));

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalledWith(
        expect.objectContaining({ images: [rerunImage], projectId: project.id, queueItemId: queueItem.id })
      );
    });
    expect(commands.reopenForRetry).toHaveBeenCalledWith({
      projectId: project.id,
      queueItemId: queueItem.id,
      retriedItems: [{ itemId: 99, sourceItemId: 41 }],
    });
    expect(getResultImages).toHaveBeenCalledWith(99, queueItem.id, queueItem.snapshot.submittedAt, {
      resultNodeIds: ['canvas_output'],
    });
    expect(commands.stageRetriedCanvasResults).not.toHaveBeenCalled();

    await runtime.dispose();
  });

  it('stages reruns through the workbench at the run bbox, including a later retry of an item the first left', async () => {
    let state = createInitialWorkbenchState();
    const projectId = state.activeProjectId;
    const { canvas } = state.projects.find((project) => project.id === projectId)!;
    const submittedBbox = { ...canvas.document.bbox };
    const queueItem: WorkbenchQueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [41, 42],
      snapshot: {
        ...createPendingQueueItem().snapshot,
        canvas: {
          document: { bbox: submittedBbox, height: canvas.document.height, width: canvas.document.width },
          documentRevision: canvas.documentRevision,
        },
      },
      status: 'failed',
    };
    state = {
      ...state,
      projects: state.projects.map((project) =>
        project.id === projectId ? { ...project, queue: { items: [queueItem] } } : project
      ),
    };
    const listeners = new Set<() => void>();
    const dispatch = (action: WorkbenchAction): void => {
      state = workbenchReducer(state, action);
      listeners.forEach((listener) => listener());
    };
    dispatch({
      mutation: { bbox: { ...submittedBbox, x: submittedBbox.x + 64 }, type: 'setCanvasBbox' },
      projectId,
      type: 'applyCanvasProjectMutation',
    });
    const handlers = new Map<string, Set<(payload: never) => void>>();
    const backend = createTestBackend({
      getItem: vi.fn((id: number) =>
        Promise.resolve({
          batchId: 'batch',
          id,
          origin: buildQueueItemOrigin(queueItem.id, projectId),
          status: 'completed' as const,
        })
      ),
      getResultImages: vi.fn((id: number) =>
        Promise.resolve([
          {
            height: 64,
            imageName: `rerun-${id}.png`,
            imageUrl: `/rerun-${id}.png`,
            queuedAt: queueItem.snapshot.submittedAt,
            sourceQueueItemId: queueItem.id,
            thumbnailUrl: `/rerun-${id}.png`,
            width: 64,
          },
        ])
      ),
      on: vi.fn((event: string, handler: (payload: never) => void) => {
        const eventHandlers = handlers.get(event) ?? new Set();
        eventHandlers.add(handler);
        handlers.set(event, eventHandlers);
        return () => eventHandlers.delete(handler);
      }),
    });
    const commands = createTestCommands({
      markBackendCancelled: (payload) => dispatch({ ...payload, type: 'markQueueItemBackendCancelled' }),
      markBackendSubmitted: (payload) => dispatch({ ...payload, type: 'markQueueItemBackendSubmitted' }),
      reopenForRetry: (payload) => dispatch({ ...payload, type: 'reopenQueueItemForRetry' }),
      routePartialResults: (payload) => dispatch({ ...payload, type: 'routeQueueItemPartialResults' }),
      routeResults: (payload) => dispatch({ ...payload, type: 'routeQueueItemResults' }),
      setStatus: (payload) => dispatch({ ...payload, type: 'setQueueItemStatus' }),
      stageRetriedCanvasResults: vi.fn(),
    });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend,
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: state.projects }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
    });
    const emitRetried = (itemId: number, sourceItemId: number) =>
      handlers
        .get('queue_items_retried')
        ?.forEach((handler) => handler(retriedEvent([{ item_id: itemId, source_item_id: sourceItemId }]) as never));
    const getRun = () => state.projects.find((project) => project.id === projectId)!.queue.items[0]!;
    const getStaged = () =>
      state.projects.find((project) => project.id === projectId)!.canvas.stagingArea.pendingImages;

    runtime.start();
    emitRetried(99, 41);
    await vi.waitFor(() => expect(getRun().status).toBe('completed'));
    emitRetried(100, 42);
    await vi.waitFor(() => expect(getStaged().map((image) => image.imageName)).toContain('rerun-100.png'));

    expect(getStaged()).toEqual([
      expect.objectContaining({
        imageName: 'rerun-99.png',
        placement: expect.objectContaining({ x: submittedBbox.x }),
      }),
      expect.objectContaining({
        imageName: 'rerun-100.png',
        placement: expect.objectContaining({ x: submittedBbox.x }),
      }),
    ]);
    expect(commands.stageRetriedCanvasResults).not.toHaveBeenCalled();

    await runtime.dispose();
  });

  it("stages what another tab's Workflow rerun produced, and nothing for a rerun that failed", async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [7], status: 'completed' };
    const image = (imageName: string, createdAt: string, isIntermediate = false) =>
      ({ createdAt, imageName, isIntermediate }) as QueueResultImage;
    const produced = image('produced.png', '2026-07-17T00:00:05.000Z');
    const getResultImages = vi.fn((_id: number, _queueItemId: string, _queuedAt: string, options?: object) =>
      Promise.resolve(
        options
          ? []
          : [
              produced,
              image('echoed-input.png', '2026-07-16T00:00:00.000Z'),
              image('preview.png', '2026-07-17T00:00:04.000Z', true),
            ]
      )
    );
    const { commands, emit, project, runtime } = createRetryHarness(queueItem, { getResultImages });
    const otherTabOrigin = buildQueueItemOrigin('run-in-another-tab', project.id);

    runtime.start();
    emit(
      'queue_items_retried',
      retriedEvent([
        { item_id: 99, source_item_id: 41 },
        { item_id: 100, source_item_id: 42 },
      ])
    );
    emit('queue_item_status_changed', { ...completedEvent(100, otherTabOrigin, 'canvas'), status: 'failed' });
    emit('queue_item_status_changed', completedEvent(99, otherTabOrigin, 'canvas'));

    await vi.waitFor(() => {
      expect(commands.stageRetriedCanvasResults).toHaveBeenCalledWith({
        backendItemId: 99,
        images: [produced],
        projectId: project.id,
        queueItemId: 'run-in-another-tab',
      });
    });
    expect(getResultImages).toHaveBeenNthCalledWith(1, 99, 'run-in-another-tab', '2026-07-17T00:00:00.000Z', {
      resultNodeIds: ['canvas_output', 'upscale_output'],
    });
    expect(commands.stageRetriedCanvasResults).toHaveBeenCalledTimes(1);

    await runtime.dispose();
  });

  it("adds a Gallery rerun its run could not adopt to that run's board", async () => {
    // Still running from an earlier retry, so this rerun of its other failed item cannot reopen it.
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [99],
      retriedBackendItemIds: [41, 42],
      snapshot: { ...createPendingQueueItem().snapshot, destination: 'gallery', galleryBoardId: 'board-1' },
      status: 'running',
    };
    const getResultImages = vi.fn().mockResolvedValue([{ imageName: 'rerun.png' } as QueueResultImage]);
    const { commands, emit, project, runtime } = createRetryHarness(queueItem, { getResultImages });
    // The destination port is shared by every test in this file.
    vi.mocked(runtimeServices.destinations.addImagesToGalleryBoard).mockClear();

    runtime.start();
    emit('queue_items_retried', retriedEvent([{ item_id: 100, source_item_id: 42 }]));
    emit('queue_item_status_changed', completedEvent(100, buildQueueItemOrigin(queueItem.id, project.id), 'gallery'));

    await vi.waitFor(() => expect(commands.refreshBackendData).toHaveBeenCalled());
    expect(commands.reopenForRetry).toHaveBeenCalledWith(expect.objectContaining({ queueItemId: queueItem.id }));
    expect(runtimeServices.destinations.addImagesToGalleryBoard).toHaveBeenCalledWith('board-1', ['rerun.png']);

    await runtime.dispose();
  });

  it("adds a video rerun its active run could not adopt to that run's board", async () => {
    // A second failed item of a video batch, retried while the first item's rerun still runs.
    const queueItem: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [99],
      retriedBackendItemIds: [41, 42],
      snapshot: {
        ...createPendingQueueItem().snapshot,
        destination: 'gallery',
        galleryBoardId: 'board-1',
        resultNodeIds: ['video_output'],
        sourceId: 'video',
      },
      status: 'running',
    };
    const getResultVideoNames = vi.fn().mockResolvedValue(['rerun.mp4']);
    const getResultVideos = vi.fn().mockResolvedValue([{ videoName: 'rerun.mp4' } as QueueResultVideo]);
    const { commands, emit, project, runtime } = createRetryHarness(queueItem, {
      getResultVideoNames,
      getResultVideos,
    });
    // The destination port is shared by every test in this file.
    vi.mocked(runtimeServices.destinations.addVideosToGalleryBoard).mockClear();

    runtime.start();
    emit('queue_items_retried', retriedEvent([{ item_id: 100, source_item_id: 42 }]));
    emit('queue_item_status_changed', completedEvent(100, buildQueueItemOrigin(queueItem.id, project.id), 'gallery'));

    await vi.waitFor(() => expect(commands.refreshBackendData).toHaveBeenCalled());
    expect(getResultVideoNames).toHaveBeenCalledWith(100, { resultNodeIds: ['video_output'] });
    expect(runtimeServices.destinations.addVideosToGalleryBoard).toHaveBeenCalledWith('board-1', ['rerun.mp4']);

    await runtime.dispose();
  });

  it("waits for the tab tracking a Canvas rerun's run, and stages the rerun if that tab lets the run go without it", async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [7], status: 'completed' };
    const getResultImages = vi.fn().mockResolvedValue([{ imageName: 'rerun.png' } as QueueResultImage]);
    let letRunGo: (() => void) | undefined;
    // Another tab tracks the run behind rerun 99; nobody tracks the one behind rerun 100 any more.
    const locks = createTestLocks(vi.fn(), {
      isHeld: (_projectId, queueItemId) => Promise.resolve(queueItemId === 'tracked-elsewhere'),
      waitForRunRelease: () =>
        new Promise((resolve) => {
          letRunGo = resolve;
        }),
    });
    const { commands, emit, project, runtime } = createRetryHarness(queueItem, { getResultImages }, { locks });
    const staged = () =>
      vi.mocked(commands.stageRetriedCanvasResults).mock.calls.map(([payload]) => payload.backendItemId);

    runtime.start();
    emit(
      'queue_items_retried',
      retriedEvent([
        { item_id: 99, source_item_id: 41 },
        { item_id: 100, source_item_id: 42 },
      ])
    );
    emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin('tracked-elsewhere', project.id), 'canvas')
    );
    emit('queue_item_status_changed', completedEvent(100, buildQueueItemOrigin('run-gone', project.id), 'canvas'));

    await vi.waitFor(() => expect(staged()).toEqual([100]));
    await settleEvents();
    expect(staged()).toEqual([100]);
    expect(getResultImages).toHaveBeenCalledTimes(1);
    letRunGo?.();

    await vi.waitFor(() => expect(staged()).toEqual([100, 99]));
    expect(locks.acquire).not.toHaveBeenCalled();

    await runtime.dispose();
  });

  /**
   * The server's delivery claims, which every tab and browser of the account shares. A claim lasts `ttlMs` here, so a
   * claimant turned away asks again within a test.
   */
  const createDeliveryServer = (ttlMs = 200) => {
    const claims = new Map<number, { claimant: string; claimedAt: number; isDelivered: boolean }>();
    const claimItemDelivery: QueueBackendPort['claimItemDelivery'] = vi.fn((itemId, action, claimant) => {
      const claim = claims.get(itemId);
      if (action === 'confirm') {
        claims.set(itemId, { claimant, claimedAt: Date.now(), isDelivered: true });
        return Promise.resolve({ kind: 'delivered' } as const);
      }
      if (action === 'release') {
        if (claim !== undefined && claim.claimant === claimant && !claim.isDelivered) {
          claims.delete(itemId);
        }
        return Promise.resolve({ kind: 'released' } as const);
      }
      if (claim?.isDelivered) {
        return Promise.resolve({ kind: 'delivered' } as const);
      }
      const age = Date.now() - (claim?.claimedAt ?? 0);
      if (claim !== undefined && claim.claimant !== claimant && age < ttlMs) {
        return Promise.resolve({ kind: 'held', retryInMs: ttlMs - age } as const);
      }
      claims.set(itemId, { claimant, claimedAt: Date.now(), isDelivered: false });
      return Promise.resolve({ kind: 'claimed' } as const);
    });
    return { claimItemDelivery, claims };
  };
  const rerunImages = (itemId: number) => Promise.resolve([{ imageName: `rerun-${itemId}.png` } as QueueResultImage]);
  const openCanvasTab = (
    server: ReturnType<typeof createDeliveryServer>,
    getResultImages: QueueBackendPort['getResultImages'] = vi.fn(rerunImages),
    locks: QueueRunLockPort = createTestLocks(vi.fn())
  ) =>
    createRetryHarness(
      { ...createPendingQueueItem(), backendItemIds: [7], status: 'completed' },
      { claimItemDelivery: server.claimItemDelivery, getResultImages },
      { locks }
    );
  /** A tab whose failed Canvas run reopens around the reruns its retry reports, and reads them as completed. */
  const openAdoptingTab = (
    server: ReturnType<typeof createDeliveryServer>,
    run: QueueItem = { ...createPendingQueueItem(), backendItemIds: [41], status: 'failed' }
  ) => {
    const tab = createRetryHarness(run, {
      claimItemDelivery: server.claimItemDelivery,
      getItem: vi.fn((id: number) =>
        Promise.resolve({
          batchId: 'batch',
          id,
          origin: buildQueueItemOrigin(run.id, 'project-1'),
          status: 'completed' as const,
        })
      ),
      getResultImages: vi.fn(rerunImages),
    });
    vi.mocked(tab.commands.reopenForRetry).mockImplementation(({ retriedItems }) => {
      run.retriedBackendItemIds = run.backendItemIds;
      run.backendItemIds = retriedItems.map(({ itemId }) => itemId);
      run.status = 'running';
      tab.notify();
    });
    return { ...tab, run };
  };
  /** The rerun images a tab routed or staged into its Canvas. */
  const stagedImageNames = (tab: ReturnType<typeof createRetryHarness>) =>
    [
      ...vi.mocked(tab.commands.stageRetriedCanvasResults).mock.calls.flatMap(([payload]) => payload.images),
      ...vi.mocked(tab.commands.routePartialResults).mock.calls.flatMap(([payload]) => payload.images),
      ...vi.mocked(tab.commands.routeResults).mock.calls.flatMap(([payload]) => payload.images),
    ].map((image) => image.imageName);
  const stagedItemIds = (tab: ReturnType<typeof createRetryHarness>) =>
    vi.mocked(tab.commands.stageRetriedCanvasResults).mock.calls.map(([payload]) => payload.backendItemId);
  const emitReruns = (tab: ReturnType<typeof createRetryHarness>, itemIds: number[]) =>
    tab.emit(
      'queue_items_retried',
      retriedEvent(itemIds.map((itemId) => ({ item_id: itemId, source_item_id: itemId + 1_000 })))
    );
  const settleEvents = () =>
    new Promise((resolve) => {
      setTimeout(resolve);
    });

  it('stages a Canvas rerun once across tabs and browsers, however late another sees it and after the first closed', async () => {
    const server = createDeliveryServer();
    const first = openCanvasTab(server);
    const second = openCanvasTab(server);
    const completed = (itemId: number) =>
      completedEvent(itemId, buildQueueItemOrigin('run-gone', first.project.id), 'canvas');

    for (const tab of [first, second]) {
      tab.runtime.start();
      emitReruns(tab, [99, 100, 101]);
    }
    first.emit('queue_item_status_changed', completed(99));
    first.emit('queue_item_status_changed', completed(100));
    await vi.waitFor(() => expect(stagedItemIds(first).sort((left, right) => left - right)).toEqual([99, 100]));
    await first.runtime.dispose();

    for (const itemId of [99, 100, 101]) {
      second.emit('queue_item_status_changed', completed(itemId));
    }
    await vi.waitFor(() => expect(stagedItemIds(second)).toEqual([101]));
    await settleEvents();
    expect(stagedItemIds(second)).toEqual([101]);

    await second.runtime.dispose();
  });

  it('stages a Canvas rerun once when tabs handle its completion at the same moment', async () => {
    const server = createDeliveryServer();
    const tabs = [0, 1, 2].map(() => openCanvasTab(server));
    const completed = completedEvent(99, buildQueueItemOrigin('run-gone', tabs[0]!.project.id), 'canvas');

    for (const tab of tabs) {
      tab.runtime.start();
      emitReruns(tab, [99]);
    }
    for (const tab of tabs) {
      tab.emit('queue_item_status_changed', completed);
    }

    await vi.waitFor(() => expect(tabs.flatMap(stagedItemIds)).toEqual([99]));
    // The winner claims, checks its claim before staging and confirms; the others ask again until it is confirmed.
    await vi.waitFor(() => expect(server.claimItemDelivery).toHaveBeenCalledTimes(7));
    expect(tabs.flatMap(stagedItemIds)).toEqual([99]);

    await Promise.all(tabs.map((tab) => tab.runtime.dispose()));
  });

  it('stages a Canvas rerun after the tab that claimed it closed before staging, once the claim expires', async () => {
    const server = createDeliveryServer();
    // This tab closes while it reads the results, so it neither stages nor gives the claim back.
    const closing = openCanvasTab(
      server,
      vi.fn(
        () =>
          new Promise<QueueResultImage[]>(() => {
            // Never settles: the tab is gone.
          })
      )
    );
    const open = openCanvasTab(server);
    const completed = completedEvent(99, buildQueueItemOrigin('run-gone', closing.project.id), 'canvas');

    for (const tab of [closing, open]) {
      tab.runtime.start();
      emitReruns(tab, [99]);
    }
    closing.emit('queue_item_status_changed', completed);
    await vi.waitFor(() => expect(server.claims.has(99)).toBe(true));
    open.emit('queue_item_status_changed', completed);
    await settleEvents();
    expect(stagedItemIds(open)).toEqual([]);

    await vi.waitFor(() => expect(stagedItemIds(open)).toEqual([99]));

    await open.runtime.dispose();
  });

  it('leaves a Canvas rerun to a tab that sees it later when this one could not read its results', async () => {
    // Claims outlast the test, so only giving the claim back lets the later tab stage it.
    const server = createDeliveryServer(60_000);
    const failing = openCanvasTab(server, vi.fn().mockRejectedValue(new Error('Server restarting')));
    const later = openCanvasTab(server);
    const completed = completedEvent(99, buildQueueItemOrigin('run-gone', failing.project.id), 'canvas');

    for (const tab of [failing, later]) {
      tab.runtime.start();
      emitReruns(tab, [99]);
    }
    failing.emit('queue_item_status_changed', completed);
    await vi.waitFor(() =>
      expect(failing.commands.recordError).toHaveBeenCalledWith(expect.objectContaining({ area: 'queue-results' }))
    );
    later.emit('queue_item_status_changed', completed);

    await vi.waitFor(() => expect(stagedItemIds(later)).toEqual([99]));
    expect(stagedItemIds(failing)).toEqual([]);

    await failing.runtime.dispose();
    await later.runtime.dispose();
  });

  it('stages a Canvas rerun when the server keeps no delivery claims', async () => {
    const tab = openCanvasTab({
      claimItemDelivery: vi.fn().mockRejectedValue(new ApiError('Not Found', 404)),
      claims: new Map(),
    });

    tab.runtime.start();
    emitReruns(tab, [99]);
    tab.emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin('run-gone', tab.project.id), 'canvas')
    );

    await vi.waitFor(() => expect(stagedItemIds(tab)).toEqual([99]));

    await tab.runtime.dispose();
  });

  it('asks again after a failed claim instead of staging a rerun another tab staged meanwhile', async () => {
    const server = createDeliveryServer();
    server.claims.set(99, { claimant: 'tab:other', claimedAt: Date.now(), isDelivered: true });
    const claimItemDelivery = vi.mocked(server.claimItemDelivery);
    claimItemDelivery.mockRejectedValueOnce(new Error('Failed to fetch'));
    const tab = openCanvasTab(server);

    tab.runtime.start();
    emitReruns(tab, [99]);
    tab.emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin('run-gone', tab.project.id), 'canvas')
    );

    await vi.waitFor(() => expect(claimItemDelivery).toHaveBeenCalledTimes(2), { timeout: 6_000 });
    await settleEvents();
    expect(stagedItemIds(tab)).toEqual([]);

    await tab.runtime.dispose();
  }, 10_000);

  it('stages a Canvas rerun that finishes while this tab is still freeing the run lock it gave up', async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [88], status: 'running' };
    let held = false;
    let freeRunLock: (() => void) | undefined;
    const locks = createTestLocks(
      vi.fn(() => {
        held = true;
        return Promise.resolve({
          kind: 'acquired',
          release: () =>
            new Promise<void>((resolve) => {
              freeRunLock = () => {
                held = false;
                resolve();
              };
            }),
        } as const);
      }),
      { isHeld: () => Promise.resolve(held) }
    );
    const { commands, emit, notify, project, runtime } = createRetryHarness(
      queueItem,
      { getResultImages: vi.fn().mockResolvedValue([{ imageName: 'rerun.png' } as QueueResultImage]) },
      { locks }
    );

    runtime.start();
    await vi.waitFor(() => expect(held).toBe(true));
    emit('queue_items_retried', retriedEvent([{ item_id: 100, source_item_id: 42 }]));
    // Removing the run gives up its lock; the browser frees it only later.
    project.queue.items = [];
    notify();
    await vi.waitFor(() => expect(freeRunLock).toBeDefined());
    emit('queue_item_status_changed', completedEvent(100, buildQueueItemOrigin(queueItem.id, project.id), 'canvas'));

    await vi.waitFor(() =>
      expect(commands.stageRetriedCanvasResults).toHaveBeenCalledWith(
        expect.objectContaining({ backendItemId: 100, queueItemId: queueItem.id })
      )
    );
    expect(held).toBe(true);

    freeRunLock?.();
    await runtime.dispose();
  });

  it('leaves a Canvas rerun to another tab when its project closed here while the results loaded', async () => {
    const server = createDeliveryServer();
    let finishRead!: (images: QueueResultImage[]) => void;
    const closing = openCanvasTab(
      server,
      vi.fn(
        () =>
          new Promise<QueueResultImage[]>((resolve) => {
            finishRead = resolve;
          })
      )
    );
    const open = openCanvasTab(server);
    const completed = completedEvent(99, buildQueueItemOrigin('run-gone', closing.project.id), 'canvas');

    for (const tab of [closing, open]) {
      tab.runtime.start();
      emitReruns(tab, [99]);
    }
    closing.emit('queue_item_status_changed', completed);
    await vi.waitFor(() => expect(finishRead).toBeDefined());
    closing.closeProject();
    finishRead([{ imageName: 'rerun-99.png' } as QueueResultImage]);
    await settleEvents();
    open.emit('queue_item_status_changed', completed);

    await vi.waitFor(() => expect(stagedItemIds(open)).toEqual([99]));
    expect(stagedItemIds(closing)).toEqual([]);

    await closing.runtime.dispose();
    await open.runtime.dispose();
  });

  it('stages a rerun at its run in one tab while another tab leaves it, whichever handles it first', async () => {
    const server = createDeliveryServer();
    const adopting = openAdoptingTab(server);
    const other = openCanvasTab(server);

    for (const tab of [adopting, other]) {
      tab.runtime.start();
      tab.emit('queue_items_retried', retriedEvent([{ item_id: 99, source_item_id: 41 }]));
    }
    await vi.waitFor(() => expect(stagedImageNames(adopting)).toContain('rerun-99.png'));
    await vi.waitFor(() => expect(server.claims.get(99)?.isDelivered).toBe(true));
    other.emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin(adopting.run.id, other.project.id), 'canvas')
    );
    await vi.waitFor(() =>
      expect(server.claimItemDelivery).toHaveBeenCalledWith(99, 'claim', expect.stringMatching(/^tab:/))
    );
    await settleEvents();

    expect(stagedItemIds(other)).toEqual([]);

    await adopting.runtime.dispose();
    await other.runtime.dispose();
  });

  it('leaves a rerun to the run reopened around it in another browser, even when this one sees it finish first', async () => {
    // Long enough that only the run confirming lets this browser stop asking.
    const server = createDeliveryServer(60_000);
    const adopting = openAdoptingTab(server);
    vi.mocked(adopting.backend.getItem).mockResolvedValue({
      batchId: 'batch',
      id: 99,
      origin: buildQueueItemOrigin(adopting.run.id, 'project-1'),
      status: 'in_progress',
    });
    const other = openCanvasTab(server);
    const completed = completedEvent(99, buildQueueItemOrigin(adopting.run.id, other.project.id), 'canvas');

    for (const tab of [adopting, other]) {
      tab.runtime.start();
      tab.emit('queue_items_retried', retriedEvent([{ item_id: 99, source_item_id: 41 }]));
    }
    // The run claims its rerun as soon as its tab holds it, long before the rerun finishes.
    await vi.waitFor(() => expect(server.claims.get(99)?.claimant).toMatch(/^run:/));
    other.emit('queue_item_status_changed', completed);
    await vi.waitFor(() =>
      expect(server.claimItemDelivery).toHaveBeenCalledWith(99, 'claim', expect.stringMatching(/^tab:/))
    );
    adopting.emit('queue_item_status_changed', completed);

    await vi.waitFor(() => expect(stagedImageNames(adopting)).toContain('rerun-99.png'));
    expect(server.claims.get(99)?.isDelivered).toBe(true);
    await vi.waitFor(
      () =>
        expect(
          vi.mocked(server.claimItemDelivery).mock.calls.filter(([, , claimant]) => claimant.startsWith('tab:'))
        ).toHaveLength(2),
      { timeout: 6_000 }
    );
    await settleEvents();
    expect(stagedItemIds(other)).toEqual([]);

    await adopting.runtime.dispose();
    await other.runtime.dispose();
  }, 10_000);

  it('keeps the claim of a rerun its run is still waiting on', async () => {
    vi.useFakeTimers({ toFake: ['setInterval'] });
    try {
      const server = createDeliveryServer(60_000);
      const adopting = openAdoptingTab(server);
      vi.mocked(adopting.backend.getItem).mockResolvedValue({
        batchId: 'batch',
        id: 99,
        origin: buildQueueItemOrigin(adopting.run.id, 'project-1'),
        status: 'pending',
      });

      adopting.runtime.start();
      adopting.emit('queue_items_retried', retriedEvent([{ item_id: 99, source_item_id: 41 }]));
      await vi.waitFor(() => expect(server.claims.get(99)?.claimant).toMatch(/^run:/));
      await settleEvents();
      // The rerun waits in the queue for longer than an unrenewed claim would last.
      server.claims.set(99, { ...server.claims.get(99)!, claimedAt: Date.now() - 50_000 });
      vi.advanceTimersByTime(40_000);

      await vi.waitFor(() => expect(Date.now() - server.claims.get(99)!.claimedAt).toBeLessThan(10_000));

      await adopting.runtime.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('gives back the claim of a rerun whose project closed before its run staged it', async () => {
    // Claims outlast the test, so only giving the claim back lets the other tab stage it.
    const server = createDeliveryServer(60_000);
    const adopting = openAdoptingTab(server);
    vi.mocked(adopting.backend.getItem).mockResolvedValue({
      batchId: 'batch',
      id: 99,
      origin: buildQueueItemOrigin(adopting.run.id, 'project-1'),
      status: 'in_progress',
    });
    const other = openCanvasTab(server);

    for (const tab of [adopting, other]) {
      tab.runtime.start();
      tab.emit('queue_items_retried', retriedEvent([{ item_id: 99, source_item_id: 41 }]));
    }
    await vi.waitFor(() => expect(server.claims.get(99)?.claimant).toMatch(/^run:/));
    adopting.closeProject();
    await vi.waitFor(() => expect(server.claims.has(99)).toBe(false));
    other.emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin(adopting.run.id, other.project.id), 'canvas')
    );

    await vi.waitFor(() => expect(stagedItemIds(other)).toEqual([99]));

    await adopting.runtime.dispose();
    await other.runtime.dispose();
  });

  it('does not stage a rerun whose claim lapsed while its results loaded and another tab staged meanwhile', async () => {
    const server = createDeliveryServer();
    let finishRead: ((images: QueueResultImage[]) => void) | undefined;
    const slow = openCanvasTab(
      server,
      vi.fn(
        () =>
          new Promise<QueueResultImage[]>((resolve) => {
            finishRead = resolve;
          })
      )
    );
    const other = openCanvasTab(server);
    const completed = completedEvent(99, buildQueueItemOrigin('run-gone', slow.project.id), 'canvas');

    for (const tab of [slow, other]) {
      tab.runtime.start();
      emitReruns(tab, [99]);
    }
    slow.emit('queue_item_status_changed', completed);
    await vi.waitFor(() => expect(finishRead).toBeDefined());
    other.emit('queue_item_status_changed', completed);
    await vi.waitFor(() => expect(stagedItemIds(other)).toEqual([99]));
    finishRead?.([{ imageName: 'rerun-99.png' } as QueueResultImage]);
    await settleEvents();
    await settleEvents();

    expect(stagedItemIds(slow)).toEqual([]);

    await slow.runtime.dispose();
    await other.runtime.dispose();
  });

  it('leaves out of a reopened run a rerun another tab or browser staged first', async () => {
    const server = createDeliveryServer();
    const staging = openCanvasTab(server);
    // Reopened around rerun 99 in a tab that closed, or by a retry this tab learned of late, and restored here.
    const restored = openAdoptingTab(server, {
      ...createPendingQueueItem(),
      backendItemIds: [99],
      retriedBackendItemIds: [41],
      status: 'running',
    });

    staging.runtime.start();
    emitReruns(staging, [99]);
    staging.emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin(restored.run.id, staging.project.id), 'canvas')
    );
    await vi.waitFor(() => expect(stagedItemIds(staging)).toEqual([99]));
    restored.runtime.start();

    await vi.waitFor(() => expect(restored.commands.routeResults).toHaveBeenCalled());
    expect(stagedImageNames(restored)).toEqual([]);

    await staging.runtime.dispose();
    await restored.runtime.dispose();
  });

  it('stages a rerun in the run restored after its tab closed while the run held the claim', async () => {
    const server = createDeliveryServer(60_000);
    const run: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [99],
      retriedBackendItemIds: [41],
      status: 'running',
    };
    // The run's tab claimed the rerun and closed before confirming; the reload restores the same run.
    server.claims.set(99, {
      claimant: `run:${JSON.stringify(['project-1', run.id])}`,
      claimedAt: Date.now(),
      isDelivered: false,
    });
    const restored = openAdoptingTab(server, run);

    restored.runtime.start();

    await vi.waitFor(() => expect(stagedImageNames(restored)).toContain('rerun-99.png'));
    await vi.waitFor(() => expect(server.claims.get(99)?.isDelivered).toBe(true));

    await restored.runtime.dispose();
  });

  it('gives back the claim of a rerun whose run ends before staging it, so another tab stages it', async () => {
    // Claims outlast the test, so only giving the claim back lets the other tab stage it.
    const server = createDeliveryServer(60_000);
    const adopting = openAdoptingTab(server);
    let finishRead: ((images: QueueResultImage[]) => void) | undefined;
    // The run is removed while it reads.
    vi.mocked(adopting.backend.getResultImages).mockReturnValue(
      new Promise((resolve) => {
        finishRead = resolve;
      })
    );
    const other = openCanvasTab(server);

    for (const tab of [adopting, other]) {
      tab.runtime.start();
      tab.emit('queue_items_retried', retriedEvent([{ item_id: 99, source_item_id: 41 }]));
    }
    await vi.waitFor(() => expect(server.claims.get(99)?.claimant).toMatch(/^run:/));
    adopting.project.queue.items = [];
    adopting.notify();
    other.emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin(adopting.run.id, other.project.id), 'canvas')
    );

    await vi.waitFor(() => expect(stagedItemIds(other)).toEqual([99]));

    finishRead?.([]);
    await adopting.runtime.dispose();
    await other.runtime.dispose();
  });

  it('stages a rerun its run refused to reopen around in a tab that has its project open', async () => {
    const server = createDeliveryServer();
    // Still running, so the run refuses to reopen around the rerun of its other failed item.
    const activeRun: QueueItem = {
      ...createPendingQueueItem(),
      backendItemIds: [99],
      retriedBackendItemIds: [41, 42],
      status: 'running',
    };
    const refusing = createRetryHarness(activeRun, { claimItemDelivery: server.claimItemDelivery });
    const other = openCanvasTab(server);
    const retried = retriedEvent([{ item_id: 100, source_item_id: 42 }]);

    for (const tab of [refusing, other]) {
      tab.runtime.start();
      tab.emit('queue_items_retried', retried);
    }
    await vi.waitFor(() => expect(refusing.commands.reopenForRetry).toHaveBeenCalled());
    other.emit(
      'queue_item_status_changed',
      completedEvent(100, buildQueueItemOrigin(activeRun.id, other.project.id), 'canvas')
    );

    await vi.waitFor(() => expect(stagedItemIds(other)).toEqual([100]));

    await refusing.runtime.dispose();
    await other.runtime.dispose();
  });

  it("handles a retry once when both the tab's own answer and the socket deliver it, and ignores others' retries", async () => {
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [41], status: 'failed' };
    const { commands, emit, notify, project, runtime } = createRetryHarness(queueItem, {}, { viewerUserId: 'me' });
    vi.mocked(commands.reopenForRetry).mockImplementation(({ retriedItems }) => {
      queueItem.retriedBackendItemIds = [...(queueItem.backendItemIds ?? [])];
      queueItem.backendItemIds = retriedItems.map(({ itemId }) => itemId);
      queueItem.status = 'running';
      notify();
    });
    const ownAnswer = retriedEvent([{ item_id: 99, source_item_id: 41 }]);

    runtime.start();
    emit('queue_items_retried', ownAnswer);
    // The socket copy an admin receives: the viewer's pair again, plus another user's.
    emit('queue_items_retried', {
      ...retriedEvent([
        { item_id: 99, source_item_id: 41 },
        { item_id: 200, source_item_id: 70 },
      ]),
      retried_item_ids_by_user: { me: [41], other: [70] },
      user_ids: ['me', 'other'],
    });
    emit(
      'queue_item_status_changed',
      completedEvent(200, buildQueueItemOrigin('their-run', 'their-project'), 'gallery')
    );

    await vi.waitFor(() => expect(project.queue.items[0]?.backendItemIds).toEqual([99]));
    expect(commands.reopenForRetry).toHaveBeenCalledTimes(1);
    expect(commands.refreshBackendData).not.toHaveBeenCalled();

    await runtime.dispose();
  });

  it('delivers reruns no run adopted to the Gallery or the current Canvas, and nothing else', async () => {
    // The owning run refuses to reopen (its Canvas document was replaced); the other rerun belongs to another tab.
    const queueItem: QueueItem = { ...createPendingQueueItem(), backendItemIds: [41], status: 'failed' };
    const rerunImage = { imageName: 'rerun.png' } as QueueResultImage;
    const getResultImages = vi.fn().mockResolvedValue([rerunImage]);
    const { commands, emit, project, runtime } = createRetryHarness(queueItem, { getResultImages });
    const canvasOrigin = buildQueueItemOrigin(queueItem.id, project.id);

    runtime.start();
    emit(
      'queue_items_retried',
      retriedEvent([
        { item_id: 99, source_item_id: 41 },
        { item_id: 100, source_item_id: 42 },
      ])
    );
    emit('queue_item_status_changed', completedEvent(99, canvasOrigin, 'canvas'));
    emit('queue_item_status_changed', completedEvent(100, buildQueueItemOrigin('other-run', project.id), 'gallery'));
    emit('queue_item_status_changed', completedEvent(101, canvasOrigin, 'canvas'));

    await vi.waitFor(() => {
      expect(commands.stageRetriedCanvasResults).toHaveBeenCalledWith({
        backendItemId: 99,
        images: [rerunImage],
        projectId: project.id,
        queueItemId: queueItem.id,
      });
    });
    expect(getResultImages).toHaveBeenCalledTimes(1);
    expect(getResultImages).toHaveBeenCalledWith(99, queueItem.id, '2026-07-17T00:00:00.000Z', {
      resultNodeIds: ['canvas_output'],
    });
    expect(commands.refreshBackendData).toHaveBeenCalledTimes(1);
    expect(queueItem.status).toBe('failed');

    await runtime.dispose();
  });
});
