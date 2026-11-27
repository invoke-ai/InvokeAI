import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueWorkflowBatchDatum } from '@features/queue/core/promptBatch';
import type {
  QueueBackendInvocation,
  QueueBackendPort,
  QueueResultImage,
  QueueResultVideo,
} from '@features/queue/core/types';
import type { BackendConnectionStatus } from '@platform/transport/types';
import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import { getQueueStagedResults } from '@app/queueStagedResults';
import { activeProgressTargetStore } from '@features/queue/data/activeProgressTargetStore';
import { buildQueueItemOrigin } from '@features/queue/data/events';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { createInitialWorkbenchState, type WorkbenchAction, workbenchReducer } from '@workbench/workbenchState.testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createQueueItemBackendSubmission,
  createQueueRuntime,
  type QueueHistoryCommands,
  type QueueHistoryPort,
  type QueueRunJournalPort,
  type QueueRunLock,
  type QueueRunLockPort,
} from './runtime';

// Spies wrap the real collector, so routing behaves as in production while tests can count rescans.
vi.mock('@features/queue/core/graphInputMedia', { spy: true });

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
  readItemsById: vi.fn().mockResolvedValue([]),
  readNext: vi.fn(),
  readStatus: vi.fn(),
  resumeProcessor: vi.fn(),
  retryItems: vi.fn(),
  ...overrides,
});

const createTestCommands = (overrides: Partial<QueueHistoryCommands> = {}): QueueHistoryCommands => ({
  markBackendCancelled: vi.fn(),
  noteFirstProgressFrame: vi.fn(),
  markBackendSubmitted: vi.fn(),
  recordSubmissionRetry: vi.fn(),
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

const createWorkbenchQueueHistory = () => {
  let state = createInitialWorkbenchState();
  const projectId = state.activeProjectId;
  const { canvas } = state.projects.find((project) => project.id === projectId)!;
  const pendingItem = createPendingQueueItem();
  const queueItem: WorkbenchQueueItem = {
    ...pendingItem,
    snapshot: {
      ...pendingItem.snapshot,
      canvas: {
        document: { bbox: { ...canvas.document.bbox }, height: canvas.document.height, width: canvas.document.width },
        documentRevision: canvas.documentRevision,
      },
    },
  };
  const setItems = (items: WorkbenchQueueItem[]): void => {
    state = {
      ...state,
      projects: state.projects.map((project) =>
        project.id === projectId ? { ...project, queue: { items } } : project
      ),
    };
  };
  setItems([queueItem]);
  const listeners = new Set<() => void>();
  const dispatch = (action: WorkbenchAction): void => {
    state = workbenchReducer(state, action);
    listeners.forEach((listener) => listener());
  };
  let connectionStatus: BackendConnectionStatus = 'connected';
  const commands = createTestCommands({
    setConnectionStatus: ({ status }) => {
      connectionStatus = status;
    },
    markBackendSubmitted: (payload) => dispatch({ ...payload, type: 'markQueueItemBackendSubmitted' }),
    setCancellationPending: (payload) => dispatch({ ...payload, type: 'setQueueItemCancellationPending' }),
    recordSubmissionRetry: (payload) => dispatch({ ...payload, type: 'recordQueueItemSubmissionRetry' }),
    restoreFromJournal: (payload) => dispatch({ ...payload, type: 'restoreQueueItemsFromJournal' }),
    setLocalRecoveryState: (payload) => dispatch({ ...payload, type: 'setQueueItemLocalRecoveryState' }),
    setStatus: vi.fn<QueueHistoryCommands['setStatus']>((payload) =>
      dispatch({ ...payload, type: 'setQueueItemStatus' })
    ),
  });
  return {
    dispatch,
    acceptedItem: {
      batchId: 'backend-batch',
      id: 88,
      origin: buildQueueItemOrigin(queueItem.id, projectId),
      status: 'pending' as const,
    },
    getRun: () =>
      state.projects.find((project) => project.id === projectId)?.queue.items.find((item) => item.id === queueItem.id),
    history: {
      commands,
      listStagedResults: () => getQueueStagedResults(state.projects),
      getSnapshot: () => ({ connectionStatus, isHydrated: true, projects: state.projects }),
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    },
    cancel: () => dispatch({ notice: null, projectId, queueItemId: queueItem.id, type: 'cancelQueueItem' }),
    /** A reload: the page forgets its runs, and only what the journal kept comes back. */
    reload: () => setItems([]),
  };
};

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

// A failing assertion must not leave fake timers or a stubbed Math.random to the next test.
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

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
    // A stepped item never takes the legacy expansion, even with a stale toggle beside it.
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
    expect(
      createQueueItemBackendSubmission(
        { id: 'project-1' },
        asWorkflow([[{ ...cfg, items: Array.from({ length: 5_001 }, (_, i) => i) }]])
      )
    ).toEqual(invalid);
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
      readItemsById: vi.fn().mockResolvedValue([]),
      readNext: vi.fn(),
      readStatus: vi.fn(),
      resumeProcessor: vi.fn(),
      retryItems: vi.fn(),
      claimItemDelivery: vi.fn().mockResolvedValue({ kind: 'claimed' }),
    };
    const commands: QueueHistoryCommands = {
      markBackendCancelled: vi.fn(),
      noteFirstProgressFrame: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      recordSubmissionRetry: vi.fn(),
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        expect.objectContaining({
          errorCode: 'project-not-saved',
          projectId: project.id,
          queueItemId: queueItem.id,
          status: 'failed',
        })
      );
    } else {
      expect(commands.setStatus).not.toHaveBeenCalled();
    }
    await runtime.dispose();
  });

  it('fails a run whose queued request is invalid with a code its notice can translate, without enqueueing', async () => {
    vi.useFakeTimers();
    const queueItem = createPendingQueueItem();
    delete (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const enqueueGenerate = vi.fn();
    const commands = createTestCommands();
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate }),
      history: {
        commands,
        listStagedResults: () => [],
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      journal: createTestJournal(),
    });

    runtime.start();
    await vi.advanceTimersByTimeAsync(1);

    expect(enqueueGenerate).not.toHaveBeenCalled();
    expect(commands.setStatus).toHaveBeenCalledWith({
      error: 'Queue item is missing a compiled backend submission.',
      errorCode: 'invalid-submission',
      projectId: project.id,
      queueItemId: queueItem.id,
      status: 'failed',
    });
    await runtime.dispose();
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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

  it('fails a run whose enqueue keeps returning a 500 instead of retrying forever', async () => {
    vi.useFakeTimers();
    const workbench = createWorkbenchQueueHistory();
    const enqueueGenerate = vi.fn().mockRejectedValue(new ApiError('graph validation crashed', 500));
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate, listItems: vi.fn().mockResolvedValue([]) }),
      history: workbench.history,
      journal: createTestJournal(),
    });

    runtime.start();

    await vi.advanceTimersByTimeAsync(20_000);
    expect(workbench.getRun()).toMatchObject({ error: 'graph validation crashed', status: 'failed' });
    expect(workbench.getRun()).not.toHaveProperty('submissionRetry');
    expect(workbench.history.commands.setStatus).toHaveBeenCalledWith(
      expect.objectContaining({ error: 'graph validation crashed', errorCode: 'server-error', status: 'failed' })
    );
    // At 0, 1, 3, 7 and 15 s: the fifth 500 is the first one at least 10 s after the first.
    expect(enqueueGenerate).toHaveBeenCalledTimes(5);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(enqueueGenerate).toHaveBeenCalledTimes(5);
    await runtime.dispose();
  });

  it('keeps the 500 streak of a run whose answer took longer than five minutes to come', async () => {
    vi.useFakeTimers();
    const workbench = createWorkbenchQueueHistory();
    const startedAt = Date.now();
    const slowServerError = () =>
      new Promise((_resolve, reject) => {
        setTimeout(() => reject(new ApiError('graph validation crashed', 500)), 6 * 60_000);
      });
    const enqueueGenerate = vi
      .fn()
      .mockRejectedValueOnce(new ApiError('graph validation crashed', 500))
      .mockImplementation(slowServerError);
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate, listItems: vi.fn().mockResolvedValue([]) }),
      history: workbench.history,
      journal: createTestJournal(),
    });

    runtime.start();
    // The second attempt starts at 1 s and its 500 arrives six minutes later.
    await vi.advanceTimersByTimeAsync(6 * 60_000 + 2_000);

    expect(enqueueGenerate).toHaveBeenCalledTimes(2);
    expect(workbench.getRun()).toMatchObject({
      status: 'pending',
      submissionRetry: { failureCount: 2, serverErrorCount: 2, serverErrorSince: startedAt },
    });
    await runtime.dispose();
  });

  it.each([
    { error: new ApiError('request timeout', 408), reason: 'unavailable', status: 408 },
    { error: new TypeError('Failed to fetch'), reason: 'unreachable', status: null },
    // An answer arrived but could not be read; the batch may have been accepted, so the run is reconciled and retried.
    { error: new SyntaxError('Unexpected token < in JSON'), reason: 'error', status: null },
  ])('keeps retrying, paced, while an enqueue fails with $error.message', async ({ error, reason, status }) => {
    vi.useFakeTimers();
    const workbench = createWorkbenchQueueHistory();
    const enqueueGenerate = vi.fn().mockRejectedValue(error);
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate, listItems: vi.fn().mockResolvedValue([]) }),
      history: workbench.history,
      journal: createTestJournal(),
    });

    runtime.start();

    // Attempts at 0, 1, 3, 7, 15, 30 and 45 s, well past the 500 window, and the run still waits for the next.
    await vi.advanceTimersByTimeAsync(59_000);
    expect(enqueueGenerate).toHaveBeenCalledTimes(7);
    expect(workbench.getRun()).toMatchObject({
      status: 'pending',
      submissionRetry: { failureCount: 7, message: error.message, reason, serverErrorCount: 0, status },
    });
    await runtime.dispose();
  });

  it('stops retrying a run cancelled while it waits to submit again', async () => {
    vi.useFakeTimers();
    const workbench = createWorkbenchQueueHistory();
    const enqueueGenerate = vi.fn().mockRejectedValue(new ApiError('unavailable', 503));
    const runtime = createQueueRuntime({
      ...runtimeServices,
      backend: createTestBackend({ enqueueGenerate, listItems: vi.fn().mockResolvedValue([]) }),
      history: workbench.history,
      journal: createTestJournal(),
    });

    runtime.start();
    // Attempts at 0 and 1 s; the next waits until 3 s.
    await vi.advanceTimersByTimeAsync(2_000);
    expect(enqueueGenerate).toHaveBeenCalledTimes(2);

    workbench.cancel();
    await vi.advanceTimersByTimeAsync(30_000);

    expect(enqueueGenerate).toHaveBeenCalledTimes(2);
    expect(workbench.getRun()).toMatchObject({ status: 'cancelled' });
    expect(workbench.getRun()?.cancellationPending).toBeUndefined();
    expect(workbench.getRun()?.error).toBeUndefined();
    expect(workbench.getRun()).not.toHaveProperty('submissionRetry');
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
        listStagedResults: () => [],
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
});

describe('queue runtime video board routing', () => {
  const createHarness = (options: {
    getResultVideoNames: QueueBackendPort['getResultVideoNames'];
    getResultVideos?: QueueBackendPort['getResultVideos'];
    galleryBoardId?: string | null;
    getResultImages?: QueueBackendPort['getResultImages'];
    graphNodes?: Record<string, QueueBackendInvocation>;
    videoBoards?: Record<string, string>;
    workflowBatchData?: QueueWorkflowBatchDatum[][];
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
    if (options.workflowBatchData && queueItem.snapshot.backendSubmission.kind !== 'invalid') {
      queueItem.snapshot.backendSubmission = {
        batchCount: 1,
        batchData: options.workflowBatchData,
        graph: queueItem.snapshot.backendSubmission.graph,
        kind: 'workflow',
      };
      queueItem.snapshot.sourceId = 'workflow';
    }
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const backend: QueueBackendPort = {
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
      noteFirstProgressFrame: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      recordSubmissionRetry: vi.fn(),
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
        listStagedResults: () => [],
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

    return { backend, commands, destinations, queueItem, runtime, videoBoards };
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
      expect(commands.routePartialResults).toHaveBeenCalledWith(routedClip);
      expect(commands.routeResults).toHaveBeenCalledWith(routedClip);
    });
    expect(destinations.addVideosToGalleryBoard).toHaveBeenCalledExactlyOnceWith('board-1', ['clip-1.mp4']);
    expect(getResultVideoNames).toHaveBeenCalledWith(77, expect.objectContaining({ excludeIntermediate: true }));

    runtime.dispose();
  });

  it('records a video-routing failure without failing the completed run', async () => {
    const getResultVideoNames = vi.fn().mockRejectedValue(new Error('transient 502'));
    const { commands, destinations, runtime } = createHarness({ getResultVideoNames });

    runtime.start();

    await vi.waitFor(() => {
      expect(commands.routeResults).toHaveBeenCalled();
      expect(commands.recordError).toHaveBeenCalledWith(expect.objectContaining({ area: 'queue-results' }));
    });
    expect(commands.setStatus).not.toHaveBeenCalledWith(expect.objectContaining({ status: 'failed' }));
    expect(destinations.addVideosToGalleryBoard).not.toHaveBeenCalled();

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
});

describe('queue runtime retries', () => {
  const createRetryHarness = (
    queueItem: QueueItem,
    backendOverrides: Partial<QueueBackendPort> = {},
    services: {
      journal?: QueueRunJournalPort;
      locks?: QueueRunLockPort;
      viewerUserId?: string | null;
      listStagedResults?: QueueHistoryPort['listStagedResults'];
    } = {}
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
    const staged: Array<{ backendItemId: number; projectId: string; queueItemId: string }> = [];
    const stageImages = (payload: {
      images: QueueResultImage[];
      backendItemId?: number;
      projectId: string;
      queueItemId: string;
    }) => {
      for (const image of payload.images) {
        const backendItemId = image.sourceBackendItemId ?? payload.backendItemId;
        if (backendItemId !== undefined) {
          staged.push({ backendItemId, projectId: payload.projectId, queueItemId: payload.queueItemId });
        }
      }
      notify();
    };
    const commands = createTestCommands({
      routePartialResults: vi.fn(stageImages),
      stageRetriedCanvasResults: vi.fn(stageImages),
      routeResults: vi.fn((payload) => {
        stageImages(payload);
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
        listStagedResults: services.listStagedResults ?? (() => staged),
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
        listStagedResults: () => getQueueStagedResults(state.projects),
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

  /** A claim lasts `ttlMs` here, so a turned-away claimant asks again within a test. */
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
  const rerunImages = (itemId: number) =>
    Promise.resolve([{ imageName: `rerun-${itemId}.png`, sourceBackendItemId: itemId } as QueueResultImage]);
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

  it('retries a lost confirmation response without staging again, including after the run ends', async () => {
    vi.useFakeTimers();
    const server = createDeliveryServer(120_000);
    const original = server.claimItemDelivery;
    let confirmations = 0;
    server.claimItemDelivery = vi.fn<QueueBackendPort['claimItemDelivery']>(async (id, action, claimant) => {
      if (action === 'confirm' && ++confirmations === 1) {
        await original(id, action, claimant);
        throw new TypeError('successful response lost');
      }
      return original(id, action, claimant);
    });
    const adopting = openAdoptingTab(server);
    adopting.runtime.start();
    adopting.emit('queue_items_retried', retriedEvent([{ item_id: 99, source_item_id: 41 }]));
    await vi.advanceTimersByTimeAsync(100);
    expect(stagedImageNames(adopting)).toContain('rerun-99.png');
    expect(server.claims.get(99)?.isDelivered).toBe(true);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(vi.mocked(server.claimItemDelivery).mock.calls.filter(([, action]) => action === 'confirm')).toHaveLength(2);
    const later = openCanvasTab(server);
    later.runtime.start();
    emitReruns(later, [99]);
    later.emit(
      'queue_item_status_changed',
      completedEvent(99, buildQueueItemOrigin('gone', later.project.id), 'canvas')
    );
    await vi.advanceTimersByTimeAsync(100);
    expect(stagedItemIds(later)).toEqual([]);
    await adopting.runtime.dispose();
    await later.runtime.dispose();
    vi.useRealTimers();
  });

  it.each([403, 422, 503] as const)(
    'bounds unsuccessful confirmation retries (%s) without releasing staged work',
    async (failure) => {
      vi.useFakeTimers();
      const server = createDeliveryServer(120_000);
      const original = server.claimItemDelivery;
      server.claimItemDelivery = vi.fn<QueueBackendPort['claimItemDelivery']>((id, action, claimant) => {
        if (action === 'confirm') {
          return Promise.reject(new ApiError('confirmation failed', failure));
        }
        return original(id, action, claimant);
      });
      const tab = openCanvasTab(server);
      tab.runtime.start();
      emitReruns(tab, [99]);
      tab.emit('queue_item_status_changed', completedEvent(99, buildQueueItemOrigin('gone', tab.project.id), 'canvas'));
      await vi.advanceTimersByTimeAsync(180_000);
      const confirmations = vi.mocked(server.claimItemDelivery).mock.calls.filter(([, action]) => action === 'confirm');
      expect(confirmations).toHaveLength(failure === 403 || failure === 422 ? 1 : 5);
      const requests = vi.mocked(server.claimItemDelivery).mock.calls.length;
      await vi.advanceTimersByTimeAsync(180_000);
      expect(server.claimItemDelivery).toHaveBeenCalledTimes(requests);
      expect(stagedItemIds(tab)).toEqual([99]);
      expect(vi.mocked(server.claimItemDelivery).mock.calls.filter(([, action]) => action === 'release')).toHaveLength(
        0
      );
      await tab.runtime.dispose();
      vi.useRealTimers();
    }
  );

  it.each(['close', 'dispose', 'account change'] as const)(
    'returns a late unadopted grant after %s, with account fencing',
    async (ending) => {
      const server = createDeliveryServer(120_000);
      const original = server.claimItemDelivery;
      let grant!: () => void;
      server.claimItemDelivery = vi.fn<QueueBackendPort['claimItemDelivery']>((id, action, claimant) =>
        action === 'claim'
          ? new Promise((resolve) => {
              grant = () => {
                void original(id, action, claimant).then(resolve);
              };
            })
          : original(id, action, claimant)
      );
      const tab = openCanvasTab(server);
      tab.runtime.start();
      emitReruns(tab, [99]);
      tab.emit('queue_item_status_changed', completedEvent(99, buildQueueItemOrigin('gone', tab.project.id), 'canvas'));
      await vi.waitFor(() => expect(grant).toBeDefined());
      let disposal: Promise<void> | undefined;
      if (ending === 'close') {
        tab.closeProject();
      }
      if (ending === 'dispose') {
        disposal = tab.runtime.dispose();
      }
      if (ending === 'account change') {
        accountLifecycle.activate('replacement-grant-account');
      }
      grant();
      await settleEvents();
      await disposal;
      expect(stagedItemIds(tab)).toEqual([]);
      const releases = vi.mocked(server.claimItemDelivery).mock.calls.filter(([, action]) => action === 'release');
      if (ending === 'account change') {
        expect(releases).toHaveLength(0);
      } else {
        expect(releases).toHaveLength(1);
        expect(server.claims.has(99)).toBe(false);
      }
      await tab.runtime.dispose();
    }
  );

  it('removes settled retry-wait listeners and aborts the pending wait on disposal', async () => {
    vi.useFakeTimers();
    const added = vi.spyOn(AbortSignal.prototype, 'addEventListener');
    const removed = vi.spyOn(AbortSignal.prototype, 'removeEventListener');
    const server = createDeliveryServer();
    const original = server.claimItemDelivery;
    let waits = 0;
    server.claimItemDelivery = vi.fn<QueueBackendPort['claimItemDelivery']>((id, action, claimant) => {
      if (action === 'claim' && waits++ < 25) {
        return Promise.resolve({ kind: 'held', retryInMs: 1 } as const);
      }
      return original(id, action, claimant);
    });
    const tab = openCanvasTab(server);
    tab.runtime.start();
    const initialRegistrations = added.mock.calls.length;
    emitReruns(tab, [99]);
    tab.emit('queue_item_status_changed', completedEvent(99, buildQueueItemOrigin('gone', tab.project.id), 'canvas'));
    await vi.advanceTimersByTimeAsync(100);
    expect(stagedItemIds(tab)).toEqual([99]);
    const waitRegistrations = () =>
      added.mock.calls.flatMap(([event, listener, options], index) =>
        index >= initialRegistrations && event === 'abort' && typeof options === 'object' && options.once
          ? [{ signal: added.mock.contexts[index], listener }]
          : []
      );
    const unremovedWaits = () =>
      waitRegistrations().filter(
        ({ signal, listener }) =>
          !(signal as AbortSignal).aborted &&
          !removed.mock.calls.some(
            ([event, removedListener], index) =>
              event === 'abort' && removedListener === listener && removed.mock.contexts[index] === signal
          )
      );
    expect(waitRegistrations().length).toBeGreaterThanOrEqual(25);
    expect(unremovedWaits()).toHaveLength(0);
    vi.mocked(server.claimItemDelivery).mockResolvedValue({ kind: 'held', retryInMs: 10_000 });
    emitReruns(tab, [100]);
    tab.emit('queue_item_status_changed', completedEvent(100, buildQueueItemOrigin('gone', tab.project.id), 'canvas'));
    await vi.advanceTimersByTimeAsync(0);
    expect(unremovedWaits()).toHaveLength(1);
    await tab.runtime.dispose();
    expect(unremovedWaits()).toHaveLength(0);
    expect(vi.getTimerCount()).toBe(0);
    added.mockRestore();
    removed.mockRestore();
    vi.useRealTimers();
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

  it('stages a Canvas rerun after the tab that claimed it closed before staging, once the claim expires', async () => {
    const server = createDeliveryServer();
    // A lost page cannot run cleanup; this abandoned claimant leaves its lease to expire.
    let finishAbandoned!: (images: QueueResultImage[]) => void;
    const closing = openCanvasTab(
      server,
      vi.fn(
        () =>
          new Promise<QueueResultImage[]>((resolve) => {
            finishAbandoned = resolve;
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

    const disposal = closing.runtime.dispose();
    finishAbandoned([{ imageName: 'late.png' } as QueueResultImage]);
    await disposal;
    await open.runtime.dispose();
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

describe('run outcomes', () => {
  const submitRun = async ({
    cancelRequested = false,
    getItem,
  }: { cancelRequested?: boolean; getItem?: QueueBackendPort['getItem'] } = {}) => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const handlers = new Map<string, Set<(payload: never) => void>>();
    const connectionListeners = new Set<(status: BackendConnectionStatus) => void>();
    const origin = buildQueueItemOrigin(queueItem.id, project.id);
    const listeners = new Set<() => void>();
    const commands = createTestCommands({
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        Object.assign(queueItem, { backendBatchId, backendItemIds, status: 'running' });
        listeners.forEach((listener) => listener());
      },
    });
    const enqueueGenerate = vi
      .fn()
      .mockResolvedValue({ batchId: 'webv2:local-queue-item', enqueued: 1, itemIds: [41], requested: 1 });
    const runtime = createQueueRuntime({
      ...runtimeServices,
      cancelRequests: { wasRequestedHere: () => cancelRequested },
      backend: createTestBackend({
        enqueueGenerate,
        ...(getItem ? { getItem } : {}),
        onConnectionChange: vi.fn((listener: (status: BackendConnectionStatus) => void) => {
          connectionListeners.add(listener);
          listener('connected');
          return () => connectionListeners.delete(listener);
        }) as QueueBackendPort['onConnectionChange'],
        listItems: vi
          .fn()
          .mockResolvedValue([{ batchId: 'webv2:local-queue-item', id: 41, origin, status: 'in_progress' }]),
        on: vi.fn((event: string, handler: (payload: never) => void) => {
          const eventHandlers = handlers.get(event) ?? new Set();
          eventHandlers.add(handler);
          handlers.set(event, eventHandlers);
          return () => eventHandlers.delete(handler);
        }) as QueueBackendPort['on'],
      }),
      history: {
        commands,
        listStagedResults: () => [],
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      journal: createTestJournal(),
    });
    const emit = (event: string, payload: unknown) =>
      handlers.get(event)?.forEach((handler) => handler(payload as never));

    runtime.start();
    await vi.waitFor(() => expect(queueItem.backendItemIds).toEqual([41]));

    const reconnect = () => connectionListeners.forEach((listener) => listener('connected'));

    return { commands, emit, origin, project, queueItem, reconnect, runtime };
  };

  it('treats a run that vanished in a clear this tab asked for as canceled, without a notice', async () => {
    const { commands, reconnect, runtime } = await submitRun({
      cancelRequested: true,
      getItem: vi.fn().mockRejectedValue(new ApiError('not found', 404)),
    });

    reconnect();

    await vi.waitFor(() =>
      expect(commands.setStatus).toHaveBeenCalledWith(expect.objectContaining({ notify: false, status: 'cancelled' }))
    );
    await runtime.dispose();
  });

  it('reports a run that vanished otherwise as failed, with nothing left to show or retry', async () => {
    const { commands, reconnect, runtime } = await submitRun({
      getItem: vi.fn().mockRejectedValue(new ApiError('not found', 404)),
    });

    reconnect();

    await vi.waitFor(() =>
      expect(commands.setStatus).toHaveBeenCalledWith(
        expect.objectContaining({ errorCode: 'removed', status: 'failed' })
      )
    );
    expect(commands.setStatus).not.toHaveBeenCalledWith(
      expect.objectContaining({ failedBackendItemIds: expect.anything() })
    );
    await runtime.dispose();
  });
});
