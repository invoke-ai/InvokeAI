import type {
  QueueBackendItem,
  QueueEnqueueResult,
  QueueEnqueueGenerateRequest,
  QueueEnqueueWorkflowRequest,
  QueueItemProgress,
  QueueProgressPreviewPayload,
  QueueResultImage,
} from '@features/queue/core/types';
import type { ActiveProgressTargetSink } from '@features/queue/data/activeProgressTargetStore';
import type { QueueItemProgressSink } from '@features/queue/data/progressStore';

import {
  activeProgressTargetStore,
  getActiveProgressTargets,
  getFollowedProgressTargets,
} from '@features/queue/data/activeProgressTargetStore';
import {
  buildQueueItemOrigin,
  buildUtilityQueueItemOrigin,
  type QueueItemStatusChangedEvent,
} from '@features/queue/data/events';
import { progressImageStore } from '@features/queue/data/progressImageStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { createSocketHub, type BackendSocket } from '@platform/transport/socketHub';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createQueueCoordinator,
  QueueItemCancelledError,
  QueueItemFailedError,
  type QueueCoordinator,
  type QueueCoordinatorBackendPort,
  type QueueCoordinatorCallbacks,
  type QueueModelLoadPort,
  type QueueNodeExecutionPort,
} from './coordinator';

const deferred = <T>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });

  return { promise, resolve };
};

class FakeSocket implements BackendSocket {
  readonly active = true;
  readonly emitted: { event: string; payload: unknown }[] = [];
  private readonly handlers = new Map<string, ((payload: never) => void)[]>();

  on(event: string, handler: (payload: never) => void): void {
    this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
  }

  off(event: string, handler: (payload: never) => void): void {
    this.handlers.set(
      event,
      (this.handlers.get(event) ?? []).filter((existing) => existing !== handler)
    );
  }

  emit(event: string, payload: unknown): void {
    this.emitted.push({ event, payload });
  }

  connect(): void {
    this.fire('connect', undefined);
  }

  disconnect(): void {
    this.fire('disconnect', 'io client disconnect');
  }

  fire(event: string, payload: unknown): void {
    for (const handler of this.handlers.get(event) ?? []) {
      (handler as (value: unknown) => void)(payload);
    }
  }
}

const createStatusEvent = (overrides: Partial<QueueItemStatusChangedEvent>): QueueItemStatusChangedEvent => ({
  batch_status: {
    batch_id: 'batch-1',
    canceled: 0,
    completed: 0,
    destination: 'gallery',
    failed: 0,
    in_progress: 1,
    origin: null,
    pending: 0,
    queue_id: 'default',
    total: 1,
    waiting: 0,
  },
  batch_id: 'batch-1',
  completed_at: null,
  created_at: '2026-06-10T00:00:00Z',
  destination: 'gallery',
  error_message: null,
  error_traceback: null,
  error_type: null,
  item_id: 1,
  origin: null,
  queue_id: 'default',
  session_id: 'session-1',
  started_at: null,
  status: 'completed',
  status_sequence: 1,
  timestamp: 1,
  updated_at: '2026-06-10T00:00:00Z',
  user_id: 'user-1',
  queue_status: {
    batch_id: 'batch-1',
    canceled: 0,
    completed: 0,
    failed: 0,
    in_progress: 1,
    item_id: 1,
    pending: 0,
    queue_id: 'default',
    session_id: 'session-1',
    total: 1,
    user_in_progress: 1,
    user_pending: 0,
    waiting: 0,
  },
  ...overrides,
});

/** The REST snapshot carries the socket event's consumer-facing fields only. */
const toPreviewPayload = (event: {
  queue_id: string;
  item_id: number;
  session_id: string;
  invocation_source_id: string;
  revision: number;
  message: string;
  percentage: number | null;
  image: { width: number; height: number; dataURL: string };
}): QueueProgressPreviewPayload => ({
  image: event.image,
  invocation_source_id: event.invocation_source_id,
  item_id: event.item_id,
  message: event.message,
  percentage: event.percentage,
  queue_id: event.queue_id,
  revision: event.revision,
  session_id: event.session_id,
});

const createQueueBackendItem = (overrides: Partial<QueueBackendItem>): QueueBackendItem => ({
  id: 1,
  status: 'in_progress',
  ...overrides,
});

const createImage = (imageName: string, sourceQueueItemId: string): QueueResultImage => ({
  height: 64,
  imageName,
  imageUrl: `https://example.test/${imageName}`,
  isIntermediate: false,
  queuedAt: '2026-06-10T00:00:00Z',
  sourceQueueItemId,
  thumbnailUrl: `https://example.test/${imageName}/thumb`,
  width: 64,
});

const generateRequest: QueueEnqueueGenerateRequest = {
  batchCount: 1,
  destination: 'gallery',
  graph: { edges: [], id: 'graph-1', nodes: {} },
  negativePrompt: '',
  negativePromptNodeId: 'negative_prompt',
  positivePrompt: 'a fjord at dawn',
  positivePromptNodeId: 'positive_prompt',
  projectId: 'project-1',
  seed: 1,
  seedNodeId: 'seed',
  seedStep: 0,
  sourceQueueItemId: 'local-1',
};

const workflowRequest: QueueEnqueueWorkflowRequest = {
  batchCount: 1,
  destination: 'gallery',
  graph: { edges: [], id: 'graph-1', nodes: {} },
  projectId: 'project-1',
  sourceQueueItemId: 'local-1',
};

interface Harness {
  activeProgressTarget: {
    clear: ReturnType<typeof vi.fn>;
    forgetRun: ReturnType<typeof vi.fn>;
    hasObservedConcurrency: ReturnType<typeof vi.fn>;
    markFramed: ReturnType<typeof vi.fn>;
    observeConcurrency: ReturnType<typeof vi.fn>;
    set: ReturnType<typeof vi.fn>;
    settle: ReturnType<typeof vi.fn>;
  };
  api: {
    [
      Key in Exclude<
        keyof QueueCoordinatorBackendPort,
        'emit' | 'on' | 'onConnectionChange' | 'getEnqueueReceipt' | 'readProgressPreviews'
      >
    ]: ReturnType<typeof vi.fn>;
  } & {
    getEnqueueReceipt?: ReturnType<typeof vi.fn>;
    readProgressPreviews?: ReturnType<typeof vi.fn<() => Promise<QueueProgressPreviewPayload[]>>>;
  };
  callbacks: { [Key in keyof QueueCoordinatorCallbacks]: ReturnType<typeof vi.fn> };
  coordinator: QueueCoordinator;
  hub: ReturnType<typeof createSocketHub>;
  modelLoads: { [Key in keyof QueueModelLoadPort]: ReturnType<typeof vi.fn> };
  nodeExecution: { [Key in keyof QueueNodeExecutionPort]: ReturnType<typeof vi.fn> };
  progressImage: {
    bindSwapImages: ReturnType<typeof vi.fn>;
    clear: ReturnType<typeof vi.fn>;
    clearHeld: ReturnType<typeof vi.fn>;
    getProgress: ReturnType<typeof vi.fn>;
    hold: ReturnType<typeof vi.fn>;
    set: ReturnType<typeof vi.fn>;
    setProgress: ReturnType<typeof vi.fn>;
  };
  progressEntries: Map<string, QueueItemProgress>;
  socket: FakeSocket;
}

const createHarness = (options: { galleryRefreshCoalesceMs?: number } = {}): Harness => {
  const socket = new FakeSocket();
  const progressEntries = new Map<string, QueueItemProgress>();
  const progress: QueueItemProgressSink = {
    clear: (queueItemId) => {
      progressEntries.delete(queueItemId);
    },
    clearAll: () => {
      progressEntries.clear();
    },
    set: (queueItemId, value) => {
      progressEntries.set(queueItemId, value);
    },
  };
  const api = {
    getEnqueueReceipt: undefined as
      | ReturnType<typeof vi.fn<(projectId: string, queueItemId: string) => Promise<QueueEnqueueResult | null>>>
      | undefined,
    cancelQueueItems: vi.fn(() => Promise.resolve()),
    cancelQueueItemsByBatchIds: vi.fn(() => Promise.resolve()),
    enqueueGenerate: vi.fn(() => Promise.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })),
    enqueueWorkflow: vi.fn(() => Promise.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })),
    getItem: vi.fn((itemId: number) => Promise.resolve(createQueueBackendItem({ id: itemId }))),
    getResultImages: vi.fn((itemId: number, sourceQueueItemId: string) =>
      Promise.resolve([createImage(`image-${itemId}.png`, sourceQueueItemId)])
    ),
    listItems: vi.fn((): Promise<QueueBackendItem[]> => Promise.resolve([])),
    readActiveItemIds: vi.fn((): Promise<{ itemIds: number[]; executingRootItemIds: number[] }> =>
      Promise.resolve({ itemIds: [], executingRootItemIds: [] })
    ),
    readProgressPreviews: undefined as
      | ReturnType<typeof vi.fn<() => Promise<QueueProgressPreviewPayload[]>>>
      | undefined,
  };
  const callbacks = {
    isRunOpen: vi.fn(() => true),
    onGalleryRefresh: vi.fn(),
  };
  const modelLoads = {
    completed: vi.fn(),
    reset: vi.fn(),
    started: vi.fn(),
  };
  const nodeExecution = {
    clearAll: vi.fn(),
    completed: vi.fn(),
    failed: vi.fn(),
    progress: vi.fn(),
    setOrigin: vi.fn(),
    settleRunning: vi.fn(),
    started: vi.fn(),
  };
  const progressImage = {
    bindSwapImages: vi.fn(),
    clear: vi.fn(),
    clearHeld: vi.fn(),
    getProgress: vi.fn<typeof progressImageStore.getProgress>(() => null),
    hold: vi.fn(),
    set: vi.fn(),
    setProgress: vi.fn(),
  };
  const concurrentRunIds = new Set<string>();
  const activeProgressTarget = {
    clear: vi.fn<ActiveProgressTargetSink['clear']>((target) => {
      if (!target) {
        concurrentRunIds.clear();
      }
    }),
    forgetRun: vi.fn((queueItemId: string) => {
      concurrentRunIds.delete(queueItemId);
    }),
    hasObservedConcurrency: vi.fn((queueItemId: string) => concurrentRunIds.has(queueItemId)),
    markFramed: vi.fn(),
    observeConcurrency: vi.fn((queueItemId: string) => {
      concurrentRunIds.add(queueItemId);
    }),
    set: vi.fn(),
    settle: vi.fn(),
  } satisfies ActiveProgressTargetSink;
  const hub = createSocketHub({ createSocket: () => socket });

  hub.connect();

  const coordinator = createQueueCoordinator(callbacks, {
    backend: {
      ...api,
      get getEnqueueReceipt() {
        return api.getEnqueueReceipt;
      },
      get readProgressPreviews() {
        return api.readProgressPreviews;
      },
      emit: hub.emit,
      on: hub.on,
      onConnectionChange: hub.onConnectionChange,
    },
    activeProgressTarget,
    galleryRefreshCoalesceMs: options.galleryRefreshCoalesceMs ?? 1,
    modelLoads,
    nodeExecution,
    progress,
    progressImage,
  });

  return {
    activeProgressTarget,
    api,
    callbacks,
    coordinator,
    hub,
    modelLoads,
    nodeExecution,
    progressImage,
    progressEntries,
    socket,
  };
};

const useRealActiveProgressTargets = (harness: Harness): void => {
  activeProgressTargetStore.clear();
  harness.activeProgressTarget.clear.mockImplementation(activeProgressTargetStore.clear);
  harness.activeProgressTarget.forgetRun.mockImplementation(activeProgressTargetStore.forgetRun);
  harness.activeProgressTarget.hasObservedConcurrency.mockImplementation(
    activeProgressTargetStore.hasObservedConcurrency
  );
  harness.activeProgressTarget.markFramed.mockImplementation(activeProgressTargetStore.markFramed);
  harness.activeProgressTarget.observeConcurrency.mockImplementation(activeProgressTargetStore.observeConcurrency);
  harness.activeProgressTarget.set.mockImplementation(activeProgressTargetStore.set);
  harness.activeProgressTarget.settle.mockImplementation(activeProgressTargetStore.settle);
};

const savedWorkflowCallRequest = (): QueueEnqueueWorkflowRequest => ({
  ...workflowRequest,
  graph: {
    ...workflowRequest.graph,
    nodes: { 'call-node': { id: 'call-node', type: 'call_saved_workflow' } },
  },
});

describe('queueCoordinator', () => {
  let harness: Harness;

  beforeEach(() => {
    harness = createHarness();
  });

  afterEach(() => {
    harness.coordinator.dispose();
    harness.hub.disconnect();
    vi.useRealTimers();
  });

  it('settles submitted runs from terminal socket events without polling', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2 }));

    const images = await resultsPromise;

    expect(images.map((image) => image.imageName)).toEqual(['image-1.png', 'image-2.png']);
    expect(harness.api.getItem).not.toHaveBeenCalled();
  });

  it('rejects with the backend error message and the item that failed when a run fails', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({ batchId: 'batch-1', enqueued: 2, itemIds: [1, 2], requested: 2 });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ error_message: 'CUDA out of memory', item_id: 2, status: 'failed' })
    );

    const failure = await resultsPromise.catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(QueueItemFailedError);
    expect(failure).toMatchObject({ failedBackendItemIds: [2], failure: 'reported', message: 'CUDA out of memory' });
  });

  it('counts a failed iteration as settled while the next iteration runs', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 4,
      itemIds: [1, 2, 3, 4],
      requested: 4,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'completed' }));
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 2, status: 'failed', error_message: 'Generation failed' })
    );
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 3, status: 'in_progress' }));
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 3 }),
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.6,
    });

    expect(harness.progressEntries.get('local-1')).toMatchObject({
      activeItemIndex: 3,
      completedItemCount: 2,
      percentage: 0.6,
      totalItemCount: 4,
    });
  });

  it('rejects with QueueItemCancelledError when the backend cancels a run', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'canceled' }));

    await expect(resultsPromise).rejects.toBeInstanceOf(QueueItemCancelledError);
  });

  it('settles tracked items from an owner-scoped bulk cancellation event', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_items_canceled', {
      canceled_item_ids: [1],
      canceled_item_ids_by_user: { 'user-1': [1] },
      queue_id: 'default',
      timestamp: 1,
      user_ids: ['user-1'],
    });

    await expect(resultsPromise).rejects.toBeInstanceOf(QueueItemCancelledError);
  });

  it('does not settle tracked items from a sanitized bulk cancellation companion', async () => {
    harness.callbacks.onBackendItemCancelled = vi.fn();
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('queue_items_canceled', {
      canceled_item_ids: [],
      canceled_item_ids_by_user: {},
      queue_id: 'default',
      timestamp: 1,
      user_ids: [],
    });

    expect(harness.callbacks.onBackendItemCancelled).not.toHaveBeenCalled();
  });

  it('ignores stale status events by status sequence', async () => {
    harness.callbacks.onBackendItemCancelled = vi.fn();
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 1, status: 'in_progress', status_sequence: 2 })
    );
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 1, status: 'canceled', status_sequence: 1 })
    );

    expect(harness.callbacks.onBackendItemCancelled).not.toHaveBeenCalled();

    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 1, status: 'canceled', status_sequence: 3 })
    );
    await expect(resultsPromise).rejects.toBeInstanceOf(QueueItemCancelledError);
  });

  it('settles runs whose terminal event arrived before tracking began', async () => {
    harness.coordinator.connect();

    // The event for item 1 lands while enqueue_batch is still resolving.
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    await harness.coordinator.submitGenerate('local-1', generateRequest);

    const images = await harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    expect(images.map((image) => image.imageName)).toEqual(['image-1.png']);
  });

  it('routes progress events to the tracked item and clears them on completion', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      message: 'Denoising',
      percentage: 0.5,
    });

    expect(harness.progressEntries.get('local-1')).toEqual({
      activeItemIndex: 1,
      completedItemCount: 0,
      message: 'Denoising',
      percentage: 0.5,
      totalItemCount: 1,
    });
    expect(harness.activeProgressTarget.set).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });

    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));
    await resultsPromise;

    expect(harness.progressEntries.has('local-1')).toBe(false);
    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    expect(harness.progressImage.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
  });

  it("signals each slot's first frame of its own once, after the frame is stored", async () => {
    harness.callbacks.onFirstProgressFrame = vi.fn(() => {
      // Live surfaces follow the slot from here; its frame must already be there to show.
      expect(harness.progressImage.set).toHaveBeenCalled();
    });
    harness.coordinator.connect();
    harness.api.enqueueGenerate.mockResolvedValueOnce({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const frame = (itemId: number, revision: number) => ({
      ...createStatusEvent({ item_id: itemId }),
      image: { dataURL: `data:image/png;base64,${itemId}-${revision}`, height: 32, width: 64 },
      message: 'Denoising',
      percentage: 0.5,
      revision,
    });

    // Model loading and text encoding report progress without a frame.
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      message: 'Loading',
      percentage: null,
    });
    expect(harness.callbacks.onFirstProgressFrame).not.toHaveBeenCalled();

    harness.socket.fire('invocation_progress', frame(1, 1));
    harness.socket.fire('invocation_progress', frame(1, 2));
    harness.socket.fire('invocation_progress', frame(2, 1));

    expect(harness.callbacks.onFirstProgressFrame).toHaveBeenCalledTimes(2);
    expect(harness.callbacks.onFirstProgressFrame).toHaveBeenNthCalledWith(1, { itemIndex: 1, queueItemId: 'local-1' });
    expect(harness.callbacks.onFirstProgressFrame).toHaveBeenNthCalledWith(2, { itemIndex: 2, queueItemId: 'local-1' });
    expect(harness.activeProgressTarget.markFramed.mock.calls).toEqual([
      [{ itemIndex: 1, queueItemId: 'local-1' }],
      [{ itemIndex: 2, queueItemId: 'local-1' }],
    ]);
  });

  it('sweeps outstanding items when the tab becomes visible again', async () => {
    // Visibility must reconcile immediately without waiting for socket reconnect backoff.
    const listeners = new Map<string, () => void>();

    vi.stubGlobal('document', {
      addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
      removeEventListener: (type: string) => listeners.delete(type),
      visibilityState: 'visible',
    });

    try {
      harness.coordinator.connect();
      await harness.coordinator.submitGenerate('local-1', generateRequest);
      harness.api.getItem.mockClear();
      harness.api.getItem.mockResolvedValueOnce(createQueueBackendItem({ id: 1, status: 'completed' }));
      const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

      listeners.get('visibilitychange')?.();

      await expect(resultsPromise).resolves.toEqual([expect.objectContaining({ imageName: 'image-1.png' })]);
      expect(harness.api.getItem).toHaveBeenCalledWith(1);

      harness.coordinator.dispose();

      expect(listeners.has('visibilitychange')).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('applies the preview snapshot on visibility and lets the revision gate drop replays', async () => {
    const listeners = new Map<string, () => void>();
    const frame = (revision: number, dataURL: string) => ({
      ...createStatusEvent({ item_id: 1 }),
      image: { dataURL, height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.5,
      revision,
    });

    vi.stubGlobal('document', {
      addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
      removeEventListener: (type: string) => listeners.delete(type),
      visibilityState: 'visible',
    });

    try {
      harness.api.readProgressPreviews = vi.fn(() =>
        Promise.resolve([toPreviewPayload(frame(2, 'data:image/png;base64,snapshot'))])
      );
      harness.coordinator.connect();
      await harness.coordinator.submitGenerate('local-1', generateRequest);
      harness.socket.fire('invocation_progress', frame(1, 'data:image/png;base64,live-1'));

      listeners.get('visibilitychange')?.();
      await Promise.resolve();
      await Promise.resolve();

      // The snapshot is newer than the last live frame: applied.
      expect(harness.progressImage.set.mock.calls.map(([image]) => (image as { dataUrl: string }).dataUrl)).toEqual([
        'data:image/png;base64,live-1',
        'data:image/png;base64,snapshot',
      ]);

      // The live stream has moved on; a second snapshot at the same revision is a replay.
      harness.socket.fire('invocation_progress', frame(3, 'data:image/png;base64,live-3'));
      listeners.get('visibilitychange')?.();
      await Promise.resolve();
      await Promise.resolve();

      expect(harness.progressImage.set).toHaveBeenCalledTimes(3);
      expect(harness.api.readProgressPreviews).toHaveBeenCalledTimes(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('keeps the completed progress image until backend item result routing finishes', async () => {
    let finishRouting: () => void = () => undefined;
    const routingPromise = new Promise<void>((resolve) => {
      finishRouting = resolve;
    });

    harness.callbacks.onBackendItemComplete = vi.fn(() => routingPromise);
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      image: { dataURL: 'data:image/png;base64,final-denoise', height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 1,
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    await resultsPromise;

    expect(harness.callbacks.onBackendItemComplete).toHaveBeenCalledWith('local-1', 1);
    // Held before routing started, so the finished image can swap in over it.
    expect(harness.progressImage.hold).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    // Execution has settled, so clear the badge while retaining the frame for routing.
    expect(harness.progressImage.setProgress).toHaveBeenLastCalledWith(null, {
      itemIndex: 1,
      queueItemId: 'local-1',
    });
    // Follow settling slots until finished-image routing completes instead of reverting to the old selection.
    expect(harness.activeProgressTarget.settle).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalled();
    expect(harness.progressImage.clear).not.toHaveBeenCalled();

    finishRouting();
    await routingPromise;
    await Promise.resolve();

    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    expect(harness.progressImage.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
  });

  it('preserves slot progress across node starts and seeds a new leg after pending', async () => {
    progressImageStore.clear();
    harness.progressImage.getProgress.mockImplementation(progressImageStore.getProgress);
    harness.progressImage.setProgress.mockImplementation(progressImageStore.setProgress);
    try {
      harness.coordinator.connect();
      await harness.coordinator.submitGenerate('local-1', generateRequest);
      const target = { itemIndex: 1, queueItemId: 'local-1' };
      const start = (nodeId: string) =>
        harness.socket.fire('invocation_started', {
          ...createStatusEvent({ item_id: 1 }),
          invocation_source_id: nodeId,
        });

      start('denoise');
      expect(progressImageStore.getProgress(target)).toEqual({ message: '', percentage: null });
      harness.socket.fire('invocation_progress', {
        ...createStatusEvent({ item_id: 1 }),
        invocation_source_id: 'denoise',
        message: 'Denoising',
        percentage: 1,
      });
      start('decode');
      expect(progressImageStore.getProgress(target)).toEqual({ message: 'Denoising', percentage: 1 });

      harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'pending' }));
      expect(progressImageStore.getProgress(target)).toBeNull();
      start('denoise');
      expect(progressImageStore.getProgress(target)).toEqual({ message: '', percentage: null });
    } finally {
      progressImageStore.clear();
    }
  });

  it.each(['socket', 'HTTP'] as const)('clears stale slot progress when %s reports a pending item', async (source) => {
    const listeners = new Map<string, () => void>();

    vi.stubGlobal('document', {
      addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
      removeEventListener: (type: string) => listeners.delete(type),
      visibilityState: 'visible',
    });

    try {
      harness.coordinator.connect();
      await harness.coordinator.submitGenerate('local-1', generateRequest);

      const target = { itemIndex: 1, queueItemId: 'local-1' };

      harness.socket.fire('invocation_progress', {
        ...createStatusEvent({ item_id: 1 }),
        message: 'Denoising',
        percentage: 0.65,
      });

      expect(harness.progressImage.setProgress).toHaveBeenLastCalledWith(
        { message: 'Denoising', percentage: 0.65 },
        target
      );

      if (source === 'socket') {
        harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'pending' }));
      } else {
        harness.api.getItem.mockResolvedValue(createQueueBackendItem({ id: 1, status: 'pending' }));

        listeners.get('visibilitychange')?.();

        await vi.waitFor(() => {
          expect(harness.progressImage.setProgress).toHaveBeenLastCalledWith(null, target);
        });
      }

      expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith(target);
      expect(harness.progressImage.setProgress).toHaveBeenLastCalledWith(null, target);
      expect(harness.progressImage.clear).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it.each([
    {
      name: 'sequential handover with out-of-order HTTP responses',
      itemIds: [1, 2],
      reconnect: false,
      startsWithInvocation: true,
      expectedSlot: 2,
      steps: [
        { kind: 'resolve', id: 2, status: 'in_progress', flush: true },
        { kind: 'resolve', id: 1, status: 'completed', flush: false },
      ],
    },
    {
      name: 'stale HTTP response overtaken by a newer socket status',
      itemIds: [1, 2, 3],
      reconnect: false,
      startsWithInvocation: false,
      expectedSlot: 3,
      steps: [
        { kind: 'resolve', id: 2, status: 'in_progress', flush: true },
        { kind: 'pending', id: 2 },
        { kind: 'resolve', id: 3, status: 'in_progress', flush: false },
        { kind: 'resolve', id: 1, status: 'completed', flush: false },
      ],
    },
    {
      name: 'live progress overtaking reconnect recovery',
      itemIds: [1, 2],
      reconnect: true,
      startsWithInvocation: false,
      expectedSlot: 2,
      steps: [
        { kind: 'progress', id: 2 },
        { kind: 'resolve', id: 1, status: 'completed', flush: false },
        { kind: 'resolve', id: 2, status: 'in_progress', flush: false },
      ],
    },
  ] as const)(
    'does not infer false concurrency during $name',
    async ({ itemIds, reconnect, startsWithInvocation, expectedSlot, steps }) => {
      useRealActiveProgressTargets(harness);
      const listeners = new Map<string, () => void>();
      const responses = new Map<number, ReturnType<typeof deferred<QueueBackendItem>>>(
        itemIds.map((id) => [id, deferred<QueueBackendItem>()])
      );

      if (!reconnect) {
        vi.stubGlobal('document', {
          addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
          removeEventListener: (type: string) => listeners.delete(type),
          visibilityState: 'visible',
        });
      }

      try {
        harness.coordinator.connect();
        harness.api.enqueueGenerate.mockResolvedValue({
          batchId: 'batch-1',
          enqueued: itemIds.length,
          itemIds: [...itemIds],
          requested: itemIds.length,
        });
        await harness.coordinator.submitGenerate('local-1', generateRequest);

        if (startsWithInvocation) {
          harness.socket.fire('invocation_started', {
            ...createStatusEvent({ item_id: 1 }),
            invocation_source_id: 'node-1',
          });
        } else {
          harness.socket.fire(
            'queue_item_status_changed',
            createStatusEvent({ item_id: 1, status: 'in_progress', status_sequence: 1 })
          );
        }
        expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);

        harness.api.getItem.mockImplementation((id: number) => responses.get(id)!.promise);

        if (reconnect) {
          harness.api.readActiveItemIds.mockResolvedValue({ itemIds: [2], executingRootItemIds: [2] });
          harness.socket.fire('disconnect', 'transport close');
          harness.socket.fire('connect', undefined);
        } else {
          listeners.get('visibilitychange')?.();
        }

        await vi.waitFor(() => {
          for (const id of itemIds) {
            expect(harness.api.getItem).toHaveBeenCalledWith(id);
          }
        });

        for (const step of steps) {
          if (step.kind === 'resolve') {
            const response = responses.get(step.id)!;
            response.resolve(createQueueBackendItem({ id: step.id, status: step.status }));
            if (step.flush) {
              await response.promise;
              await Promise.resolve();
            }
          } else if (step.kind === 'pending') {
            harness.socket.fire(
              'queue_item_status_changed',
              createStatusEvent({ item_id: step.id, status: 'pending', status_sequence: 2 })
            );
          } else {
            harness.socket.fire('invocation_progress', {
              ...createStatusEvent({ item_id: step.id }),
              invocation_source_id: 'denoise',
              message: 'Denoising',
              percentage: 0.25,
            });
            expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);
          }
        }

        await vi.waitFor(() => {
          expect(getActiveProgressTargets()).toEqual([{ itemIndex: expectedSlot, queueItemId: 'local-1' }]);
        });
        if (reconnect) {
          await vi.waitFor(() => expect(harness.api.readActiveItemIds.mock.calls.length).toBeGreaterThan(1));
        }
        expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);
      } finally {
        if (!reconnect) {
          vi.unstubAllGlobals();
        }
      }
    }
  );

  it('does not count a waiting workflow parent as a second executing iteration', async () => {
    useRealActiveProgressTargets(harness);
    harness.api.enqueueWorkflow.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', savedWorkflowCallRequest());

    const status = (itemId: number, state: QueueItemStatusChangedEvent['status'], sequence: number) =>
      harness.socket.fire(
        'queue_item_status_changed',
        createStatusEvent({ item_id: itemId, status: state, status_sequence: sequence })
      );

    status(1, 'in_progress', 1);
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      image: { dataURL: 'data:image/png;base64,parent-frame', height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.5,
    });
    status(1, 'waiting', 2);

    // The first parent still owns a followable preview, but its worker has been released.
    expect(getFollowedProgressTargets()).toContainEqual({ itemIndex: 1, queueItemId: 'local-1' });
    status(2, 'in_progress', 1);
    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);

    // A single worker can then execute each parent's child in turn.
    status(2, 'waiting', 2);
    status(3, 'in_progress', 1);
    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 3 }),
      invocation_source_id: 'child-node',
      root_item_id: 1,
      workflow_call_parent_source_id: 'call-node',
    });
    status(3, 'completed', 2);
    status(4, 'in_progress', 1);
    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 4 }),
      invocation_source_id: 'child-node',
      root_item_id: 2,
      workflow_call_parent_source_id: 'call-node',
    });
    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);
  });

  it('observes genuine overlapping execution and remembers it after one iteration completes', async () => {
    useRealActiveProgressTargets(harness);
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);

    const first = createStatusEvent({ item_id: 1, status: 'in_progress', status_sequence: 1 });
    harness.socket.fire('queue_item_status_changed', {
      ...first,
      batch_status: { ...first.batch_status, in_progress: 1, pending: 1, total: 2 },
    });
    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);

    const reportProgress = (itemId: number, percentage: number, message: string) => {
      harness.socket.fire('invocation_progress', {
        ...createStatusEvent({ item_id: itemId }),
        invocation_source_id: 'denoise',
        message,
        percentage,
      });
    };

    reportProgress(1, 0.2, 'Sampling first image');
    expect(harness.progressEntries.get('local-1')).toMatchObject({
      activeItemIndex: 1,
      message: 'Sampling first image',
      percentage: 0.2,
    });

    const second = createStatusEvent({ item_id: 2, status: 'in_progress', status_sequence: 1 });
    harness.socket.fire('queue_item_status_changed', {
      ...second,
      batch_status: { ...second.batch_status, in_progress: 2, total: 2 },
    });
    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true);

    const expectBatchProgress = (completedItemCount: number, percentage: number | null) => {
      expect(harness.progressEntries.get('local-1')).toMatchObject({
        activeItemIndex: undefined,
        completedItemCount,
        message: '',
        percentage,
        totalItemCount: 2,
      });
    };

    // Switch immediately to batch progress when overlap is observed.
    expectBatchProgress(0, null);

    for (const [itemId, percentage] of [
      [1, 0.2],
      [2, 0.8],
      [1, 0.25],
      [2, 0.85],
    ] as const) {
      const message = `Sampling image ${itemId}`;
      reportProgress(itemId, percentage, message);
      expectBatchProgress(0, null);
      expect(harness.progressImage.setProgress).toHaveBeenLastCalledWith(
        { message, percentage },
        { itemIndex: itemId, queueItemId: 'local-1' }
      );
    }

    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 1, status: 'completed', status_sequence: 2 })
    );
    expect(getActiveProgressTargets()).toEqual([{ itemIndex: 2, queueItemId: 'local-1' }]);
    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true);
    expectBatchProgress(1, 0.5);

    // Reconnecting must restore aggregate progress without another settlement.
    harness.socket.fire('disconnect', 'transport close');
    harness.socket.fire('connect', undefined);

    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true);
    expectBatchProgress(1, 0.5);

    reportProgress(2, 0.9, 'Finishing second image');
    expectBatchProgress(1, 0.5);

    reportProgress(2, 1, 'Decoding second image');
    expectBatchProgress(1, 0.5);

    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 2, status: 'completed', status_sequence: 2 })
    );
    expectBatchProgress(2, 0.99);
  });

  it.each(['completed', 'failed'] as const)(
    'caps concurrent progress at 99% before and after the last iteration is %s',
    async (finalStatus) => {
      useRealActiveProgressTargets(harness);
      const itemIds = Array.from({ length: 200 }, (_, index) => index + 1);
      harness.api.enqueueGenerate.mockResolvedValue({
        batchId: 'batch-1',
        enqueued: itemIds.length,
        itemIds,
        requested: itemIds.length,
      });
      harness.coordinator.connect();
      await harness.coordinator.submitGenerate('local-1', generateRequest);

      for (const itemId of [1, 2]) {
        const event = createStatusEvent({ item_id: itemId, status: 'in_progress', status_sequence: 1 });
        harness.socket.fire('queue_item_status_changed', {
          ...event,
          batch_status: { ...event.batch_status, in_progress: itemId, pending: 200 - itemId, total: 200 },
        });
      }
      expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true);

      for (const itemId of itemIds.slice(0, -1)) {
        harness.socket.fire(
          'queue_item_status_changed',
          createStatusEvent({ item_id: itemId, status: 'completed', status_sequence: 2 })
        );
      }
      expect(harness.progressEntries.get('local-1')).toMatchObject({
        completedItemCount: 199,
        percentage: 0.99,
        totalItemCount: 200,
      });

      harness.socket.fire(
        'queue_item_status_changed',
        createStatusEvent({ item_id: 200, status: finalStatus, status_sequence: 2 })
      );
      expect(harness.progressEntries.get('local-1')).toMatchObject({
        completedItemCount: 200,
        percentage: 0.99,
        totalItemCount: 200,
      });
    }
  );

  it('does not observe concurrency during sequential result routing', async () => {
    useRealActiveProgressTargets(harness);
    const routing = deferred<void>();
    harness.callbacks.onBackendItemComplete = vi.fn(() => routing.promise);
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });

    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 1 }),
      invocation_source_id: 'node-1',
    });

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 2 }),
      invocation_source_id: 'node-1',
    });

    expect(getFollowedProgressTargets()).toHaveLength(2);
    expect(getActiveProgressTargets()).toEqual([{ itemIndex: 2, queueItemId: 'local-1' }]);
    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);

    routing.resolve();
    await routing.promise;
    await Promise.resolve();

    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);
  });

  it.each([
    {
      name: 'retains concurrency through result delivery until detach',
      firstStatus: 'in_progress',
      executingRoots: [1, 2],
      concurrent: true,
      finish: 'results',
    },
    {
      name: 'forgets concurrency and active slots on immediate detach',
      firstStatus: 'in_progress',
      executingRoots: [1, 2],
      concurrent: true,
      finish: 'detach',
    },
    {
      name: 'does not infer overlap from separate HTTP reads spanning a hand-off',
      firstStatus: 'in_progress',
      executingRoots: [2],
      concurrent: false,
      finish: 'none',
    },
    {
      name: 'does not infer overlap from a recovered sequential batch',
      firstStatus: 'completed',
      executingRoots: [2],
      concurrent: false,
      finish: 'none',
    },
  ] as const)('$name', async ({ firstStatus, executingRoots, concurrent, finish }) => {
    useRealActiveProgressTargets(harness);
    harness.api.readActiveItemIds.mockResolvedValue({
      itemIds: [...executingRoots],
      executingRootItemIds: [...executingRoots],
    });
    harness.api.getItem.mockImplementation((id: number) =>
      Promise.resolve(
        createQueueBackendItem({
          id,
          origin: buildQueueItemOrigin('local-1'),
          status: id === 1 ? firstStatus : 'in_progress',
        })
      )
    );

    if (finish === 'results') {
      harness.coordinator.connect();
    }
    await harness.coordinator.reconcile([{ backendItemIds: [1, 2], id: 'local-1', status: 'running' }]);

    if (concurrent) {
      await vi.waitFor(() => expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true));
    } else {
      await vi.waitFor(() => expect(harness.api.readActiveItemIds).toHaveBeenCalled());
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
      expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);
    }

    if (firstStatus === 'completed') {
      expect(getActiveProgressTargets()).toEqual([{ itemIndex: 2, queueItemId: 'local-1' }]);
    }

    if (finish === 'results') {
      expect(getActiveProgressTargets()).toHaveLength(2);
      const results = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

      harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

      expect(getActiveProgressTargets()).toHaveLength(1);
      expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true);

      harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2 }));

      await expect(results).resolves.toHaveLength(2);
      // The runtime still has to deliver results and close the local queue item.
      expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true);
    }

    if (finish !== 'none') {
      harness.coordinator.detachRun('local-1');
      expect(getActiveProgressTargets()).toEqual([]);
      expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);
    }
  });

  it('recognizes distinct executing workflow roots from a consistent snapshot', async () => {
    useRealActiveProgressTargets(harness);
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 2, itemIds: [1, 2], requested: 2 });
    harness.coordinator.connect();
    // Two children of the same root are not two batch iterations.
    harness.api.readActiveItemIds.mockResolvedValue({ itemIds: [4, 3, 1, 2], executingRootItemIds: [1] });
    await harness.coordinator.submitWorkflow('local-1', savedWorkflowCallRequest());
    const child = createStatusEvent({ item_id: 3, status: 'in_progress', status_sequence: 1 });
    harness.socket.fire('queue_item_status_changed', {
      ...child,
      batch_status: { ...child.batch_status, in_progress: 2, total: 4, waiting: 2 },
    });
    await vi.waitFor(() => expect(harness.api.readActiveItemIds.mock.calls.length).toBeGreaterThan(1));
    expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);

    harness.api.readActiveItemIds.mockResolvedValue({ itemIds: [5, 3, 1, 2], executingRootItemIds: [1, 2] });
    harness.socket.fire('queue_item_status_changed', {
      ...child,
      item_id: 5,
      batch_status: { ...child.batch_status, in_progress: 2, total: 5, waiting: 2 },
    });
    await vi.waitFor(() => expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(true));
  });

  it.each(['detaches', 'closes'] as const)(
    'ignores concurrent execution evidence when the run %s before the snapshot arrives',
    async (action) => {
      useRealActiveProgressTargets(harness);
      const snapshot = deferred<{ itemIds: number[]; executingRootItemIds: number[] }>();
      harness.api.readActiveItemIds.mockImplementation(() => snapshot.promise);
      harness.api.enqueueGenerate.mockResolvedValue({ batchId: 'batch-1', enqueued: 2, itemIds: [1, 2], requested: 2 });
      await harness.coordinator.submitGenerate('local-1', generateRequest);
      expect(harness.api.readActiveItemIds).toHaveBeenCalledTimes(1);

      if (action === 'detaches') {
        harness.coordinator.detachRun('local-1');
      } else {
        harness.callbacks.isRunOpen?.mockReturnValue(false);
      }

      snapshot.resolve({ itemIds: [1, 2], executingRootItemIds: [1, 2] });
      await snapshot.promise;
      // Flush the deadline wrapper and the coordinator's continuation.
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
      expect(harness.activeProgressTarget.observeConcurrency).not.toHaveBeenCalled();
      expect(activeProgressTargetStore.hasObservedConcurrency('local-1')).toBe(false);
    }
  );

  it('does not let an old routing finalizer clear replacement-account progress', async () => {
    const routing = deferred<void>();

    harness.callbacks.onBackendItemComplete = vi.fn(() => routing.promise);
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));
    await resultsPromise;
    harness.coordinator.dispose();
    const clearsAfterDispose = harness.progressImage.clear.mock.calls.length;

    accountLifecycle.invalidate();
    accountLifecycle.activate('replacement-account');
    routing.resolve();
    await routing.promise;
    await Promise.resolve();

    expect(harness.progressImage.clear).toHaveBeenCalledTimes(clearsAfterDispose);
  });

  it('routes model load socket events to the model-load port', () => {
    harness.coordinator.connect();

    harness.socket.fire('model_load_started', { config: { name: 'model-a' } });
    harness.socket.fire('model_load_complete', { config: { name: 'model-a' } });

    expect(harness.modelLoads.started).toHaveBeenCalledWith({ config: { name: 'model-a' } });
    expect(harness.modelLoads.completed).toHaveBeenCalledWith({ config: { name: 'model-a' } });
  });

  it('replays node events that landed before the enqueue response and settles the nodes', async () => {
    const acceptance = deferred<QueueEnqueueResult>();

    harness.api.enqueueWorkflow.mockReturnValue(acceptance.promise);
    harness.coordinator.connect();
    const submission = harness.coordinator.submitWorkflow('local-1', workflowRequest);

    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 1 }), invocation_source_id: 'node-1' });
    harness.socket.fire('invocation_complete', {
      ...createStatusEvent({ item_id: 1 }),
      invocation_source_id: 'node-1',
      result: { type: 'integer_output', value: 7 },
    });
    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 1 }), invocation_source_id: 'node-2' });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'completed' }));

    expect(harness.nodeExecution.started).not.toHaveBeenCalled();

    acceptance.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    await submission;

    expect(harness.nodeExecution.started.mock.calls.map(([event]) => event.invocation_source_id)).toEqual([
      'node-1',
      'node-2',
    ]);
    expect(harness.nodeExecution.completed).toHaveBeenCalledTimes(1);
    expect(harness.nodeExecution.settleRunning).toHaveBeenCalledWith(new Set(['node-1', 'node-2']), 'completed');
    await expect(harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z')).resolves.toHaveLength(1);
  });

  it('routes child invocation events to the parent call node', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 2 }),
      invocation_source_id: 'child-node',
      root_item_id: 1,
      workflow_call_parent_source_id: 'call-node',
    });

    expect(harness.nodeExecution.started).toHaveBeenCalledWith(
      expect.objectContaining({
        invocation_source_id: 'call-node',
        item_id: 1,
      })
    );
  });

  it('routes child preview frames to the parent call node and root progress slot', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2 }),
      image: { dataURL: 'data:image/png;base64,child', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      root_item_id: 1,
      workflow_call_parent_source_id: 'call-node',
    });

    expect(harness.nodeExecution.progress).toHaveBeenCalledWith('call-node', 0.5, 'Child sampling');
    expect(harness.progressImage.set).toHaveBeenCalledWith(
      { dataUrl: 'data:image/png;base64,child', height: 32, width: 64 },
      { itemIndex: 1, queueItemId: 'local-1' }
    );
    // The child (item 2) and root (item 1) must update the same parent Canvas slot.
    expect(harness.progressImage.setProgress).toHaveBeenCalledWith(
      { message: 'Child sampling', percentage: 0.5 },
      { itemIndex: 1, queueItemId: 'local-1' }
    );
  });

  it('clears workflow preview state on disposal and ignores later events', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('workflow-root', savedWorkflowCallRequest());
    const childFrame = {
      ...createStatusEvent({ item_id: 3, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,child', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 1,
      session_id: 'child-session',
      workflow_call_parent_source_id: 'call-node',
    };
    harness.socket.fire('invocation_progress', childFrame);
    const progressSetCount = harness.progressImage.set.mock.calls.length;

    harness.coordinator.dispose();

    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith();
    expect(harness.progressImage.clear).toHaveBeenCalledWith();
    harness.socket.fire('invocation_progress', { ...childFrame, revision: 2 });
    expect(harness.progressImage.set).toHaveBeenCalledTimes(progressSetCount);
  });

  it('follows whichever item runs live, even a lower id after a higher one', async () => {
    harness.api.enqueueWorkflow
      .mockResolvedValueOnce({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })
      .mockResolvedValueOnce({ batchId: 'batch-2', enqueued: 1, itemIds: [2], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);
    await harness.coordinator.submitWorkflow('local-2', { ...workflowRequest, sourceQueueItemId: 'local-2' });

    // Model affinity ran item 2 first.
    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 2 }), invocation_source_id: 'node-1' });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'completed' }));
    harness.nodeExecution.clearAll.mockClear();
    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 1 }), invocation_source_id: 'node-1' });
    harness.socket.fire('invocation_complete', {
      ...createStatusEvent({ item_id: 1 }),
      invocation_source_id: 'node-1',
      result: { type: 'integer_output', value: 1 },
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'completed' }));

    expect(harness.nodeExecution.clearAll).toHaveBeenCalledTimes(1);
    expect(harness.nodeExecution.completed).toHaveBeenCalledTimes(1);
    expect(harness.nodeExecution.settleRunning).toHaveBeenLastCalledWith(new Set(['node-1']), 'completed');
  });

  it('resets node state when a new item starts and settles running nodes to the item outcome', async () => {
    harness.api.enqueueWorkflow
      .mockResolvedValueOnce({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })
      .mockResolvedValueOnce({ batchId: 'batch-2', enqueued: 1, itemIds: [2], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);
    await harness.coordinator.submitWorkflow('local-2', { ...workflowRequest, sourceQueueItemId: 'local-2' });
    harness.nodeExecution.clearAll.mockClear();

    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 1 }), invocation_source_id: 'node-1' });
    expect(harness.nodeExecution.clearAll).toHaveBeenCalledTimes(1);
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      invocation_source_id: 'node-1',
      message: 'sampling',
      percentage: 0.5,
    });
    expect(harness.nodeExecution.clearAll).toHaveBeenCalledTimes(1);
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ error_message: 'boom', item_id: 1, status: 'failed' })
    );

    expect(harness.nodeExecution.settleRunning).toHaveBeenLastCalledWith(new Set(['node-1']), 'failed', 'boom');

    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 2 }), invocation_source_id: 'node-1' });
    expect(harness.nodeExecution.clearAll).toHaveBeenCalledTimes(2);

    harness.coordinator.detachRun('local-2');
    expect(harness.nodeExecution.settleRunning).toHaveBeenLastCalledWith(new Set(['node-1']), 'canceled');
  });

  it('ignores untracked queue events before mutating local execution state', () => {
    vi.useFakeTimers();
    harness = createHarness({ galleryRefreshCoalesceMs: 400 });
    harness.coordinator.connect();
    harness.callbacks.onGalleryRefresh.mockClear();

    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 99 }),
      invocation_source_id: 'node-1',
    });
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 99 }),
      invocation_source_id: 'node-1',
      message: 'other user progress',
      percentage: 0.5,
    });
    harness.socket.fire('invocation_complete', {
      ...createStatusEvent({ item_id: 99 }),
      invocation_source_id: 'node-1',
      result: { type: 'image_output' },
    });
    harness.socket.fire('invocation_error', {
      ...createStatusEvent({ item_id: 99 }),
      error_message: 'other user failure',
      error_type: 'Error',
      invocation_source_id: 'node-1',
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 99 }));

    expect(harness.nodeExecution.started).not.toHaveBeenCalled();
    expect(harness.nodeExecution.progress).not.toHaveBeenCalled();
    expect(harness.nodeExecution.completed).not.toHaveBeenCalled();
    expect(harness.nodeExecution.failed).not.toHaveBeenCalled();
    expect(harness.nodeExecution.settleRunning).not.toHaveBeenCalled();
    expect(harness.progressImage.clear).not.toHaveBeenCalled();
    expect(harness.progressImage.set).not.toHaveBeenCalled();
    expect(harness.callbacks.onGalleryRefresh).not.toHaveBeenCalled();
  });

  describe('reconcile', () => {
    it('looks up the exact receipt for pending runs without downloading queue history', async () => {
      harness.api.getEnqueueReceipt = vi.fn().mockResolvedValue({
        batchId: 'batch-9',
        enqueued: 1,
        itemIds: [7],
        requested: 1,
      });
      harness.api.getItem.mockResolvedValue(
        createQueueBackendItem({ batchId: 'batch-9', id: 7, origin: buildQueueItemOrigin('local-1', 'project-1') })
      );

      const outcomes = await harness.coordinator.reconcile([
        { id: 'local-1', projectId: 'project-1', status: 'pending' },
      ]);

      expect(harness.api.getEnqueueReceipt).toHaveBeenCalledWith('project-1', 'local-1');
      expect(harness.api.listItems).not.toHaveBeenCalled();
      expect(outcomes.get('local-1')).toEqual({ backendBatchId: 'batch-9', backendItemIds: [7], kind: 'adopted' });
    });

    it('does not resubmit an accepted run whose backend items were cleared', async () => {
      harness.api.getEnqueueReceipt = vi.fn().mockResolvedValue({
        batchId: 'batch-9',
        enqueued: 1,
        itemIds: [7],
        requested: 1,
      });
      harness.api.getItem.mockRejectedValue(new ApiError('not found', 404));

      const outcomes = await harness.coordinator.reconcile([
        { id: 'local-1', projectId: 'project-1', status: 'pending' },
      ]);

      expect(outcomes.get('local-1')).toEqual({
        backendBatchId: 'batch-9',
        backendItemIds: [7],
        kind: 'missing',
      });
      expect(harness.api.listItems).not.toHaveBeenCalled();
    });
    it('adopts pending items the backend already accepted, by origin', async () => {
      harness.api.listItems.mockResolvedValue([
        createQueueBackendItem({ batchId: 'batch-9', id: 7, origin: buildQueueItemOrigin('local-1') }),
      ]);

      const outcomes = await harness.coordinator.reconcile([{ id: 'local-1', status: 'pending' }]);

      expect(outcomes.get('local-1')).toEqual({ backendBatchId: 'batch-9', backendItemIds: [7], kind: 'adopted' });
      expect(harness.api.enqueueGenerate).not.toHaveBeenCalled();
    });

    it('never adopts persisted backend ids from another project or local run', async () => {
      harness.api.getItem.mockResolvedValue(
        createQueueBackendItem({ id: 7, origin: buildQueueItemOrigin('other-local', 'other-project') })
      );

      const outcomes = await harness.coordinator.reconcile([
        { backendItemIds: [7], id: 'local-1', projectId: 'project-1', status: 'running' },
      ]);

      expect(outcomes.get('local-1')).toEqual({ backendItemIds: [7], kind: 'missing' });
    });

    it('marks running items missing when their backend items vanished', async () => {
      harness.api.getItem.mockRejectedValue(new ApiError('not found', 404));

      const outcomes = await harness.coordinator.reconcile([{ backendItemIds: [7], id: 'local-1', status: 'running' }]);

      expect(outcomes.get('local-1')).toEqual({ backendItemIds: [7], kind: 'missing' });
      expect(harness.api.listItems).not.toHaveBeenCalled();
    });

    // Exercise real reconciliation with a completed utility item to prove it cannot enter project adoption or result
    // routing.
    it('never adopts a completed, result-carrying utility-origin item into a project queue item', async () => {
      harness.api.listItems.mockResolvedValue([
        createQueueBackendItem({
          batchId: 'util-batch',
          id: 99,
          origin: buildUtilityQueueItemOrigin('util-run-1'),
          status: 'completed',
        }),
      ]);

      const outcomes = await harness.coordinator.reconcile([{ id: 'local-1', status: 'pending' }]);

      // Utility origins cannot satisfy a pending project's backend trace; enqueue that project item fresh.
      expect(outcomes.get('local-1')).toEqual({ kind: 'enqueue' });
      expect(harness.api.getResultImages).not.toHaveBeenCalled();
      expect(harness.api.getItem).not.toHaveBeenCalledWith(99);
    });
  });

  it('fails runs whose backend items were pruned (404) during a sweep', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.api.getItem.mockRejectedValue(new ApiError('not found', 404));
    harness.socket.fire('connect', undefined);

    const failure = await resultsPromise.catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(QueueItemFailedError);
    // Vanished, not failed: nothing is left to show or retry, and a clear this tab asked for explains it.
    expect(failure).toMatchObject({
      failedBackendItemIds: [],
      failure: 'removed',
      message: expect.stringContaining('no longer on the backend'),
    });
  });

  it('prefers one batch cancellation request, falling back to item ids', async () => {
    await harness.coordinator.cancelRun({ backendBatchId: 'batch-1', backendItemIds: [1, 2] });

    expect(harness.api.cancelQueueItemsByBatchIds).toHaveBeenCalledWith(['batch-1']);
    expect(harness.api.cancelQueueItems).not.toHaveBeenCalled();

    await harness.coordinator.cancelRun({ backendItemIds: [1, 2] });

    expect(harness.api.cancelQueueItems).toHaveBeenCalledWith([1, 2]);
  });

  it('detaches socket listeners after dispose', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    expect(harness.progressEntries.size).toBe(1);

    harness.coordinator.dispose();

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    expect(harness.nodeExecution.settleRunning).not.toHaveBeenCalled();
    expect(harness.progressEntries.size).toBe(0);
  });

  it('does not adopt an enqueue response that completes after its account scope expires', async () => {
    const acceptance = deferred<{ batchId: string; enqueued: number; itemIds: number[]; requested: number }>();

    harness.api.enqueueGenerate.mockReturnValue(acceptance.promise);
    const submission = harness.coordinator.submitGenerate('local-1', generateRequest);

    accountLifecycle.invalidate();
    acceptance.resolve({ batchId: 'old-account-batch', enqueued: 1, itemIds: [42], requested: 1 });

    await expect(submission).rejects.toBeInstanceOf(QueueItemCancelledError);
    expect(harness.progressEntries.size).toBe(0);
  });
});
