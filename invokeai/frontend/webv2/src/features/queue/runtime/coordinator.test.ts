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
  buildQueueItemOrigin,
  buildUtilityQueueItemOrigin,
  type QueueItemStatusChangedEvent,
} from '@features/queue/data/events';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { createSocketHub, type BackendSocket } from '@platform/transport/socketHub';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createQueueCoordinator,
  QueueEnqueueNotAcceptedError,
  QueueItemCancelledError,
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
    hold: ReturnType<typeof vi.fn>;
    set: ReturnType<typeof vi.fn>;
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
    readProgressPreviews: undefined as
      | ReturnType<typeof vi.fn<() => Promise<QueueProgressPreviewPayload[]>>>
      | undefined,
  };
  const callbacks = {
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
  const progressImage = { bindSwapImages: vi.fn(), clear: vi.fn(), clearHeld: vi.fn(), hold: vi.fn(), set: vi.fn() };
  const activeProgressTarget = { clear: vi.fn(), set: vi.fn(), settle: vi.fn() } satisfies ActiveProgressTargetSink;
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

  it('rejects with the backend error message when a run fails', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ error_message: 'CUDA out of memory', item_id: 1, status: 'failed' })
    );

    await expect(resultsPromise).rejects.toThrow('CUDA out of memory');
  });

  it('rejects with QueueItemCancelledError when the backend cancels a run', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'canceled' }));

    await expect(resultsPromise).rejects.toBeInstanceOf(QueueItemCancelledError);
  });

  it('returns completed batch item results when sibling backend items are canceled', async () => {
    harness.callbacks.onBackendItemCancelled = vi.fn();
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 3,
      itemIds: [1, 2, 3],
      requested: 3,
    });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'canceled' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2 }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 3, status: 'canceled' }));

    const images = await resultsPromise;

    expect(images.map((image) => image.imageName)).toEqual(['image-2.png']);
    expect(harness.api.getResultImages).toHaveBeenCalledTimes(1);
    expect(harness.api.getResultImages).toHaveBeenCalledWith(2, 'local-1', '2026-06-10T00:00:00Z');
    expect(harness.callbacks.onBackendItemCancelled).toHaveBeenCalledWith('local-1', 1);
    expect(harness.callbacks.onBackendItemCancelled).toHaveBeenCalledWith('local-1', 3);
  });

  it('forwards result image extraction options when waiting for results', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z', {
      resultNodeIds: ['canvas_output'],
    });

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    await resultsPromise;

    expect(harness.api.getResultImages).toHaveBeenCalledWith(1, 'local-1', '2026-06-10T00:00:00Z', {
      resultNodeIds: ['canvas_output'],
    });
  });

  it('rejects with QueueItemCancelledError when every backend item in a batch is canceled', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'canceled' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'canceled' }));

    await expect(resultsPromise).rejects.toBeInstanceOf(QueueItemCancelledError);
    expect(harness.api.getResultImages).not.toHaveBeenCalled();
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

  it('rejects runs when the backend queue accepts no items', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({ batchId: 'batch-1', enqueued: 0, itemIds: [], requested: 1 });

    await expect(harness.coordinator.submitGenerate('local-1', generateRequest)).rejects.toBeInstanceOf(
      QueueEnqueueNotAcceptedError
    );
  });

  it('tracks every item when the backend queue accepts only part of the batch', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 2 });

    await expect(harness.coordinator.submitGenerate('local-1', generateRequest)).resolves.toEqual({
      batchId: 'batch-1',
      enqueued: 1,
      itemIds: [1],
      requested: 2,
    });
  });

  it('coalesces gallery refreshes across a burst of completions', async () => {
    vi.useFakeTimers();
    harness = createHarness({ galleryRefreshCoalesceMs: 400 });
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 10,
      itemIds: Array.from({ length: 10 }, (_, index) => index + 1),
      requested: 10,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    await vi.advanceTimersByTimeAsync(500); // flush the on-connect refresh
    harness.callbacks.onGalleryRefresh.mockClear();

    for (let itemId = 1; itemId <= 10; itemId += 1) {
      harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: itemId }));
    }

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.callbacks.onGalleryRefresh).toHaveBeenCalledTimes(1);
  });

  it('does not refresh the gallery for failed or canceled items', async () => {
    vi.useFakeTimers();
    harness = createHarness({ galleryRefreshCoalesceMs: 400 });
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    await vi.advanceTimersByTimeAsync(500); // flush the on-connect refresh
    harness.callbacks.onGalleryRefresh.mockClear();

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'failed' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'canceled' }));

    await vi.advanceTimersByTimeAsync(500);

    expect(harness.callbacks.onGalleryRefresh).not.toHaveBeenCalled();
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

  it('detaches one local run without disturbing another', async () => {
    harness.coordinator.connect();
    harness.api.enqueueGenerate
      .mockResolvedValueOnce({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })
      .mockResolvedValueOnce({ batchId: 'batch-2', enqueued: 1, itemIds: [2], requested: 1 });
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    await harness.coordinator.submitGenerate('local-2', { ...generateRequest, sourceQueueItemId: 'local-2' });
    const detachedResults = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');
    const survivingResults = harness.coordinator.waitForResults('local-2', '2026-06-10T00:00:00Z');

    harness.coordinator.detachRun('local-1');

    await expect(detachedResults).rejects.toBeInstanceOf(QueueItemCancelledError);
    expect(harness.progressImage.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    expect(harness.progressImage.clear).not.toHaveBeenCalledWith();

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'completed' }));
    await expect(survivingResults).resolves.toEqual([expect.objectContaining({ imageName: 'image-2.png' })]);
  });

  it('publishes the active target before a progress image is available', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      message: 'Starting denoise',
      percentage: 0.01,
    });

    expect(harness.activeProgressTarget.set).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    expect(harness.progressImage.set).not.toHaveBeenCalled();
  });

  it('keeps the followed slot and its last frame across a connection drop', async () => {
    // Keep the frame while disconnected; the backend continues and reconciliation finds its outcome.
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      image: { dataURL: 'data:image/png;base64,frame', height: 32, width: 64 },
      message: 'Denoising',
      percentage: 0.5,
    });

    harness.hub.disconnect();

    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalled();
    expect(harness.progressImage.clear).not.toHaveBeenCalled();
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

  it('runs a sweep requested while one is in flight, instead of dropping it', async () => {
    // The visibility sweep often fires while the network is still coming back;
    // the reconnect sweep a second later is the one that can reach the backend.
    const firstRead = deferred<QueueBackendItem>();

    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    harness.api.getItem.mockClear();
    harness.api.getItem
      .mockReturnValueOnce(firstRead.promise)
      .mockResolvedValueOnce(createQueueBackendItem({ id: 1, status: 'completed' }));
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.hub.disconnect();
    harness.hub.connect();
    await Promise.resolve();
    expect(harness.api.getItem).toHaveBeenCalledTimes(1);

    harness.hub.disconnect();
    harness.hub.connect();
    firstRead.resolve(createQueueBackendItem({ id: 1, status: 'in_progress' }));

    await expect(resultsPromise).resolves.toEqual([expect.objectContaining({ imageName: 'image-1.png' })]);
    expect(harness.api.getItem).toHaveBeenCalledTimes(2);
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

  it('fetches the preview snapshot after re-adopting runs on reload', async () => {
    harness.api.readProgressPreviews = vi.fn(() =>
      Promise.resolve([
        toPreviewPayload({
          ...createStatusEvent({ item_id: 1 }),
          image: { dataURL: 'data:image/png;base64,snapshot', height: 32, width: 64 },
          invocation_source_id: 'denoise',
          message: 'Denoising',
          percentage: 0.5,
          revision: 4,
        }),
      ])
    );
    harness.api.getItem.mockResolvedValue(
      createQueueBackendItem({ id: 1, origin: buildQueueItemOrigin('local-1', 'project-1'), status: 'in_progress' })
    );
    harness.coordinator.connect();

    const outcomes = await harness.coordinator.reconcile([
      { backendItemIds: [1], id: 'local-1', projectId: 'project-1', status: 'running' },
    ]);
    await Promise.resolve();
    await Promise.resolve();

    expect(outcomes.get('local-1')?.kind).toBe('resumed');
    expect(harness.progressImage.set).toHaveBeenCalledWith(
      { dataUrl: 'data:image/png;base64,snapshot', height: 32, width: 64 },
      { itemIndex: 1, queueItemId: 'local-1' }
    );
  });

  it('publishes a running session before its first preview and on reconnect', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    expect(harness.activeProgressTarget.set).not.toHaveBeenCalled();
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'in_progress' }));
    expect(harness.activeProgressTarget.set).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    expect(harness.progressImage.set).not.toHaveBeenCalled();
    harness.activeProgressTarget.set.mockClear();
    harness.api.getItem.mockResolvedValue(
      createQueueBackendItem({ id: 1, origin: buildQueueItemOrigin('local-1', 'project-1'), status: 'in_progress' })
    );
    await harness.coordinator.reconcile([
      { backendItemIds: [1], id: 'local-1', projectId: 'project-1', status: 'running' },
    ]);
    expect(harness.activeProgressTarget.set).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
  });

  it.each(['pending', 'waiting'] as const)(
    'clears a running session when reconciliation discovers a missed %s transition',
    async (status) => {
      harness.coordinator.connect();
      await harness.coordinator.submitGenerate('local-1', generateRequest);
      harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'in_progress' }));
      harness.api.getItem.mockResolvedValue(
        createQueueBackendItem({ id: 1, origin: buildQueueItemOrigin('local-1', 'project-1'), status })
      );
      await harness.coordinator.reconcile([
        { backendItemIds: [1], id: 'local-1', projectId: 'project-1', status: 'running' },
      ]);
      expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
    }
  );

  it('reopens the revision gate when an item goes back to waiting', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const frame = (revision: number, dataURL: string) => ({
      ...createStatusEvent({ item_id: 1 }),
      image: { dataURL, height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.5,
      revision,
    });

    harness.socket.fire('invocation_progress', frame(5, 'data:image/png;base64,first-leg'));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'waiting' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'in_progress' }));
    harness.socket.fire('invocation_progress', frame(1, 'data:image/png;base64,second-leg'));

    expect(harness.progressImage.set.mock.calls.map(([image]) => (image as { dataUrl: string }).dataUrl)).toEqual([
      'data:image/png;base64,first-leg',
      'data:image/png;base64,second-leg',
    ]);
  });

  it('drops a preview frame whose revision is not newer than the one already shown', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const frame = (revision: number, dataURL: string) => ({
      ...createStatusEvent({ item_id: 1 }),
      image: { dataURL, height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.5,
      revision,
    });

    harness.socket.fire('invocation_progress', frame(2, 'data:image/png;base64,second'));
    harness.socket.fire('invocation_progress', frame(2, 'data:image/png;base64,duplicate-second'));
    harness.socket.fire('invocation_progress', frame(1, 'data:image/png;base64,first'));
    harness.socket.fire('invocation_progress', frame(3, 'data:image/png;base64,third'));
    // A new session on the same item starts over.
    harness.socket.fire('invocation_progress', { ...frame(1, 'data:image/png;base64,retry'), session_id: 'session-2' });

    expect(harness.progressImage.set.mock.calls.map(([image]) => (image as { dataUrl: string }).dataUrl)).toEqual([
      'data:image/png;base64,second',
      'data:image/png;base64,third',
      'data:image/png;base64,retry',
    ]);
    expect(harness.activeProgressTarget.set).toHaveBeenCalledTimes(3);
    expect(harness.nodeExecution.progress).toHaveBeenCalledTimes(3);
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

  it('notifies when one backend item in a batch completes', async () => {
    harness.callbacks.onBackendItemComplete = vi.fn();
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    expect(harness.callbacks.onBackendItemComplete).toHaveBeenCalledWith('local-1', 1);
  });

  it('routes progress images to the active image slot inside a submitted batch', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2 }),
      image: { dataURL: 'data:image/png;base64,abc', height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.5,
    });

    expect(harness.progressImage.set).toHaveBeenCalledWith(
      { dataUrl: 'data:image/png;base64,abc', height: 32, width: 64 },
      { itemIndex: 2, queueItemId: 'local-1' }
    );
  });

  it('routes model load socket events to the model-load port', () => {
    harness.coordinator.connect();

    harness.socket.fire('model_load_started', { config: { name: 'model-a' } });
    harness.socket.fire('model_load_complete', { config: { name: 'model-a' } });

    expect(harness.modelLoads.started).toHaveBeenCalledWith({ config: { name: 'model-a' } });
    expect(harness.modelLoads.completed).toHaveBeenCalledWith({ config: { name: 'model-a' } });
  });

  it('resets model-load activity when the connection status changes', () => {
    harness.coordinator.connect();
    harness.modelLoads.reset.mockClear();

    harness.hub.disconnect();

    expect(harness.modelLoads.reset).toHaveBeenCalled();
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

  it('replays a child preview received before the enqueue response', async () => {
    const acceptance = deferred<QueueEnqueueResult>();

    harness.api.enqueueWorkflow.mockReturnValue(acceptance.promise);
    harness.coordinator.connect();
    const submission = harness.coordinator.submitWorkflow('local-1', workflowRequest);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2 }),
      image: { dataURL: 'data:image/png;base64:child', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 1,
      session_id: 'child-session',
      workflow_call_parent_source_id: 'call-node',
    });

    expect(harness.progressImage.set).not.toHaveBeenCalled();

    acceptance.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    await submission;

    expect(harness.progressImage.set).toHaveBeenCalledWith(
      { dataUrl: 'data:image/png;base64:child', height: 32, width: 64 },
      { itemIndex: 1, queueItemId: 'local-1' }
    );
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

  it('does not let child terminal events settle the visible call node', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    const child = {
      ...createStatusEvent({ item_id: 2 }),
      invocation_source_id: 'child-node',
      root_item_id: 1,
      workflow_call_parent_source_id: 'call-node',
    };
    harness.socket.fire('invocation_complete', { ...child, result: { type: 'integer_output', value: 1 } });
    harness.socket.fire('invocation_error', { ...child, error_message: 'child failed', error_type: 'ValueError' });

    expect(harness.nodeExecution.completed).not.toHaveBeenCalled();
    expect(harness.nodeExecution.failed).not.toHaveBeenCalled();
    expect(harness.nodeExecution.settleRunning).not.toHaveBeenCalled();
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
  });

  it('routes sequential child and grandchild preview frames to the same root slot', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('workflow-root', savedWorkflowCallRequest());
    const target = { itemIndex: 1, queueItemId: 'workflow-root' };
    const frame = (itemId: number, parentItemId: number, revision: number, dataURL: string) => ({
      ...createStatusEvent({ item_id: itemId, status: 'in_progress' }),
      image: { dataURL, height: 32, width: 64 },
      invocation_source_id: 'shared-child-node',
      message: 'Child sampling',
      parent_item_id: parentItemId,
      percentage: revision / 4,
      revision,
      root_item_id: 1,
      session_id: `child-session-${itemId}`,
      workflow_call_parent_source_id: 'call-node',
    });

    harness.socket.fire('invocation_progress', frame(2, 1, 1, 'data:image/png;base64,child-first'));
    harness.socket.fire('invocation_progress', frame(2, 1, 2, 'data:image/png;base64,child-second'));
    harness.socket.fire('invocation_progress', frame(3, 2, 1, 'data:image/png;base64,grandchild'));

    expect(harness.activeProgressTarget.set.mock.calls.map(([activeTarget]) => activeTarget)).toEqual([
      target,
      target,
      target,
    ]);
    expect(
      harness.progressImage.set.mock.calls.map(([image, imageTarget]) => ({ image, target: imageTarget }))
    ).toEqual([
      {
        image: { dataUrl: 'data:image/png;base64,child-first', height: 32, width: 64 },
        target,
      },
      {
        image: { dataUrl: 'data:image/png;base64,child-second', height: 32, width: 64 },
        target,
      },
      {
        image: { dataUrl: 'data:image/png;base64,grandchild', height: 32, width: 64 },
        target,
      },
    ]);
  });

  it('keeps identical child node ids isolated across concurrent roots', async () => {
    harness.api.enqueueWorkflow
      .mockResolvedValueOnce({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })
      .mockResolvedValueOnce({ batchId: 'batch-2', enqueued: 1, itemIds: [2], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('root-one', savedWorkflowCallRequest());
    await harness.coordinator.submitWorkflow('root-two', savedWorkflowCallRequest());
    const targetOne = { itemIndex: 1, queueItemId: 'root-one' };
    const targetTwo = { itemIndex: 1, queueItemId: 'root-two' };
    const frame = (itemId: number, rootItemId: number, label: string) => ({
      ...createStatusEvent({ item_id: itemId, status: 'in_progress' }),
      image: { dataURL: `data:image/png;base64,${label}`, height: 32, width: 64 },
      invocation_source_id: 'shared-child-node',
      message: 'Child sampling',
      parent_item_id: rootItemId,
      percentage: 0.5,
      revision: 1,
      root_item_id: rootItemId,
      session_id: `child-session-${rootItemId}`,
      workflow_call_parent_source_id: 'same-call-node',
    });

    harness.socket.fire('invocation_progress', frame(11, 1, 'root-one-frame'));
    harness.socket.fire('invocation_progress', frame(12, 2, 'root-two-frame'));

    expect(harness.progressImage.set.mock.calls.map(([image, target]) => ({ image, target }))).toEqual([
      {
        image: { dataUrl: 'data:image/png;base64,root-one-frame', height: 32, width: 64 },
        target: targetOne,
      },
      {
        image: { dataUrl: 'data:image/png;base64,root-two-frame', height: 32, width: 64 },
        target: targetTwo,
      },
    ]);
  });

  it.each(['socket status', 'visibility sweep'] as const)(
    'keeps a child preview eligible when parent waiting arrives through %s',
    async (delivery) => {
      const listeners = new Map<string, () => void>();
      vi.stubGlobal('document', {
        addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
        removeEventListener: (type: string) => listeners.delete(type),
        visibilityState: 'visible',
      });

      try {
        harness.coordinator.connect();
        await harness.coordinator.submitWorkflow('local-1', workflowRequest);
        const target = { itemIndex: 1, queueItemId: 'local-1' };
        const childFrame = (revision: number, dataURL: string) => ({
          ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
          image: { dataURL, height: 32, width: 64 },
          invocation_source_id: 'child-node',
          message: 'Child sampling',
          percentage: 0.5,
          revision,
          root_item_id: 1,
          session_id: 'child-session',
          workflow_call_parent_source_id: 'call-node',
        });

        harness.socket.fire('invocation_progress', childFrame(1, 'data:image/png;base64,first'));

        if (delivery === 'socket status') {
          harness.socket.fire(
            'queue_item_status_changed',
            createStatusEvent({ item_id: 1, status: 'waiting', status_sequence: 2 })
          );
        } else {
          const waitingRead = deferred<QueueBackendItem>();
          harness.api.getItem.mockReturnValueOnce(waitingRead.promise);
          listeners.get('visibilitychange')?.();
          expect(harness.api.getItem).toHaveBeenCalledWith(1);
          waitingRead.resolve(createQueueBackendItem({ id: 1, status: 'waiting' }));
          await Promise.resolve();
        }

        expect(harness.activeProgressTarget.clear).not.toHaveBeenCalledWith(target);
        expect(harness.progressImage.clear).not.toHaveBeenCalledWith(target);

        harness.socket.fire('invocation_progress', childFrame(2, 'data:image/png;base64,second'));

        expect(harness.activeProgressTarget.set).toHaveBeenLastCalledWith(target);
        expect(harness.progressImage.set.mock.calls.map(([image]) => (image as { dataUrl: string }).dataUrl)).toEqual([
          'data:image/png;base64,first',
          'data:image/png;base64,second',
        ]);
      } finally {
        vi.unstubAllGlobals();
      }
    }
  );

  it('keeps the last root frame while waiting before the first child frame', async () => {
    harness.coordinator.connect();
    const request = {
      ...workflowRequest,
      graph: {
        ...workflowRequest.graph,
        nodes: { 'call-node': { id: 'call-node', type: 'call_saved_workflow' } },
      },
    };
    await harness.coordinator.submitWorkflow('local-1', request);
    const target = { itemIndex: 1, queueItemId: 'local-1' };

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,root', height: 32, width: 64 },
      invocation_source_id: 'call-node',
      message: 'Calling workflow',
      percentage: 0.4,
      revision: 1,
      session_id: 'root-session',
    });
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 1, status: 'waiting', status_sequence: 2 })
    );

    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalledWith(target);
    expect(harness.progressImage.clear).not.toHaveBeenCalledWith(target);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,child', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 1,
      session_id: 'child-session',
      workflow_call_parent_source_id: 'call-node',
    });

    expect(harness.activeProgressTarget.set).toHaveBeenLastCalledWith(target);
    expect(harness.progressImage.set).toHaveBeenLastCalledWith(
      { dataUrl: 'data:image/png;base64,child', height: 32, width: 64 },
      target
    );
  });

  it('keeps the root preview eligible through child completion and parent resume', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);
    const target = { itemIndex: 1, queueItemId: 'local-1' };

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,child', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 1,
      session_id: 'child-session',
      workflow_call_parent_source_id: 'call-node',
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'waiting' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'completed' }));
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 1, status: 'pending', status_sequence: 2 })
    );
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ item_id: 1, status: 'in_progress', status_sequence: 3 })
    );

    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalledWith(target);
    expect(harness.activeProgressTarget.set).toHaveBeenLastCalledWith(target);
    expect(harness.progressImage.clear).not.toHaveBeenCalledWith(target);
  });

  it('recognizes a queued saved-workflow call before its first child event', async () => {
    const request = {
      ...workflowRequest,
      graph: {
        ...workflowRequest.graph,
        nodes: { call: { id: 'call', type: 'call_saved_workflow' } },
      },
    };
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', request);
    const target = { itemIndex: 1, queueItemId: 'local-1' };

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,parent-frame', height: 32, width: 64 },
      invocation_source_id: 'call',
      message: 'Calling saved workflow',
      percentage: 0.5,
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'waiting' }));

    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalledWith(target);
    expect(harness.progressImage.clear).not.toHaveBeenCalledWith(target);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,child-frame', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      root_item_id: 1,
      workflow_call_parent_source_id: 'call',
    });

    expect(harness.activeProgressTarget.set).toHaveBeenLastCalledWith(target);
    expect(harness.progressImage.set).toHaveBeenLastCalledWith(
      { dataUrl: 'data:image/png;base64,child-frame', height: 32, width: 64 },
      target
    );
  });

  it('preserves a recognized child route when reconciliation finds the parent waiting', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);
    const target = { itemIndex: 1, queueItemId: 'local-1' };

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,child-frame', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      root_item_id: 1,
      workflow_call_parent_source_id: 'call-node',
    });
    harness.activeProgressTarget.clear.mockClear();
    harness.api.getItem.mockResolvedValue(
      createQueueBackendItem({
        id: 1,
        origin: buildQueueItemOrigin('local-1', 'project-1'),
        status: 'waiting',
      })
    );

    await harness.coordinator.reconcile([
      { backendItemIds: [1], id: 'local-1', projectId: 'project-1', status: 'running' },
    ]);

    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalledWith(target);
    expect(harness.progressImage.clear).not.toHaveBeenCalledWith(target);
  });

  it('still clears a workflow preview while waiting when no child call is known', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);
    const target = { itemIndex: 1, queueItemId: 'local-1' };

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,parent-frame', height: 32, width: 64 },
      invocation_source_id: 'ordinary-node',
      message: 'Running workflow',
      percentage: 0.5,
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'waiting' }));

    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith(target);
    expect(harness.progressImage.set).toHaveBeenCalledWith(
      { dataUrl: 'data:image/png;base64,parent-frame', height: 32, width: 64 },
      target
    );
    expect(harness.progressImage.clear).not.toHaveBeenCalledWith(target);
  });

  it('clears the root preview target when a saved-workflow run reaches terminal status', async () => {
    const request = {
      ...workflowRequest,
      graph: {
        ...workflowRequest.graph,
        nodes: { call: { id: 'call', type: 'call_saved_workflow' } },
      },
    };
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', request);
    const target = { itemIndex: 1, queueItemId: 'local-1' };

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,parent-frame', height: 32, width: 64 },
      invocation_source_id: 'call',
      message: 'Calling saved workflow',
      percentage: 0.5,
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'waiting' }));

    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalledWith(target);

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'completed' }));

    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith(target);
    expect(harness.progressImage.clear).toHaveBeenCalledWith(target);
  });

  it('ignores a child preview whose root is not tracked', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,unrelated', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 99,
      session_id: 'unrelated-child-session',
      workflow_call_parent_source_id: 'call-node',
    });

    expect(harness.activeProgressTarget.set).not.toHaveBeenCalled();
    expect(harness.progressImage.set).not.toHaveBeenCalled();
    expect(harness.nodeExecution.progress).not.toHaveBeenCalled();
  });

  it('isolates a saved-workflow preview from status changes on another root', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-other',
      enqueued: 1,
      itemIds: [2],
      requested: 1,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('workflow-root', savedWorkflowCallRequest());
    await harness.coordinator.submitGenerate('other-root', generateRequest);

    const workflowTarget = { itemIndex: 1, queueItemId: 'workflow-root' };
    const otherTarget = { itemIndex: 1, queueItemId: 'other-root' };
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,workflow-root', height: 32, width: 64 },
      invocation_source_id: 'call-node',
      message: 'Calling workflow',
      percentage: 0.4,
      revision: 1,
      session_id: 'root-session',
    });
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,other-root', height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.5,
      revision: 1,
      session_id: 'other-session',
    });

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'waiting' }));

    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith(otherTarget);
    expect(harness.activeProgressTarget.clear).not.toHaveBeenCalledWith(workflowTarget);
    expect(harness.progressImage.clear).not.toHaveBeenCalledWith(workflowTarget);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 3, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,workflow-child', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 1,
      session_id: 'child-session',
      workflow_call_parent_source_id: 'call-node',
    });

    expect(harness.progressImage.set).toHaveBeenLastCalledWith(
      { dataUrl: 'data:image/png;base64,workflow-child', height: 32, width: 64 },
      workflowTarget
    );
  });

  it('routes sibling child frames to their own batch slots when node ids match', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', {
      ...savedWorkflowCallRequest(),
      batchCount: 2,
    });

    const frame = (itemId: number, rootItemId: number, label: string) => ({
      ...createStatusEvent({ item_id: itemId, status: 'in_progress' }),
      image: { dataURL: `data:image/png;base64,${label}`, height: 32, width: 64 },
      invocation_source_id: 'same-child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: rootItemId,
      session_id: `child-session-${itemId}`,
      workflow_call_parent_source_id: 'call-node',
    });

    harness.socket.fire('invocation_progress', frame(11, 1, 'slot-one'));
    harness.socket.fire('invocation_progress', frame(12, 2, 'slot-two'));

    expect(harness.progressImage.set.mock.calls.map(([, target]) => target)).toEqual([
      { itemIndex: 1, queueItemId: 'local-1' },
      { itemIndex: 2, queueItemId: 'local-1' },
    ]);
    expect(harness.progressImage.set.mock.calls.map(([image]) => (image as { dataUrl: string }).dataUrl)).toEqual([
      'data:image/png;base64,slot-one',
      'data:image/png;base64,slot-two',
    ]);
  });

  it('does not borrow another root frame when this workflow root has no frame', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-other',
      enqueued: 1,
      itemIds: [2],
      requested: 1,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('workflow-root', savedWorkflowCallRequest());
    await harness.coordinator.submitGenerate('other-root', generateRequest);

    const otherTarget = { itemIndex: 1, queueItemId: 'other-root' };
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,other-root-only', height: 32, width: 64 },
      invocation_source_id: 'denoise',
      message: 'Denoising',
      percentage: 0.5,
      revision: 1,
      session_id: 'other-session',
    });
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1, status: 'in_progress' }),
      image: null,
      invocation_source_id: 'call-node',
      message: 'Calling workflow',
      percentage: 0.4,
      session_id: 'root-session',
    });
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 3, status: 'in_progress' }),
      image: null,
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      root_item_id: 1,
      session_id: 'child-session',
      workflow_call_parent_source_id: 'call-node',
    });

    expect(harness.progressImage.set.mock.calls).toEqual([
      [{ dataUrl: 'data:image/png;base64,other-root-only', height: 32, width: 64 }, otherTarget],
    ]);
  });

  it.each(['completed', 'failed', 'canceled'] as const)(
    'retires a saved-workflow preview after root %s and ignores late child frames',
    async (status) => {
      harness.coordinator.connect();
      await harness.coordinator.submitWorkflow('workflow-root', savedWorkflowCallRequest());
      const target = { itemIndex: 1, queueItemId: 'workflow-root' };
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
      harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status }));

      expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith(target);
      expect(harness.progressImage.clear).toHaveBeenCalledWith(target);
      const progressSetCount = harness.progressImage.set.mock.calls.length;
      const activeSetCount = harness.activeProgressTarget.set.mock.calls.length;
      const nodeProgressCount = harness.nodeExecution.progress.mock.calls.length;

      harness.socket.fire('invocation_progress', { ...childFrame, revision: 2 });

      expect(harness.progressImage.set).toHaveBeenCalledTimes(progressSetCount);
      expect(harness.activeProgressTarget.set).toHaveBeenCalledTimes(activeSetCount);
      expect(harness.nodeExecution.progress).toHaveBeenCalledTimes(nodeProgressCount);
    }
  );

  it('retires a saved-workflow preview on owner-scoped bulk cancellation', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('workflow-root', savedWorkflowCallRequest());
    const target = { itemIndex: 1, queueItemId: 'workflow-root' };
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 3, status: 'in_progress' }),
      image: { dataURL: 'data:image/png;base64,child', height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 1,
      session_id: 'child-session',
      workflow_call_parent_source_id: 'call-node',
    });

    harness.socket.fire('queue_items_canceled', {
      canceled_item_ids: [1],
      canceled_item_ids_by_user: { 'user-1': [1] },
      queue_id: 'default',
      timestamp: 2,
      user_ids: ['user-1'],
    });

    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith(target);
    expect(harness.progressImage.clear).toHaveBeenCalledWith(target);
  });

  it('retires a detached workflow preview and ignores late child frames', async () => {
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('workflow-root', savedWorkflowCallRequest());
    const target = { itemIndex: 1, queueItemId: 'workflow-root' };
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

    harness.coordinator.detachRun('workflow-root');

    expect(harness.activeProgressTarget.clear).toHaveBeenCalledWith(target);
    expect(harness.progressImage.clear).toHaveBeenCalledWith(target);
    harness.socket.fire('invocation_progress', { ...childFrame, revision: 2 });
    expect(harness.progressImage.set).toHaveBeenCalledTimes(progressSetCount);
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

  it('allows a child preview revision again after the child reaches terminal status', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    const childFrame = {
      ...createStatusEvent({ item_id: 2 }),
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
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'completed' }));
    harness.socket.fire('invocation_progress', childFrame);

    expect(harness.progressImage.set).toHaveBeenCalledTimes(2);
  });

  it('bounds preview gates when child terminal events are missed', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    const frame = (item_id: number) => ({
      ...createStatusEvent({ item_id }),
      image: { dataURL: `data:image/png;base64,child-${item_id}`, height: 32, width: 64 },
      invocation_source_id: 'child-node',
      message: 'Child sampling',
      percentage: 0.5,
      revision: 1,
      root_item_id: 1,
      session_id: `child-session-${item_id}`,
      workflow_call_parent_source_id: 'call-node',
    });

    for (let itemId = 2; itemId <= 1026; itemId += 1) {
      harness.socket.fire('invocation_progress', frame(itemId));
    }

    // Item 2's gate was evicted by the bounded fallback, so a missed terminal
    // event cannot leave it permanently stale if the id is reused.
    harness.socket.fire('invocation_progress', frame(2));

    expect(harness.progressImage.set).toHaveBeenCalledTimes(1026);
  });

  it('does not buffer node events while no enqueue request is in flight', async () => {
    harness.api.enqueueWorkflow.mockResolvedValueOnce({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 1 }), invocation_source_id: 'node-1' });

    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    expect(harness.nodeExecution.started).not.toHaveBeenCalled();
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

  it('skips replayed events and a late settle from an item another item has since taken over', async () => {
    const acceptance = deferred<QueueEnqueueResult>();

    harness.api.enqueueWorkflow
      .mockReturnValueOnce(acceptance.promise)
      .mockResolvedValueOnce({ batchId: 'batch-2', enqueued: 1, itemIds: [2], requested: 1 });
    harness.coordinator.connect();
    const submission = harness.coordinator.submitWorkflow('local-1', workflowRequest);

    // Item 1 ran while its enqueue response was still in flight; those events are buffered.
    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 1 }), invocation_source_id: 'node-1' });
    harness.socket.fire('invocation_complete', {
      ...createStatusEvent({ item_id: 1 }),
      invocation_source_id: 'node-1',
      result: { type: 'integer_output', value: 1 },
    });
    await harness.coordinator.submitWorkflow('local-2', { ...workflowRequest, sourceQueueItemId: 'local-2' });
    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 2 }), invocation_source_id: 'node-1' });
    harness.nodeExecution.clearAll.mockClear();

    acceptance.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    await submission;
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'completed' }));

    expect(harness.nodeExecution.clearAll).not.toHaveBeenCalled();
    expect(harness.nodeExecution.completed).not.toHaveBeenCalled();
    expect(harness.nodeExecution.settleRunning).not.toHaveBeenCalled();

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'completed' }));
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

  it('names the project workflow a run came from when its nodes start, and forgets it for unattributed runs', async () => {
    harness.api.enqueueWorkflow
      .mockResolvedValueOnce({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })
      .mockResolvedValueOnce({ batchId: 'batch-2', enqueued: 1, itemIds: [2], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest, {
      projectId: 'project-1',
      workflowId: 'wf-a',
    });
    await harness.coordinator.submitWorkflow('local-2', { ...workflowRequest, sourceQueueItemId: 'local-2' });

    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 1 }), invocation_source_id: 'node-1' });
    expect(harness.nodeExecution.setOrigin).toHaveBeenLastCalledWith({ projectId: 'project-1', workflowId: 'wf-a' });

    // A second copy with the same node id starts: the store is cleared and re-attributed before its state lands.
    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 2 }), invocation_source_id: 'node-1' });
    expect(harness.nodeExecution.clearAll).toHaveBeenCalled();
    expect(harness.nodeExecution.setOrigin).toHaveBeenLastCalledWith(null);
  });

  it('attributes reconciled runs from their recorded origin', async () => {
    harness.api.getItem.mockResolvedValue(
      createQueueBackendItem({ id: 7, origin: buildQueueItemOrigin('local-7', 'project-1'), status: 'in_progress' })
    );
    harness.coordinator.connect();
    await harness.coordinator.reconcile([
      {
        backendBatchId: 'batch-1',
        backendItemIds: [7],
        id: 'local-7',
        origin: { projectId: 'project-1', workflowId: 'wf-b' },
        projectId: 'project-1',
        status: 'running',
      },
    ]);

    harness.socket.fire('invocation_started', { ...createStatusEvent({ item_id: 7 }), invocation_source_id: 'node-1' });

    expect(harness.nodeExecution.setOrigin).toHaveBeenLastCalledWith({ projectId: 'project-1', workflowId: 'wf-b' });
  });

  it('preserves a root failure when queue status precedes the root invocation error', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 1 }),
      invocation_source_id: 'call-node',
    });
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ error_message: 'Workflow failed', item_id: 1, status: 'failed' })
    );

    expect(harness.nodeExecution.settleRunning).toHaveBeenLastCalledWith(
      new Set(['call-node']),
      'failed',
      'Workflow failed'
    );
  });

  it('settles the root node after child activity when the root queue status arrives first', async () => {
    harness.api.enqueueWorkflow.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 });
    harness.coordinator.connect();
    await harness.coordinator.submitWorkflow('local-1', workflowRequest);

    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 1 }),
      invocation_source_id: 'call-node',
    });
    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 2 }),
      invocation_source_id: 'child-node',
      root_item_id: 1,
      workflow_call_parent_source_id: 'call-node',
    });
    harness.socket.fire(
      'queue_item_status_changed',
      createStatusEvent({ error_message: 'Child workflow failed', item_id: 1, status: 'failed' })
    );

    expect(harness.nodeExecution.settleRunning).toHaveBeenLastCalledWith(
      new Set(['call-node']),
      'failed',
      'Child workflow failed'
    );
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

  it('tracks the active image index inside a submitted batch', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);

    expect(harness.progressEntries.get('local-1')).toEqual({
      activeItemIndex: undefined,
      completedItemCount: 0,
      message: '',
      percentage: null,
      totalItemCount: 2,
    });

    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

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
      totalItemCount: 2,
    });

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    // Clear active slot index between items while preserving completion tracking.
    expect(harness.progressEntries.get('local-1')).toEqual({
      activeItemIndex: undefined,
      completedItemCount: 1,
      message: '',
      percentage: null,
      totalItemCount: 2,
    });

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2 }),
      message: 'Denoising',
      percentage: 0.25,
    });

    expect(harness.progressEntries.get('local-1')).toEqual({
      activeItemIndex: 2,
      completedItemCount: 1,
      message: 'Denoising',
      percentage: 0.25,
      totalItemCount: 2,
    });

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2 }));

    await resultsPromise;
  });

  it('does not clear a newer active slot when an older terminal event arrives late', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({
      batchId: 'batch-1',
      enqueued: 2,
      itemIds: [1, 2],
      requested: 2,
    });
    harness.coordinator.connect();
    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 1 }),
      message: 'First slot',
      percentage: 0.9,
    });
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 2 }),
      message: 'Second slot',
      percentage: 0.2,
    });
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    expect(harness.progressEntries.get('local-1')).toEqual({
      activeItemIndex: 2,
      completedItemCount: 1,
      message: 'Second slot',
      percentage: 0.2,
      totalItemCount: 2,
    });
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

    it('resumes running items and settles them from their listed terminal status', async () => {
      harness.api.getItem.mockResolvedValue(
        createQueueBackendItem({ id: 7, origin: buildQueueItemOrigin('local-1'), status: 'completed' })
      );

      const outcomes = await harness.coordinator.reconcile([{ backendItemIds: [7], id: 'local-1', status: 'running' }]);

      expect(outcomes.get('local-1')).toEqual({ kind: 'resumed' });
      expect(harness.api.listItems).not.toHaveBeenCalled();

      const images = await harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

      expect(images.map((image) => image.imageName)).toEqual(['image-7.png']);
    });

    it('bounds backend reads while reconciling and collecting a large run', async () => {
      const backendItemIds = Array.from({ length: 64 }, (_, index) => index + 1);
      let activeItemReads = 0;
      let maxItemReads = 0;
      let activeResultReads = 0;
      let maxResultReads = 0;
      harness.api.getItem.mockImplementation(async (itemId: number) => {
        activeItemReads += 1;
        maxItemReads = Math.max(maxItemReads, activeItemReads);
        await new Promise((resolve) => {
          setTimeout(resolve, 1);
        });
        activeItemReads -= 1;
        return createQueueBackendItem({
          id: itemId,
          origin: buildQueueItemOrigin('local-1', 'project-1'),
          status: 'completed',
        });
      });
      harness.api.getResultImages.mockImplementation(async (itemId: number, sourceQueueItemId: string) => {
        activeResultReads += 1;
        maxResultReads = Math.max(maxResultReads, activeResultReads);
        await new Promise((resolve) => {
          setTimeout(resolve, 1);
        });
        activeResultReads -= 1;
        return [createImage(`image-${itemId}.png`, sourceQueueItemId)];
      });

      await harness.coordinator.reconcile([
        { backendItemIds, id: 'local-1', projectId: 'project-1', status: 'running' },
      ]);
      const images = await harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

      expect(images).toHaveLength(64);
      expect(maxItemReads).toBeLessThanOrEqual(16);
      expect(maxResultReads).toBeLessThanOrEqual(16);
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

    it('resumes the surviving items when part of an accepted batch was pruned', async () => {
      harness.api.getItem.mockImplementation((itemId: number) =>
        itemId === 7
          ? Promise.reject(new ApiError('not found', 404))
          : Promise.resolve(
              createQueueBackendItem({
                id: itemId,
                origin: buildQueueItemOrigin('local-1', 'project-1'),
                status: 'completed',
              })
            )
      );

      const outcomes = await harness.coordinator.reconcile([
        { backendItemIds: [7, 8], id: 'local-1', projectId: 'project-1', status: 'running' },
      ]);

      expect(outcomes.get('local-1')).toEqual({
        backendItemIds: [8],
        kind: 'resumed',
        missingBackendItemIds: [7],
      });
      await expect(harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z')).resolves.toEqual([
        expect.objectContaining({ imageName: 'image-8.png' }),
      ]);
    });

    it('asks for a fresh enqueue when a pending item left no backend trace', async () => {
      harness.api.listItems.mockResolvedValue([]);

      const outcomes = await harness.coordinator.reconcile([{ id: 'local-1', status: 'pending' }]);

      expect(outcomes.get('local-1')).toEqual({ kind: 'enqueue' });
    });

    it('skips the backend round-trip when there is nothing to reconcile', async () => {
      const outcomes = await harness.coordinator.reconcile([]);

      expect(outcomes.size).toBe(0);
      expect(harness.api.listItems).not.toHaveBeenCalled();
    });

    // Exercise real reconciliation with a completed utility item to prove it cannot enter project adoption or
    // result routing.
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

  it('settles missed events through the safety sweep on reconnect', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.api.getItem.mockResolvedValue(createQueueBackendItem({ id: 1, status: 'completed' }));
    harness.socket.fire('connect', undefined);

    const images = await resultsPromise;

    expect(images.map((image) => image.imageName)).toEqual(['image-1.png']);
  });

  it('fails runs whose backend items were pruned (404) during a sweep', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.api.getItem.mockRejectedValue(new ApiError('not found', 404));
    harness.socket.fire('connect', undefined);

    await expect(resultsPromise).rejects.toThrow('no longer on the backend queue');
  });

  it('prefers one batch cancellation request, falling back to item ids', async () => {
    await harness.coordinator.cancelRun({ backendBatchId: 'batch-1', backendItemIds: [1, 2] });

    expect(harness.api.cancelQueueItemsByBatchIds).toHaveBeenCalledWith(['batch-1']);
    expect(harness.api.cancelQueueItems).not.toHaveBeenCalled();

    await harness.coordinator.cancelRun({ backendItemIds: [1, 2] });

    expect(harness.api.cancelQueueItems).toHaveBeenCalledWith([1, 2]);
  });

  it('treats stale missing backend items as already cancelled', async () => {
    harness.api.cancelQueueItems.mockRejectedValue(
      new ApiError('Queue item with id 42 not found in queue default', 404)
    );

    await expect(harness.coordinator.cancelRun({ backendItemIds: [42] })).resolves.toBeUndefined();
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
