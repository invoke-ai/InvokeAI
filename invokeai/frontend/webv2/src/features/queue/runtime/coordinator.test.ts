import type {
  QueueBackendItem,
  QueueEnqueueGenerateRequest,
  QueueItemProgress,
  QueueResultImage,
} from '@features/queue/core/types';
import type { QueueItemProgressSink } from '@features/queue/data/progressStore';

import {
  buildQueueItemOrigin,
  buildUtilityQueueItemOrigin,
  type QueueItemStatusChangedEvent,
} from '@features/queue/data/events';
import { ApiError } from '@platform/transport/http';
import { createSocketHub, type BackendSocket } from '@platform/transport/socketHub';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createQueueCoordinator,
  QueueItemCancelledError,
  type QueueCoordinator,
  type QueueCoordinatorBackendPort,
  type QueueCoordinatorCallbacks,
  type QueueModelLoadPort,
  type QueueNodeExecutionPort,
} from './coordinator';

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
  shouldRandomizeSeed: false,
  sourceQueueItemId: 'local-1',
};

interface Harness {
  api: {
    [Key in Exclude<keyof QueueCoordinatorBackendPort, 'emit' | 'on' | 'onConnectionChange'>]: ReturnType<typeof vi.fn>;
  };
  callbacks: { [Key in keyof QueueCoordinatorCallbacks]: ReturnType<typeof vi.fn> };
  coordinator: QueueCoordinator;
  hub: ReturnType<typeof createSocketHub>;
  modelLoads: { [Key in keyof QueueModelLoadPort]: ReturnType<typeof vi.fn> };
  nodeExecution: { [Key in keyof QueueNodeExecutionPort]: ReturnType<typeof vi.fn> };
  progressImage: { clear: ReturnType<typeof vi.fn>; set: ReturnType<typeof vi.fn> };
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
    set: (queueItemId, value) => {
      progressEntries.set(queueItemId, value);
    },
  };
  const api = {
    cancelQueueItems: vi.fn(() => Promise.resolve()),
    cancelQueueItemsByBatchIds: vi.fn(() => Promise.resolve()),
    enqueueGenerate: vi.fn(() => Promise.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })),
    enqueueWorkflow: vi.fn(() => Promise.resolve({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 1 })),
    getItem: vi.fn((itemId: number) => Promise.resolve(createQueueBackendItem({ id: itemId }))),
    getResultImages: vi.fn((itemId: number, sourceQueueItemId: string) =>
      Promise.resolve([createImage(`image-${itemId}.png`, sourceQueueItemId)])
    ),
    listItems: vi.fn((): Promise<QueueBackendItem[]> => Promise.resolve([])),
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
    settleRunning: vi.fn(),
    started: vi.fn(),
  };
  const progressImage = { clear: vi.fn(), set: vi.fn() };
  const hub = createSocketHub({ createSocket: () => socket });

  hub.connect();

  const coordinator = createQueueCoordinator(callbacks, {
    backend: {
      ...api,
      emit: hub.emit,
      on: hub.on,
      onConnectionChange: hub.onConnectionChange,
    },
    galleryRefreshCoalesceMs: options.galleryRefreshCoalesceMs ?? 1,
    modelLoads,
    nodeExecution,
    progress,
    progressImage,
  });

  return { api, callbacks, coordinator, hub, modelLoads, nodeExecution, progressImage, progressEntries, socket };
};

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

    await expect(harness.coordinator.submitGenerate('local-1', generateRequest)).rejects.toThrow(
      'The backend queue did not accept this generation.'
    );
  });

  it('rejects runs when the backend queue accepts only part of the batch', async () => {
    harness.api.enqueueGenerate.mockResolvedValue({ batchId: 'batch-1', enqueued: 1, itemIds: [1], requested: 2 });

    await expect(harness.coordinator.submitGenerate('local-1', generateRequest)).rejects.toThrow(
      'The backend queue accepted 1 of 2 requested items.'
    );
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

    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));
    await resultsPromise;

    expect(harness.progressEntries.has('local-1')).toBe(false);
    expect(harness.progressImage.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
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
    expect(harness.progressImage.clear).not.toHaveBeenCalled();

    finishRouting();
    await routingPromise;
    await Promise.resolve();

    expect(harness.progressImage.clear).toHaveBeenCalledWith({ itemIndex: 1, queueItemId: 'local-1' });
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
      activeItemIndex: 1,
      completedItemCount: 0,
      message: '',
      percentage: null,
      totalItemCount: 2,
    });

    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    expect(harness.progressEntries.get('local-1')).toEqual({
      activeItemIndex: 2,
      completedItemCount: 1,
      message: '',
      percentage: null,
      totalItemCount: 2,
    });

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2 }));

    await resultsPromise;
  });

  describe('reconcile', () => {
    it('adopts pending items the backend already accepted, by origin', async () => {
      harness.api.listItems.mockResolvedValue([
        createQueueBackendItem({ batchId: 'batch-9', id: 7, origin: buildQueueItemOrigin('local-1') }),
      ]);

      const outcomes = await harness.coordinator.reconcile([{ id: 'local-1', status: 'pending' }]);

      expect(outcomes.get('local-1')).toEqual({ backendBatchId: 'batch-9', backendItemIds: [7], kind: 'adopted' });
      expect(harness.api.enqueueGenerate).not.toHaveBeenCalled();
    });

    it('resumes running items and settles them from their listed terminal status', async () => {
      harness.api.getItem.mockResolvedValue(createQueueBackendItem({ id: 7, status: 'completed' }));

      const outcomes = await harness.coordinator.reconcile([{ backendItemIds: [7], id: 'local-1', status: 'running' }]);

      expect(outcomes.get('local-1')).toEqual({ kind: 'resumed' });
      expect(harness.api.listItems).not.toHaveBeenCalled();

      const images = await harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

      expect(images.map((image) => image.imageName)).toEqual(['image-7.png']);
    });

    it('marks running items missing when their backend items vanished', async () => {
      harness.api.getItem.mockRejectedValue(new ApiError('not found', 404));

      const outcomes = await harness.coordinator.reconcile([{ backendItemIds: [7], id: 'local-1', status: 'running' }]);

      expect(outcomes.get('local-1')).toEqual({ kind: 'missing' });
      expect(harness.api.listItems).not.toHaveBeenCalled();
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

    // Review fix (Task 38, finding 3): the prior "Risk-4" coverage only asserted
    // the parse primitive (`parseQueueItemOrigin(utilityOrigin) === null`) in
    // isolation. This drives the REAL `reconcile` path with a completed,
    // result-carrying `webv2:util:<uuid>`-origin backend item mixed into a live
    // `listItems` response, proving the coordinator itself — not just the
    // helper it calls — never adopts it into a project queue item (which is the
    // only way a utility item's images could ever reach `routeQueueItemResults`
    // and land in canvas staging or the gallery).
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

      // The utility item parses to no local queue item id, so it can never be
      // bucketed under 'local-1' by origin: the pending project item finds no
      // backend trace of its own and is asked to enqueue fresh — never
      // "adopted" with the utility item's id, and never routed anywhere.
      expect(outcomes.get('local-1')).toEqual({ kind: 'enqueue' });
      expect(harness.api.getResultImages).not.toHaveBeenCalled();
      expect(harness.api.getItem).not.toHaveBeenCalledWith(99);
    });
  });

  it('settles missed events through the safety sweep on reconnect', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    // The completion event was lost to a disconnect; the reconnect sweep
    // re-checks every outstanding item.
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

  it('prefers precise item ids for cancellation, falling back to batch id', async () => {
    await harness.coordinator.cancelRun({ backendBatchId: 'batch-1', backendItemIds: [1, 2] });

    expect(harness.api.cancelQueueItems).toHaveBeenCalledWith([1, 2]);
    expect(harness.api.cancelQueueItemsByBatchIds).not.toHaveBeenCalled();

    await harness.coordinator.cancelRun({ backendBatchId: 'batch-2' });

    expect(harness.api.cancelQueueItemsByBatchIds).toHaveBeenCalledWith(['batch-2']);
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
    harness.coordinator.dispose();

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    expect(harness.nodeExecution.settleRunning).not.toHaveBeenCalled();
  });
});
