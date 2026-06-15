import type { EnqueueGenerateRequest, ImageDTO, QueueItemDTO } from '@workbench/generation/types';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { NodeExecutionSink } from './nodeExecutionStore';
import type { QueueItemProgress, QueueItemProgressSink } from './progressStore';

import { buildQueueItemOrigin, type QueueItemStatusChangedEvent } from './events';
import { ApiError } from './http';
import {
  createQueueCoordinator,
  QueueItemCancelledError,
  type BackendSocket,
  type QueueCoordinator,
  type QueueCoordinatorApi,
  type QueueCoordinatorCallbacks,
} from './queueCoordinator';

class FakeSocket implements BackendSocket {
  readonly emitted: { event: string; payload: unknown }[] = [];
  private readonly handlers = new Map<string, ((payload: never) => void)[]>();

  on(event: string, handler: (payload: never) => void): void {
    this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
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
  batch_id: 'batch-1',
  completed_at: null,
  created_at: '2026-06-10T00:00:00Z',
  destination: 'gallery',
  error_message: null,
  error_type: null,
  item_id: 1,
  origin: null,
  queue_id: 'default',
  session_id: 'session-1',
  started_at: null,
  status: 'completed',
  updated_at: '2026-06-10T00:00:00Z',
  ...overrides,
});

const createQueueItemDTO = (overrides: Partial<QueueItemDTO>): QueueItemDTO => ({
  item_id: 1,
  session: { results: {} },
  status: 'in_progress',
  ...overrides,
});

const createImage = (imageName: string, sourceQueueItemId: string): ImageDTO => ({
  height: 64,
  imageName,
  imageUrl: `https://example.test/${imageName}`,
  isIntermediate: false,
  queuedAt: '2026-06-10T00:00:00Z',
  sourceQueueItemId,
  thumbnailUrl: `https://example.test/${imageName}/thumb`,
  width: 64,
});

const generateRequest: EnqueueGenerateRequest = {
  batchCount: 1,
  destination: 'gallery',
  graph: { edges: [], id: 'graph-1', nodes: {} },
  negativePrompt: '',
  negativePromptNodeId: 'negative_prompt',
  positivePrompt: 'a fjord at dawn',
  positivePromptNodeId: 'positive_prompt',
  seed: 1,
  seedNodeId: 'seed',
  shouldRandomizeSeed: false,
  sourceQueueItemId: 'local-1',
};

interface Harness {
  api: { [Key in keyof QueueCoordinatorApi]: ReturnType<typeof vi.fn> };
  callbacks: { [Key in keyof QueueCoordinatorCallbacks]: ReturnType<typeof vi.fn> };
  coordinator: QueueCoordinator;
  nodeExecution: { [Key in keyof NodeExecutionSink]: ReturnType<typeof vi.fn> };
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
    enqueueGenerateGraph: vi.fn(() => Promise.resolve({ batchId: 'batch-1', itemIds: [1] })),
    enqueueWorkflowGraph: vi.fn(() => Promise.resolve({ batchId: 'batch-1', itemIds: [1] })),
    getQueueItem: vi.fn((itemId: number) => Promise.resolve(createQueueItemDTO({ item_id: itemId }))),
    getQueueItemResultImages: vi.fn((itemId: number, sourceQueueItemId: string) =>
      Promise.resolve([createImage(`image-${itemId}.png`, sourceQueueItemId)])
    ),
    listAllQueueItems: vi.fn((): Promise<QueueItemDTO[]> => Promise.resolve([])),
  };
  const callbacks = {
    onConnectionChange: vi.fn(),
    onGalleryRefresh: vi.fn(),
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
  const coordinator = createQueueCoordinator(callbacks, {
    api,
    createSocket: () => socket,
    galleryRefreshCoalesceMs: options.galleryRefreshCoalesceMs ?? 1,
    nodeExecution,
    progress,
    progressImage,
  });

  return { api, callbacks, coordinator, nodeExecution, progressImage, progressEntries, socket };
};

describe('queueCoordinator', () => {
  let harness: Harness;

  beforeEach(() => {
    harness = createHarness();
  });

  afterEach(() => {
    harness.coordinator.dispose();
    vi.useRealTimers();
  });

  it('reports connection lifecycle and subscribes to the queue', () => {
    harness.coordinator.connect();

    expect(harness.callbacks.onConnectionChange).toHaveBeenNthCalledWith(1, 'connecting', undefined);
    expect(harness.callbacks.onConnectionChange).toHaveBeenNthCalledWith(2, 'connected', undefined);
    expect(harness.socket.emitted).toContainEqual({ event: 'subscribe_queue', payload: { queue_id: 'default' } });
  });

  it('settles submitted runs from terminal socket events without polling', async () => {
    harness.api.enqueueGenerateGraph.mockResolvedValue({ batchId: 'batch-1', itemIds: [1, 2] });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2 }));

    const images = await resultsPromise;

    expect(images.map((image) => image.imageName)).toEqual(['image-1.png', 'image-2.png']);
    expect(harness.api.getQueueItem).not.toHaveBeenCalled();
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
    harness.api.enqueueGenerateGraph.mockResolvedValue({ batchId: 'batch-1', itemIds: [1, 2, 3] });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'canceled' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2 }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 3, status: 'canceled' }));

    const images = await resultsPromise;

    expect(images.map((image) => image.imageName)).toEqual(['image-2.png']);
    expect(harness.api.getQueueItemResultImages).toHaveBeenCalledTimes(1);
    expect(harness.api.getQueueItemResultImages).toHaveBeenCalledWith(2, 'local-1', '2026-06-10T00:00:00Z');
    expect(harness.callbacks.onBackendItemCancelled).toHaveBeenCalledWith('local-1', 1);
    expect(harness.callbacks.onBackendItemCancelled).toHaveBeenCalledWith('local-1', 3);
  });

  it('rejects with QueueItemCancelledError when every backend item in a batch is canceled', async () => {
    harness.api.enqueueGenerateGraph.mockResolvedValue({ batchId: 'batch-1', itemIds: [1, 2] });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'canceled' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 2, status: 'canceled' }));

    await expect(resultsPromise).rejects.toBeInstanceOf(QueueItemCancelledError);
    expect(harness.api.getQueueItemResultImages).not.toHaveBeenCalled();
  });

  it('settles runs whose terminal event arrived before tracking began', async () => {
    harness.coordinator.connect();

    // The event for item 1 lands while enqueue_batch is still resolving.
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    await harness.coordinator.submitGenerate('local-1', generateRequest);

    const images = await harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    expect(images.map((image) => image.imageName)).toEqual(['image-1.png']);
  });

  it('coalesces gallery refreshes across a burst of completions', async () => {
    vi.useFakeTimers();
    harness = createHarness({ galleryRefreshCoalesceMs: 400 });
    harness.api.enqueueGenerateGraph.mockResolvedValue({
      batchId: 'batch-1',
      itemIds: Array.from({ length: 10 }, (_, index) => index + 1),
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
    harness.api.enqueueGenerateGraph.mockResolvedValue({ batchId: 'batch-1', itemIds: [1, 2] });
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

  it('notifies when one backend item in a batch completes', async () => {
    harness.callbacks.onBackendItemComplete = vi.fn();
    harness.api.enqueueGenerateGraph.mockResolvedValue({ batchId: 'batch-1', itemIds: [1, 2] });
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1 }));

    expect(harness.callbacks.onBackendItemComplete).toHaveBeenCalledWith('local-1', 1);
  });

  it('routes progress images to the active image slot inside a submitted batch', async () => {
    harness.api.enqueueGenerateGraph.mockResolvedValue({ batchId: 'batch-1', itemIds: [1, 2] });
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
    harness.api.enqueueGenerateGraph.mockResolvedValue({ batchId: 'batch-1', itemIds: [1, 2] });
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
      harness.api.listAllQueueItems.mockResolvedValue([
        createQueueItemDTO({ batch_id: 'batch-9', item_id: 7, origin: buildQueueItemOrigin('local-1') }),
      ]);

      const outcomes = await harness.coordinator.reconcile([{ id: 'local-1', status: 'pending' }]);

      expect(outcomes.get('local-1')).toEqual({ backendBatchId: 'batch-9', backendItemIds: [7], kind: 'adopted' });
      expect(harness.api.enqueueGenerateGraph).not.toHaveBeenCalled();
    });

    it('resumes running items and settles them from their listed terminal status', async () => {
      harness.api.getQueueItem.mockResolvedValue(createQueueItemDTO({ item_id: 7, status: 'completed' }));

      const outcomes = await harness.coordinator.reconcile([{ backendItemIds: [7], id: 'local-1', status: 'running' }]);

      expect(outcomes.get('local-1')).toEqual({ kind: 'resumed' });
      expect(harness.api.listAllQueueItems).not.toHaveBeenCalled();

      const images = await harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

      expect(images.map((image) => image.imageName)).toEqual(['image-7.png']);
    });

    it('marks running items missing when their backend items vanished', async () => {
      harness.api.getQueueItem.mockRejectedValue(new ApiError('not found', 404));

      const outcomes = await harness.coordinator.reconcile([{ backendItemIds: [7], id: 'local-1', status: 'running' }]);

      expect(outcomes.get('local-1')).toEqual({ kind: 'missing' });
      expect(harness.api.listAllQueueItems).not.toHaveBeenCalled();
    });

    it('asks for a fresh enqueue when a pending item left no backend trace', async () => {
      harness.api.listAllQueueItems.mockResolvedValue([]);

      const outcomes = await harness.coordinator.reconcile([{ id: 'local-1', status: 'pending' }]);

      expect(outcomes.get('local-1')).toEqual({ kind: 'enqueue' });
    });

    it('skips the backend round-trip when there is nothing to reconcile', async () => {
      const outcomes = await harness.coordinator.reconcile([]);

      expect(outcomes.size).toBe(0);
      expect(harness.api.listAllQueueItems).not.toHaveBeenCalled();
    });
  });

  it('settles missed events through the safety sweep on reconnect', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    // The completion event was lost to a disconnect; the reconnect sweep
    // re-checks every outstanding item.
    harness.api.getQueueItem.mockResolvedValue(createQueueItemDTO({ item_id: 1, status: 'completed' }));
    harness.socket.fire('connect', undefined);

    const images = await resultsPromise;

    expect(images.map((image) => image.imageName)).toEqual(['image-1.png']);
  });

  it('fails runs whose backend items were pruned (404) during a sweep', async () => {
    harness.coordinator.connect();

    await harness.coordinator.submitGenerate('local-1', generateRequest);
    const resultsPromise = harness.coordinator.waitForResults('local-1', '2026-06-10T00:00:00Z');

    harness.api.getQueueItem.mockRejectedValue(new ApiError('not found', 404));
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

  it('stops notifying after dispose', () => {
    harness.coordinator.connect();
    harness.callbacks.onConnectionChange.mockClear();

    harness.coordinator.dispose();

    expect(harness.callbacks.onConnectionChange).not.toHaveBeenCalled();
  });
});
