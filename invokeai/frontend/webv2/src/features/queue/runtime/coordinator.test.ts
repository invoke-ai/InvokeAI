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
    markFramed: ReturnType<typeof vi.fn>;
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
  const activeProgressTarget = {
    clear: vi.fn(),
    markFramed: vi.fn(),
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
      slotPercentages: { 1: 0.5 },
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

  it('clears stored slot percentage when an active item is requeued via socket or HTTP sweep', async () => {
    const listeners = new Map<string, () => void>();
    vi.stubGlobal('document', {
      addEventListener: (type: string, listener: () => void) => listeners.set(type, listener),
      removeEventListener: (type: string) => listeners.delete(type),
      visibilityState: 'visible',
    });

    try {
      harness.coordinator.connect();
      await harness.coordinator.submitGenerate('local-1', generateRequest);

      harness.socket.fire('invocation_progress', {
        ...createStatusEvent({ item_id: 1 }),
        message: 'Denoising',
        percentage: 0.88,
      });
      expect(harness.progressEntries.get('local-1')).toMatchObject({
        activeItemIndex: 1,
        percentage: 0.88,
      });

      // Socket requeue back to pending clears the stored percentage and active slot index.
      harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 1, status: 'pending' }));
      expect(harness.progressEntries.get('local-1')).toEqual({
        activeItemIndex: undefined,
        completedItemCount: 0,
        message: '',
        percentage: null,
        totalItemCount: 1,
      });

      // HTTP reconciliation sweep returning pending clears a restarted attempt's percentage as well.
      harness.socket.fire('invocation_progress', {
        ...createStatusEvent({ item_id: 1 }),
        message: 'Denoising',
        percentage: 0.66,
      });
      expect(harness.progressEntries.get('local-1')).toMatchObject({
        activeItemIndex: 1,
        percentage: 0.66,
      });

      harness.api.getItem.mockResolvedValueOnce(createQueueBackendItem({ id: 1, status: 'pending' }));
      listeners.get('visibilitychange')?.();
      await vi.waitFor(() =>
        expect(harness.progressEntries.get('local-1')).toEqual({
          activeItemIndex: undefined,
          completedItemCount: 0,
          message: '',
          percentage: null,
          totalItemCount: 1,
        })
      );
    } finally {
      harness.coordinator.dispose();
      vi.unstubAllGlobals();
    }
  });

  it('retains independent per-slot percentages for concurrent call_saved_workflow batch items', async () => {
    harness.coordinator.connect();
    harness.api.enqueueGenerate.mockResolvedValueOnce({
      batchId: 'batch-wf',
      enqueued: 2,
      itemIds: [701, 702],
      requested: 2,
    });
    await harness.coordinator.submitGenerate('local-wf', {
      ...generateRequest,
      graph: { edges: [], id: 'g', nodes: { call: { id: 'call', type: 'call_saved_workflow' } } },
    });

    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 701, status: 'waiting' }));
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 702, status: 'waiting' }));

    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 801 }),
      message: 'Slot 1 child',
      percentage: 0.45,
      root_item_id: 701,
      workflow_call_parent_source_id: 'call',
    });
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 802 }),
      message: 'Slot 2 child',
      percentage: 0.72,
      root_item_id: 702,
      workflow_call_parent_source_id: 'call',
    });
    // Sibling slot 1 ticks again while slot 2 is still running: slot 2 must retain 0.72 in slotPercentages.
    harness.socket.fire('invocation_progress', {
      ...createStatusEvent({ item_id: 801 }),
      message: 'Slot 1 child',
      percentage: 0.5,
      root_item_id: 701,
      workflow_call_parent_source_id: 'call',
    });

    expect(harness.progressEntries.get('local-wf')).toEqual({
      activeItemIndex: 1,
      completedItemCount: 0,
      message: 'Slot 1 child',
      percentage: 0.5,
      slotPercentages: { 1: 0.5, 2: 0.72 },
      totalItemCount: 2,
    });

    // Starting the next invocation (e.g. VAE decode after denoise) resets that slot's percentage to null.
    harness.socket.fire('invocation_started', {
      ...createStatusEvent({ item_id: 801 }),
      invocation_source_id: 'vae_decode',
      root_item_id: 701,
      session_id: 'child-1',
      workflow_call_parent_source_id: 'call',
    });
    expect(harness.progressEntries.get('local-wf')?.slotPercentages).toEqual({ 1: null, 2: 0.72 });

    // Completing slot 2 removes its per-slot entry while keeping slot 1 intact.
    harness.socket.fire('queue_item_status_changed', createStatusEvent({ item_id: 702, status: 'completed' }));
    expect(harness.progressEntries.get('local-wf')?.slotPercentages).toEqual({ 1: null });
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
