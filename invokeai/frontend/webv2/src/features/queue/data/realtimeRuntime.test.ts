import type { QueueItemReadModel, QueueReadModel } from '@features/queue/core/types';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { QueueItemStatusChangedEvent } from './events';

import { applyQueueItemStatusToReadModels, queueKeys } from './queries';
import { createQueueRealtimeRuntime } from './realtimeRuntime';

describe('queue realtime runtime', () => {
  const handlers = new Map<string, (payload: never) => void>();
  const detachers: Array<ReturnType<typeof vi.fn>> = [];
  const backend = {
    on: vi.fn((event: string, handler: (payload: never) => void) => {
      handlers.set(event, handler);
      const detach = vi.fn(() => handlers.delete(event));
      detachers.push(detach);
      return detach;
    }),
    onConnectionChange: vi.fn((handler: (status: 'connected') => void) => {
      handler('connected');
      const detach = vi.fn();
      detachers.push(detach);
      return detach;
    }),
  };
  const applyStatusChange = vi.fn();
  const invalidate = vi.fn();
  const progress = { clear: vi.fn(), clearAll: vi.fn(), set: vi.fn() };
  const refreshModelCache = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    handlers.clear();
    detachers.length = 0;
    backend.on.mockClear();
    backend.onConnectionChange.mockClear();
    applyStatusChange.mockReset();
    invalidate.mockReset();
    progress.clear.mockReset();
    progress.clearAll.mockReset();
    progress.set.mockReset();
    refreshModelCache.mockReset();
  });

  afterEach(() => vi.useRealTimers());

  it('attaches once and coalesces a burst of queue events into one invalidation', () => {
    const runtime = createQueueRealtimeRuntime({ applyStatusChange, backend, invalidate, progress, refreshModelCache });

    runtime.start();
    runtime.start();
    handlers.get('batch_enqueued')?.({} as never);
    handlers.get('queue_cleared')?.({} as never);
    handlers.get('queue_items_retried')?.({} as never);
    handlers.get('queue_items_canceled')?.({ canceled_item_ids: [] } as never);

    expect(backend.onConnectionChange).toHaveBeenCalledTimes(1);
    expect(backend.on).toHaveBeenCalledTimes(8);
    expect(invalidate).not.toHaveBeenCalled();

    vi.advanceTimersByTime(50);

    expect(invalidate).toHaveBeenCalledTimes(1);
    progress.clearAll.mockClear();
    runtime.dispose();
    expect(detachers.every((detach) => detach.mock.calls.length === 1)).toBe(true);
    expect(progress.clearAll).toHaveBeenCalledOnce();
  });

  it('updates transient progress without invalidating the queue list', () => {
    const runtime = createQueueRealtimeRuntime({ applyStatusChange, backend, invalidate, progress, refreshModelCache });

    runtime.start();
    vi.advanceTimersByTime(50);
    invalidate.mockClear();

    handlers.get('invocation_progress')?.({
      image: { dataURL: 'data:image/png;base64,preview', height: 32, width: 64 },
      item_id: 42,
      message: 'Denoising',
      percentage: 0.5,
    } as never);

    expect(progress.set).toHaveBeenCalledWith(42, {
      image: { dataUrl: 'data:image/png;base64,preview', height: 32, width: 64 },
      message: 'Denoising',
      percentage: 0.5,
    });
    expect(invalidate).not.toHaveBeenCalled();
    runtime.dispose();
  });

  it('forwards the reporting GPU so concurrent sessions can be told apart', () => {
    const runtime = createQueueRealtimeRuntime({ applyStatusChange, backend, invalidate, progress, refreshModelCache });

    runtime.start();
    vi.advanceTimersByTime(50);

    // Preserve device tags so concurrent progress tiles can identify their accelerator.
    handlers.get('invocation_progress')?.({
      device: 'cuda:1',
      item_id: 42,
      message: 'Denoising',
      percentage: 0.5,
    } as never);

    expect(progress.set).toHaveBeenCalledWith(42, {
      device: 'cuda:1',
      image: undefined,
      message: 'Denoising',
      percentage: 0.5,
    });
    runtime.dispose();
  });

  it('keeps completed previews until result hydration and clears failed previews immediately', () => {
    const runtime = createQueueRealtimeRuntime({ applyStatusChange, backend, invalidate, progress, refreshModelCache });

    runtime.start();
    handlers.get('queue_item_status_changed')?.({ item_id: 42, status: 'completed' } as never);
    handlers.get('queue_item_status_changed')?.({ item_id: 43, status: 'failed' } as never);

    expect(progress.clear).not.toHaveBeenCalledWith(42);
    expect(progress.clear).toHaveBeenCalledWith(43);
    runtime.dispose();
  });

  it('ignores timers and socket callbacks after its account scope expires', () => {
    const runtime = createQueueRealtimeRuntime({ applyStatusChange, backend, invalidate, progress, refreshModelCache });

    runtime.start();
    progress.set.mockClear();
    accountLifecycle.invalidate();

    handlers.get('invocation_progress')?.({
      item_id: 42,
      message: 'Old account',
      percentage: 0.5,
    } as never);
    vi.advanceTimersByTime(50);

    expect(progress.set).not.toHaveBeenCalled();
    expect(invalidate).not.toHaveBeenCalled();
    runtime.dispose();
  });

  describe('live read model', () => {
    const readItem = (id: number, status: QueueItemReadModel['status']): QueueItemReadModel => ({
      batchId: 'batch-1',
      createdAt: '2026-11-14T00:00:00.000Z',
      id,
      origin: 'webv2:p:project-1:q:run-1',
      resultImageNames: [],
      sessionId: `session-${id}`,
      status,
      updatedAt: '2026-11-14T00:00:00.000Z',
    });
    const counts = (inProgress: number, pending: number, completed: number) => ({
      canceled: 0,
      completed,
      failed: 0,
      inProgress,
      pending,
      queueId: 'default',
      total: inProgress + pending + completed,
      waiting: 0,
    });
    const statusEvent = (
      itemId: number,
      status: QueueItemStatusChangedEvent['status'],
      sequence: number,
      queueCounts: ReturnType<typeof counts>
    ) =>
      ({
        batch_id: 'batch-1',
        completed_at: null,
        created_at: '2026-11-14T00:00:00',
        destination: 'gallery',
        error_message: null,
        error_traceback: null,
        error_type: null,
        item_id: itemId,
        origin: 'webv2:p:project-1:q:run-1',
        queue_id: 'default',
        queue_status: {
          canceled: queueCounts.canceled,
          completed: queueCounts.completed,
          failed: queueCounts.failed,
          in_progress: queueCounts.inProgress,
          pending: queueCounts.pending,
          queue_id: 'default',
          total: queueCounts.total,
          user_in_progress: null,
          user_pending: null,
          waiting: 0,
        },
        session_id: `session-${itemId}`,
        started_at: null,
        status,
        status_sequence: sequence,
        timestamp: 0,
        updated_at: '2026-11-14T00:00:01',
        user_id: 'user-1',
      }) as unknown as QueueItemStatusChangedEvent;

    it('moves the current and next items as statuses change, before the reconciling read', () => {
      const client = new QueryClient();
      const scope = {};
      // A batch of three: item 1 running, items 2 and 3 waiting.
      client.setQueryData<QueueReadModel>(queueKeys.readModel(scope), {
        current: readItem(1, 'in_progress'),
        items: [readItem(3, 'pending'), readItem(2, 'pending'), readItem(1, 'in_progress')],
        next: readItem(2, 'pending'),
        scope,
        status: { processor: { isProcessing: true, isStarted: true }, queue: counts(1, 2, 0) },
      });
      const runtime = createQueueRealtimeRuntime({
        applyStatusChange: (change) => applyQueueItemStatusToReadModels(client, change),
        backend,
        invalidate,
        progress,
        refreshModelCache,
      });
      const model = () => client.getQueryData<QueueReadModel>(queueKeys.readModel(scope))!;
      const fire = (event: QueueItemStatusChangedEvent) => handlers.get('queue_item_status_changed')?.(event as never);

      runtime.start();
      vi.advanceTimersByTime(50);
      invalidate.mockClear();

      // Between items: nothing runs, and item 2 is next.
      fire(statusEvent(1, 'completed', 3, counts(0, 2, 1)));
      expect(model().current).toBeNull();
      expect(model().next?.id).toBe(2);
      expect(model().status.queue).toMatchObject({ completed: 1, inProgress: 0, pending: 2 });

      fire(statusEvent(2, 'in_progress', 2, counts(1, 1, 1)));
      expect(model().current).toMatchObject({ id: 2, status: 'in_progress' });
      expect(model().next?.id).toBe(3);
      // The read still follows to reconcile, coalesced.
      expect(invalidate).not.toHaveBeenCalled();
      vi.advanceTimersByTime(50);
      expect(invalidate).toHaveBeenCalledOnce();

      // A replayed older event for item 2 does not move it back to waiting.
      fire(statusEvent(2, 'pending', 1, counts(0, 2, 1)));
      expect(model().current?.id).toBe(2);
      runtime.dispose();
    });

    it("follows a restarted server's lower status sequences once a read has brought them in", () => {
      const client = new QueryClient();
      const scope = {};
      const seed = (items: QueueItemReadModel[]) =>
        client.setQueryData<QueueReadModel>(queueKeys.readModel(scope), {
          current: items.find((item) => item.status === 'in_progress') ?? null,
          items,
          next: items.find((item) => item.status === 'pending') ?? null,
          scope,
          status: { processor: { isProcessing: true, isStarted: true }, queue: counts(0, 1, 0) },
        });
      const runtime = createQueueRealtimeRuntime({
        applyStatusChange: (change) => applyQueueItemStatusToReadModels(client, change),
        backend,
        invalidate,
        progress,
        refreshModelCache,
      });
      const fire = (event: QueueItemStatusChangedEvent) => handlers.get('queue_item_status_changed')?.(event as never);
      const model = () => client.getQueryData<QueueReadModel>(queueKeys.readModel(scope))!;

      runtime.start();
      // Before the restart, item 8 started at its ninth status change.
      seed([{ ...readItem(8, 'pending'), statusSequence: 8 }]);
      fire(statusEvent(8, 'in_progress', 9, counts(1, 0, 0)));
      expect(model().current?.statusSequence).toBe(9);
      // The server came back with a reset queue that reuses the id; the read after reconnecting brings its item in.
      seed([{ ...readItem(8, 'pending'), statusSequence: 1 }]);
      fire(statusEvent(8, 'in_progress', 2, counts(1, 0, 0)));

      expect(model().current).toMatchObject({ id: 8, status: 'in_progress', statusSequence: 2 });
      runtime.dispose();
    });
  });
});
