import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
  const invalidate = vi.fn();
  const progress = { clear: vi.fn(), clearAll: vi.fn(), set: vi.fn() };
  const refreshModelCache = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    handlers.clear();
    detachers.length = 0;
    backend.on.mockClear();
    backend.onConnectionChange.mockClear();
    invalidate.mockReset();
    progress.clear.mockReset();
    progress.clearAll.mockReset();
    progress.set.mockReset();
    refreshModelCache.mockReset();
  });

  afterEach(() => vi.useRealTimers());

  it('attaches once and coalesces a burst of queue events into one invalidation', () => {
    const runtime = createQueueRealtimeRuntime({ backend, invalidate, progress, refreshModelCache });

    runtime.start();
    runtime.start();
    handlers.get('batch_enqueued')?.({} as never);
    handlers.get('queue_cleared')?.({} as never);
    handlers.get('queue_items_retried')?.({} as never);

    expect(backend.onConnectionChange).toHaveBeenCalledTimes(1);
    expect(backend.on).toHaveBeenCalledTimes(7);
    expect(invalidate).not.toHaveBeenCalled();

    vi.advanceTimersByTime(50);

    expect(invalidate).toHaveBeenCalledTimes(1);
    runtime.dispose();
    expect(detachers.every((detach) => detach.mock.calls.length === 1)).toBe(true);
  });

  it('updates transient progress without invalidating the queue list', () => {
    const runtime = createQueueRealtimeRuntime({ backend, invalidate, progress, refreshModelCache });

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

  it('keeps completed previews until result hydration and clears failed previews immediately', () => {
    const runtime = createQueueRealtimeRuntime({ backend, invalidate, progress, refreshModelCache });

    runtime.start();
    handlers.get('queue_item_status_changed')?.({ item_id: 42, status: 'completed' } as never);
    handlers.get('queue_item_status_changed')?.({ item_id: 43, status: 'failed' } as never);

    expect(progress.clear).not.toHaveBeenCalledWith(42);
    expect(progress.clear).toHaveBeenCalledWith(43);
    runtime.dispose();
  });
});
