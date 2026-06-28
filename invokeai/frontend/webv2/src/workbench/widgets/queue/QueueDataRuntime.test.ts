import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  clearAllProgress: vi.fn(),
  clearProgress: vi.fn(),
  setProgress: vi.fn(),
  onConnectionChange: vi.fn(),
  refreshModelCacheStats: vi.fn(),
  refreshQueue: vi.fn(),
  socketOn: vi.fn(),
}));

vi.mock('@workbench/backend/events', () => ({
  isTerminalBackendStatus: (status: string) => status === 'completed' || status === 'failed' || status === 'canceled',
}));

vi.mock('@workbench/backend/itemProgressStore', () => ({
  itemProgressStore: {
    clear: mocks.clearProgress,
    clearAll: mocks.clearAllProgress,
    set: mocks.setProgress,
  },
}));

vi.mock('@workbench/backend/socketHub', () => ({
  socketHub: {
    on: mocks.socketOn,
    onConnectionChange: mocks.onConnectionChange,
  },
}));

vi.mock('./modelCacheStore', () => ({
  refreshModelCacheStats: mocks.refreshModelCacheStats,
}));

vi.mock('./queueDataStore', () => ({
  ensureQueueLoaded: vi.fn(),
  refreshQueue: mocks.refreshQueue,
  setQueueScope: vi.fn(),
  useCurrentBatchItems: vi.fn(() => []),
  useQueueCounts: vi.fn(() => ({
    canceled: 0,
    completed: 0,
    failed: 0,
    in_progress: 0,
    pending: 0,
    queue_id: 'default',
    total: 0,
  })),
  useRecentItems: vi.fn(() => []),
}));

vi.mock('@workbench/settings/store', () => ({
  useWorkbenchPreferences: vi.fn(() => ({ queueJobsScope: 'all' })),
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useOptionalWorkbenchSelector: vi.fn((_selector: unknown, fallback: unknown) => fallback),
}));

describe('QueueDataRuntime queue refresh events', () => {
  beforeEach(() => {
    mocks.clearAllProgress.mockReset();
    mocks.clearProgress.mockReset();
    mocks.onConnectionChange.mockReset();
    mocks.refreshModelCacheStats.mockReset();
    mocks.refreshQueue.mockReset();
    mocks.setProgress.mockReset();
    mocks.socketOn.mockReset();
    mocks.socketOn.mockReturnValue(vi.fn());
    mocks.onConnectionChange.mockReturnValue(vi.fn());
  });

  it('refreshes queue data for all queue mutation socket events', async () => {
    const { attachQueueDataRuntime } = await import('./QueueDataRuntime');

    attachQueueDataRuntime();

    const handlersByEvent = new Map<string, (payload: never) => void>(
      mocks.socketOn.mock.calls.map((call) => [call[0] as string, call[1] as (payload: never) => void])
    );

    handlersByEvent.get('batch_enqueued')?.({} as never);
    handlersByEvent.get('queue_cleared')?.({} as never);
    handlersByEvent.get('queue_items_retried')?.({} as never);

    expect(mocks.refreshQueue).toHaveBeenCalledTimes(3);
  });

  it('stores live denoise preview images by backend queue item id', async () => {
    const { attachQueueDataRuntime } = await import('./QueueDataRuntime');

    attachQueueDataRuntime();

    const handlersByEvent = new Map<string, (payload: never) => void>(
      mocks.socketOn.mock.calls.map((call) => [call[0] as string, call[1] as (payload: never) => void])
    );

    handlersByEvent.get('invocation_progress')?.({
      image: { dataURL: 'data:image/png;base64,preview', height: 32, width: 64 },
      item_id: 42,
      message: 'Denoising',
      percentage: 0.5,
    } as never);

    expect(mocks.setProgress).toHaveBeenCalledWith(42, {
      image: { dataUrl: 'data:image/png;base64,preview', height: 32, width: 64 },
      message: 'Denoising',
      percentage: 0.5,
    });
  });

  it('does not clear the last live preview when later invocations start for the same queue item', async () => {
    const { attachQueueDataRuntime } = await import('./QueueDataRuntime');

    attachQueueDataRuntime();

    const handlersByEvent = new Map<string, (payload: never) => void>(
      mocks.socketOn.mock.calls.map((call) => [call[0] as string, call[1] as (payload: never) => void])
    );

    handlersByEvent.get('invocation_progress')?.({
      image: { dataURL: 'data:image/png;base64,last-denoise', height: 32, width: 64 },
      item_id: 42,
      message: 'Denoising',
      percentage: 1,
    } as never);
    handlersByEvent.get('invocation_started')?.({ item_id: 42 } as never);

    expect(mocks.setProgress).toHaveBeenLastCalledWith(42, {
      message: '',
      percentage: null,
    });
  });

  it('keeps completed item progress images until the refreshed queue item has a final result', async () => {
    const { attachQueueDataRuntime } = await import('./QueueDataRuntime');

    attachQueueDataRuntime();

    const handlersByEvent = new Map<string, (payload: never) => void>(
      mocks.socketOn.mock.calls.map((call) => [call[0] as string, call[1] as (payload: never) => void])
    );

    handlersByEvent.get('queue_item_status_changed')?.({ item_id: 42, status: 'completed' } as never);
    handlersByEvent.get('queue_item_status_changed')?.({ item_id: 43, status: 'failed' } as never);

    expect(mocks.clearProgress).not.toHaveBeenCalledWith(42);
    expect(mocks.clearProgress).toHaveBeenCalledWith(43);
  });
});
