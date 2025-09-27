import { createAction } from '@reduxjs/toolkit';
import type { AppStore, RootState } from 'app/store/store';
import type { EnqueueBatchArg, EnqueueBatchResponse } from './executeEnqueue';
import { executeEnqueue } from './executeEnqueue';
import { describe, expect, it, vi } from 'vitest';

const createTestStore = () => {
  const state = {} as RootState;
  const dispatch = vi.fn<(action: unknown) => unknown>((action) => {
    if (typeof action === 'object' && action !== null && 'type' in action) {
      return undefined;
    }
    const unwrap = vi.fn<() => Promise<EnqueueBatchResponse>>().mockResolvedValue({
      batch_id: 'batch-1',
      item_ids: ['item-1'],
    } as EnqueueBatchResponse);
    return { unwrap };
  });
  const getState = vi.fn(() => state);
  return { dispatch, getState } as unknown as AppStore;
};

const createBatchArg = (prepend: boolean): EnqueueBatchArg => ({
  prepend,
  batch: {
    graph: {} as EnqueueBatchArg['batch']['graph'],
    runs: 1,
    data: [],
    origin: 'test',
    destination: 'test',
  },
});

describe('executeEnqueue', () => {
  it('dispatches enqueue flow and invokes success callback', async () => {
    const store = createTestStore();
    const requestedAction = createAction('test/enqueue');
    const options = { prepend: false } as const;
    const batchConfig = createBatchArg(options.prepend);
    const onSuccess = vi.fn();
    const build = vi.fn(async () => ({ batchConfig }));
    const prepareBatch = vi.fn(() => batchConfig);

    const result = await executeEnqueue({
      store,
      options,
      requestedAction,
      build,
      prepareBatch,
      onSuccess,
      log: { error: vi.fn() },
    });

    expect(store.dispatch).toHaveBeenCalledWith(requestedAction());
    expect(build).toHaveBeenCalledWith({ store, options });
    expect(prepareBatch).toHaveBeenCalledWith({ store, options, buildResult: { batchConfig } });
    expect(onSuccess).toHaveBeenCalled();
    expect(result?.batchConfig).toBe(batchConfig);
  });

  it('stops when build returns null', async () => {
    const store = createTestStore();
    const requestedAction = createAction('test/enqueue');
    const options = { prepend: true } as const;
    const build = vi.fn(async () => null);
    const prepareBatch = vi.fn();

    const result = await executeEnqueue({
      store,
      options,
      requestedAction,
      build,
      prepareBatch,
      log: { error: vi.fn() },
    });

    expect(result).toBeNull();
    expect(build).toHaveBeenCalled();
    expect(prepareBatch).not.toHaveBeenCalled();
  });

  it('invokes onError when build throws', async () => {
    const store = createTestStore();
    const requestedAction = createAction('test/enqueue');
    const options = { prepend: false } as const;
    const error = new Error('boom');
    const build = vi.fn(async () => {
      throw error;
    });
    const onError = vi.fn();
    const logError = vi.fn();

    const result = await executeEnqueue({
      store,
      options,
      requestedAction,
      build,
      prepareBatch: vi.fn(),
      onError,
      log: { error: logError },
    });

    expect(result).toBeNull();
    expect(onError).toHaveBeenCalledWith({ store, options, error });
    expect(logError).toHaveBeenCalled();
  });
});
