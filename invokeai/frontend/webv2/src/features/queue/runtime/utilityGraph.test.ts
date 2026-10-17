import type { InvocationCompleteEvent, QueueItemStatusChangedEvent } from '@features/queue/contracts';
import type { SocketHub } from '@platform/transport/socketHub';
import type { BackendConnectionStatus } from '@platform/transport/types';

import {
  buildQueueItemOrigin,
  buildUtilityQueueItemOrigin,
  isUtilityQueueItemOrigin,
  parseQueueItemOrigin,
} from '@features/queue/contracts';
import { ApiError } from '@platform/transport/http';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { UtilityEnqueue, UtilityItemStatusReader } from './utilityGraph';

import {
  DEFAULT_UTILITY_QUEUE_TIMEOUT_MS,
  DEFAULT_UTILITY_RECONCILE_INTERVAL_MS,
  runUtilityGraph,
  submitUtilityGraph,
  UtilityQueueError,
} from './utilityGraph';

/** A fake socket hub that records handlers per event and can emit to them, mirroring `hub.on`. */
const createFakeHub = () => {
  const handlers = new Map<string, Set<(payload: unknown) => void>>();
  let detachCount = 0;

  const connectionHandlers = new Set<(status: BackendConnectionStatus) => void>();

  const hub: Pick<SocketHub, 'on' | 'onConnectionChange'> = {
    on: (event, handler) => {
      const set = handlers.get(event) ?? new Set();
      set.add(handler as (payload: unknown) => void);
      handlers.set(event, set);
      return () => {
        detachCount += 1;
        set.delete(handler as (payload: unknown) => void);
      };
    },
    onConnectionChange: (handler) => {
      connectionHandlers.add(handler);
      handler('connected');
      return () => connectionHandlers.delete(handler);
    },
  };

  const emit = (event: string, payload: unknown): void => {
    for (const handler of handlers.get(event) ?? []) {
      handler(payload);
    }
  };

  return {
    connection: (status: BackendConnectionStatus): void => {
      for (const handler of connectionHandlers) {
        handler(status);
      }
    },
    connectionHandlerCount: (): number => connectionHandlers.size,
    emit,
    get detachCount() {
      return detachCount;
    },
    handlerCount: (event: string): number => handlers.get(event)?.size ?? 0,
    hub,
  };
};

const UTIL_ID = 'fixed-util-id';
const ORIGIN = buildUtilityQueueItemOrigin(UTIL_ID);

const completeEvent = (overrides: Partial<InvocationCompleteEvent> = {}): InvocationCompleteEvent => ({
  batch_id: 'b',
  destination: null,
  invocation_source_id: 'control_filter',
  item_id: 1,
  origin: ORIGIN,
  queue_id: 'default',
  result: { height: 48, image: { image_name: 'filtered.png' }, type: 'image_output', width: 64 },
  session_id: 's',
  timestamp: 1,
  user_id: 'user-1',
  ...overrides,
});

const statusEvent = (
  status: QueueItemStatusChangedEvent['status'],
  overrides: Partial<QueueItemStatusChangedEvent> = {}
): QueueItemStatusChangedEvent => ({
  batch_status: {
    batch_id: 'b',
    canceled: 0,
    completed: 0,
    destination: null,
    failed: 0,
    in_progress: 1,
    origin: ORIGIN,
    pending: 0,
    queue_id: 'default',
    total: 1,
    waiting: 0,
  },
  batch_id: 'b',
  completed_at: null,
  created_at: '',
  destination: null,
  error_message: null,
  error_traceback: null,
  error_type: null,
  item_id: 1,
  origin: ORIGIN,
  queue_id: 'default',
  session_id: 's',
  started_at: null,
  status,
  status_sequence: 1,
  timestamp: 1,
  updated_at: '',
  user_id: 'user-1',
  queue_status: {
    batch_id: 'b',
    canceled: 0,
    completed: 0,
    failed: 0,
    in_progress: 1,
    item_id: 1,
    pending: 0,
    queue_id: 'default',
    session_id: 's',
    total: 1,
    user_in_progress: 1,
    user_pending: 0,
    waiting: 0,
  },
  ...overrides,
});

const okEnqueue: UtilityEnqueue = () => Promise.resolve({ enqueued: 1, itemIds: [1] });

const createDeferred = <T>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const createReconcileScheduler = () => {
  const pending: Array<{ callback: () => void; canceled: boolean }> = [];
  const cancelers: ReturnType<typeof vi.fn>[] = [];
  const schedule = vi.fn((callback: () => void, _delayMs: number) => {
    const entry = { callback, canceled: false };
    pending.push(entry);
    const cancel = vi.fn(() => {
      entry.canceled = true;
    });
    cancelers.push(cancel);
    return cancel;
  });
  return {
    cancelers,
    pending,
    runNext() {
      const entry = pending.shift();
      if (!entry) {
        throw new Error('No reconciliation retry is scheduled.');
      }
      if (!entry.canceled) {
        entry.callback();
      }
    },
    schedule,
  };
};

afterEach(() => {
  vi.useRealTimers();
});

describe('runUtilityGraph — await + settle', () => {
  it('resolves with the captured image name and the isolated origin on completion', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      outputNodeId: 'control_filter',
    });

    // Listeners are attached before enqueue resolves (fast-finish race closed).
    expect(fake.handlerCount('invocation_complete')).toBe(1);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);

    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).resolves.toEqual({ height: 48, imageName: 'filtered.png', origin: ORIGIN, width: 64 });
    // Both listeners detached on settle.
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(fake.detachCount).toBe(2);
  });

  it('ignores completions from other output nodes when outputNodeId is set', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      outputNodeId: 'control_filter',
    });

    // A different node's completion is ignored; the target node's is captured.
    fake.emit(
      'invocation_complete',
      completeEvent({
        invocation_source_id: 'other',
        result: { image: { image_name: 'wrong.png' }, type: 'image_output' },
      })
    );
    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
  });

  it('allows delayed socket output to win after a successful null reconciliation', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => Promise.resolve(null),
      scheduleReconcileRetry: scheduler.schedule,
    });
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());

    fake.emit('invocation_complete', completeEvent());

    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
    expect(scheduler.cancelers[0]).toHaveBeenCalledOnce();
  });

  it('reconciles the target output when completed status arrives before invocation_complete', async () => {
    const fake = createFakeHub();
    const reconcileCompletedOutput = vi.fn(() =>
      Promise.resolve({ height: 48, imageName: 'reconciled.png', width: 64 })
    );
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      outputNodeId: 'control_filter',
      reconcileCompletedOutput,
    });

    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).resolves.toEqual({
      height: 48,
      imageName: 'reconciled.png',
      origin: ORIGIN,
      width: 64,
    });
    expect(reconcileCompletedOutput).toHaveBeenCalledExactlyOnceWith([1], 'control_filter');
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('resolves when a later reconciliation finds output after an initial successful null', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const reconcileCompletedOutput = vi
      .fn()
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce({ height: 48, imageName: 'later.png', width: 64 });
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      outputNodeId: 'control_filter',
      reconcileCompletedOutput,
      scheduleReconcileRetry: scheduler.schedule,
    });

    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());
    scheduler.runNext();

    await expect(promise).resolves.toMatchObject({ imageName: 'later.png' });
    expect(reconcileCompletedOutput).toHaveBeenCalledTimes(2);
  });

  it('rejects no-output only after exhausting successful null reconciliations', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const reconcileCompletedOutput = vi.fn(() => Promise.resolve(null));
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput,
      reconcileRetryPolicy: { delayMs: 10, maxAttempts: 3 },
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledTimes(1));
    scheduler.runNext();
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledTimes(2));
    scheduler.runNext();

    await expect(promise).rejects.toMatchObject({ reason: 'no-output' });
    expect(reconcileCompletedOutput).toHaveBeenCalledTimes(3);
  });

  it('aborts and fully cleans up while completed-output reconciliation is pending', async () => {
    const fake = createFakeHub();
    const reconciliation = createDeferred<{ height: number; imageName: string; width: number } | null>();
    const controller = new AbortController();
    const cancel = vi.fn(() => Promise.resolve());
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      outputNodeId: 'control_filter',
      reconcileCompletedOutput: () => reconciliation.promise,
      signal: controller.signal,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    controller.abort();

    await expect(promise).rejects.toMatchObject({ reason: 'aborted' });
    await vi.waitFor(() => expect(cancel).toHaveBeenCalledExactlyOnceWith([1]));
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    reconciliation.resolve({ height: 48, imageName: 'late.png', width: 64 });
    await Promise.resolve();
    expect(fake.detachCount).toBe(2);
  });

  it('allows delayed socket output to win after a transient reconciliation failure', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const reconcileCompletedOutput = vi.fn(() => Promise.reject(new TypeError('gateway unavailable')));
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      outputNodeId: 'control_filter',
      reconcileCompletedOutput,
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());

    fake.emit('invocation_complete', completeEvent());

    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
    expect(reconcileCompletedOutput).toHaveBeenCalledOnce();
    expect(scheduler.cancelers[0]).toHaveBeenCalledOnce();
    expect(fake.handlerCount('invocation_complete')).toBe(0);
  });

  it('retries bounded reconciliation failures and resolves from a later successful attempt', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const reconcileCompletedOutput = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('first failure'))
      .mockRejectedValueOnce(new TypeError('second failure'))
      .mockResolvedValueOnce({ height: 48, imageName: 'reconciled.png', width: 64 });
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      outputNodeId: 'control_filter',
      reconcileCompletedOutput,
      reconcileRetryPolicy: { delayMs: 25, maxAttempts: 3 },
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledTimes(1));
    scheduler.runNext();
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledTimes(2));
    scheduler.runNext();

    await expect(promise).resolves.toMatchObject({ imageName: 'reconciled.png' });
    expect(reconcileCompletedOutput).toHaveBeenCalledTimes(3);
    expect(scheduler.schedule).toHaveBeenNthCalledWith(1, expect.any(Function), 25);
  });

  it('rejects with reconcile reason and preserves the final failure after bounded attempts are exhausted', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const finalCause = new TypeError('connection failed');
    const reconcileCompletedOutput = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('temporary failure'))
      .mockRejectedValueOnce(finalCause);
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput,
      reconcileRetryPolicy: { delayMs: 10, maxAttempts: 2 },
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());
    scheduler.runNext();

    await expect(promise).rejects.toMatchObject({
      cause: finalCause,
      message: 'Failed to reconcile utility graph output after 2 attempts: connection failed',
      reason: 'reconcile',
    });
    expect(fake.handlerCount('invocation_complete')).toBe(0);
  });

  it('aborts during a scheduled reconciliation retry and cancels the retry timer', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const controller = new AbortController();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => Promise.reject(new TypeError('offline')),
      scheduleReconcileRetry: scheduler.schedule,
      signal: controller.signal,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());

    controller.abort();

    await expect(promise).rejects.toMatchObject({ reason: 'aborted' });
    expect(scheduler.cancelers[0]).toHaveBeenCalledOnce();
    scheduler.runNext();
    expect(fake.handlerCount('invocation_complete')).toBe(0);
  });

  it('cancels a scheduled reconciliation retry when terminal failure wins', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => Promise.reject(new TypeError('offline')),
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());

    fake.emit('queue_item_status_changed', statusEvent('failed', { error_message: 'backend failed' }));

    await expect(promise).rejects.toMatchObject({ message: 'backend failed', reason: 'failed' });
    expect(scheduler.cancelers[0]).toHaveBeenCalledOnce();
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('fails an authentication reconciliation error immediately without retrying', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const cause = new ApiError('unauthorized', 401);
    const reconcileCompletedOutput = vi.fn(() => Promise.reject(cause));
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput,
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).rejects.toMatchObject({ cause, reason: 'reconcile' });
    expect(reconcileCompletedOutput).toHaveBeenCalledOnce();
    expect(scheduler.schedule).not.toHaveBeenCalled();
  });

  it('retries a transient 503 reconciliation error', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const reconcileCompletedOutput = vi
      .fn()
      .mockRejectedValueOnce(new ApiError('unavailable', 503))
      .mockResolvedValueOnce({ height: 48, imageName: 'recovered.png', width: 64 });
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput,
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());
    scheduler.runNext();

    await expect(promise).resolves.toMatchObject({ imageName: 'recovered.png' });
    expect(reconcileCompletedOutput).toHaveBeenCalledTimes(2);
  });

  it('fails malformed protocol data immediately without retrying', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const cause = new Error('Queue item 1 returned malformed reconciliation data.');
    const reconcileCompletedOutput = vi.fn(() => Promise.reject(cause));
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput,
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).rejects.toMatchObject({ cause, reason: 'reconcile' });
    expect(reconcileCompletedOutput).toHaveBeenCalledOnce();
    expect(scheduler.schedule).not.toHaveBeenCalled();
  });

  it('captures a synchronous reconciler throw triggered after enqueue completion', async () => {
    const fake = createFakeHub();
    const enqueue = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const cause = new Error('sync after enqueue');
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => enqueue.promise,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => {
        throw cause;
      },
    });
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    enqueue.resolve({ enqueued: 1, itemIds: [1] });

    await expect(promise).rejects.toMatchObject({ cause, reason: 'reconcile' });
  });

  it('captures a synchronous reconciler throw triggered from the socket callback', async () => {
    const fake = createFakeHub();
    const cause = new Error('sync from socket');
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => {
        throw cause;
      },
    });
    await Promise.resolve();

    expect(() => fake.emit('queue_item_status_changed', statusEvent('completed'))).not.toThrow();
    await expect(promise).rejects.toMatchObject({ cause, reason: 'reconcile' });
  });

  it('captures a synchronous reconciler throw triggered from the retry timer', async () => {
    const fake = createFakeHub();
    const scheduler = createReconcileScheduler();
    const cause = new Error('sync from retry');
    const reconcileCompletedOutput = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('offline'))
      .mockImplementationOnce(() => {
        throw cause;
      });
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput,
      scheduleReconcileRetry: scheduler.schedule,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduler.schedule).toHaveBeenCalledOnce());

    expect(() => scheduler.runNext()).not.toThrow();
    await expect(promise).rejects.toMatchObject({ cause, reason: 'reconcile' });
  });

  it('rejects with the failure reason + message on a failed item', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });
    fake.emit('queue_item_status_changed', statusEvent('failed', { error_message: 'boom' }));
    await expect(promise).rejects.toMatchObject({ message: 'boom', reason: 'failed' });
  });

  it.each(['', '   '])('uses a trimmed backend error type when the message is %j', async (error_message) => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });

    fake.emit(
      'queue_item_status_changed',
      statusEvent('failed', { error_message, error_traceback: 'giant traceback', error_type: '  AttributeError  ' })
    );

    await expect(promise).rejects.toMatchObject({ message: 'AttributeError', reason: 'failed' });
  });

  it('prefers and trims a nonempty backend message over its error type', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });

    fake.emit(
      'queue_item_status_changed',
      statusEvent('failed', { error_message: '  useful detail  ', error_type: 'AttributeError' })
    );

    await expect(promise).rejects.toMatchObject({ message: 'useful detail', reason: 'failed' });
  });

  it('uses the generic backend-failed fallback when message and type are missing', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });

    fake.emit('queue_item_status_changed', statusEvent('failed'));

    await expect(promise).rejects.toMatchObject({ message: 'The utility graph failed.', reason: 'failed' });
  });

  it('ignores malformed non-string backend failure fields', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });

    fake.emit(
      'queue_item_status_changed',
      statusEvent('failed', { error_message: 42, error_type: { name: 'AttributeError' } } as never)
    );

    await expect(promise).rejects.toMatchObject({ message: 'The utility graph failed.', reason: 'failed' });
  });

  it('rejects canceled on a canceled item', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });
    fake.emit('queue_item_status_changed', statusEvent('canceled'));
    await expect(promise).rejects.toMatchObject({ reason: 'canceled' });
  });
});

describe('runUtilityGraph — timeout + cancellation', () => {
  it('rejects with timeout and cancels every accepted backend item', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const cancel = vi.fn((_itemIds: number[]) => Promise.resolve());
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 3, itemIds: [11, 12, 13] }),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      timeoutMs: 1000,
    });
    const assertion = expect(promise).rejects.toMatchObject({ reason: 'timeout' });
    await vi.advanceTimersByTimeAsync(1000);
    await assertion;
    expect(cancel).toHaveBeenCalledExactlyOnceWith([11, 12, 13]);
    // Listeners cleaned up after timeout.
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('rejects immediately without enqueueing or canceling when the signal is already aborted', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const cancel = vi.fn((_itemIds: number[]) => Promise.resolve());
    const enqueue = vi.fn(okEnqueue);
    controller.abort();
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
    });
    await expect(promise).rejects.toMatchObject({ reason: 'aborted' });
    expect(enqueue).not.toHaveBeenCalled();
    expect(cancel).not.toHaveBeenCalled();
  });

  it('rejects aborted, cancels accepted backend items, and detaches listeners', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const cancel = vi.fn((_itemIds: number[]) => Promise.resolve());
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 2, itemIds: [21, 22] }),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
    });
    await Promise.resolve();
    controller.abort();
    await expect(promise).rejects.toMatchObject({ reason: 'aborted' });
    await vi.waitFor(() => expect(cancel).toHaveBeenCalledExactlyOnceWith([21, 22]));
    expect(fake.handlerCount('invocation_complete')).toBe(0);
  });

  it('rejects promptly, then cancels accepted items when a pending enqueue resolves', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const enqueue = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const cancel = vi.fn((_itemIds: number[]) => Promise.resolve());
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: () => enqueue.promise,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
    });
    const assertion = expect(promise).rejects.toMatchObject({ reason: 'aborted' });

    controller.abort();
    await assertion;
    expect(cancel).not.toHaveBeenCalled();

    enqueue.resolve({ enqueued: 2, itemIds: [31, 32] });
    await vi.waitFor(() => expect(cancel).toHaveBeenCalledExactlyOnceWith([31, 32]));
  });

  it('cancels at most once across abort, timeout, terminal, and cancellation-failure races', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const controller = new AbortController();
    const enqueue = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const cancel = vi.fn((_itemIds: number[]) => Promise.reject(new Error('cancel failed')));
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: () => enqueue.promise,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
      timeoutMs: 1000,
    });
    const assertion = expect(promise).rejects.toMatchObject({ reason: 'aborted' });

    controller.abort();
    await vi.advanceTimersByTimeAsync(1000);
    fake.emit('queue_item_status_changed', statusEvent('failed'));
    enqueue.resolve({ enqueued: 2, itemIds: [41, 42] });

    await assertion;
    await vi.runAllTimersAsync();
    expect(cancel).toHaveBeenCalledExactlyOnceWith([41, 42]);
  });

  it('does not cancel when a terminal success wins before a later abort', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const cancel = vi.fn((_itemIds: number[]) => Promise.resolve());
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 2, itemIds: [51, 52] }),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
    });
    await Promise.resolve();

    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
    controller.abort();
    await Promise.resolve();

    expect(cancel).not.toHaveBeenCalled();
  });

  it('does not cancel when an aborted pending enqueue later accepts zero items', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const enqueue = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const cancel = vi.fn((_itemIds: number[]) => Promise.resolve());
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: () => enqueue.promise,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
    });
    const assertion = expect(promise).rejects.toMatchObject({ reason: 'aborted' });

    controller.abort();
    enqueue.resolve({ enqueued: 0, itemIds: [] });

    await assertion;
    await Promise.resolve();
    expect(cancel).not.toHaveBeenCalled();
  });

  it('does not use a timeout when timeoutMs is 0', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      timeoutMs: 0,
    });
    await vi.advanceTimersByTimeAsync(DEFAULT_UTILITY_QUEUE_TIMEOUT_MS * 2);
    // Still pending — settle it so the promise doesn't leak.
    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
  });
});

describe('runUtilityGraph — enqueue failures', () => {
  it.each([
    { enqueued: 1, itemIds: [], label: 'accepted count without ids', canceled: [] },
    { enqueued: 1, itemIds: [Number.NaN], label: 'non-finite id', canceled: [] },
    { enqueued: 1, itemIds: [0], label: 'non-positive id', canceled: [] },
    { enqueued: Number.NaN, itemIds: [11], label: 'non-finite accepted count', canceled: [11] },
    { enqueued: 1.5, itemIds: [11], label: 'fractional accepted count', canceled: [11] },
    { enqueued: -1, itemIds: [11], label: 'negative accepted count', canceled: [11] },
    { enqueued: 2, itemIds: [11, 11], label: 'duplicate ids', canceled: [11] },
    { enqueued: 2, itemIds: [11], label: 'count mismatch', canceled: [11] },
  ])('immediately rejects $label and fully cleans up', async ({ canceled, enqueued, itemIds }) => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const cancel = vi.fn((_itemIds: number[]) => Promise.resolve());
    const promise = runUtilityGraph({
      cancel,
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued, itemIds }),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });

    await expect(promise).rejects.toMatchObject({ reason: 'enqueue' });
    await Promise.resolve();

    expect(cancel).toHaveBeenCalledTimes(canceled.length > 0 ? 1 : 0);
    if (canceled.length > 0) {
      expect(cancel).toHaveBeenCalledWith(canceled);
    }
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('accepts a valid multi-item enqueue response', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 2, itemIds: [11, 12] }),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });
    await Promise.resolve();

    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
  });

  it('contains a synchronous enqueue throw and cleans listeners with timeout disabled', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const cause = new Error('synchronous enqueue failure');
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => {
        throw cause;
      },
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      timeoutMs: 0,
    });

    await expect(promise).rejects.toMatchObject({ cause, message: cause.message, reason: 'enqueue' });
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('contains a synchronous reconciliation scheduler throw and fully settles with timeout disabled', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const cause = new Error('scheduler unavailable');
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => Promise.resolve(null),
      scheduleReconcileRetry: () => {
        throw cause;
      },
      timeoutMs: 0,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).rejects.toMatchObject({
      cause,
      message: 'Failed to schedule utility graph output reconciliation: scheduler unavailable',
      reason: 'reconcile',
    });
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('contains a throwing reconciliation retry cancellation seam during socket success', async () => {
    const fake = createFakeHub();
    const cancelRetry = vi.fn(() => {
      throw new Error('cancel retry failed');
    });
    const scheduleReconcileRetry = vi.fn(() => cancelRetry);
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => Promise.resolve(null),
      scheduleReconcileRetry,
      timeoutMs: 0,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.waitFor(() => expect(scheduleReconcileRetry).toHaveBeenCalledOnce());

    expect(() => fake.emit('invocation_complete', completeEvent())).not.toThrow();
    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
    expect(cancelRetry).toHaveBeenCalledOnce();
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('contains a synchronous id-source throw as a setup error', async () => {
    const cause = new Error('id source failed');
    const promise = runUtilityGraph({
      createId: () => {
        throw cause;
      },
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: createFakeHub().hub,
      timeoutMs: 0,
    });

    await expect(promise).rejects.toMatchObject({ cause, reason: 'setup' });
  });

  it('cleans a partially attached listener when the second socket subscription throws', async () => {
    const fake = createFakeHub();
    const cause = new Error('subscription failed');
    let calls = 0;
    const hub: Pick<SocketHub, 'on'> = {
      on: (event, handler) => {
        calls += 1;
        if (calls === 2) {
          throw cause;
        }
        return fake.hub.on(event, handler);
      },
    };
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub,
      timeoutMs: 0,
    });

    await expect(promise).rejects.toMatchObject({ cause, reason: 'setup' });
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('contains a synchronous injected reconciliation classifier throw', async () => {
    const fake = createFakeHub();
    const cause = new Error('classifier failed');
    const promise = runUtilityGraph({
      classifyReconcileError: () => {
        throw cause;
      },
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      reconcileCompletedOutput: () => Promise.reject(new TypeError('network failed')),
      timeoutMs: 0,
    });
    await Promise.resolve();
    fake.emit('queue_item_status_changed', statusEvent('completed'));

    await expect(promise).rejects.toMatchObject({ cause, reason: 'reconcile' });
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('cleans listeners when abort-signal subscription throws synchronously', async () => {
    const fake = createFakeHub();
    const cause = new Error('abort subscription failed');
    const signal = {
      aborted: false,
      addEventListener: () => {
        throw cause;
      },
      removeEventListener: vi.fn(),
    } as unknown as AbortSignal;
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal,
      timeoutMs: 0,
    });

    await expect(promise).rejects.toMatchObject({ cause, reason: 'setup' });
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('rejects enqueue when the backend accepts zero items', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 0, itemIds: [] }),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });
    await expect(promise).rejects.toMatchObject({ reason: 'enqueue' });
  });

  it('rejects enqueue when the enqueue call throws', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.reject(new Error('network down')),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });
    await expect(promise).rejects.toMatchObject({ message: 'network down', reason: 'enqueue' });
  });
});

describe('runUtilityGraph — origin isolation (Risk 4)', () => {
  it('mints a webv2:util: origin that project routing provably ignores', () => {
    // Utility origins must remain invisible to project adoption and staging/gallery routing.
    expect(isUtilityQueueItemOrigin(ORIGIN)).toBe(true);
    expect(parseQueueItemOrigin(ORIGIN)).toBeNull();

    const projectOrigin = buildQueueItemOrigin('local-1', 'project-1');
    expect(isUtilityQueueItemOrigin(projectOrigin)).toBe(false);
    expect(parseQueueItemOrigin(projectOrigin)).toBe('local-1');
  });

  it('ignores socket events whose origin is not this utility run', async () => {
    const fake = createFakeHub();
    let settled = false;
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      timeoutMs: 0,
    }).then((result) => {
      settled = true;
      return result;
    });

    // Events for a DIFFERENT origin (another util run or a project item) must not settle us.
    const otherOrigin = buildUtilityQueueItemOrigin('other-util');
    fake.emit(
      'invocation_complete',
      completeEvent({ origin: otherOrigin, result: { image: { image_name: 'other.png' }, type: 'image_output' } })
    );
    fake.emit(
      'queue_item_status_changed',
      statusEvent('completed', { origin: buildQueueItemOrigin('local-1', 'project-1') })
    );
    await Promise.resolve();
    expect(settled).toBe(false);

    // Our origin's events do settle us.
    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
  });
});

describe('UtilityQueueError', () => {
  it('carries a reason and a name', () => {
    const error = new UtilityQueueError('timeout', 'nope');
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe('UtilityQueueError');
    expect(error.reason).toBe('timeout');
  });
});

describe('submitUtilityGraph — fire, then watch settlement', () => {
  const graph = { edges: [], id: 'pbr', nodes: {} };

  it('resolves once the backend accepts the graph and settles when the item completes', async () => {
    const fake = createFakeHub();
    const enqueue = vi.fn<UtilityEnqueue>(() => Promise.resolve({ enqueued: 1, itemIds: [7] }));

    const submission = await submitUtilityGraph({ createId: () => UTIL_ID, enqueue, graph, hub: fake.hub });

    expect(enqueue).toHaveBeenCalledExactlyOnceWith({ graph, origin: ORIGIN });
    expect(submission).toMatchObject({ itemIds: [7], origin: ORIGIN });
    // Only status changes are watched; no output image is expected.
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);

    fake.emit('queue_item_status_changed', statusEvent('completed', { origin: 'other-origin' }));
    fake.emit('queue_item_status_changed', statusEvent('completed', { item_id: 7 }));

    await expect(submission.settled).resolves.toBeUndefined();
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('settles a completion that arrives before the enqueue response', async () => {
    const fake = createFakeHub();
    const deferred = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const pending = submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => deferred.promise,
      graph,
      hub: fake.hub,
    });

    fake.emit('queue_item_status_changed', statusEvent('completed'));
    deferred.resolve({ enqueued: 1, itemIds: [1] });

    await expect((await pending).settled).resolves.toBeUndefined();
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('rejects settlement with the backend failure message, or as canceled', async () => {
    const failed = createFakeHub();
    const failedSubmission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph,
      hub: failed.hub,
    });
    failed.emit('queue_item_status_changed', statusEvent('failed', { error_message: 'No model' }));
    await expect(failedSubmission.settled).rejects.toMatchObject({ message: 'No model', reason: 'failed' });

    const canceled = createFakeHub();
    const canceledSubmission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph,
      hub: canceled.hub,
    });
    canceled.emit('queue_item_status_changed', statusEvent('canceled'));
    await expect(canceledSubmission.settled).rejects.toMatchObject({ reason: 'canceled' });
    expect(canceled.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('rejects the submission, and detaches, when the enqueue fails or answers inconsistently', async () => {
    const failing = createFakeHub();
    const cause = new ApiError('{"detail":"queue paused"}', 409);
    await expect(
      submitUtilityGraph({
        createId: () => UTIL_ID,
        enqueue: () => Promise.reject(cause),
        graph,
        hub: failing.hub,
      })
    ).rejects.toMatchObject({ cause, message: 'queue paused', reason: 'enqueue' });
    expect(failing.handlerCount('queue_item_status_changed')).toBe(0);
    expect(failing.handlerCount('queue_cleared')).toBe(0);

    const inconsistent = createFakeHub();
    const cancel = vi.fn(() => Promise.resolve());
    await expect(
      submitUtilityGraph({
        cancel,
        createId: () => UTIL_ID,
        enqueue: () => Promise.resolve({ enqueued: 0, itemIds: [] }),
        graph,
        hub: inconsistent.hub,
      })
    ).rejects.toMatchObject({ reason: 'enqueue' });
    expect(inconsistent.handlerCount('queue_item_status_changed')).toBe(0);
    expect(cancel).not.toHaveBeenCalled();

    // Items the backend did accept alongside an inconsistent count are canceled rather than left running unwatched.
    await expect(
      submitUtilityGraph({
        cancel,
        createId: () => UTIL_ID,
        enqueue: () => Promise.resolve({ enqueued: 2, itemIds: [5] }),
        graph,
        hub: inconsistent.hub,
      })
    ).rejects.toMatchObject({ reason: 'enqueue' });
    await vi.waitFor(() => expect(cancel).toHaveBeenCalledExactlyOnceWith([5]));
  });

  it('passes prepend through to the enqueue so the item runs ahead of queued generations', async () => {
    const fake = createFakeHub();
    const enqueue = vi.fn<UtilityEnqueue>(() => Promise.resolve({ enqueued: 1, itemIds: [1] }));

    await submitUtilityGraph({ createId: () => UTIL_ID, enqueue, graph, hub: fake.hub, prepend: true });

    expect(enqueue).toHaveBeenCalledExactlyOnceWith({ graph, origin: ORIGIN, prepend: true });
  });

  it('aborting ends the watch: the listener detaches, the timer clears, and the item is not canceled', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const controller = new AbortController();
    const readItemStatus = vi.fn();
    const submission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph,
      hub: fake.hub,
      readItemStatus,
      signal: controller.signal,
    });
    const outcome = submission.settled.then(
      () => 'resolved',
      (error: UtilityQueueError) => error.reason
    );

    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);
    controller.abort();

    await expect(outcome).resolves.toBe('aborted');
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(fake.handlerCount('queue_cleared')).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
    expect(readItemStatus).not.toHaveBeenCalled();

    // An already-aborted signal refuses the submission before anything is enqueued.
    const enqueue = vi.fn<UtilityEnqueue>(okEnqueue);
    await expect(
      submitUtilityGraph({ createId: () => UTIL_ID, enqueue, graph, hub: fake.hub, signal: AbortSignal.abort() })
    ).rejects.toMatchObject({ reason: 'aborted' });
    expect(enqueue).not.toHaveBeenCalled();
  });

  it('removes its abort listener once the watch settles, so each submission leaves nothing on the signal', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const added = vi.spyOn(controller.signal, 'addEventListener');
    const removed = vi.spyOn(controller.signal, 'removeEventListener');
    const submission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph,
      hub: fake.hub,
      signal: controller.signal,
    });

    expect(added).toHaveBeenCalledExactlyOnceWith('abort', expect.any(Function));
    expect(removed).not.toHaveBeenCalled();

    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(submission.settled).resolves.toBeUndefined();

    const handler = added.mock.calls[0]?.[1];
    expect(removed).toHaveBeenCalledExactlyOnceWith('abort', handler);
  });

  // Vitest fails a run on an unhandled rejection, so settling `settled` before the caller holds it must stay quiet.
  it('settles quietly when the account changes while the enqueue is still pending', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const controller = new AbortController();
    const deferred = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const pending = submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => deferred.promise,
      graph,
      hub: fake.hub,
      signal: controller.signal,
    });

    controller.abort();
    await Promise.resolve();
    deferred.resolve({ enqueued: 1, itemIds: [1] });

    const submission = await pending;
    await expect(submission.settled).rejects.toMatchObject({ reason: 'aborted' });
    expect(submission.itemIds).toEqual([1]);
    // The settled watch arms no reconciliation timer for the late acceptance.
    expect(vi.getTimerCount()).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('settles quietly when the item fails before the enqueue response arrives', async () => {
    const fake = createFakeHub();
    const deferred = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const pending = submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => deferred.promise,
      graph,
      hub: fake.hub,
    });

    fake.emit('queue_item_status_changed', statusEvent('failed', { error_message: 'No model' }));
    await Promise.resolve();
    deferred.resolve({ enqueued: 1, itemIds: [1] });

    const submission = await pending;
    await expect(submission.settled).rejects.toMatchObject({ message: 'No model', reason: 'failed' });
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('re-reads the item after a reconnect so a missed completion still settles', async () => {
    const fake = createFakeHub();
    const readItemStatus = vi.fn<UtilityItemStatusReader>(() => Promise.resolve({ status: 'in_progress' }));
    const submission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 1, itemIds: [9] }),
      graph,
      hub: fake.hub,
      readItemStatus,
    });
    const outcome = submission.settled.then(
      () => 'resolved',
      (error: UtilityQueueError) => error.reason
    );

    // The initial status replay is not a reconnect.
    fake.connection('connected');
    expect(readItemStatus).not.toHaveBeenCalled();

    fake.connection('disconnected');
    fake.connection('connected');
    await vi.waitFor(() => expect(readItemStatus).toHaveBeenCalledExactlyOnceWith(9, undefined));
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);

    readItemStatus.mockResolvedValueOnce({ status: 'completed' });
    fake.connection('disconnected');
    fake.connection('connected');

    await expect(outcome).resolves.toBe('resolved');
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(fake.connectionHandlerCount()).toBe(0);
  });

  it('re-reads the item after a queue clear and settles as removed when the item is gone', async () => {
    const fake = createFakeHub();
    const readItemStatus = vi.fn<UtilityItemStatusReader>(() => Promise.resolve({ status: 'pending' }));
    const submission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 1, itemIds: [9] }),
      graph,
      hub: fake.hub,
      readItemStatus,
    });
    const outcome = submission.settled.then(
      () => 'resolved',
      (error: UtilityQueueError) => error.reason
    );
    const clearedEvent = { queue_id: 'default', timestamp: 1, user_id: null };

    // A clear that only removed other items leaves this one watched.
    fake.emit('queue_cleared', clearedEvent);
    await vi.waitFor(() => expect(readItemStatus).toHaveBeenCalledExactlyOnceWith(9, undefined));
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);

    // A pending item a clear deleted answers 404: there will never be a terminal event for it.
    readItemStatus.mockRejectedValueOnce(new ApiError('{"detail":"Queue item not found"}', 404));
    fake.emit('queue_cleared', clearedEvent);

    await expect(outcome).resolves.toBe('removed');
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(fake.handlerCount('queue_cleared')).toBe(0);
  });

  it('defers a clear that arrives before the enqueue response until the item id is known', async () => {
    const fake = createFakeHub();
    const deferred = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const readItemStatus = vi.fn<UtilityItemStatusReader>(() =>
      Promise.reject(new ApiError('{"detail":"Queue item not found"}', 404))
    );
    const pending = submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => deferred.promise,
      graph,
      hub: fake.hub,
      readItemStatus,
    });

    fake.emit('queue_cleared', { queue_id: 'default', timestamp: 1, user_id: null });
    await Promise.resolve();
    expect(readItemStatus).not.toHaveBeenCalled();

    deferred.resolve({ enqueued: 1, itemIds: [12] });
    const submission = await pending;

    await expect(submission.settled).rejects.toMatchObject({ reason: 'removed' });
    expect(readItemStatus).toHaveBeenCalledExactlyOnceWith(12, undefined);
  });

  it('reads the item at the reconcile interval: a finished item settles, an open one re-arms the read', async () => {
    vi.useFakeTimers();
    for (const [status, expected] of [
      ['completed', 'resolved'],
      ['failed', 'failed'],
    ] as const) {
      const fake = createFakeHub();
      const submission = await submitUtilityGraph({
        createId: () => UTIL_ID,
        enqueue: okEnqueue,
        graph,
        hub: fake.hub,
        readItemStatus: () => Promise.resolve({ error_message: 'boom', status }),
      });
      const outcome = submission.settled.then(
        () => 'resolved',
        (error: UtilityQueueError) => error.reason
      );

      vi.advanceTimersByTime(DEFAULT_UTILITY_RECONCILE_INTERVAL_MS - 1);
      expect(fake.handlerCount('queue_item_status_changed')).toBe(1);
      vi.advanceTimersByTime(1);

      await expect(outcome).resolves.toBe(expected);
      expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    }

    // A first run may still be downloading models at the interval: the watch reads again later instead of giving up.
    const fake = createFakeHub();
    const readItemStatus = vi.fn<UtilityItemStatusReader>(() => Promise.resolve({ status: 'in_progress' }));
    const submission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph,
      hub: fake.hub,
      readItemStatus,
    });
    const outcome = submission.settled.then(
      () => 'resolved',
      (error: UtilityQueueError) => error.reason
    );

    await vi.advanceTimersByTimeAsync(DEFAULT_UTILITY_RECONCILE_INTERVAL_MS);
    expect(readItemStatus).toHaveBeenCalledTimes(1);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);
    expect(vi.getTimerCount()).toBe(1);

    readItemStatus.mockResolvedValueOnce({ status: 'completed' });
    await vi.advanceTimersByTimeAsync(DEFAULT_UTILITY_RECONCILE_INTERVAL_MS);

    await expect(outcome).resolves.toBe('resolved');
    expect(readItemStatus).toHaveBeenCalledTimes(2);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not arm a reconcile read when the interval is disabled', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const submission = await submitUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph,
      hub: fake.hub,
      reconcileIntervalMs: 0,
    });

    expect(vi.getTimerCount()).toBe(0);
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(submission.settled).resolves.toBeUndefined();
  });
});
