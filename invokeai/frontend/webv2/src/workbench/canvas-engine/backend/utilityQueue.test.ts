import type { InvocationCompleteEvent, QueueItemStatusChangedEvent } from '@workbench/backend/events';
import type { SocketHub } from '@workbench/backend/socketHub';

import {
  buildQueueItemOrigin,
  buildUtilityQueueItemOrigin,
  isUtilityQueueItemOrigin,
  parseQueueItemOrigin,
} from '@workbench/backend/events';
import { ApiError } from '@workbench/backend/http';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { UtilityEnqueue } from './utilityQueue';

import { DEFAULT_UTILITY_QUEUE_TIMEOUT_MS, runUtilityGraph, UtilityQueueError } from './utilityQueue';

/** A fake socket hub that records handlers per event and can emit to them, mirroring `hub.on`. */
const createFakeHub = () => {
  const handlers = new Map<string, Set<(payload: unknown) => void>>();
  let detachCount = 0;

  const hub: Pick<SocketHub, 'on'> = {
    on: (event, handler) => {
      const set = handlers.get(event) ?? new Set();
      set.add(handler as (payload: unknown) => void);
      handlers.set(event, set);
      return () => {
        detachCount += 1;
        set.delete(handler as (payload: unknown) => void);
      };
    },
  };

  const emit = (event: string, payload: unknown): void => {
    for (const handler of handlers.get(event) ?? []) {
      handler(payload);
    }
  };

  return {
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
  ...overrides,
});

const statusEvent = (
  status: QueueItemStatusChangedEvent['status'],
  overrides: Partial<QueueItemStatusChangedEvent> = {}
): QueueItemStatusChangedEvent => ({
  batch_id: 'b',
  completed_at: null,
  created_at: '',
  destination: null,
  error_message: null,
  error_type: null,
  item_id: 1,
  origin: ORIGIN,
  queue_id: 'default',
  session_id: 's',
  started_at: null,
  status,
  updated_at: '',
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
    // The utility origin resolves to NO local queue item, so `queueCoordinator`
    // (which keys backend items by `parseQueueItemOrigin`) and `routeQueueItemResults`
    // never adopt a utility item into project staging / gallery routing.
    expect(isUtilityQueueItemOrigin(ORIGIN)).toBe(true);
    expect(parseQueueItemOrigin(ORIGIN)).toBeNull();

    // A real project origin, by contrast, DOES parse to its local queue item id —
    // demonstrating the coordinator adopts those but not utility items.
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
