import type { InvocationCompleteEvent, QueueItemStatusChangedEvent } from '@features/queue/contracts';
import type { QueueItemsCanceledEvent } from '@features/queue/data/events';
import type { SocketHub } from '@platform/transport/socketHub';
import type { BackendConnectionStatus } from '@platform/transport/types';

import {
  buildQueueItemOrigin,
  buildUtilityQueueItemOrigin,
  isUtilityQueueItemOrigin,
  parseQueueItemOrigin,
} from '@features/queue/contracts';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { UtilityEnqueue, UtilityItemStatusReader, UtilityQueueError } from './utilityGraph';

import { DEFAULT_UTILITY_RECONCILE_INTERVAL_MS, runUtilityGraph, submitUtilityGraph } from './utilityGraph';

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

const bulkCanceledEvent = (itemIds: number[]): QueueItemsCanceledEvent => ({
  canceled_item_ids: itemIds,
  canceled_item_ids_by_user: { 'user-1': itemIds },
  queue_id: 'default',
  timestamp: 1,
  user_ids: ['user-1'],
});

/** Yields a real macrotask, so everything the previous tasks queued (an unhandled rejection report included) runs. */
const nextMacrotask = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
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
  accountLifecycle.invalidate();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('runUtilityGraph — await + settle', () => {
  it.each([false, true])('requests front-of-queue execution only when prepend is %s', async (prepend) => {
    const fake = createFakeHub();
    const acknowledge = vi.fn();
    const enqueue = vi.fn<UtilityEnqueue>(async (options) => ({ ...(await okEnqueue(options)), acknowledge }));
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      prepend,
    });
    await Promise.resolve();
    expect(enqueue).toHaveBeenCalledWith({
      graph: { edges: [], id: 'g', nodes: {} },
      origin: ORIGIN,
      idempotencyKey: ORIGIN,
      ...(prepend ? { prepend: true } : {}),
    });
    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
    expect(acknowledge).toHaveBeenCalledOnce();
  });

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
    // Every listener detached on settle: output, status, queue clear, and bulk cancel.
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(fake.detachCount).toBe(4);
    expect(fake.connectionHandlerCount()).toBe(0);
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
    expect(reconcileCompletedOutput).toHaveBeenCalledExactlyOnceWith([1], 'control_filter', expect.any(AbortSignal));
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
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

  it.each<[string, Partial<QueueItemStatusChangedEvent>, string]>([
    ['the backend message', { error_message: 'boom' }, 'boom'],
    [
      'a trimmed backend error type when the message is empty',
      { error_message: '', error_traceback: 'giant traceback', error_type: '  AttributeError  ' },
      'AttributeError',
    ],
    [
      'a trimmed backend error type when the message is blank',
      { error_message: '   ', error_traceback: 'giant traceback', error_type: '  AttributeError  ' },
      'AttributeError',
    ],
    [
      'a trimmed nonempty backend message over its error type',
      { error_message: '  useful detail  ', error_type: 'AttributeError' },
      'useful detail',
    ],
    ['the generic fallback when message and type are missing', {}, 'The utility graph failed.'],
    [
      'the generic fallback for malformed non-string failure fields',
      { error_message: 42, error_type: { name: 'AttributeError' } } as never,
      'The utility graph failed.',
    ],
  ])('rejects a failed item with %s', async (_label, fields, message) => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });

    fake.emit('queue_item_status_changed', statusEvent('failed', fields));

    await expect(promise).rejects.toMatchObject({ message, reason: 'failed' });
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
      readItemStatus: () => Promise.resolve({ status: 'pending' }),
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
});

describe('runUtilityGraph — enqueue failures', () => {
  it.each([
    { enqueued: 1, itemIds: [], label: 'accepted count without ids', canceled: [] },
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
});

describe('runUtilityGraph — missed settlement', () => {
  it('re-reads the item after a queue clear, bulk cancel, or reconnect instead of waiting for the timeout', async () => {
    const graph = { edges: [], id: 'g', nodes: {} };
    const removed = createFakeHub();
    const readRemoved = vi.fn<UtilityItemStatusReader>(() =>
      Promise.reject(new ApiError('{"detail":"Queue item not found"}', 404))
    );
    const removedRun = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 1, itemIds: [3] }),
      graph,
      hub: removed.hub,
      readItemStatus: readRemoved,
      timeoutMs: 0,
    });
    await vi.waitFor(() => expect(removed.handlerCount('queue_cleared')).toBe(1));
    await Promise.resolve();
    removed.emit('queue_cleared', { queue_id: 'default', timestamp: 1, user_id: null });
    await expect(removedRun).rejects.toMatchObject({ reason: 'removed' });
    expect(readRemoved).toHaveBeenCalledExactlyOnceWith(3, expect.any(AbortSignal));
    expect(removed.handlerCount('queue_items_canceled')).toBe(0);

    const canceled = createFakeHub();
    const canceledRun = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 1, itemIds: [3] }),
      graph,
      hub: canceled.hub,
      readItemStatus: () => Promise.resolve({ status: 'canceled' }),
      timeoutMs: 0,
    });
    await nextMacrotask();
    canceled.emit('queue_items_canceled', bulkCanceledEvent([3]));
    await expect(canceledRun).rejects.toMatchObject({ reason: 'canceled' });

    // A reconnect that finds the item completed reconciles its output as a socket completion would.
    const reconnected = createFakeHub();
    const reconnectedRun = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 1, itemIds: [3] }),
      graph,
      hub: reconnected.hub,
      readItemStatus: () => Promise.resolve({ status: 'completed' }),
      reconcileCompletedOutput: () => Promise.resolve({ height: 8, imageName: 'out.png', width: 8 }),
      timeoutMs: 0,
    });
    await nextMacrotask();
    reconnected.connection('disconnected');
    reconnected.connection('connected');
    await expect(reconnectedRun).resolves.toEqual({ height: 8, imageName: 'out.png', origin: ORIGIN, width: 8 });
    expect(reconnected.connectionHandlerCount()).toBe(0);
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

describe('submitUtilityGraph — fire, then watch settlement', () => {
  const graph = { edges: [], id: 'pbr', nodes: {} };

  it('resolves once the backend accepts the graph and settles when the item completes', async () => {
    const fake = createFakeHub();
    const enqueue = vi.fn<UtilityEnqueue>(() => Promise.resolve({ enqueued: 1, itemIds: [7] }));

    const submission = await submitUtilityGraph({ createId: () => UTIL_ID, enqueue, graph, hub: fake.hub });

    expect(enqueue).toHaveBeenCalledExactlyOnceWith({ graph, origin: ORIGIN, idempotencyKey: ORIGIN });
    expect(submission).toMatchObject({ itemIds: [7], origin: ORIGIN });
    // Only status changes are watched; no output image is expected.
    expect(fake.handlerCount('invocation_complete')).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);

    fake.emit('queue_item_status_changed', statusEvent('completed', { origin: 'other-origin' }));
    fake.emit('queue_item_status_changed', statusEvent('completed', { item_id: 7 }));

    await expect(submission.settled).resolves.toBeUndefined();
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

  // `settled` can reject before the caller holds it. Vitest fails the run on an unhandled rejection, and a real
  // macrotask between the rejection and the caller's handler gives one the chance to be reported.

  it('settles quietly when the account changes while the enqueue is still pending', async () => {
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
    await nextMacrotask();
    vi.useFakeTimers();
    deferred.resolve({ enqueued: 1, itemIds: [1] });

    const submission = await pending;
    await expect(submission.settled).rejects.toMatchObject({ reason: 'aborted' });
    expect(submission.itemIds).toEqual([1]);
    // The settled watch arms no reconciliation timer for the late acceptance.
    expect(vi.getTimerCount()).toBe(0);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });
});

describe('utility execution progress', () => {
  it('observes execution before the enqueue response without regressing on a stale pending read', async () => {
    const fake = createFakeHub();
    const accepted = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const read = createDeferred<{ status: 'pending' }>();
    const onStatus = vi.fn();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => accepted.promise,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      onStatus,
      readItemStatus: () => read.promise,
    });
    expect(onStatus).not.toHaveBeenCalled();
    fake.emit('queue_item_status_changed', statusEvent('pending', { origin: 'another-utility' }));
    expect(onStatus).not.toHaveBeenCalled();
    fake.emit('queue_item_status_changed', statusEvent('in_progress'));
    expect(onStatus.mock.calls).toEqual([['in_progress']]);
    accepted.resolve({ enqueued: 1, itemIds: [1] });
    await nextMacrotask();
    read.resolve({ status: 'pending' });
    await nextMacrotask();
    expect(onStatus.mock.calls).toEqual([['in_progress']]);
    fake.emit('invocation_complete', completeEvent());
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(promise).resolves.toMatchObject({ imageName: 'filtered.png' });
    fake.emit('queue_item_status_changed', statusEvent('pending'));
    expect(onStatus.mock.calls).toEqual([['in_progress']]);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('repairs missed execution events by polling and stops reads and callbacks after cancellation', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const abort = new AbortController();
    const onStatus = vi.fn();
    const readItemStatus = vi
      .fn<UtilityItemStatusReader>()
      .mockResolvedValueOnce({ status: 'pending' })
      .mockResolvedValueOnce({ status: 'in_progress' })
      .mockResolvedValue({ status: 'pending' });
    const cancel = vi.fn(() => Promise.resolve());
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      onStatus,
      readItemStatus,
      cancel,
      signal: abort.signal,
    });
    const outcome = promise.catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(0);
    expect(onStatus.mock.calls).toEqual([['pending']]);
    await vi.advanceTimersByTimeAsync(DEFAULT_UTILITY_RECONCILE_INTERVAL_MS);
    expect(onStatus.mock.calls).toEqual([['pending'], ['in_progress']]);
    fake.connection('disconnected');
    fake.connection('connected');
    await vi.advanceTimersByTimeAsync(0);
    expect(onStatus.mock.calls).toEqual([['pending'], ['in_progress']]);
    abort.abort();
    await expect(outcome).resolves.toMatchObject({ reason: 'aborted' });
    await vi.advanceTimersByTimeAsync(0);
    const reads = readItemStatus.mock.calls.length;
    await vi.advanceTimersByTimeAsync(10_000);
    expect(readItemStatus).toHaveBeenCalledTimes(reads);
    expect(cancel).toHaveBeenCalledWith([1]);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('keeps observer read failures within the shared health budget and fences account cancellation', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const onStatus = vi.fn();
    const cancel = vi.fn().mockResolvedValue(undefined);
    const readItemStatus = vi.fn<UtilityItemStatusReader>().mockRejectedValue(new TypeError('offline'));
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      enqueue: okEnqueue,
      onStatus,
      readItemStatus,
      cancel,
    });
    const outcome = promise.catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i += 1) {
      fake.emit('queue_cleared', {});
      await vi.advanceTimersByTimeAsync(0);
    }
    expect(onStatus).not.toHaveBeenCalled();
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);
    await vi.advanceTimersByTimeAsync(DEFAULT_UTILITY_RECONCILE_INTERVAL_MS);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(1);
    readItemStatus.mockResolvedValueOnce({ status: 'in_progress' });
    fake.connection('disconnected');
    fake.connection('connected');
    await vi.advanceTimersByTimeAsync(0);
    expect(onStatus.mock.calls).toEqual([['in_progress']]);
    accountLifecycle.activate('replacement-progress-owner');
    await expect(outcome).resolves.toMatchObject({ reason: 'aborted' });
    const reads = readItemStatus.mock.calls.length;
    fake.emit('queue_item_status_changed', statusEvent('pending'));
    await vi.advanceTimersByTimeAsync(2 * DEFAULT_UTILITY_RECONCILE_INTERVAL_MS);
    expect(readItemStatus).toHaveBeenCalledTimes(reads);
    expect(onStatus.mock.calls).toEqual([['in_progress']]);
    expect(cancel).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not treat another accepted item still pending as a return to waiting', async () => {
    const fake = createFakeHub();
    const onStatus = vi.fn();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: () => Promise.resolve({ enqueued: 2, itemIds: [1, 2] }),
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      onStatus,
      readItemStatus: (id) => Promise.resolve({ status: id === 1 ? 'in_progress' : 'pending' }),
    });
    await nextMacrotask();
    expect(onStatus.mock.calls).toEqual([['in_progress']]);
    fake.emit('queue_item_status_changed', statusEvent('pending', { item_id: 2 }));
    expect(onStatus.mock.calls).toEqual([['in_progress']]);
    fake.emit('queue_item_status_changed', statusEvent('failed', { error_message: 'Resize failed' }));
    await expect(promise).rejects.toMatchObject({ reason: 'failed' });
    expect(fake.connectionHandlerCount()).toBe(0);
  });
});

describe('utility recovery lifetimes', () => {
  const graph = { edges: [], id: 'utility-recovery', nodes: {} };

  it.each(['submit', 'run'] as const)(
    'surfaces persistent status-read failure for %s without exhausting on event bursts',
    async (api) => {
      vi.useFakeTimers();
      const fake = createFakeHub();
      const readItemStatus = vi.fn<UtilityItemStatusReader>().mockRejectedValue(new TypeError('backend down'));
      const operation =
        api === 'submit'
          ? (await submitUtilityGraph({ graph, hub: fake.hub, enqueue: okEnqueue, readItemStatus })).settled
          : runUtilityGraph({ graph, hub: fake.hub, enqueue: okEnqueue, readItemStatus });
      const assertion = expect(operation).rejects.toMatchObject({
        reason: 'reconcile',
        message: expect.stringContaining('Backend work may still be running'),
      });
      for (let i = 0; i < 10; i += 1) {
        fake.emit('queue_cleared', {});
        await vi.advanceTimersByTimeAsync(0);
      }
      expect(fake.handlerCount('queue_item_status_changed')).toBe(1);
      await vi.advanceTimersByTimeAsync(30_000);
      await assertion;
      expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
      expect(vi.getTimerCount()).toBe(0);
    }
  );

  it('recovers a missing terminal event within one status interval and cleans up', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const readItemStatus = vi.fn<UtilityItemStatusReader>().mockResolvedValue({ status: 'completed' });
    const reconcileCompletedOutput = vi.fn().mockResolvedValue({ imageName: 'recovered.png', width: 64, height: 64 });
    const result = runUtilityGraph({
      graph,
      hub: fake.hub,
      enqueue: okEnqueue,
      createId: () => UTIL_ID,
      readItemStatus,
      reconcileCompletedOutput,
    });
    await vi.advanceTimersByTimeAsync(DEFAULT_UTILITY_RECONCILE_INTERVAL_MS - 1);
    expect(readItemStatus).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    await expect(result).resolves.toMatchObject({ imageName: 'recovered.png' });
    expect(readItemStatus).toHaveBeenCalledTimes(1);
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('aborts outstanding status reads when a watch stops and ignores their late terminal result', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const status = createDeferred<{ status: 'completed' }>();
    const readItemStatus = vi.fn<UtilityItemStatusReader>().mockReturnValue(status.promise);
    const submission = await submitUtilityGraph({
      graph,
      hub: fake.hub,
      enqueue: okEnqueue,
      readItemStatus,
      signal: controller.signal,
    });
    fake.connection('disconnected');
    fake.connection('connected');
    const requestSignal = readItemStatus.mock.calls[0]?.[1];
    expect(requestSignal?.aborted).toBe(false);
    controller.abort();
    await expect(submission.settled).rejects.toMatchObject({ reason: 'aborted' });
    expect(requestSignal?.aborted).toBe(true);
    status.resolve({ status: 'completed' });
    await nextMacrotask();
    expect(readItemStatus).toHaveBeenCalledTimes(1);
    expect(fake.connectionHandlerCount()).toBe(0);
  });

  it('aborts an outstanding output read when its account expires without retrying', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const output = createDeferred<null>();
    const reconcileCompletedOutput = vi.fn().mockReturnValue(output.promise);
    const result = runUtilityGraph({
      graph,
      hub: fake.hub,
      enqueue: okEnqueue,
      createId: () => UTIL_ID,
      reconcileCompletedOutput,
    });
    await vi.advanceTimersByTimeAsync(0);
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await vi.advanceTimersByTimeAsync(0);
    const requestSignal = reconcileCompletedOutput.mock.calls[0]?.[2] as AbortSignal;
    expect(requestSignal.aborted).toBe(false);
    accountLifecycle.activate('next-account');
    await expect(result).rejects.toMatchObject({ reason: 'aborted' });
    expect(requestSignal.aborted).toBe(true);
    output.resolve(null);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(reconcileCompletedOutput).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not start a deferred admission under a replacement account', async () => {
    const fake = createFakeHub();
    const enqueue = vi.fn(okEnqueue);
    const result = runUtilityGraph({ graph, hub: fake.hub, enqueue });
    accountLifecycle.activate('replacement-before-admission');
    await expect(result).rejects.toMatchObject({ reason: 'aborted' });
    await nextMacrotask();
    expect(enqueue).not.toHaveBeenCalled();
  });

  it('never cancels old-account ids through the replacement account after a late admission', async () => {
    const fake = createFakeHub();
    const admission = createDeferred<{ enqueued: number; itemIds: number[] }>();
    const cancel = vi.fn().mockResolvedValue(undefined);
    const result = runUtilityGraph({ graph, hub: fake.hub, enqueue: () => admission.promise, cancel });
    await Promise.resolve();
    accountLifecycle.activate('replacement');
    await expect(result).rejects.toMatchObject({ reason: 'aborted' });
    admission.resolve({ enqueued: 1, itemIds: [72] });
    await nextMacrotask();
    expect(cancel).not.toHaveBeenCalled();
  });
});
