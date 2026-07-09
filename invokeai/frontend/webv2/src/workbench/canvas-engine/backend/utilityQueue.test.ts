import type { InvocationCompleteEvent, QueueItemStatusChangedEvent } from '@workbench/backend/events';
import type { SocketHub } from '@workbench/backend/socketHub';

import {
  buildQueueItemOrigin,
  buildUtilityQueueItemOrigin,
  isUtilityQueueItemOrigin,
  parseQueueItemOrigin,
} from '@workbench/backend/events';
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
  result: { image: { image_name: 'filtered.png' }, type: 'image_output' },
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

    await expect(promise).resolves.toEqual({ imageName: 'filtered.png', origin: ORIGIN });
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

  it('rejects no-output when the item completes without an image', async () => {
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
    });
    fake.emit('queue_item_status_changed', statusEvent('completed'));
    await expect(promise).rejects.toMatchObject({ name: 'UtilityQueueError', reason: 'no-output' });
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
  it('rejects with timeout after timeoutMs with no terminal event', async () => {
    vi.useFakeTimers();
    const fake = createFakeHub();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      timeoutMs: 1000,
    });
    const assertion = expect(promise).rejects.toMatchObject({ reason: 'timeout' });
    await vi.advanceTimersByTimeAsync(1000);
    await assertion;
    // Listeners cleaned up after timeout.
    expect(fake.handlerCount('queue_item_status_changed')).toBe(0);
  });

  it('rejects immediately when the signal is already aborted', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    controller.abort();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
    });
    await expect(promise).rejects.toMatchObject({ reason: 'aborted' });
  });

  it('rejects aborted when the signal fires mid-flight and detaches listeners', async () => {
    const fake = createFakeHub();
    const controller = new AbortController();
    const promise = runUtilityGraph({
      createId: () => UTIL_ID,
      enqueue: okEnqueue,
      graph: { edges: [], id: 'g', nodes: {} },
      hub: fake.hub,
      signal: controller.signal,
    });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ reason: 'aborted' });
    expect(fake.handlerCount('invocation_complete')).toBe(0);
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
