import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { noteQueueCancelRequested, queueCancelRequests } from './cancelRequests';

const submittedBefore = new Date(Date.now() - 1_000).toISOString();
const run = {
  backendBatchId: 'batch-1',
  backendItemIds: [41, 42],
  origin: 'webv2:p:project-1:q:item-1',
  submittedAt: submittedBefore,
};
const wasRequested = (candidate = run) => queueCancelRequests.wasRequestedHere(candidate);

afterEach(() => {
  vi.useRealTimers();
  accountLifecycle.invalidate();
});

describe('queue cancel requests', () => {
  it('covers a run by one of its items, its batch, or a scope its origin falls in', () => {
    expect(wasRequested()).toBe(false);

    noteQueueCancelRequested({ itemId: 42, kind: 'item' });
    expect(wasRequested()).toBe(true);

    accountLifecycle.invalidate();
    noteQueueCancelRequested({ batchId: 'batch-1', kind: 'batch' });
    expect(wasRequested()).toBe(true);

    accountLifecycle.invalidate();
    noteQueueCancelRequested({ kind: 'scope', originPrefix: 'webv2:p:project-2:q:' });
    expect(wasRequested()).toBe(false);
    noteQueueCancelRequested({ kind: 'scope', originPrefix: 'webv2:p:project-1:q:' });
    expect(wasRequested()).toBe(true);
  });

  it('leaves out the running item a cancel-all-but-current kept', () => {
    noteQueueCancelRequested({ exceptItemId: 41, kind: 'scope', originPrefix: null });

    expect(wasRequested()).toBe(false);
    expect(wasRequested({ ...run, backendItemIds: [43] })).toBe(true);
  });

  it('covers only runs submitted before the request', () => {
    noteQueueCancelRequested({ kind: 'scope', originPrefix: null });

    expect(wasRequested({ ...run, submittedAt: new Date(Date.now() + 1_000).toISOString() })).toBe(false);
  });

  it('forgets a request whose call failed', () => {
    const drop = noteQueueCancelRequested({ kind: 'scope', originPrefix: null });

    drop();

    expect(wasRequested()).toBe(false);
  });

  it('lets a request lapse, so a cancel long after still reports', () => {
    vi.useFakeTimers();
    noteQueueCancelRequested({ kind: 'scope', originPrefix: null });

    vi.advanceTimersByTime(2 * 60_000);

    expect(wasRequested()).toBe(false);
  });

  it('forgets requests with the account', () => {
    noteQueueCancelRequested({ kind: 'scope', originPrefix: null });

    accountLifecycle.invalidate();

    expect(wasRequested()).toBe(false);
  });
});
