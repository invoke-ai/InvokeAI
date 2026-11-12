import type { QueueBackendPort, QueueItemReadModel } from '@features/queue/core/types';

import { buildQueueItemOrigin } from '@features/queue/data/events';
import { ApiError } from '@platform/transport/http';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDeliveryConfirmations, type StagedQueueResult } from './deliveryClaims';

const candidate = (id: number): StagedQueueResult => ({ backendItemId: id, projectId: 'project', queueItemId: 'run' });
const item = (id: number, retry: number | null = 1): QueueItemReadModel =>
  ({
    id,
    retriedFromItemId: retry,
    destination: 'canvas',
    origin: buildQueueItemOrigin('run', 'project'),
  }) as QueueItemReadModel;
const harness = (overrides: Partial<QueueBackendPort> = {}) => {
  const backend = {
    claimItemDelivery: vi.fn<QueueBackendPort['claimItemDelivery']>().mockResolvedValue({ kind: 'delivered' }),
    readItemsById: vi
      .fn<QueueBackendPort['readItemsById']>()
      .mockImplementation((ids) => Promise.resolve(ids.map((id) => item(id)))),
    ...overrides,
  };
  const reportError = vi.fn();
  const owner = new AbortController();
  const confirmations = createDeliveryConfirmations({
    backend,
    isActive: () => !owner.signal.aborted,
    signal: owner.signal,
    reportError,
  });
  return { backend, confirmations, reportError, owner };
};

afterEach(() => vi.useRealTimers());

describe('delivery confirmations', () => {
  it('reads recovered candidates serially in bounded batches and caps confirmation writes at eight', async () => {
    let activeReads = 0;
    let maxReads = 0;
    let activeWrites = 0;
    let maxWrites = 0;
    const releases: Array<() => void> = [];
    const { backend, confirmations } = harness({
      readItemsById: vi.fn<QueueBackendPort['readItemsById']>(async (ids) => {
        activeReads += 1;
        maxReads = Math.max(maxReads, activeReads);
        await Promise.resolve();
        activeReads -= 1;
        return ids.map((id) => item(id, id % 2 ? 1 : null));
      }),
      claimItemDelivery: vi.fn<QueueBackendPort['claimItemDelivery']>(() => {
        activeWrites += 1;
        maxWrites = Math.max(maxWrites, activeWrites);
        return new Promise((resolve) => {
          releases.push(() => {
            activeWrites -= 1;
            resolve({ kind: 'delivered' });
          });
        });
      }),
    });
    const candidates = Array.from({ length: 200 }, (_, index) => candidate(index + 1));
    confirmations.recover(candidates);
    confirmations.recover(candidates);
    await vi.waitFor(() => expect(backend.readItemsById).toHaveBeenCalledTimes(4));
    expect(vi.mocked(backend.readItemsById).mock.calls.map(([ids]) => ids.length)).toEqual([64, 64, 64, 8]);
    expect(maxReads).toBe(1);
    expect(backend.claimItemDelivery).toHaveBeenCalledTimes(8);
    for (let i = 0; i < 13; i += 1) {
      for (const release of releases.splice(0)) {
        release();
      }
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    }
    await vi.waitFor(() => expect(backend.claimItemDelivery).toHaveBeenCalledTimes(100));
    expect(maxWrites).toBe(8);
    expect(
      vi.mocked(backend.claimItemDelivery).mock.calls.every(([id, action]) => id % 2 === 1 && action === 'confirm')
    ).toBe(true);
    confirmations.recover(candidates);
    await Promise.resolve();
    expect(backend.readItemsById).toHaveBeenCalledTimes(4);
    confirmations.dispose();
    releases.splice(0).forEach((release) => release());
  });

  it('does not confirm ordinary, missing, foreign-project, foreign-run or non-Canvas rows after reload/history clearing', async () => {
    const { backend, confirmations } = harness({
      readItemsById: vi
        .fn()
        .mockResolvedValue([
          item(1, null),
          { ...item(3), origin: buildQueueItemOrigin('run', 'other-project') },
          { ...item(4), origin: buildQueueItemOrigin('other-run', 'project') },
          { ...item(5), destination: 'gallery' },
          item(6),
        ]),
    });
    const candidates = [1, 2, 3, 4, 5, 6].map(candidate);
    confirmations.recover(candidates);
    await vi.waitFor(() => expect(backend.claimItemDelivery).toHaveBeenCalledTimes(1));
    expect(backend.claimItemDelivery).toHaveBeenCalledWith(6, 'confirm', 'run:["project","run"]');
    confirmations.recover(candidates);
    expect(backend.readItemsById).toHaveBeenCalledTimes(1);
    confirmations.dispose();
  });

  it.each([false, true])(
    'keeps only surviving ordinary candidates when routing throws (staged: %s)',
    async (staged) => {
      const { backend, confirmations } = harness();
      expect(() =>
        confirmations.routeOrdinaryResults([99], () => {
          if (staged) {
            confirmations.recover([candidate(99)]);
          }
          throw new Error('routing failed');
        })
      ).toThrow('routing failed');
      if (staged) {
        confirmations.recover([candidate(99)]);
        await Promise.resolve();
        expect(backend.readItemsById).not.toHaveBeenCalled();
        // Removing the candidate ends its classification, so a later restored candidate must be validated again.
        confirmations.recover([]);
      }
      confirmations.recover([candidate(99)]);
      await vi.waitFor(() =>
        expect(backend.claimItemDelivery).toHaveBeenCalledWith(99, 'confirm', 'run:["project","run"]')
      );
      expect(backend.readItemsById).toHaveBeenCalledExactlyOnceWith([99], expect.any(AbortSignal));
      confirmations.dispose();
    }
  );

  it('shares one five-attempt budget across run/tab claimants and synchronous candidate notifications', async () => {
    vi.useFakeTimers();
    const { backend, confirmations, reportError } = harness({
      claimItemDelivery: vi.fn().mockRejectedValue(new TypeError('offline')),
    });
    confirmations.confirm(99, 'tab:one');
    confirmations.recover([candidate(99)]);
    confirmations.confirm(99, 'run:two');
    await vi.advanceTimersByTimeAsync(30_000);
    confirmations.recover([candidate(99)]);
    confirmations.confirm(99, 'tab:one');
    await vi.advanceTimersByTimeAsync(180_000);
    expect(backend.claimItemDelivery).toHaveBeenCalledTimes(5);
    expect(reportError).toHaveBeenCalledOnce();
    expect(reportError).toHaveBeenCalledWith(expect.stringContaining('are staged'));
    expect(backend.readItemsById).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
    confirmations.dispose();
  });

  it('bounds failed recovery reads and reports exhausted recovery without arbitrary claim renewals', async () => {
    vi.useFakeTimers();
    const { backend, confirmations, reportError } = harness({
      readItemsById: vi.fn().mockRejectedValue(new TypeError('offline')),
    });
    const candidates = Array.from({ length: 1_000 }, (_, index) => candidate(index + 1));
    confirmations.recover(candidates);
    await vi.advanceTimersByTimeAsync(180_000);
    confirmations.recover(candidates);
    expect(backend.readItemsById).toHaveBeenCalledTimes(3);
    expect(backend.claimItemDelivery).not.toHaveBeenCalled();
    expect(reportError).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    confirmations.dispose();
  });

  it('clears retry timers and queued confirmations when its account ends', async () => {
    vi.useFakeTimers();
    const { backend, confirmations, owner, reportError } = harness({
      claimItemDelivery: vi.fn().mockRejectedValue(new TypeError('offline')),
    });
    for (let id = 1; id <= 20; id += 1) {
      confirmations.confirm(id, 'tab:owner');
    }
    await vi.advanceTimersByTimeAsync(0);
    expect(vi.getTimerCount()).toBe(20);
    owner.abort();
    expect(vi.getTimerCount()).toBe(0);
    const requests = vi.mocked(backend.claimItemDelivery).mock.calls.length;
    await vi.advanceTimersByTimeAsync(180_000);
    expect(backend.claimItemDelivery).toHaveBeenCalledTimes(requests);
    expect(reportError).not.toHaveBeenCalled();
    expect(confirmations.isStaged(1)).toBe(false);
  });

  it('drains candidates arriving as the previous recovery pass is finishing', async () => {
    const { backend, confirmations } = harness();
    confirmations.recover([]);
    confirmations.recover([candidate(99)]);
    await vi.waitFor(() =>
      expect(backend.claimItemDelivery).toHaveBeenCalledWith(99, 'confirm', 'run:["project","run"]')
    );
    confirmations.dispose();
  });

  it('renews a granted live lease behind a recovery backlog without claiming recovered candidates', async () => {
    vi.useFakeTimers();
    let claimedAt = Date.now();
    let holder = 'tab:actual';
    const backend = {
      readItemsById: vi
        .fn<QueueBackendPort['readItemsById']>()
        .mockImplementation((ids) => Promise.resolve(ids.map((id) => item(id)))),
      claimItemDelivery: vi.fn<QueueBackendPort['claimItemDelivery']>((_id, action, claimant) => {
        if (action === 'confirm') {
          return new Promise(() => {
            /* Keep confirmation transport deferred while the lease-renewal queue drains. */
          });
        }
        if (holder !== claimant && Date.now() - claimedAt < 120_000) {
          return Promise.resolve({ kind: 'held', retryInMs: 120_000 - (Date.now() - claimedAt) });
        }
        holder = claimant;
        claimedAt = Date.now();
        return Promise.resolve({ kind: 'claimed' });
      }),
    };
    const { confirmations } = harness(backend);
    confirmations.recover(Array.from({ length: 30 }, (_, index) => candidate(index + 1)));
    await vi.advanceTimersByTimeAsync(0);
    confirmations.confirm(999, 'run:recovery');
    confirmations.confirm(999, 'tab:actual', true);
    confirmations.confirm(999, 'tab:fail-open');
    const timer = setInterval(confirmations.renewOwned, 40_000);
    await vi.advanceTimersByTimeAsync(160_000);
    const renewals = backend.claimItemDelivery.mock.calls.filter(([, action]) => action === 'claim');
    expect(renewals).toEqual(Array.from({ length: 4 }, () => [999, 'claim', 'tab:actual']));
    expect(backend.claimItemDelivery.mock.calls.filter(([, action]) => action === 'confirm')).toHaveLength(8);
    await expect(backend.claimItemDelivery(999, 'claim', 'another-browser')).resolves.toMatchObject({ kind: 'held' });
    confirmations.dispose();
    clearInterval(timer);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('keeps an upgraded live holder when the previous holder renewal responds late', async () => {
    let finishOldRenewal!: () => void;
    const { backend, confirmations } = harness({
      claimItemDelivery: vi.fn<QueueBackendPort['claimItemDelivery']>((_id, action, claimant) => {
        if (action === 'confirm') {
          return new Promise(() => {
            /* Confirmation remains unresolved while delivery ownership changes. */
          });
        }
        if (claimant === 'tab:old') {
          return new Promise((resolve) => {
            finishOldRenewal = () => resolve({ kind: 'held', retryInMs: 120_000 });
          });
        }
        return Promise.resolve({ kind: 'claimed' });
      }),
    });
    confirmations.confirm(99, 'tab:old', true);
    confirmations.renewOwned();
    confirmations.confirm(99, 'tab:new', true);
    finishOldRenewal();
    // Let the old response and its in-flight cleanup settle before the next renewal tick.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    confirmations.renewOwned();
    expect(backend.claimItemDelivery).toHaveBeenCalledWith(99, 'claim', 'tab:new');
    confirmations.dispose();
  });

  it('bounds and coalesces owned lease renewal requests', async () => {
    let active = 0;
    let peak = 0;
    const finishers: Array<() => void> = [];
    const { backend, confirmations } = harness({
      claimItemDelivery: vi.fn<QueueBackendPort['claimItemDelivery']>((_id, action) => {
        if (action === 'confirm') {
          return new Promise(() => {
            /* Keep confirmation transport deferred while the lease-renewal queue drains. */
          });
        }
        active += 1;
        peak = Math.max(peak, active);
        return new Promise((resolve) => {
          finishers.push(() => {
            active -= 1;
            resolve({ kind: 'claimed' });
          });
        });
      }),
    });
    for (let id = 1; id <= 20; id += 1) {
      confirmations.confirm(id, 'tab:owner', true);
    }
    confirmations.renewOwned();
    confirmations.renewOwned();
    expect(active).toBe(8);
    for (let i = 0; i < 3; i += 1) {
      finishers.splice(0).forEach((finish) => finish());
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    }
    await vi.waitFor(() =>
      expect(vi.mocked(backend.claimItemDelivery).mock.calls.filter(([, action]) => action === 'claim')).toHaveLength(
        20
      )
    );
    expect(peak).toBe(8);
    confirmations.dispose();
  });

  it('keeps the unsupported-endpoint fallback quiet', async () => {
    const { backend, confirmations, reportError } = harness({
      claimItemDelivery: vi.fn().mockRejectedValue(new ApiError('unsupported', 404)),
    });
    confirmations.confirm(99, 'tab:one');
    await vi.waitFor(() => expect(backend.claimItemDelivery).toHaveBeenCalledOnce());
    expect(reportError).not.toHaveBeenCalled();
    confirmations.dispose();
  });
});
