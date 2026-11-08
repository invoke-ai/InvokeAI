import type { QueueBackendPort } from '@features/queue/core/types';

import { parseQueueItemOrigin, parseQueueItemOriginProjectId } from '@features/queue/data/events';
import { ApiError, sleep } from '@platform/transport/http';

export type StagedQueueResult = { backendItemId: number; projectId: string; queueItemId: string };

/** Owns bounded, claimant-independent confirmation and authoritative candidate recovery for one account runtime. */
export const createDeliveryConfirmations = ({
  backend,
  isActive,
  signal,
  reportError,
}: {
  backend: Pick<QueueBackendPort, 'claimItemDelivery' | 'readItemsById'>;
  isActive: () => boolean;
  signal: AbortSignal;
  reportError: (message: string) => void;
}) => {
  const confirmations = new Map<
    number,
    {
      attempts: number;
      claimant: string;
      settled: boolean;
      leaseClaimant?: string;
      timer?: ReturnType<typeof setTimeout>;
    }
  >();
  const ready = new Set<number>();
  const renewalQueue = new Set<number>();
  const renewing = new Set<number>();
  let activeRenewals = 0;
  let candidates = new Map<number, StagedQueueResult>();
  const checkedCandidates = new Set<number>();
  const recovery = new AbortController();
  let activeRequests = 0;
  let recovering = false;
  let recoverAgain = false;
  let recoveryFailed = false;
  let disposed = false;
  const active = () => !disposed && isActive();

  const drain = (): void => {
    if (!active()) {
      return;
    }
    for (const itemId of ready) {
      const pending = confirmations.get(itemId)!;
      if (activeRequests >= 8) {
        break;
      }
      ready.delete(itemId);
      pending.attempts += 1;
      activeRequests += 1;
      void backend
        .claimItemDelivery(itemId, 'confirm', pending.claimant)
        .then((result) => {
          if (result.kind !== 'delivered') {
            throw new Error('Invalid delivery confirmation response');
          }
          pending.settled = true;
        })
        .catch((error: unknown) => {
          if (!active()) {
            return;
          }
          const definitive =
            error instanceof ApiError && error.status < 500 && error.status !== 408 && error.status !== 429;
          if (definitive || pending.attempts >= 5) {
            pending.settled = true;
            if (error instanceof ApiError && error.status === 404) {
              return;
            }
            reportError(
              'Canvas results are staged, but server delivery could not be confirmed. Another browser may deliver them again.'
            );
            return;
          }
          pending.timer = setTimeout(
            () => {
              pending.timer = undefined;
              ready.add(itemId);
              drain();
            },
            1_000 * 2 ** (pending.attempts - 1)
          );
        })
        .finally(() => {
          activeRequests -= 1;
          drain();
        });
    }
  };

  const confirm = (itemId: number, claimant: string, ownsClaim = false): void => {
    if (!active()) {
      return;
    }
    const existing = confirmations.get(itemId);
    if (existing) {
      if (ownsClaim && !existing.settled) {
        existing.leaseClaimant = claimant;
      }
      return;
    }
    confirmations.set(itemId, {
      attempts: 0,
      claimant,
      settled: false,
      ...(ownsClaim ? { leaseClaimant: claimant } : {}),
    });
    ready.add(itemId);
    drain();
  };

  const drainRenewals = (): void => {
    if (!active()) {
      return;
    }
    for (const itemId of renewalQueue) {
      if (activeRenewals >= 8) {
        break;
      }
      renewalQueue.delete(itemId);
      const pending = confirmations.get(itemId);
      if (!pending?.leaseClaimant || pending.settled) {
        continue;
      }
      const leaseClaimant = pending.leaseClaimant;
      activeRenewals += 1;
      renewing.add(itemId);
      void backend
        .claimItemDelivery(itemId, 'claim', leaseClaimant)
        .then((result) => {
          if (
            confirmations.get(itemId) === pending &&
            pending.leaseClaimant === leaseClaimant &&
            result.kind !== 'claimed'
          ) {
            pending.leaseClaimant = undefined;
          }
        })
        .catch(() => undefined)
        .finally(() => {
          renewing.delete(itemId);
          activeRenewals -= 1;
          drainRenewals();
        });
    }
  };
  const renewOwned = (): void => {
    if (!active()) {
      return;
    }
    for (const [id, pending] of confirmations) {
      if (pending.leaseClaimant && !pending.settled && !renewing.has(id)) {
        renewalQueue.add(id);
      }
    }
    drainRenewals();
  };

  const routeOrdinaryResults = (itemIds: number[], route: () => void): void => {
    // Mark before synchronous staging notifications; retain only candidates the route actually leaves behind.
    for (const id of itemIds) {
      checkedCandidates.add(id);
    }
    try {
      route();
    } finally {
      for (const id of itemIds) {
        if (!candidates.has(id)) {
          checkedCandidates.delete(id);
        }
      }
    }
  };

  const recover = (staged: StagedQueueResult[]): void => {
    if (!active()) {
      return;
    }
    candidates = new Map(staged.map((candidate) => [candidate.backendItemId, candidate]));
    for (const id of checkedCandidates) {
      if (!candidates.has(id)) {
        checkedCandidates.delete(id);
      }
    }
    // Retain exhausted budgets while the candidate exists; synchronous staging notifications cannot restart them.
    for (const [id, pending] of confirmations) {
      if (!candidates.has(id) && pending.settled) {
        confirmations.delete(id);
      }
    }
    if (recovering) {
      recoverAgain = true;
      return;
    }
    if (recoveryFailed) {
      return;
    }
    recovering = true;
    void (async () => {
      while (active()) {
        const batch = [...candidates.values()]
          .filter(
            (candidate) =>
              !checkedCandidates.has(candidate.backendItemId) && !confirmations.has(candidate.backendItemId)
          )
          .slice(0, 64);
        if (batch.length === 0) {
          return;
        }
        for (const candidate of batch) {
          checkedCandidates.add(candidate.backendItemId);
        }
        for (let attempt = 0; attempt < 3 && active(); attempt += 1) {
          try {
            const items = await backend.readItemsById(
              batch.map((candidate) => candidate.backendItemId),
              AbortSignal.any([signal, recovery.signal, AbortSignal.timeout(10_000)])
            );
            if (!active()) {
              return;
            }
            for (const item of items) {
              const candidate = candidates.get(item.id);
              // Queue history is transient; backend retry ancestry survives reload and completed-history clearing.
              if (
                candidate &&
                item.retriedFromItemId !== null &&
                item.retriedFromItemId !== undefined &&
                item.destination === 'canvas' &&
                parseQueueItemOrigin(item.origin) === candidate.queueItemId &&
                parseQueueItemOriginProjectId(item.origin) === candidate.projectId
              ) {
                confirm(item.id, `run:${JSON.stringify([candidate.projectId, candidate.queueItemId])}`);
              }
            }
            break;
          } catch {
            if (!active()) {
              return;
            }
            if (attempt === 2) {
              recoveryFailed = true;
              reportError(
                'Staged Canvas results could not be checked for delivery recovery. Reload the editor to retry.'
              );
              return;
            } else {
              await sleep(1_000 * 2 ** attempt, AbortSignal.any([signal, recovery.signal]));
            }
          }
        }
      }
    })()
      .catch(() => undefined)
      .finally(() => {
        recovering = false;
        if (recoverAgain) {
          recoverAgain = false;
          recover([...candidates.values()]);
        }
      });
  };

  const dispose = (): void => {
    disposed = true;
    recoverAgain = false;
    signal.removeEventListener('abort', dispose);
    recovery.abort();
    for (const pending of confirmations.values()) {
      clearTimeout(pending.timer);
    }
    confirmations.clear();
    ready.clear();
    renewalQueue.clear();
    renewing.clear();
    candidates.clear();
    checkedCandidates.clear();
  };
  signal.addEventListener('abort', dispose, { once: true });
  if (signal.aborted) {
    dispose();
  }

  return {
    confirm,
    renewOwned,
    routeOrdinaryResults,
    isStaged: (itemId: number) => confirmations.has(itemId) || candidates.has(itemId),
    recover,
    dispose,
  };
};
