import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';

/**
 * Cancellations (and clears) this tab asked for with a notice of its own ("Cancellation requested"). A run they end
 * reports nothing more, so one cancel gives one notice. A request covers only runs submitted before it, and lapses
 * after a while: a run canceled long after, by someone else, still reports. Session-only and per account.
 */
export type QueueCancelRequest =
  | { kind: 'item'; itemId: number }
  | { kind: 'batch'; batchId: string }
  /**
   * `originPrefix` null: every item the caller may cancel. `exceptItemId`: the running item a cancel-all-but-current
   * left alone.
   */
  | { exceptItemId?: number; kind: 'scope'; originPrefix: string | null };

/** A run the queue runtime asks about: its backend items, batch, origin, and when it was submitted. */
export interface QueueCancelRequestRun {
  backendBatchId?: string;
  backendItemIds: readonly number[];
  origin: string;
  submittedAt: string;
}

/** The runtime's view of the requests: whether this tab already announced the cancel that ended a run. */
export interface QueueCancelRequestPort {
  wasRequestedHere(run: QueueCancelRequestRun): boolean;
}

/** Long enough for a backend to settle a cancel it acknowledged. */
const REQUEST_LIFETIME_MS = 2 * 60_000;

let requests: { at: number; request: QueueCancelRequest }[] = [];

const prune = (now: number): void => {
  requests = requests.filter(({ at }) => now - at < REQUEST_LIFETIME_MS);
};

const covers = (request: QueueCancelRequest, run: QueueCancelRequestRun): boolean => {
  switch (request.kind) {
    case 'item':
      return run.backendItemIds.includes(request.itemId);
    case 'batch':
      return request.batchId === run.backendBatchId;
    case 'scope':
      return (
        (request.originPrefix === null || run.origin.startsWith(request.originPrefix)) &&
        (request.exceptItemId === undefined || !run.backendItemIds.includes(request.exceptItemId))
      );
  }
};

/**
 * Records a request before it is sent, so a cancel event that beats the response is still covered. Returns the way
 * to drop it again when the request fails.
 */
export const noteQueueCancelRequested = (request: QueueCancelRequest): (() => void) => {
  const now = Date.now();
  const entry = { at: now, request };

  prune(now);
  requests.push(entry);

  return () => {
    requests = requests.filter((candidate) => candidate !== entry);
  };
};

export const queueCancelRequests: QueueCancelRequestPort = {
  wasRequestedHere: (run) => {
    prune(Date.now());

    const submittedAt = Date.parse(run.submittedAt);

    return requests.some(
      ({ at, request }) => Number.isFinite(submittedAt) && submittedAt <= at && covers(request, run)
    );
  },
};

registerAccountOwnedResource({
  clear: () => {
    requests = [];
  },
  name: 'queue-cancel-requests',
});
