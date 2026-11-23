import type {
  QueueItemReadModel,
  QueueQueryScope,
  QueueReadModel,
  QueueStatusReadModel,
} from '@features/queue/core/types';
import type { QueryClient } from '@tanstack/react-query';

import { CANCEL_ITEM_TIMEOUT_MS } from '@features/queue/data/httpRealtimeQueueBackend';
import {
  applyQueueItemStatusGuess,
  applyQueueItemStatusToReadModels,
  compareQueuedItems,
  invalidateQueueQueries,
  queueKeys,
} from '@features/queue/data/queries';
import { getQueueReadModelOptions, queueCommands } from '@features/queue/publicApi';
import { noteQueueCancelRequested } from '@features/queue/runtime/cancelRequests';
import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { sleep, withRequestDeadline } from '@platform/transport/http';

import { readQueueReadModel, readWholeQueueStatus } from './queueDataStore';

/** A scoped reading counts only its scope, so a scoped view needs the whole-queue status read. Null while unknown. */
export const getInProgressAnywhere = (
  scope: QueueQueryScope,
  model: Pick<QueueReadModel, 'status'> | null | undefined,
  wholeQueueStatus: QueueStatusReadModel | null | undefined
): number | null =>
  scope.originPrefix ? (wholeQueueStatus?.queue.inProgress ?? null) : (model?.status.queue.inProgress ?? null);

/**
 * The manageable in-progress item, else the next waiting item only in the gap between items (processor running,
 * nothing in progress anywhere). `skippedItemId` leaves out an item a cancel just found already finished.
 */
export const getCancelCurrentTarget = (
  model: Pick<QueueReadModel, 'current' | 'items' | 'next' | 'status'> | null | undefined,
  {
    canManageItem,
    inProgressAnywhere,
    skippedItemId,
  }: {
    canManageItem: (item: QueueItemReadModel) => boolean;
    inProgressAnywhere: number | null;
    skippedItemId?: number;
  }
): QueueItemReadModel | null => {
  if (!model) {
    return null;
  }

  const inRunOrder = [...model.items].sort(compareQueuedItems);
  const isTarget = (item: QueueItemReadModel | null, status: 'in_progress' | 'pending'): item is QueueItemReadModel =>
    item?.status === status && item.id !== skippedItemId && canManageItem(item);
  const running = [model.current, ...inRunOrder].find((item) => isTarget(item, 'in_progress'));

  if (running) {
    return running;
  }

  const isBetweenItems =
    model.status.processor.isStarted &&
    model.current === null &&
    model.status.queue.inProgress === 0 &&
    inProgressAnywhere === 0;

  if (!isBetweenItems) {
    return null;
  }

  const windowHoldsEveryPending =
    inRunOrder.filter((item) => item.status === 'pending').length === model.status.queue.pending;

  return (
    (isTarget(model.next, 'pending') ? model.next : null) ??
    (windowHoldsEveryPending ? (inRunOrder.find((item) => isTarget(item, 'pending')) ?? null) : null)
  );
};

export type CancelCurrentOutcome =
  | { kind: 'busy' }
  | { item: QueueItemReadModel; kind: 'canceled' }
  | { error: unknown; kind: 'failed' }
  /** The item it read as current had already finished, and what is current now could not be read: nothing stopped. */
  | { item: QueueItemReadModel; kind: 'finished' }
  | { kind: 'nothing' }
  /** The account changed meanwhile; nothing is reported for the one that asked. */
  | { kind: 'superseded' };

/** Long enough to absorb a double click or double tap; it also waits for the reconciling read, up to the limit. */
export const CANCEL_CURRENT_SETTLE_MS = 400;
const CANCEL_CURRENT_SETTLE_LIMIT_MS = 2_000;

/** One at a time, so a held key, a double click or a double tap stops one item. */
let inFlightOwner: AccountScope | null = null;

/** A failed request takes the guess back only while the same account is signed in and no event confirmed the cancel. */
const cancelItem = async (client: QueryClient, owner: AccountScope, item: QueueItemReadModel): Promise<boolean> => {
  const dropNote = noteQueueCancelRequested({ itemId: item.id, kind: 'item' });
  const guess = applyQueueItemStatusGuess(client, {
    item: { ...item, status: 'canceled', statusSequence: null },
    queueCounts: null,
  });

  try {
    const answer = await queueCommands.cancelItem(item.id);

    guess.settle();
    // The server's answer, a canceled item or one that had already finished, supersedes the guess.
    applyQueueItemStatusToReadModels(client, { item: answer, queueCounts: null });
    if (answer.status !== 'canceled') {
      dropNote();
    }

    return answer.status === 'canceled';
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      guess.settle();
    } else if (guess.revert()) {
      return true;
    }
    dropNote();
    throw error;
  }
};

/** Shared by every surface; picks its target when it runs. Calls while busy or settling return `busy` at once. */
export const cancelCurrentQueueItem = async ({
  canManageItem,
  client,
  scope,
}: {
  canManageItem: (item: QueueItemReadModel) => boolean;
  client: QueryClient;
  scope: QueueQueryScope;
}): Promise<CancelCurrentOutcome> => {
  if (inFlightOwner && isAccountScopeCurrent(inFlightOwner)) {
    return { kind: 'busy' };
  }

  const owner = captureAccountScope();
  inFlightOwner = owner;
  let outcome: CancelCurrentOutcome = { kind: 'nothing' };

  try {
    outcome = await cancelCurrent(client, owner, scope, canManageItem);
    return outcome;
  } finally {
    const reconcile = isAccountScopeCurrent(owner) ? invalidateQueueQueries(client) : Promise.resolve();
    const release = () => {
      if (inFlightOwner === owner) {
        inFlightOwner = null;
      }
    };

    if (outcome.kind === 'canceled') {
      void Promise.race([
        Promise.all([sleep(CANCEL_CURRENT_SETTLE_MS), reconcile.catch(() => undefined)]),
        sleep(CANCEL_CURRENT_SETTLE_LIMIT_MS),
      ]).then(release);
    } else {
      release();
    }
  }
};

const cancelCurrent = async (
  client: QueryClient,
  owner: AccountScope,
  scope: QueueQueryScope,
  canManageItem: (item: QueueItemReadModel) => boolean
): Promise<CancelCurrentOutcome> => {
  try {
    const model = client.getQueryData<QueueReadModel>(getQueueReadModelOptions(scope).queryKey);
    const target = getCancelCurrentTarget(model, {
      canManageItem,
      inProgressAnywhere: getInProgressAnywhere(
        scope,
        model,
        client.getQueryData<QueueStatusReadModel>(queueKeys.status({}))
      ),
    });

    if (!target) {
      return { kind: 'nothing' };
    }
    if (await cancelItem(client, owner, target)) {
      return { item: target, kind: 'canceled' };
    }
    if (!isAccountScopeCurrent(owner)) {
      return { kind: 'superseded' };
    }

    // Read afresh, bounded so a stalled read cannot hold the command busy; a failed read means it already finished.
    const reading = await withRequestDeadline(
      (signal) =>
        Promise.all([
          readQueueReadModel(client, scope, signal),
          scope.originPrefix ? readWholeQueueStatus(client, signal) : Promise.resolve(null),
        ]),
      CANCEL_ITEM_TIMEOUT_MS
    ).catch(() => null);

    if (!isAccountScopeCurrent(owner)) {
      return { kind: 'superseded' };
    }
    if (!reading) {
      return { item: target, kind: 'finished' };
    }

    const [fresh, wholeQueueStatus] = reading;

    const retarget = getCancelCurrentTarget(fresh, {
      canManageItem,
      inProgressAnywhere: getInProgressAnywhere(scope, fresh, wholeQueueStatus),
      skippedItemId: target.id,
    });

    return retarget && (await cancelItem(client, owner, retarget))
      ? { item: retarget, kind: 'canceled' }
      : { kind: 'nothing' };
  } catch (error) {
    return isAccountScopeCurrent(owner) ? { error, kind: 'failed' } : { kind: 'superseded' };
  }
};
