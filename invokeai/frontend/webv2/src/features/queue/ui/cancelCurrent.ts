import type {
  QueueItemReadModel,
  QueueQueryScope,
  QueueReadModel,
  QueueStatusReadModel,
} from '@features/queue/core/types';
import type { QueryClient } from '@tanstack/react-query';

import { CANCEL_ITEM_TIMEOUT_MS } from '@features/queue/data/httpRealtimeQueueBackend';
import {
  applyQueueItemStatusToReadModels,
  compareQueuedItems,
  invalidateQueueQueries,
  queueKeys,
  readQueueReadModels,
} from '@features/queue/data/queries';
import { getQueueReadModelOptions, queueCommands } from '@features/queue/publicApi';
import { noteQueueCancelRequested } from '@features/queue/runtime/cancelRequests';
import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { sleep, withRequestDeadline } from '@platform/transport/http';

import { readQueueReadModel, readWholeQueueStatus } from './queueDataStore';

/**
 * What runs anywhere in the queue, whatever the view's scope: a scoped reading counts only its scope, so another
 * project's or another user's item running would read as nothing. An unscoped reading's own count is the whole
 * queue's; a scoped one needs the whole-queue status read. Null while that is unknown.
 */
export const getInProgressAnywhere = (
  scope: QueueQueryScope,
  model: Pick<QueueReadModel, 'status'> | null | undefined,
  wholeQueueStatus: QueueStatusReadModel | null | undefined
): number | null =>
  scope.originPrefix ? (wholeQueueStatus?.queue.inProgress ?? null) : (model?.status.queue.inProgress ?? null);

/**
 * What cancel-current stops: the item in progress the viewer can manage (the server's current one, else one of theirs
 * the read window shows running beside someone else's), else, only in the gap between two items, the next one waiting.
 * The gap is the processor running (not paused) with nothing in progress anywhere in the queue (`inProgressAnywhere`,
 * unknown counting as something): the hand-off between items, a cached reading that is behind, or a canceled item's
 * worker still finishing. Stopping the next item is then what stops the run from the viewer's side. While the
 * processor is paused, or anything runs that it would not cancel, nothing waiting is stopped. The next item is the
 * server's; the read window stands in for it (the server's belonging to someone else) only when it holds exactly the
 * pending items the counts report, since the window is the newest items and the true next may be older.
 * `skippedItemId` leaves out an item a cancel just found already finished.
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
  /** Another cancel-current is still in flight; this one did nothing. */
  | { kind: 'busy' }
  | { item: QueueItemReadModel; kind: 'canceled' }
  | { error: unknown; kind: 'failed' }
  | { kind: 'nothing' }
  /** The account changed meanwhile; nothing is reported for the one that asked. */
  | { kind: 'superseded' };

/**
 * How long a cancel-current that stopped an item keeps the command busy, at least: a double click, a double tap of the
 * shortcut or a double activation in the palette lands well inside it, while a deliberate second press after it stops
 * the next item. It also waits for the read that reconciles the queue, up to `CANCEL_CURRENT_SETTLE_LIMIT_MS`.
 */
export const CANCEL_CURRENT_SETTLE_MS = 400;
const CANCEL_CURRENT_SETTLE_LIMIT_MS = 2_000;

/**
 * The account whose cancel-current is in flight or settling; one at a time, so a held key, a double click or a
 * double tap stops one item.
 */
let inFlightOwner: AccountScope | null = null;

/**
 * Shows the item canceled in every loaded read model at once, then asks the server; true when the server canceled
 * it. An item that had already finished comes back unchanged: the cancel did nothing. A failed request puts back each
 * read model this write changed, only while the account that made it is still signed in (a sign-out clears the cache
 * first, and the next account must not see this one's queue) and only where nothing changed it since (an event that
 * arrived meanwhile stays).
 */
const cancelItem = async (client: QueryClient, owner: AccountScope, item: QueueItemReadModel): Promise<boolean> => {
  const dropNote = noteQueueCancelRequested({ itemId: item.id, kind: 'item' });
  const before = readQueueReadModels(client);

  applyQueueItemStatusToReadModels(client, {
    item: { ...item, status: 'canceled', statusSequence: null },
    queueCounts: null,
  });
  const written = new Map(readQueueReadModels(client).map(([queryKey, model]) => [JSON.stringify(queryKey), model]));

  try {
    const answer = await queueCommands.cancelItem(item.id);

    // The server's answer, a canceled item or one that had already finished, supersedes the guess.
    applyQueueItemStatusToReadModels(client, { item: answer, queueCounts: null });
    if (answer.status !== 'canceled') {
      dropNote();
    }

    return answer.status === 'canceled';
  } catch (error) {
    dropNote();
    if (isAccountScopeCurrent(owner)) {
      for (const [queryKey, model] of before) {
        const optimistic = written.get(JSON.stringify(queryKey));
        if (optimistic !== model && client.getQueryData(queryKey) === optimistic) {
          client.setQueryData(queryKey, model);
        }
      }
    }
    throw error;
  }
};

/**
 * Cancel-current, for every surface: the top bar, the queue widget, the shortcut and the palette. Picks its target
 * from the loaded read model when it runs, not when a control rendered. When the item it read as current had already
 * finished, it reads the queue afresh and stops what is current now, once. A call while another is in flight, or
 * within the settle window after one stopped an item (`CANCEL_CURRENT_SETTLE_MS`), returns `busy` at once: one
 * mechanism for every entry point, so a double click, a double tap of the shortcut or a double activation in the
 * palette stops one item.
 */
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

    // The reading the target came from was out of date: read afresh, and for a scoped view the whole queue's status
    // too. Bounded, so a stalled read cannot hold the command busy.
    const [fresh, wholeQueueStatus] = await withRequestDeadline(
      (signal) =>
        Promise.all([
          readQueueReadModel(client, scope, signal),
          scope.originPrefix ? readWholeQueueStatus(client, signal) : Promise.resolve(null),
        ]),
      CANCEL_ITEM_TIMEOUT_MS
    );

    if (!isAccountScopeCurrent(owner)) {
      return { kind: 'superseded' };
    }

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
