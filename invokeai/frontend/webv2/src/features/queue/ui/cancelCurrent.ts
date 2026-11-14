import type { QueueItemReadModel, QueueQueryScope, QueueReadModel } from '@features/queue/core/types';
import type { QueryClient } from '@tanstack/react-query';

import {
  applyQueueItemStatusToReadModels,
  compareQueuedItems,
  invalidateQueueQueries,
  readQueueReadModels,
} from '@features/queue/data/queries';
import { getQueueReadModelOptions, queueCommands } from '@features/queue/publicApi';
import { noteQueueCancelRequested } from '@features/queue/runtime/cancelRequests';
import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';

import { readQueueReadModel } from './queueDataStore';

/**
 * What cancel-current stops: the running item the viewer can manage, else the next one waiting, in the order the
 * server runs them. Nothing runs during the hand-off between two items, when the cached reading is behind, or while a
 * canceled item's worker is still finishing; stopping the next item is then what stops the run from the viewer's side.
 * The server's own current and next come first; the read window stands in when the server's belongs to someone else.
 * `skippedItemId` leaves out an item a cancel just found already finished.
 */
export const getCancelCurrentTarget = (
  model: Pick<QueueReadModel, 'current' | 'items' | 'next'> | null | undefined,
  canManageItem: (item: QueueItemReadModel) => boolean,
  skippedItemId?: number
): QueueItemReadModel | null => {
  if (!model) {
    return null;
  }

  const inRunOrder = [...model.items].sort(compareQueuedItems);
  const isTarget = (item: QueueItemReadModel | null, status: 'in_progress' | 'pending'): item is QueueItemReadModel =>
    item?.status === status && item.id !== skippedItemId && canManageItem(item);

  return (
    [model.current, ...inRunOrder].find((item) => isTarget(item, 'in_progress')) ??
    [model.next, ...inRunOrder].find((item) => isTarget(item, 'pending')) ??
    null
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

/** The account whose cancel-current is in flight; one at a time, so a held key or a double click stops one item. */
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
    const canceled = (await queueCommands.cancelItem(item.id)).status === 'canceled';

    if (!canceled) {
      dropNote();
    }

    return canceled;
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
 * finished, it reads the queue afresh and stops what is current now, once. A call while another is in flight returns
 * `busy` at once.
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

  try {
    const target = getCancelCurrentTarget(
      client.getQueryData<QueueReadModel>(getQueueReadModelOptions(scope).queryKey),
      canManageItem
    );

    if (!target) {
      return { kind: 'nothing' };
    }
    if (await cancelItem(client, owner, target)) {
      return { item: target, kind: 'canceled' };
    }
    if (!isAccountScopeCurrent(owner)) {
      return { kind: 'superseded' };
    }

    // The reading the target came from was out of date.
    const fresh = await readQueueReadModel(client, scope);

    if (!isAccountScopeCurrent(owner)) {
      return { kind: 'superseded' };
    }

    const retarget = getCancelCurrentTarget(fresh, canManageItem, target.id);

    return retarget && (await cancelItem(client, owner, retarget))
      ? { item: retarget, kind: 'canceled' }
      : { kind: 'nothing' };
  } catch (error) {
    return isAccountScopeCurrent(owner) ? { error, kind: 'failed' } : { kind: 'superseded' };
  } finally {
    if (inFlightOwner === owner) {
      inFlightOwner = null;
    }
    if (isAccountScopeCurrent(owner)) {
      void invalidateQueueQueries(client);
    }
  }
};
