import type {
  QueueBackendPort,
  QueueCounts,
  QueueItemReadModel,
  QueueItemDetailReadModel,
  QueueQueryScope,
  QueueReadModel,
  QueueStatusReadModel,
} from '@features/queue/core/types';

import { isTerminalBackendStatus } from '@features/queue/data/events';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { queryOptions, type QueryClient } from '@tanstack/react-query';

export const QUEUE_RECENT_WINDOW = 50;

export const queueKeys = {
  all: ['queue'] as const,
  itemDetail: (itemId: number) => [...queueKeys.all, 'item-detail', itemId] as const,
  readModel: (scope: QueueQueryScope) => [...queueKeys.all, 'read-model', scope.originPrefix ?? 'all'] as const,
  status: (scope: QueueQueryScope) => [...queueKeys.all, 'status', scope.originPrefix ?? 'all'] as const,
};

/** Counts and processor state alone, for summaries that would otherwise load and hydrate the recent window. */
export const queueStatusOptions = (backend: QueueBackendPort, scope: QueueQueryScope) =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<QueueStatusReadModel> => {
        const status = await backend.readStatus(scope, AbortSignal.any([signal, owner.signal]));

        assertAccountScopeCurrent(owner);

        return status;
      },
      queryKey: queueKeys.status(scope),
      staleTime: 5_000,
    });
  })();

export const queueReadModelOptions = (
  backend: QueueBackendPort,
  scope: QueueQueryScope,
  onRead?: (model: QueueReadModel) => void
) =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<QueueReadModel> => {
        const requestSignal = AbortSignal.any([signal, owner.signal]);
        const [status, current, next, idsResult] = await Promise.all([
          backend.readStatus(scope, requestSignal),
          backend.readCurrent(scope, requestSignal),
          backend.readNext(scope, requestSignal),
          backend.readItemIds('desc', scope, requestSignal, QUEUE_RECENT_WINDOW),
        ]);

        assertAccountScopeCurrent(owner);
        requestSignal.throwIfAborted();
        // A server that predates the id limit answers with every id.
        const items = await backend.readItemsById(idsResult.itemIds.slice(0, QUEUE_RECENT_WINDOW), requestSignal);

        assertAccountScopeCurrent(owner);
        requestSignal.throwIfAborted();
        const model = { current, items, next, scope, status };

        onRead?.(model);

        return model;
      },
      queryKey: queueKeys.readModel(scope),
      staleTime: 5_000,
    });
  })();

/** One item's saved execution data, read when a details view opens; an item's session changes only while it runs. */
export const queueItemDetailOptions = (backend: QueueBackendPort, itemId: number) =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<QueueItemDetailReadModel> => {
        const detail = await backend.readItemDetail(itemId, AbortSignal.any([signal, owner.signal]));

        assertAccountScopeCurrent(owner);

        return detail;
      },
      queryKey: queueKeys.itemDetail(itemId),
      staleTime: 5_000,
    });
  })();

/**
 * Refresh queue reads after queue activity; coalesced by QueryClient, and only active observers refetch. A finished
 * item's execution data no longer changes, so its open details are not read again on every queue event; a details
 * view opened again reads them afresh.
 */
export const invalidateQueueQueries = async (queryClient: QueryClient): Promise<void> => {
  await queryClient.invalidateQueries({
    predicate: ({ queryKey, state }) =>
      queryKey[0] === queueKeys.all[0] &&
      (queryKey[1] !== 'item-detail' ||
        !isTerminalBackendStatus((state.data as QueueItemDetailReadModel | undefined)?.status ?? 'pending')),
  });
};

/** An item's new status as the server reported it in a status event, or as the viewer's own cancel assumes it. */
export interface QueueItemStatusChange {
  item: QueueItemReadModel;
  /**
   * The whole queue's counts after the change, as a status event carries them; null when unknown. Their own-item
   * fields belong to the item's owner, whoever receives the event (an administrator receives every user's).
   */
  queueCounts: QueueCounts | null;
}

const isFinished = (status: QueueItemReadModel['status']): boolean =>
  status === 'completed' || status === 'failed' || status === 'canceled';

/** The order the server runs pending items in: highest priority first, then oldest first. */
export const compareQueuedItems = (left: QueueItemReadModel, right: QueueItemReadModel): number =>
  (right.priority ?? 0) - (left.priority ?? 0) || left.id - right.id;

/**
 * Whether `change` is newer than the reading the model holds. Both carry a status sequence: the higher one wins, so
 * a late event never undoes a read, and a server that started over (a reset queue reuses item ids) is followed once a
 * read has brought its own sequences in. Without both: a finished item never runs again.
 */
const isNewerReading = (known: QueueItemReadModel, change: QueueItemReadModel): boolean =>
  known.statusSequence !== null &&
  known.statusSequence !== undefined &&
  change.statusSequence !== null &&
  change.statusSequence !== undefined
    ? change.statusSequence > known.statusSequence
    : !isFinished(known.status);

const COUNT_KEYS: Record<QueueItemReadModel['status'], keyof QueueCounts> = {
  canceled: 'canceled',
  completed: 'completed',
  failed: 'failed',
  in_progress: 'inProgress',
  pending: 'pending',
  waiting: 'waiting',
};

/** The whole-queue counts an event carries, over the reading's own counts of the viewer's items. */
const withEventCounts = (counts: QueueCounts, event: QueueCounts): QueueCounts => ({
  ...counts,
  canceled: event.canceled,
  completed: event.completed,
  failed: event.failed,
  inProgress: event.inProgress,
  pending: event.pending,
  total: event.total,
  waiting: event.waiting,
});

/**
 * Moves one item between two count buckets, as a change of an item in the counts' scope does; the viewer's own counts
 * wait for the read, since whose item it is is not known here.
 */
const shiftCounts = (
  counts: QueueCounts,
  from: QueueItemReadModel['status'],
  to: QueueItemReadModel['status']
): QueueCounts =>
  from === to
    ? counts
    : {
        ...counts,
        [COUNT_KEYS[from]]: Math.max(0, (counts[COUNT_KEYS[from]] as number) - 1),
        [COUNT_KEYS[to]]: (counts[COUNT_KEYS[to]] as number) + 1,
      };

/**
 * The read model with one item's status change applied at once, so the current and next items and the counts are
 * right before the read that reconciles them lands. A change outside the model's scope, or not newer than what the
 * model holds, changes no item. Counts: an unscoped model takes an event's whole-queue counts, even when it already
 * shows the item so (as after the viewer's own cancel); otherwise a change to an item the model knows moves it between
 * buckets. The viewer's own counts wait for the read: an event's are the item owner's.
 */
export const applyQueueItemStatusChange = (
  model: QueueReadModel,
  { item, queueCounts }: QueueItemStatusChange
): QueueReadModel => {
  const { originPrefix } = model.scope;

  if (originPrefix && !item.origin?.startsWith(originPrefix)) {
    return model;
  }

  const known = [model.current, model.next, ...model.items].find((candidate) => candidate?.id === item.id) ?? null;
  const eventCounts = queueCounts && !originPrefix ? withEventCounts(model.status.queue, queueCounts) : null;

  if (known && (known.status === item.status || !isNewerReading(known, item))) {
    return eventCounts && known.status === item.status
      ? { ...model, status: { ...model.status, queue: eventCounts } }
      : model;
  }

  const updated: QueueItemReadModel = known
    ? {
        ...known,
        completedAt: item.completedAt ?? known.completedAt,
        errorMessage: item.errorMessage ?? known.errorMessage,
        errorType: item.errorType ?? known.errorType,
        startedAt: item.startedAt ?? known.startedAt,
        status: item.status,
        statusSequence: item.statusSequence ?? known.statusSequence,
        updatedAt: item.updatedAt,
      }
    : item;
  const items = model.items.map((candidate) => (candidate.id === item.id ? updated : candidate));
  const others = items.filter((candidate) => candidate.id !== item.id);
  const current =
    item.status === 'in_progress'
      ? (model.current ?? updated)
      : model.current?.id === item.id
        ? (others.filter((candidate) => candidate.status === 'in_progress').sort(compareQueuedItems)[0] ?? null)
        : model.current;
  const counts =
    eventCounts ?? (known ? shiftCounts(model.status.queue, known.status, item.status) : model.status.queue);
  // Until the read names the server's next item, the window's first pending item in run order stands in, but only
  // when the window holds exactly the pending items the counts report: the window is the newest items, and the true
  // next may be older than all of them (a long batch, or a member's own items among everyone's). An item the window
  // does not list turning pending, or counts that disagree with the window, leave the next item unknown until then.
  const pendingInWindow = items.filter((candidate) => candidate.status === 'pending');
  const standIn =
    (known || item.status !== 'pending') && pendingInWindow.length === counts.pending
      ? (pendingInWindow.sort(compareQueuedItems)[0] ?? null)
      : null;
  const next =
    model.next?.id === item.id
      ? item.status === 'pending'
        ? updated
        : standIn
      : model.next === null && item.status === 'pending'
        ? standIn
        : model.next;

  return { ...model, current, items, next, status: { ...model.status, queue: counts } };
};

/**
 * Apply a status change to every loaded read model at once, and to the whole queue's status read, so the current and
 * next items and what runs anywhere are right before the coalesced invalidation's read reconciles them. Only data
 * already in the cache changes.
 */
export const applyQueueItemStatusToReadModels = (queryClient: QueryClient, change: QueueItemStatusChange): void => {
  const previous = readQueueReadModels(queryClient)
    .flatMap(([, model]) => (model ? [model.current, model.next, ...model.items] : []))
    .find((candidate) => candidate?.id === change.item.id)?.status;

  queryClient.setQueriesData<QueueReadModel>({ queryKey: [...queueKeys.all, 'read-model'] }, (model) =>
    model ? applyQueueItemStatusChange(model, change) : model
  );
  queryClient.setQueryData<QueueStatusReadModel>(queueKeys.status({}), (status) =>
    status
      ? {
          ...status,
          queue: change.queueCounts
            ? withEventCounts(status.queue, change.queueCounts)
            : previous
              ? shiftCounts(status.queue, previous, change.item.status)
              : status.queue,
        }
      : status
  );
};

/** Every loaded read model, to put back as they were when a change applied ahead of the server turns out wrong. */
export const readQueueReadModels = (queryClient: QueryClient) =>
  queryClient.getQueriesData<QueueReadModel>({ queryKey: [...queueKeys.all, 'read-model'] });
