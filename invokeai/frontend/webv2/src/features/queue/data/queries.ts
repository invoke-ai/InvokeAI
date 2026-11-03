import type {
  QueueBackendPort,
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
