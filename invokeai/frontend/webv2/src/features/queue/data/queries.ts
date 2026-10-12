import type {
  QueueBackendPort,
  QueueQueryScope,
  QueueReadModel,
  QueueStatusReadModel,
} from '@features/queue/core/types';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { queryOptions, type QueryClient } from '@tanstack/react-query';

export const QUEUE_RECENT_WINDOW = 50;

export const queueKeys = {
  all: ['queue'] as const,
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

/** Coalesced by QueryClient; only active queue observers refetch. */
export const invalidateQueueReadModels = async (queryClient: QueryClient): Promise<void> => {
  await queryClient.invalidateQueries({ queryKey: queueKeys.all });
};
