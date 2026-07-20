import type { QueueBackendPort, QueueQueryScope, QueueReadModel } from '@features/queue/core/types';

import { queryOptions, type QueryClient } from '@tanstack/react-query';

export const QUEUE_RECENT_WINDOW = 50;

export const queueKeys = {
  all: ['queue'] as const,
  readModel: (scope: QueueQueryScope) => [...queueKeys.all, 'read-model', scope.originPrefix ?? 'all'] as const,
};

export const queueReadModelOptions = (
  backend: QueueBackendPort,
  scope: QueueQueryScope,
  onRead?: (model: QueueReadModel) => void
) =>
  queryOptions({
    queryFn: async (): Promise<QueueReadModel> => {
      const [status, current, next, idsResult] = await Promise.all([
        backend.readStatus(scope),
        backend.readCurrent(scope),
        backend.readNext(scope),
        backend.readItemIds('desc', scope),
      ]);
      const items = await backend.readItemsById(idsResult.itemIds.slice(0, QUEUE_RECENT_WINDOW));

      const model = { current, items, next, scope, status };

      onRead?.(model);

      return model;
    },
    queryKey: queueKeys.readModel(scope),
    staleTime: 5_000,
  });

/** Coalesced by QueryClient; only active queue observers refetch. */
export const invalidateQueueReadModels = async (queryClient: QueryClient): Promise<void> => {
  await queryClient.invalidateQueries({ queryKey: queueKeys.all });
};
