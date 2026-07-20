import type { QueueQueryScope } from '@features/queue/core/types';
import type {
  QueueAndProcessorStatusDTO,
  QueueItemIdsResultDTO,
  QueueServerItemDTO,
} from '@features/queue/data/serverTypes';

import { apiFetchJson } from '@platform/transport/http';

const QUEUE_ID = 'default';
const buildQueueUrl = (path = ''): string => `/api/v1/queue/${QUEUE_ID}/${path}`;

const buildQueryString = (params: Record<string, string | undefined>): string => {
  const searchParams = new URLSearchParams();

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      searchParams.set(key, value);
    }
  }

  const queryString = searchParams.toString();

  return queryString ? `?${queryString}` : '';
};

export const getQueueStatus = (scope: QueueQueryScope = {}): Promise<QueueAndProcessorStatusDTO> =>
  apiFetchJson<QueueAndProcessorStatusDTO>(
    buildQueueUrl(`status${buildQueryString({ origin_prefix: scope.originPrefix })}`)
  );

export const getCurrentQueueItem = (scope: QueueQueryScope = {}): Promise<QueueServerItemDTO | null> =>
  apiFetchJson<QueueServerItemDTO | null>(
    buildQueueUrl(`current${buildQueryString({ origin_prefix: scope.originPrefix })}`)
  );

export const getNextQueueItem = (scope: QueueQueryScope = {}): Promise<QueueServerItemDTO | null> =>
  apiFetchJson<QueueServerItemDTO | null>(
    buildQueueUrl(`next${buildQueryString({ origin_prefix: scope.originPrefix })}`)
  );

export const getQueueItem = (itemId: number): Promise<QueueServerItemDTO> =>
  apiFetchJson<QueueServerItemDTO>(buildQueueUrl(`i/${itemId}`));

export const listAllQueueItems = (): Promise<QueueServerItemDTO[]> =>
  apiFetchJson<QueueServerItemDTO[]>(buildQueueUrl('list_all'));

export const getQueueItemIds = (
  orderDir: 'asc' | 'desc' = 'desc',
  scope: QueueQueryScope = {}
): Promise<QueueItemIdsResultDTO> =>
  apiFetchJson<QueueItemIdsResultDTO>(
    buildQueueUrl(
      `item_ids${buildQueryString({ order_dir: orderDir.toUpperCase(), origin_prefix: scope.originPrefix })}`
    )
  );

export const getQueueItemsByIds = (itemIds: number[]): Promise<QueueServerItemDTO[]> => {
  if (itemIds.length === 0) {
    return Promise.resolve([]);
  }

  return apiFetchJson<QueueServerItemDTO[]>(buildQueueUrl('items_by_ids'), {
    body: JSON.stringify({ item_ids: itemIds }),
    method: 'POST',
  });
};

export const clearQueue = (): Promise<unknown> => apiFetchJson(buildQueueUrl('clear'), { method: 'PUT' });

export const pruneQueue = (): Promise<unknown> => apiFetchJson(buildQueueUrl('prune'), { method: 'PUT' });

export const deleteQueueItem = (itemId: number): Promise<unknown> =>
  apiFetchJson(buildQueueUrl(`i/${itemId}`), { method: 'DELETE' });

export const deleteQueueItems = async (itemIds: number[]): Promise<void> => {
  await Promise.all(itemIds.map(deleteQueueItem));
};

export const cancelQueueItems = async (itemIds: number[]): Promise<void> => {
  await Promise.all(itemIds.map(cancelQueueItem));
};

export const clearFailedQueueItems = async (scope: QueueQueryScope = {}): Promise<void> => {
  const idsResult = await getQueueItemIds('desc', scope);
  const items = await getQueueItemsByIds(idsResult.item_ids);
  const failedItemIds = items.filter((item) => item.status === 'failed').map((item) => item.item_id);

  await deleteQueueItems(failedItemIds);
};

export const clearScopedQueue = async (scope: QueueQueryScope = {}): Promise<void> => {
  if (!scope.originPrefix) {
    await clearQueue();
    return;
  }

  const idsResult = await getQueueItemIds('desc', scope);
  await deleteQueueItems(idsResult.item_ids);
};

export const cancelAllExceptCurrent = (): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('cancel_all_except_current'), { method: 'PUT' });

export const cancelQueueItem = (itemId: number): Promise<unknown> =>
  apiFetchJson(buildQueueUrl(`i/${itemId}/cancel`), { method: 'PUT' });

export const cancelCurrentQueueItem = async (): Promise<void> => {
  const current = await getCurrentQueueItem();

  if (current) {
    await cancelQueueItem(current.item_id);
  }
};

export const cancelScopedQueueItems = async (
  scope: QueueQueryScope = {},
  currentItemId?: number | null
): Promise<void> => {
  const idsResult = await getQueueItemIds('desc', scope);
  const items = await getQueueItemsByIds(idsResult.item_ids);
  const cancellableItemIds = items
    .filter((item) => item.item_id !== currentItemId && (item.status === 'pending' || item.status === 'in_progress'))
    .map((item) => item.item_id);

  await cancelQueueItems(cancellableItemIds);
};

export const cancelByBatchIds = (batchIds: string[]): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('cancel_by_batch_ids'), {
    body: JSON.stringify({ batch_ids: batchIds }),
    method: 'PUT',
  });

export const retryItemsById = (itemIds: number[]): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('retry_items_by_id'), {
    body: JSON.stringify(itemIds),
    method: 'PUT',
  });

export const pauseQueueProcessor = (): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('processor/pause'), { method: 'PUT' });

export const resumeQueueProcessor = (): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('processor/resume'), { method: 'PUT' });
