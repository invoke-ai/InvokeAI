import type { QueueQueryScope } from '@features/queue/core/types';
import type {
  QueueAndProcessorStatusDTO,
  QueueItemIdsResultDTO,
  QueueServerItemDTO,
} from '@features/queue/data/serverTypes';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
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

export const getQueueStatus = (
  scope: QueueQueryScope = {},
  signal?: AbortSignal
): Promise<QueueAndProcessorStatusDTO> =>
  apiFetchJson<QueueAndProcessorStatusDTO>(
    buildQueueUrl(`status${buildQueryString({ origin_prefix: scope.originPrefix })}`),
    { signal }
  );

export const getCurrentQueueItem = (
  scope: QueueQueryScope = {},
  signal?: AbortSignal
): Promise<QueueServerItemDTO | null> =>
  apiFetchJson<QueueServerItemDTO | null>(
    buildQueueUrl(`current${buildQueryString({ origin_prefix: scope.originPrefix })}`),
    { signal }
  );

export const getNextQueueItem = (
  scope: QueueQueryScope = {},
  signal?: AbortSignal
): Promise<QueueServerItemDTO | null> =>
  apiFetchJson<QueueServerItemDTO | null>(
    buildQueueUrl(`next${buildQueryString({ origin_prefix: scope.originPrefix })}`),
    { signal }
  );

export const getQueueItem = (itemId: number, signal?: AbortSignal): Promise<QueueServerItemDTO> =>
  apiFetchJson<QueueServerItemDTO>(buildQueueUrl(`i/${itemId}`), { signal });

export const listAllQueueItems = (signal?: AbortSignal): Promise<QueueServerItemDTO[]> =>
  apiFetchJson<QueueServerItemDTO[]>(buildQueueUrl('list_all'), { signal });

export const getQueueItemIds = (
  orderDir: 'asc' | 'desc' = 'desc',
  scope: QueueQueryScope = {},
  signal?: AbortSignal
): Promise<QueueItemIdsResultDTO> =>
  apiFetchJson<QueueItemIdsResultDTO>(
    buildQueueUrl(
      `item_ids${buildQueryString({ order_dir: orderDir.toUpperCase(), origin_prefix: scope.originPrefix })}`
    ),
    { signal }
  );

export const getQueueItemsByIds = (itemIds: number[], signal?: AbortSignal): Promise<QueueServerItemDTO[]> => {
  if (itemIds.length === 0) {
    return Promise.resolve([]);
  }

  return apiFetchJson<QueueServerItemDTO[]>(buildQueueUrl('items_by_ids'), {
    body: JSON.stringify({ item_ids: itemIds }),
    method: 'POST',
    signal,
  });
};

export const clearQueue = (signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('clear'), { method: 'PUT', signal });

export const pruneQueue = (signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('prune'), { method: 'PUT', signal });

export const deleteQueueItem = (itemId: number, signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl(`i/${itemId}`), { method: 'DELETE', signal });

export const deleteQueueItems = async (itemIds: number[], signal?: AbortSignal): Promise<void> => {
  await Promise.all(itemIds.map((itemId) => deleteQueueItem(itemId, signal)));
};

export const cancelQueueItems = async (itemIds: number[], signal?: AbortSignal): Promise<void> => {
  await Promise.all(itemIds.map((itemId) => cancelQueueItem(itemId, signal)));
};

export const clearFailedQueueItems = async (scope: QueueQueryScope = {}): Promise<void> => {
  const owner = captureAccountScope();
  const idsResult = await getQueueItemIds('desc', scope, owner.signal);

  assertAccountScopeCurrent(owner);
  const items = await getQueueItemsByIds(idsResult.item_ids, owner.signal);

  assertAccountScopeCurrent(owner);
  const failedItemIds = items.filter((item) => item.status === 'failed').map((item) => item.item_id);

  await deleteQueueItems(failedItemIds, owner.signal);
  assertAccountScopeCurrent(owner);
};

export const clearScopedQueue = async (scope: QueueQueryScope = {}): Promise<void> => {
  const owner = captureAccountScope();

  if (!scope.originPrefix) {
    await clearQueue(owner.signal);
    assertAccountScopeCurrent(owner);
    return;
  }

  const idsResult = await getQueueItemIds('desc', scope, owner.signal);

  assertAccountScopeCurrent(owner);
  await deleteQueueItems(idsResult.item_ids, owner.signal);
  assertAccountScopeCurrent(owner);
};

export const cancelAllExceptCurrent = (signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('cancel_all_except_current'), { method: 'PUT', signal });

export const cancelQueueItem = (itemId: number, signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl(`i/${itemId}/cancel`), { method: 'PUT', signal });

export const cancelCurrentQueueItem = async (): Promise<void> => {
  const owner = captureAccountScope();
  const current = await getCurrentQueueItem({}, owner.signal);

  assertAccountScopeCurrent(owner);
  if (current) {
    await cancelQueueItem(current.item_id, owner.signal);
    assertAccountScopeCurrent(owner);
  }
};

export const cancelScopedQueueItems = async (
  scope: QueueQueryScope = {},
  currentItemId?: number | null
): Promise<void> => {
  const owner = captureAccountScope();
  const idsResult = await getQueueItemIds('desc', scope, owner.signal);

  assertAccountScopeCurrent(owner);
  const items = await getQueueItemsByIds(idsResult.item_ids, owner.signal);

  assertAccountScopeCurrent(owner);
  const cancellableItemIds = items
    .filter(
      (item) =>
        item.user_id !== 'redacted' &&
        item.item_id !== currentItemId &&
        (item.status === 'pending' || item.status === 'in_progress')
    )
    .map((item) => item.item_id);

  await cancelQueueItems(cancellableItemIds, owner.signal);
  assertAccountScopeCurrent(owner);
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
