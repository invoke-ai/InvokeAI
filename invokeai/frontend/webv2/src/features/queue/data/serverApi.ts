import type { QueueProgressPreviewPayload, QueueQueryScope } from '@features/queue/core/types';
import type {
  CancelByBatchIdsResultDTO,
  InvocationCacheStatusDTO,
  QueueAndProcessorStatusDTO,
  QueueItemDeliveryResultDTO,
  QueueItemIdsResultDTO,
  QueueServerItemDTO,
  RetryItemsResultDTO,
} from '@features/queue/data/serverTypes';

import { assertAccountScopeCurrent, captureAccountScope, type AccountScope } from '@platform/state/accountLifecycle';
import { apiFetchJson } from '@platform/transport/http';

const QUEUE_ID = 'default';
const QUEUE_MUTATION_CONCURRENCY = 8;
/** The backend's per-request cap on hydrating items by id (`MAX_QUEUE_ITEM_IDS_PER_REQUEST`). */
const QUEUE_ITEM_IDS_PER_REQUEST = 1000;
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

const runConcurrent = async <T>(items: readonly T[], visit: (item: T) => Promise<unknown>): Promise<void> => {
  let nextIndex = 0;
  const worker = async (): Promise<void> => {
    while (nextIndex < items.length) {
      const item = items[nextIndex];
      nextIndex += 1;
      if (item !== undefined) {
        await visit(item);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(QUEUE_MUTATION_CONCURRENCY, items.length) }, worker));
};

export const getQueueStatus = (
  scope: QueueQueryScope = {},
  signal?: AbortSignal
): Promise<QueueAndProcessorStatusDTO> =>
  apiFetchJson<QueueAndProcessorStatusDTO>(
    buildQueueUrl(`status${buildQueryString({ origin_prefix: scope.originPrefix })}`),
    { signal }
  );

/** The latest `invocation_progress` payload of each running item the user owns, with revisions. */
export const getProgressPreviews = (signal?: AbortSignal): Promise<QueueProgressPreviewPayload[]> =>
  apiFetchJson<QueueProgressPreviewPayload[]>(buildQueueUrl('previews'), { signal });

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

/** Every matching id, or with `limit` (at most `QUEUE_ITEM_IDS_PER_REQUEST`) the first ids of that order. */
export const getQueueItemIds = (
  orderDir: 'asc' | 'desc' = 'desc',
  scope: QueueQueryScope = {},
  signal?: AbortSignal,
  limit?: number
): Promise<QueueItemIdsResultDTO> =>
  apiFetchJson<QueueItemIdsResultDTO>(
    buildQueueUrl(
      `item_ids${buildQueryString({
        limit: limit === undefined ? undefined : String(limit),
        order_dir: orderDir.toUpperCase(),
        origin_prefix: scope.originPrefix,
      })}`
    ),
    { signal }
  );

/** Reads `action`'s answer for `itemIds` in id order, one bounded request per page of ids. */
const readQueueItemIdPages = async <T>(
  action: 'item_summaries_by_ids' | 'items_by_ids',
  itemIds: number[],
  signal?: AbortSignal
): Promise<T[]> => {
  const results: T[] = [];

  for (let start = 0; start < itemIds.length; start += QUEUE_ITEM_IDS_PER_REQUEST) {
    const page = await apiFetchJson<T[]>(buildQueueUrl(action), {
      body: JSON.stringify({ item_ids: itemIds.slice(start, start + QUEUE_ITEM_IDS_PER_REQUEST) }),
      method: 'POST',
      signal,
    });
    results.push(...page);
  }

  return results;
};

/** Hydrates items in id order. */
export const getQueueItemsByIds = (itemIds: number[], signal?: AbortSignal): Promise<QueueServerItemDTO[]> =>
  readQueueItemIdPages<QueueServerItemDTO>('items_by_ids', itemIds, signal);

type QueueItemSummaryDTO = Pick<QueueServerItemDTO, 'item_id' | 'parent_item_id' | 'status'>;

/** The roots in `scope` whose status `include` accepts; deleting a root removes its whole workflow-call chain. */
const readScopedRootIds = async (
  scope: QueueQueryScope,
  include: (status: QueueServerItemDTO['status']) => boolean,
  owner: AccountScope
): Promise<number[]> => {
  const idsResult = await getQueueItemIds('desc', scope, owner.signal);

  assertAccountScopeCurrent(owner);
  // Just status and lineage: choosing roots needs no sessions.
  const summaries = await readQueueItemIdPages<QueueItemSummaryDTO>(
    'item_summaries_by_ids',
    idsResult.item_ids,
    owner.signal
  );

  assertAccountScopeCurrent(owner);
  return summaries
    .filter((item) => include(item.status) && (item.parent_item_id ?? null) === null)
    .map((item) => item.item_id);
};

export const clearQueue = (signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('clear'), { method: 'PUT', signal });

export const pruneQueue = (signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('prune'), { method: 'PUT', signal });

export const deleteQueueItem = (itemId: number, signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl(`i/${itemId}`), { method: 'DELETE', signal });

export const deleteQueueItems = async (itemIds: number[], signal?: AbortSignal): Promise<void> => {
  await runConcurrent(itemIds, (itemId) => deleteQueueItem(itemId, signal));
};

export const cancelQueueItems = async (itemIds: number[], signal?: AbortSignal): Promise<void> => {
  await runConcurrent(itemIds, (itemId) => cancelQueueItem(itemId, signal));
};

/**
 * Deletes failed roots with their workflow-call chains. A child that failed under a root that is still active, or was
 * then canceled, stays with its root, so Clear Failed never deletes a running tree.
 */
export const clearFailedQueueItems = async (scope: QueueQueryScope = {}): Promise<void> => {
  const owner = captureAccountScope();

  await deleteQueueItems(await readScopedRootIds(scope, (status) => status === 'failed', owner), owner.signal);
  assertAccountScopeCurrent(owner);
};

const isTerminalStatus = (status: QueueServerItemDTO['status']): boolean =>
  status === 'completed' || status === 'canceled' || status === 'failed';

/**
 * Removes finished history and keeps pending, waiting and running items. Without a scope the server prunes in one
 * sweep; a scoped prune deletes the scope's terminal root items individually, as the clear commands do. Deleting a
 * root removes its whole workflow-call chain, so children are never addressed on their own: a second delete would
 * miss, and a finished child must not take a still-waiting parent with it.
 */
export const clearFinishedQueueItems = async (scope: QueueQueryScope = {}): Promise<void> => {
  const owner = captureAccountScope();

  if (!scope.originPrefix) {
    await pruneQueue(owner.signal);
    assertAccountScopeCurrent(owner);
    return;
  }

  await deleteQueueItems(await readScopedRootIds(scope, isTerminalStatus, owner), owner.signal);
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

export const cancelAllExceptCurrent = (scope: QueueQueryScope = {}, signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl(`cancel_all_except_current${buildQueryString({ origin_prefix: scope.originPrefix })}`), {
    method: 'PUT',
    signal,
  });

export const cancelAll = (scope: QueueQueryScope = {}, signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(buildQueueUrl(`cancel_all${buildQueryString({ origin_prefix: scope.originPrefix })}`), {
    method: 'PUT',
    signal,
  });

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

/** Cancels the scope in one server-side sweep; `keepCurrent` spares whatever is in progress. */
export const cancelScopedQueueItems = async (
  scope: QueueQueryScope = {},
  options: { keepCurrent?: boolean } = {}
): Promise<void> => {
  const owner = captureAccountScope();

  await (options.keepCurrent ? cancelAllExceptCurrent(scope, owner.signal) : cancelAll(scope, owner.signal));
  assertAccountScopeCurrent(owner);
};

export const cancelByBatchIds = async (batchIds: string[]): Promise<CancelByBatchIdsResultDTO> => {
  const owner = captureAccountScope();
  const result = await apiFetchJson<CancelByBatchIdsResultDTO>(buildQueueUrl('cancel_by_batch_ids'), {
    body: JSON.stringify({ batch_ids: batchIds }),
    method: 'PUT',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);

  return result;
};

export const retryItemsById = async (itemIds: number[]): Promise<RetryItemsResultDTO> => {
  const owner = captureAccountScope();
  const result = await apiFetchJson<RetryItemsResultDTO>(buildQueueUrl('retry_items_by_id'), {
    body: JSON.stringify(itemIds),
    method: 'PUT',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);

  return result;
};

export const claimQueueItemDelivery = async (
  itemId: number,
  action: 'claim' | 'confirm' | 'release',
  claimant: string
): Promise<QueueItemDeliveryResultDTO> => {
  const owner = captureAccountScope();
  const result = await apiFetchJson<QueueItemDeliveryResultDTO>(buildQueueUrl(`i/${itemId}/delivery`), {
    body: JSON.stringify({ action, claimant }),
    method: 'POST',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);

  return result;
};

const INVOCATION_CACHE_URL = '/api/v1/app/invocation_cache';

/** Admin-only: the invocation (node output) cache, distinct from the model cache. */
export const getInvocationCacheStatus = (signal?: AbortSignal): Promise<InvocationCacheStatusDTO> =>
  apiFetchJson<InvocationCacheStatusDTO>(`${INVOCATION_CACHE_URL}/status`, { signal });

export const clearInvocationCache = (signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(INVOCATION_CACHE_URL, { method: 'DELETE', signal });

export const setInvocationCacheEnabled = (enabled: boolean, signal?: AbortSignal): Promise<unknown> =>
  apiFetchJson(`${INVOCATION_CACHE_URL}/${enabled ? 'enable' : 'disable'}`, { method: 'PUT', signal });

export const pauseQueueProcessor = (): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('processor/pause'), { method: 'PUT' });

export const resumeQueueProcessor = (): Promise<unknown> =>
  apiFetchJson(buildQueueUrl('processor/resume'), { method: 'PUT' });
