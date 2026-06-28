import type { BackendQueueItemStatus } from '@workbench/backend/events';

import { apiFetchJson } from '@workbench/backend/http';

/**
 * Server-wide session-queue REST layer for the Queue widget. This mirrors the
 * legacy v6 queue API surface (`frontend/web/src/services/api/endpoints/queue.ts`)
 * — the same status/list/current/next reads, the `item_ids → items_by_ids`
 * windowed pagination, and the clear/prune/cancel/pause mutations — ported to
 * webv2's `apiFetchJson` idiom. Socket events drive refresh (see
 * `queueDataStore` + `QueueDataRuntime`) in place of RTK Query tag invalidation.
 *
 * Unlike the local in-session queue (`project.queue.items`, which only tracks
 * this client's submissions), this reads the whole backend queue: counts and
 * items from every source and every client.
 */

const QUEUE_ID = 'default';
const buildQueueUrl = (path = ''): string => `/api/v1/queue/${QUEUE_ID}/${path}`;

export interface QueueQueryScope {
  originPrefix?: string;
}

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

/** A single batch substitution recorded for a queue item (prompt / seed / image field). */
export interface QueueNodeFieldValue {
  node_path: string;
  field_name: string;
  value: string | number | { image_name?: string } | null;
}

/**
 * The backend `SessionQueueItem` shape, narrowed to the fields the widget reads.
 * `session` is the heavy execution graph; we only ever touch `session.results`
 * to resolve a result image, so it is loosely typed here.
 */
export interface QueueServerItem {
  item_id: number;
  status: BackendQueueItemStatus;
  batch_id: string;
  origin?: string | null;
  destination?: string | null;
  session_id: string;
  error_type?: string | null;
  error_message?: string | null;
  error_traceback?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  field_values?: QueueNodeFieldValue[] | null;
  retried_from_item_id?: number | null;
  session?: { results?: Record<string, unknown> };
}

export interface QueueStatusCounts {
  queue_id: string;
  item_id?: number | null;
  batch_id?: string | null;
  session_id?: string | null;
  pending: number;
  in_progress: number;
  completed: number;
  failed: number;
  canceled: number;
  total: number;
}

export interface QueueProcessorStatus {
  is_started: boolean;
  is_processing: boolean;
}

export interface QueueAndProcessorStatus {
  queue: QueueStatusCounts;
  processor: QueueProcessorStatus;
}

export interface QueueItemIdsResult {
  item_ids: number[];
  total_count: number;
}

// --- Reads -----------------------------------------------------------------

export const getQueueStatus = (scope: QueueQueryScope = {}): Promise<QueueAndProcessorStatus> =>
  apiFetchJson<QueueAndProcessorStatus>(
    buildQueueUrl(`status${buildQueryString({ origin_prefix: scope.originPrefix })}`)
  );

export const getCurrentQueueItem = (scope: QueueQueryScope = {}): Promise<QueueServerItem | null> =>
  apiFetchJson<QueueServerItem | null>(
    buildQueueUrl(`current${buildQueryString({ origin_prefix: scope.originPrefix })}`)
  );

export const getNextQueueItem = (scope: QueueQueryScope = {}): Promise<QueueServerItem | null> =>
  apiFetchJson<QueueServerItem | null>(buildQueueUrl(`next${buildQueryString({ origin_prefix: scope.originPrefix })}`));

export const getQueueItem = (itemId: number): Promise<QueueServerItem> =>
  apiFetchJson<QueueServerItem>(buildQueueUrl(`i/${itemId}`));

export const getQueueItemIds = (
  orderDir: 'asc' | 'desc' = 'desc',
  scope: QueueQueryScope = {}
): Promise<QueueItemIdsResult> =>
  apiFetchJson<QueueItemIdsResult>(
    buildQueueUrl(
      `item_ids${buildQueryString({ order_dir: orderDir.toUpperCase(), origin_prefix: scope.originPrefix })}`
    )
  );

/** Fetch full items for the given ids, preserving order. Mirrors legacy `items_by_ids`. */
export const getQueueItemsByIds = (itemIds: number[]): Promise<QueueServerItem[]> => {
  if (itemIds.length === 0) {
    return Promise.resolve([]);
  }

  return apiFetchJson<QueueServerItem[]>(buildQueueUrl('items_by_ids'), {
    body: JSON.stringify({ item_ids: itemIds }),
    method: 'POST',
  });
};

// --- Mutations -------------------------------------------------------------

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
