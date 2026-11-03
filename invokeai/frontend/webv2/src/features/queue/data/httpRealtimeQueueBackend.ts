import type { QueueBackendPort } from '@features/queue/core/types';
import type { QueueItemsRetriedEvent } from '@features/queue/data/events';

import { socketHub } from '@platform/transport/socketHub';

import {
  mapQueueBackendItemDTO,
  mapQueueItemDetailDTO,
  mapQueueItemDTO,
  mapQueueItemIdsDTO,
  mapQueueStatusDTO,
} from './mappers';
import {
  cancelByBatchIds,
  cancelCurrentQueueItem,
  claimQueueItemDelivery,
  cancelQueueItem,
  cancelQueueItems,
  cancelScopedQueueItems,
  clearFailedQueueItems,
  clearFinishedQueueItems,
  clearScopedQueue,
  getCurrentQueueItem,
  getNextQueueItem,
  getProgressPreviews,
  getQueueItem,
  getQueueItemIds,
  getQueueItemsByIds,
  getQueueStatus,
  listAllQueueItems,
  pauseQueueProcessor,
  retryItemsById,
  resumeQueueProcessor,
} from './serverApi';
import {
  acknowledgeQueueEnqueue,
  enqueueGenerate,
  enqueueWorkflow,
  getQueueEnqueueReceipt,
  getResultImages,
  getResultVideoNames,
  getResultVideos,
} from './submissionApi';

/**
 * This tab's own retry answers, delivered like the socket's `queue_items_retried` so a rerun is adopted even when
 * the socket misses that event; handlers ignore the pairs they have already seen.
 */
const ownRetryHandlers = new Set<(payload: never) => void>();

/** Production adapter for the queue backend port. */
export const queueBackend: QueueBackendPort = {
  acknowledgeEnqueue: acknowledgeQueueEnqueue,
  cancelBatch: async (batchId) => (await cancelByBatchIds([batchId])).canceled,
  cancelCurrentItem: cancelCurrentQueueItem,
  cancelQueueItems,
  cancelQueueItemsByBatchIds: async (batchIds) => {
    await cancelByBatchIds(batchIds);
  },
  cancelItem: async (itemId) => {
    await cancelQueueItem(itemId);
  },
  cancelScopedItems: cancelScopedQueueItems,
  clearFailedItems: clearFailedQueueItems,
  clearFinishedItems: clearFinishedQueueItems,
  clearItems: clearScopedQueue,
  emit: socketHub.emit,
  enqueueGenerate,
  enqueueWorkflow,
  getEnqueueReceipt: getQueueEnqueueReceipt,
  getItem: async (itemId) => mapQueueBackendItemDTO(await getQueueItem(itemId)),
  getResultImages,
  getResultVideoNames,
  getResultVideos,
  listItems: async () => (await listAllQueueItems()).map(mapQueueBackendItemDTO),
  on: (event, handler) => {
    const detachSocket = socketHub.on(event, handler);
    if (event !== 'queue_items_retried') {
      return detachSocket;
    }
    ownRetryHandlers.add(handler);
    return () => {
      detachSocket();
      ownRetryHandlers.delete(handler);
    };
  },
  onConnectionChange: socketHub.onConnectionChange,
  pauseProcessor: async () => {
    await pauseQueueProcessor();
  },
  readItemDetail: async (itemId, signal) => mapQueueItemDetailDTO(await getQueueItem(itemId, signal)),
  readCurrent: async (scope, signal) => {
    const item = await getCurrentQueueItem(scope, signal);

    return item ? mapQueueItemDTO(item) : null;
  },
  readItemIds: async (order, scope, signal, limit) =>
    mapQueueItemIdsDTO(await getQueueItemIds(order, scope, signal, limit)),
  readItemsById: async (itemIds, signal) => (await getQueueItemsByIds(itemIds, signal)).map(mapQueueItemDTO),
  readProgressPreviews: getProgressPreviews,
  readNext: async (scope, signal) => {
    const item = await getNextQueueItem(scope, signal);

    return item ? mapQueueItemDTO(item) : null;
  },
  readStatus: async (scope, signal) => mapQueueStatusDTO(await getQueueStatus(scope, signal)),
  claimItemDelivery: async (itemId, action, claimant) => {
    const result = await claimQueueItemDelivery(itemId, action, claimant);

    return result.status === 'held'
      ? { kind: 'held', retryInMs: Math.max(1, result.retry_after_ms ?? 0) }
      : { kind: result.status };
  },
  retryItems: async (itemIds) => {
    const result = await retryItemsById(itemIds);
    if (result.retried_items?.length) {
      const event: QueueItemsRetriedEvent = {
        queue_id: result.queue_id,
        retried_item_ids: result.retried_item_ids,
        retried_item_ids_by_user: {},
        retried_items: result.retried_items,
        timestamp: Date.now(),
        user_ids: [],
      };
      for (const handler of ownRetryHandlers) {
        handler(event as never);
      }
    }
    return result.retried_item_ids;
  },
  resumeProcessor: async () => {
    await resumeQueueProcessor();
  },
};
