import type { QueueBackendPort } from '@features/queue/core/types';

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
  on: socketHub.on,
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
  retryItems: async (itemIds) => (await retryItemsById(itemIds)).retried_item_ids,
  resumeProcessor: async () => {
    await resumeQueueProcessor();
  },
};
