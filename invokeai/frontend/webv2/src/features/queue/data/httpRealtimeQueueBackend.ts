import type { QueueBackendPort } from '@features/queue/core/types';

import { socketHub } from '@platform/transport/socketHub';

import { mapQueueBackendItemDTO, mapQueueItemDTO, mapQueueItemIdsDTO, mapQueueStatusDTO } from './mappers';
import {
  cancelByBatchIds,
  cancelCurrentQueueItem,
  cancelQueueItem,
  cancelQueueItems,
  cancelScopedQueueItems,
  clearFailedQueueItems,
  clearScopedQueue,
  getCurrentQueueItem,
  getNextQueueItem,
  getQueueItem,
  getQueueItemIds,
  getQueueItemsByIds,
  getQueueStatus,
  listAllQueueItems,
  pauseQueueProcessor,
  retryItemsById,
  resumeQueueProcessor,
} from './serverApi';
import { enqueueGenerate, enqueueWorkflow, getResultImages, getResultVideoNames } from './submissionApi';

/** Production adapter for the queue backend port. */
export const queueBackend: QueueBackendPort = {
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
  clearItems: clearScopedQueue,
  emit: socketHub.emit,
  enqueueGenerate,
  enqueueWorkflow,
  getItem: async (itemId) => mapQueueBackendItemDTO(await getQueueItem(itemId)),
  getResultImages,
  getResultVideoNames,
  listItems: async () => (await listAllQueueItems()).map(mapQueueBackendItemDTO),
  on: socketHub.on,
  onConnectionChange: socketHub.onConnectionChange,
  pauseProcessor: async () => {
    await pauseQueueProcessor();
  },
  readCurrent: async (scope, signal) => {
    const item = await getCurrentQueueItem(scope, signal);

    return item ? mapQueueItemDTO(item) : null;
  },
  readItemIds: async (order, scope, signal) => mapQueueItemIdsDTO(await getQueueItemIds(order, scope, signal)),
  readItemsById: async (itemIds, signal) => (await getQueueItemsByIds(itemIds, signal)).map(mapQueueItemDTO),
  readNext: async (scope, signal) => {
    const item = await getNextQueueItem(scope, signal);

    return item ? mapQueueItemDTO(item) : null;
  },
  readStatus: async (scope, signal) => mapQueueStatusDTO(await getQueueStatus(scope, signal)),
  retryItems: retryItemsById,
  resumeProcessor: async () => {
    await resumeQueueProcessor();
  },
};
