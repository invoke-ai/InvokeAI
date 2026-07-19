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
import { enqueueGenerate, enqueueWorkflow, getResultImages } from './submissionApi';

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
  listItems: async () => (await listAllQueueItems()).map(mapQueueBackendItemDTO),
  on: socketHub.on,
  onConnectionChange: socketHub.onConnectionChange,
  pauseProcessor: async () => {
    await pauseQueueProcessor();
  },
  readCurrent: async (scope) => {
    const item = await getCurrentQueueItem(scope);

    return item ? mapQueueItemDTO(item) : null;
  },
  readItemIds: async (order, scope) => mapQueueItemIdsDTO(await getQueueItemIds(order, scope)),
  readItemsById: async (itemIds) => (await getQueueItemsByIds(itemIds)).map(mapQueueItemDTO),
  readNext: async (scope) => {
    const item = await getNextQueueItem(scope);

    return item ? mapQueueItemDTO(item) : null;
  },
  readStatus: async (scope) => mapQueueStatusDTO(await getQueueStatus(scope)),
  retryItems: retryItemsById,
  resumeProcessor: async () => {
    await resumeQueueProcessor();
  },
};
