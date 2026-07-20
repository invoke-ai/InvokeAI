import type { QueueItemReadModel } from '@features/queue/core/types';

import { getResultImageName } from '@features/queue/core/generationMeta';
import { itemProgressStore } from '@features/queue/data/itemProgressStore';

export const getQueueItemIdsWithResultImages = (items: QueueItemReadModel[]): number[] => {
  const itemIds: number[] = [];

  for (const item of items) {
    if (getResultImageName(item)) {
      itemIds.push(item.id);
    }
  }

  return itemIds;
};

export const clearProgressForQueueItemsWithResults = (items: QueueItemReadModel[]): void => {
  for (const itemId of getQueueItemIdsWithResultImages(items)) {
    itemProgressStore.clear(itemId);
  }
};
