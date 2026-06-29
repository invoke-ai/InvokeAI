import { itemProgressStore } from '@workbench/backend/itemProgressStore';

import type { QueueServerItem } from './queueServerApi';

import { getResultImageName } from './fieldValues';

export const getQueueItemIdsWithResultImages = (items: QueueServerItem[]): number[] => {
  const itemIds: number[] = [];

  for (const item of items) {
    if (getResultImageName(item)) {
      itemIds.push(item.item_id);
    }
  }

  return itemIds;
};

export const clearProgressForQueueItemsWithResults = (items: QueueServerItem[]): void => {
  for (const itemId of getQueueItemIdsWithResultImages(items)) {
    itemProgressStore.clear(itemId);
  }
};
