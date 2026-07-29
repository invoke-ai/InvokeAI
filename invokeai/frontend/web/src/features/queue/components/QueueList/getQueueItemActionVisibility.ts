import type { S } from 'services/api/types';

const isChildQueueItem = (item: S['SessionQueueItem']): boolean =>
  item.parent_item_id !== null && item.parent_item_id !== undefined;

export const getQueueItemActionVisibility = (item: S['SessionQueueItem']) => ({
  canShowCancelQueueItem: true,
  canShowRetryQueueItem: !isChildQueueItem(item),
});
