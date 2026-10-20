import type { S } from 'services/api/types';

/**
 * Takes the fields common to a full queue item and a list summary: the list rows decide from a
 * summary, the expanded detail view from the same summary it was handed.
 */
type QueueItemActionSubject = Pick<S['SessionQueueItem'], 'parent_item_id'>;

const isChildQueueItem = (item: QueueItemActionSubject): boolean =>
  item.parent_item_id !== null && item.parent_item_id !== undefined;

export const getQueueItemActionVisibility = (item: QueueItemActionSubject) => ({
  canShowCancelQueueItem: true,
  canShowRetryQueueItem: !isChildQueueItem(item),
});
