import type { ApiTagDescription } from 'services/api';
import { LIST_ALL_TAG, LIST_TAG } from 'services/api';

/**
 * The queue caches that every queue-changing socket event must refetch. Shared by the
 * queue_item_status_changed (owner and non-owner), queue_cleared, batch_enqueued, and bulk
 * retried/canceled handlers: each spreads this base and appends its event-specific tags
 * (id-scoped BatchStatus / QueueCountsByDestination, per-item SessionQueueItem ids, etc.).
 * Keeping the shared core in one place prevents the handlers drifting apart when a new queue
 * cache is added — one class of recipient silently missing an invalidation is a stale-UI bug.
 */
export const QUEUE_CHANGED_TAGS: readonly ApiTagDescription[] = [
  'SessionQueueStatus',
  'CurrentSessionQueueItem',
  'NextSessionQueueItem',
  'SessionQueueItemIdList',
  { type: 'SessionQueueItem', id: LIST_TAG },
  { type: 'SessionQueueItem', id: LIST_ALL_TAG },
];
