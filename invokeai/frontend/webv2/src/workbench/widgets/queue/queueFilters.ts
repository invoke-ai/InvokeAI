import type { BackendQueueItemStatus } from '@workbench/backend/events';

import type { QueueStatusCounts } from './queueServerApi';

/** Status filters for the RECENT list, matching the screenshot's tab row. */
export type QueueFilterId = 'all' | 'active' | 'completed' | 'failed' | 'canceled';

export const QUEUE_FILTERS: { id: QueueFilterId; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'active', label: 'Active' },
  { id: 'completed', label: 'Done' },
  { id: 'failed', label: 'Failed' },
  { id: 'canceled', label: 'Canceled' },
];

export const matchesFilter = (status: BackendQueueItemStatus, filter: QueueFilterId): boolean => {
  if (filter === 'all') {
    return true;
  }

  if (filter === 'active') {
    return status === 'pending' || status === 'in_progress';
  }

  return status === filter;
};

/** Count for a filter, derived from the server-wide status counts (not the windowed list). */
export const getFilterCount = (counts: QueueStatusCounts, filter: QueueFilterId): number => {
  switch (filter) {
    case 'all':
      return counts.total;
    case 'active':
      return counts.pending + counts.in_progress;
    case 'completed':
      return counts.completed;
    case 'failed':
      return counts.failed;
    case 'canceled':
      return counts.canceled;
    default:
      return 0;
  }
};
