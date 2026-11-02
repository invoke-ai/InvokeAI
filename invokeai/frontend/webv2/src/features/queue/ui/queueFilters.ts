import type { QueueCounts, QueueItemStatus } from '@features/queue/core/types';

/** Status filters for the RECENT list, matching the screenshot's tab row. */
export type QueueFilterId = 'all' | 'active' | 'completed' | 'failed' | 'canceled';

export const QUEUE_FILTERS: { id: QueueFilterId; labelKey: string }[] = [
  { id: 'all', labelKey: 'common.all' },
  { id: 'active', labelKey: 'common.active' },
  { id: 'completed', labelKey: 'common.done' },
  { id: 'failed', labelKey: 'common.failed' },
  { id: 'canceled', labelKey: 'common.canceled' },
];

export const matchesFilter = (status: QueueItemStatus, filter: QueueFilterId): boolean => {
  if (filter === 'all') {
    return true;
  }

  if (filter === 'active') {
    return status === 'pending' || status === 'in_progress' || status === 'waiting';
  }

  return status === filter;
};

/**
 * Count for a filter, derived from the scope's status counts (not the windowed list). Every tab counts what its list
 * shows, everyone's items alike, so Active is the pending, running and waiting items `matchesFilter` keeps.
 */
export const getFilterCount = (counts: QueueCounts, filter: QueueFilterId): number => {
  switch (filter) {
    case 'all':
      return counts.total;
    case 'active':
      return counts.pending + counts.inProgress + counts.waiting;
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
