import type { QueueCounts } from '@features/queue/core/types';

import { describe, expect, it } from 'vitest';

import { getFilterCount, matchesFilter, QUEUE_FILTERS } from './queueFilters';

const counts: QueueCounts = {
  canceled: 1,
  completed: 2,
  failed: 3,
  inProgress: 5,
  pending: 7,
  queueId: 'default',
  total: 19,
  userInProgress: 1,
  userPending: 2,
  waiting: 1,
};

describe('queue filters', () => {
  it('counts every tab the way its list filters: the same items, waiting ones among the active', () => {
    expect(QUEUE_FILTERS.map(({ id }) => getFilterCount(counts, id))).toEqual([19, 13, 2, 3, 1]);
    expect(matchesFilter('waiting', 'active')).toBe(true);
  });
});
