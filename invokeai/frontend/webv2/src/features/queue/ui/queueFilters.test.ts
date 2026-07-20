import type { QueueCounts } from '@features/queue/core/types';

import { describe, expect, it } from 'vitest';

import { getFilterCount, matchesFilter } from './queueFilters';

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
  it('uses personal counts for active work and global counts for history', () => {
    expect(getFilterCount(counts, 'active')).toBe(3);
    expect(getFilterCount(counts, 'all')).toBe(19);
    expect(getFilterCount(counts, 'completed')).toBe(2);
  });

  it('treats waiting workflow items as active', () => {
    expect(matchesFilter('waiting', 'active')).toBe(true);
  });

  it('falls back to global active counts in single-user responses', () => {
    expect(getFilterCount({ ...counts, userInProgress: null, userPending: null }, 'active')).toBe(12);
  });
});
