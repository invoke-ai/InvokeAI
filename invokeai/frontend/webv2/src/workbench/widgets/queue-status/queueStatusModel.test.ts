import { describe, expect, it } from 'vitest';

import { getQueueStatusChip } from './queueStatusModel';

describe('getQueueStatusChip', () => {
  it('is idle with an empty queue', () => {
    expect(getQueueStatusChip({ remaining: 0, total: 0 }, false)).toEqual({
      count: 0,
      labelKey: 'idle',
      tone: 'neutral',
    });
  });

  it('shows the remaining count while running', () => {
    expect(getQueueStatusChip({ remaining: 3, total: 5 }, false)).toEqual({
      count: 3,
      labelKey: 'queued',
      tone: 'running',
    });
  });

  it('flags a paused processor with open work', () => {
    expect(getQueueStatusChip({ remaining: 3, total: 5 }, true)).toEqual({
      count: 3,
      labelKey: 'paused',
      tone: 'paused',
    });
  });
});
