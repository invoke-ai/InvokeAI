import { describe, expect, it } from 'vitest';

import type { QueueItem } from './historyTypes';

import {
  getQueueItemSnapshotBatchCount,
  getQueueItemSnapshotDimensions,
  getQueueItemSnapshotPositivePrompt,
} from './historySnapshot';

describe('queue history snapshot presentation', () => {
  it('uses safe shell fallbacks for legacy active and history items without presentation metadata', () => {
    const item = {
      id: 'legacy',
      snapshot: { submittedAt: '2026-07-01T00:00:00.000Z' },
      status: 'pending',
    } as unknown as QueueItem;

    expect(getQueueItemSnapshotBatchCount(item)).toBe(1);
    expect(getQueueItemSnapshotDimensions(item, { height: 640, width: 960 })).toEqual({ height: 640, width: 960 });
    expect(getQueueItemSnapshotPositivePrompt(item)).toBe('');
  });
});
