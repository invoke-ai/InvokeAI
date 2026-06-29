import { describe, expect, it } from 'vitest';

import type { QueueServerItem } from './queueServerApi';

import { getQueueItemIdsWithResultImages } from './queueProgressCleanup';

const createItem = (itemId: number, results: Record<string, unknown> = {}): QueueServerItem => ({
  batch_id: 'batch-1',
  created_at: '2026-06-27T00:00:00Z',
  item_id: itemId,
  session: { results },
  session_id: 'session-1',
  status: 'completed',
  updated_at: '2026-06-27T00:00:00Z',
});

describe('getQueueItemIdsWithResultImages', () => {
  it('returns queue item ids that have final image results', () => {
    expect(
      getQueueItemIdsWithResultImages([
        createItem(1),
        createItem(2, { node: { image: { image_name: 'result.png' } } }),
        createItem(3, { node: { collection: [{ image_name: 'batch-result.png' }] } }),
      ])
    ).toEqual([2, 3]);
  });
});
