import type { QueueItemReadModel } from '@features/queue';

import { describe, expect, it } from 'vitest';

import { getQueueItemIdsWithResultImages } from './queueProgressCleanup';

const createItem = (itemId: number, results: Record<string, unknown> = {}): QueueItemReadModel => ({
  batchId: 'batch-1',
  createdAt: '2026-06-27T00:00:00Z',
  id: itemId,
  resultImageNames: Object.keys(results).length > 0 ? ['result.png'] : [],
  sessionId: 'session-1',
  status: 'completed',
  updatedAt: '2026-06-27T00:00:00Z',
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
