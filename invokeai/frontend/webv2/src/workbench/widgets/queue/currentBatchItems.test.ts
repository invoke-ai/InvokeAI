import { describe, expect, it } from 'vitest';

import type { QueueServerItem } from './queueServerApi';

import { getCurrentBatchItems } from './currentBatchItems';

const createItem = (item_id: number, batch_id: string, status: QueueServerItem['status']): QueueServerItem => ({
  batch_id,
  created_at: '2026-06-26T00:00:00Z',
  item_id,
  session_id: `session-${item_id}`,
  status,
  updated_at: '2026-06-26T00:00:00Z',
});

describe('getCurrentBatchItems', () => {
  it('shows the current item and every pending item from the current batch', () => {
    const current = createItem(2, 'batch-a', 'in_progress');
    const items = [
      createItem(4, 'batch-b', 'pending'),
      createItem(3, 'batch-a', 'pending'),
      current,
      createItem(1, 'batch-a', 'completed'),
    ];

    expect(
      getCurrentBatchItems({ current, items, next: createItem(3, 'batch-a', 'pending') }).map((item) => item.item_id)
    ).toEqual([2, 3]);
  });

  it('uses the next item batch when there is no current item', () => {
    const next = createItem(5, 'batch-c', 'pending');
    const items = [createItem(6, 'batch-c', 'pending'), next, createItem(4, 'batch-b', 'pending')];

    expect(getCurrentBatchItems({ current: null, items, next }).map((item) => item.item_id)).toEqual([5, 6]);
  });
});
