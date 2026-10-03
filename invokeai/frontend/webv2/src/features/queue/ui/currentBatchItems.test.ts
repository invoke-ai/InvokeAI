import type { QueueItemReadModel } from '@features/queue';

import { describe, expect, it } from 'vitest';

import { getCurrentBatchItems } from './currentBatchItems';

const createItem = (id: number, batchId: string, status: QueueItemReadModel['status']): QueueItemReadModel => ({
  batchId,
  createdAt: '2026-06-26T00:00:00Z',
  id,
  resultImageNames: [],
  sessionId: `session-${id}`,
  status,
  updatedAt: '2026-06-26T00:00:00Z',
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
      getCurrentBatchItems({ current, items, next: createItem(3, 'batch-a', 'pending') }).map((item) => item.id)
    ).toEqual([2, 3]);
  });

  it('uses the next item batch when there is no current item', () => {
    const next = createItem(5, 'batch-c', 'pending');
    const items = [createItem(6, 'batch-c', 'pending'), next, createItem(4, 'batch-b', 'pending')];

    expect(getCurrentBatchItems({ current: null, items, next }).map((item) => item.id)).toEqual([5, 6]);
  });

  it('keeps concurrently running items from another batch out of recent history', () => {
    const current = createItem(10, 'batch-local', 'in_progress');
    const remoteRunning = createItem(11, 'batch-remote', 'in_progress');
    const unrelatedPending = createItem(12, 'batch-remote', 'pending');

    expect(
      getCurrentBatchItems({
        current,
        items: [remoteRunning, unrelatedPending],
        next: null,
      }).map((item) => item.id)
    ).toEqual([10, 11]);
  });

  it('shows running items even when the backend has no single current or next item', () => {
    const running = createItem(20, 'batch-remote', 'in_progress');

    expect(getCurrentBatchItems({ current: null, items: [running], next: null }).map((item) => item.id)).toEqual([20]);
  });
});
