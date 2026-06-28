import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@workbench/backend/http', () => ({
  apiFetchJson: mocks.apiFetchJson,
}));

describe('queueServerApi', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ item_ids: [], total_count: 0 });
  });

  it('uses backend SQLiteDirection values for queue item ordering', async () => {
    const { getQueueItemIds } = await import('./queueServerApi');

    await getQueueItemIds('desc');

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/queue/default/item_ids?order_dir=DESC');
  });

  it('deletes only failed queue items when clearing failed items', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url === '/api/v1/queue/default/item_ids?order_dir=DESC') {
        return Promise.resolve({ item_ids: [3, 2, 1], total_count: 3 });
      }

      if (url === '/api/v1/queue/default/items_by_ids') {
        return Promise.resolve([
          { item_id: 3, status: 'completed' },
          { item_id: 2, status: 'failed' },
          { item_id: 1, status: 'pending' },
        ]);
      }

      return Promise.resolve({});
    });

    const { clearFailedQueueItems } = await import('./queueServerApi');

    await clearFailedQueueItems();

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/queue/default/i/2', { method: 'DELETE' });
    expect(mocks.apiFetchJson).not.toHaveBeenCalledWith('/api/v1/queue/default/i/1', { method: 'DELETE' });
    expect(mocks.apiFetchJson).not.toHaveBeenCalledWith('/api/v1/queue/default/i/3', { method: 'DELETE' });
  });
});
