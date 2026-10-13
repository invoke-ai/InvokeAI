import { afterEach, describe, expect, it, vi } from 'vitest';

const server = vi.hoisted(() => ({ claimQueueItemDelivery: vi.fn(), retryItemsById: vi.fn() }));
vi.mock('./serverApi', async (importOriginal) => ({ ...(await importOriginal<object>()), ...server }));

import { queueBackend } from './httpRealtimeQueueBackend';

afterEach(() => {
  vi.clearAllMocks();
});

describe('queueBackend retries', () => {
  it("hands the tab's own retry answer to retry listeners, so adoption does not hang on the socket", async () => {
    const pairs = [{ item_id: 99, source_item_id: 41 }];
    server.retryItemsById.mockResolvedValue({ queue_id: 'default', retried_item_ids: [41], retried_items: pairs });
    const onRetried = vi.fn();
    const onOther = vi.fn();
    const detach = queueBackend.on('queue_items_retried', onRetried);
    const detachOther = queueBackend.on('queue_item_status_changed', onOther);

    await expect(queueBackend.retryItems([41])).resolves.toEqual([41]);
    expect(onRetried).toHaveBeenCalledWith(
      expect.objectContaining({ queue_id: 'default', retried_items: pairs, user_ids: [] })
    );
    expect(onOther).not.toHaveBeenCalled();

    detach();
    detachOther();
    await queueBackend.retryItems([41]);
    expect(onRetried).toHaveBeenCalledTimes(1);
  });

  it('announces nothing for a server that does not pair retries', async () => {
    server.retryItemsById.mockResolvedValue({ queue_id: 'default', retried_item_ids: [41] });
    const onRetried = vi.fn();
    const detach = queueBackend.on('queue_items_retried', onRetried);

    await queueBackend.retryItems([41]);

    expect(onRetried).not.toHaveBeenCalled();
    detach();
  });
});

describe('queueBackend delivery claims', () => {
  it("passes the server's answer on, with how long a held claim keeps the item", async () => {
    server.claimQueueItemDelivery
      .mockResolvedValueOnce({ retry_after_ms: null, status: 'claimed' })
      .mockResolvedValueOnce({ retry_after_ms: 1_500, status: 'held' })
      .mockResolvedValueOnce({ retry_after_ms: null, status: 'held' })
      .mockResolvedValueOnce({ status: 'delivered' });

    await expect(queueBackend.claimItemDelivery(99, 'claim', 'tab:a')).resolves.toEqual({ kind: 'claimed' });
    await expect(queueBackend.claimItemDelivery(99, 'claim', 'tab:b')).resolves.toEqual({
      kind: 'held',
      retryInMs: 1_500,
    });
    // A held answer without a time is asked again at once rather than never.
    await expect(queueBackend.claimItemDelivery(99, 'claim', 'tab:b')).resolves.toEqual({ kind: 'held', retryInMs: 1 });
    await expect(queueBackend.claimItemDelivery(99, 'confirm', 'tab:a')).resolves.toEqual({ kind: 'delivered' });
    expect(server.claimQueueItemDelivery).toHaveBeenCalledWith(99, 'confirm', 'tab:a');
  });
});
