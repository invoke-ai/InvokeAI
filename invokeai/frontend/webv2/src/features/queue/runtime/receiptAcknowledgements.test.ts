import { afterEach, describe, expect, it, vi } from 'vitest';

import { createQueueReceiptAcknowledgements } from './receiptAcknowledgements';

afterEach(() => vi.useRealTimers());

describe('queue receipt acknowledgements', () => {
  it('retains failed cleanup and retries without needing an active queue run', async () => {
    vi.useFakeTimers();
    const receipt = { projectId: 'project-1', queueItemId: 'run-1' };
    let pending = [receipt];
    const acknowledge = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(undefined);
    const remove = vi.fn(() => {
      pending = [];
      return Promise.resolve({ kind: 'removed' as const });
    });
    const worker = createQueueReceiptAcknowledgements({
      acknowledge,
      isActive: () => true,
      store: {
        acknowledgeReceipt: remove,
        listPendingReceipts: () => Promise.resolve({ kind: 'available', entries: pending }),
      },
    });

    await worker.flush();
    expect(remove).not.toHaveBeenCalled();
    await worker.flush();
    await worker.flush();
    expect(acknowledge).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(acknowledge).toHaveBeenCalledTimes(2);
    expect(remove).toHaveBeenCalledWith('project-1', 'run-1');
    worker.dispose();
  });

  it('does not mutate storage after its owner is disposed during acknowledgement', async () => {
    let finish!: () => void;
    const acknowledge = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        })
    );
    const remove = vi.fn();
    const worker = createQueueReceiptAcknowledgements({
      acknowledge,
      isActive: () => true,
      store: {
        acknowledgeReceipt: remove,
        listPendingReceipts: () =>
          Promise.resolve({ kind: 'available', entries: [{ projectId: 'p', queueItemId: 'q' }] }),
      },
    });
    const flushing = worker.flush();
    await vi.waitFor(() => expect(acknowledge).toHaveBeenCalledOnce());
    worker.dispose();
    finish();
    await flushing;
    expect(remove).not.toHaveBeenCalled();
  });
});
