import { mapWithConcurrency } from '@platform/core/concurrency';

export interface QueueReceiptStorePort {
  acknowledgeReceipt(projectId: string, queueItemId: string): Promise<{ kind: 'removed' | 'unavailable' }>;
  listPendingReceipts(): Promise<
    { entries: Array<{ projectId: string; queueItemId: string }>; kind: 'available' } | { kind: 'unavailable' }
  >;
}

export const createQueueReceiptAcknowledgements = ({
  acknowledge,
  isActive,
  store,
}: {
  acknowledge(projectId: string, queueItemId: string): Promise<void>;
  isActive(): boolean;
  store: QueueReceiptStorePort;
}) => {
  let disposed = false;
  let pending: Promise<void> | undefined;
  let requested = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let delayMs = 1_000;
  const active = () => !disposed && isActive();

  const retry = (): void => {
    if (timer !== undefined || !active()) {
      return;
    }
    timer = setTimeout(() => {
      timer = undefined;
      void flush();
    }, delayMs);
    delayMs = Math.min(delayMs * 2, 30_000);
  };

  const drain = async (): Promise<void> => {
    while (active()) {
      requested = false;
      const result = await store.listPendingReceipts();
      if (!active()) {
        return;
      }
      if (result.kind !== 'available') {
        retry();
        return;
      }
      if (result.entries.length === 0) {
        delayMs = 1_000;
        return;
      }
      const results = await mapWithConcurrency(result.entries, 8, async (entry) => {
        if (!active()) {
          return false;
        }
        try {
          await acknowledge(entry.projectId, entry.queueItemId);
          return active() && (await store.acknowledgeReceipt(entry.projectId, entry.queueItemId)).kind === 'removed';
        } catch {
          return false;
        }
      });
      if (results.some((removed) => !removed)) {
        retry();
        return;
      }
    }
  };

  const flush = (): Promise<void> => {
    if (!active()) {
      return Promise.resolve();
    }
    requested = true;
    if (pending) {
      return pending;
    }
    if (timer !== undefined) {
      return Promise.resolve();
    }
    pending = drain()
      .catch(retry)
      .finally(() => {
        pending = undefined;
        if (requested && timer === undefined && active()) {
          void flush();
        }
      });
    return pending;
  };

  return {
    dispose() {
      disposed = true;
      if (timer !== undefined) {
        clearTimeout(timer);
      }
    },
    flush,
  };
};
