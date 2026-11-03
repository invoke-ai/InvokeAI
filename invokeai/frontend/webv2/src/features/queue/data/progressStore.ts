import type { QueueItemProgress } from '@features/queue/core/types';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createKeyedTransientStore } from '@platform/state/externalStore';

/**
 * Store high-frequency progress outside workbench state to avoid global rerenders and autosave; subscribe per
 * local item ID.
 */

export interface QueueItemProgressSink {
  clearAll?(): void;
  set(queueItemId: string, progress: QueueItemProgress): void;
  clear(queueItemId: string): void;
}

const progressByQueueItemId = createKeyedTransientStore<string, QueueItemProgress>();

export const queueItemProgressStore: QueueItemProgressSink = {
  clearAll() {
    progressByQueueItemId.clear();
  },
  clear(queueItemId) {
    progressByQueueItemId.delete(queueItemId);
  },
  set(queueItemId, progress) {
    progressByQueueItemId.set(queueItemId, progress);
  },
};

registerAccountOwnedResource({
  clear: () => queueItemProgressStore.clearAll?.(),
  name: 'queue-local-item-progress',
});

export const useQueueItemProgress = (queueItemId: string): QueueItemProgress | null =>
  progressByQueueItemId.useValue(queueItemId) ?? null;
