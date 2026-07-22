import type { QueueItemProgressTarget } from '@features/queue/core/types';

import { createExternalStore } from '@platform/state/externalStore';

export interface ActiveProgressTargetSink {
  clear(target?: QueueItemProgressTarget): void;
  set(target: QueueItemProgressTarget): void;
}

const store = createExternalStore<{ target: QueueItemProgressTarget | null }>({ target: null });

const isSameTarget = (left: QueueItemProgressTarget | null, right: QueueItemProgressTarget): boolean =>
  left?.queueItemId === right.queueItemId && left.itemIndex === right.itemIndex;

export const activeProgressTargetStore: ActiveProgressTargetSink = {
  clear(target) {
    const current = store.getSnapshot().target;

    if (current && (!target || isSameTarget(current, target))) {
      store.patchSnapshot({ target: null });
    }
  },
  set(target) {
    if (!isSameTarget(store.getSnapshot().target, target)) {
      store.patchSnapshot({ target });
    }
  },
};

export const useActiveProgressTarget = (): QueueItemProgressTarget | null =>
  store.useSelector((snapshot) => snapshot.target);
