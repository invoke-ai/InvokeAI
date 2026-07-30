import type { QueueItemProgressTarget } from '@features/queue/core/types';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * The slots currently reporting progress.
 *
 * A list rather than a single value because of multi-GPU: with `generation_devices`
 * (default `auto`) the backend runs one session per GPU, so a batch of four across
 * two GPUs has two slots live at once. Holding only the most recent target meant
 * concurrent sessions overwrote each other on every progress frame, and the preview
 * flipped between them several times a second.
 *
 * Order is the order sessions started, which keeps the single-target accessor below
 * stable for as long as that session runs.
 */
export interface ActiveProgressTargetSink {
  clear(target?: QueueItemProgressTarget): void;
  set(target: QueueItemProgressTarget): void;
}

const store = createExternalStore<{ targets: QueueItemProgressTarget[] }>({ targets: [] });

const isSameTarget = (left: QueueItemProgressTarget, right: QueueItemProgressTarget): boolean =>
  left.queueItemId === right.queueItemId && left.itemIndex === right.itemIndex;

export const activeProgressTargetStore: ActiveProgressTargetSink = {
  clear(target) {
    const { targets } = store.getSnapshot();

    if (!target) {
      if (targets.length > 0) {
        store.patchSnapshot({ targets: [] });
      }

      return;
    }

    const remaining = targets.filter((candidate) => !isSameTarget(candidate, target));

    if (remaining.length !== targets.length) {
      store.patchSnapshot({ targets: remaining });
    }
  },
  set(target) {
    const { targets } = store.getSnapshot();

    // Progress frames arrive many times a second per session; re-appending an
    // already-tracked target would publish a fresh array identity every frame and
    // re-render every consumer.
    if (targets.some((candidate) => isSameTarget(candidate, target))) {
      return;
    }

    store.patchSnapshot({ targets: [...targets, target] });
  },
};

registerAccountOwnedResource({
  clear: () => activeProgressTargetStore.clear(),
  name: 'queue-active-progress-target',
});

/**
 * The slot to follow where a surface can only show one.
 *
 * The oldest still-running slot rather than the most recent to report: following the
 * most recent is what made the preview flip between concurrent sessions. Behaviour is
 * identical to the previous single-value store whenever one session runs at a time,
 * which is every single-GPU install.
 */
export const useActiveProgressTarget = (): QueueItemProgressTarget | null =>
  store.useSelector((snapshot) => snapshot.targets[0] ?? null);

/** Every slot reporting progress, in the order its session started. */
export const useActiveProgressTargets = (): QueueItemProgressTarget[] =>
  store.useSelector((snapshot) => snapshot.targets);

export const getActiveProgressTargets = (): QueueItemProgressTarget[] => store.getSnapshot().targets;
