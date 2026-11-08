import type { QueueItemProgressTarget } from '@features/queue/core/types';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Track concurrent slots in start order. Settling slots preserve the final frame until results land; multi-slot
 * surfaces exclude them to avoid transient extra tiles.
 */
export interface ActiveProgressTargetSink {
  clear(target?: QueueItemProgressTarget): void;
  set(target: QueueItemProgressTarget): void;
  settle(target: QueueItemProgressTarget): void;
}

interface ActiveProgressTargetsSnapshot {
  settlingTargets: QueueItemProgressTarget[];
  targets: QueueItemProgressTarget[];
}

const store = createExternalStore<ActiveProgressTargetsSnapshot>({ settlingTargets: [], targets: [] });

const isSameTarget = (left: QueueItemProgressTarget, right: QueueItemProgressTarget): boolean =>
  left.queueItemId === right.queueItemId && left.itemIndex === right.itemIndex;

const includes = (targets: QueueItemProgressTarget[], target: QueueItemProgressTarget): boolean =>
  targets.some((candidate) => isSameTarget(candidate, target));

const without = (targets: QueueItemProgressTarget[], target: QueueItemProgressTarget): QueueItemProgressTarget[] =>
  targets.filter((candidate) => !isSameTarget(candidate, target));

export const activeProgressTargetStore: ActiveProgressTargetSink = {
  clear(target) {
    const { settlingTargets, targets } = store.getSnapshot();

    if (!target) {
      if (targets.length > 0 || settlingTargets.length > 0) {
        store.patchSnapshot({ settlingTargets: [], targets: [] });
      }

      return;
    }

    const remaining = without(targets, target);
    const remainingSettling = without(settlingTargets, target);

    store.patchSnapshot({
      ...(remaining.length !== targets.length ? { targets: remaining } : {}),
      ...(remainingSettling.length !== settlingTargets.length ? { settlingTargets: remainingSettling } : {}),
    });
  },
  set(target) {
    const { settlingTargets, targets } = store.getSnapshot();

    // Progress frames arrive many times a second per session; re-appending an
    // already-tracked target would publish a fresh array identity every frame and
    // re-render every consumer.
    if (includes(targets, target)) {
      return;
    }

    store.patchSnapshot({
      targets: [...targets, target],
      // A settled slot reporting progress again is running again.
      ...(includes(settlingTargets, target) ? { settlingTargets: without(settlingTargets, target) } : {}),
    });
  },
  settle(target) {
    const { settlingTargets, targets } = store.getSnapshot();

    // A slot that never reported progress was never followed; nothing to keep up.
    if (!includes(targets, target)) {
      return;
    }

    store.patchSnapshot({
      settlingTargets: includes(settlingTargets, target) ? settlingTargets : [...settlingTargets, target],
      targets: without(targets, target),
    });
  },
};

registerAccountOwnedResource({
  clear: () => activeProgressTargetStore.clear(),
  name: 'queue-active-progress-target',
});

/**
 * Running slots first: a settling slot is only worth following while nothing is
 * running, or a concurrent session's live stream would sit unseen behind a
 * static frame for the whole routing window.
 */
const selectFollowedTargets = ({
  settlingTargets,
  targets,
}: ActiveProgressTargetsSnapshot): QueueItemProgressTarget[] =>
  settlingTargets.length === 0 ? targets : [...targets, ...settlingTargets];

/**
 * Single-slot surfaces follow the oldest active or settling slot until its result lands, avoiding
 * concurrent-session flicker.
 */
export const useActiveProgressTarget = (): QueueItemProgressTarget | null =>
  store.useSelector((snapshot) => selectFollowedTargets(snapshot)[0] ?? null);

/** Every slot currently running, in the order its session started. */
export const useActiveProgressTargets = (): QueueItemProgressTarget[] =>
  store.useSelector((snapshot) => snapshot.targets);

/** Every followable slot — running ones first, then settling ones. */
export const useFollowedProgressTargets = (): QueueItemProgressTarget[] => store.useSelector(selectFollowedTargets);

export const getActiveProgressTargets = (): QueueItemProgressTarget[] => store.getSnapshot().targets;

export const getFollowedProgressTargets = (): QueueItemProgressTarget[] => selectFollowedTargets(store.getSnapshot());
