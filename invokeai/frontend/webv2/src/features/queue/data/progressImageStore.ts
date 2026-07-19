import type { QueueProgressImage } from '@features/queue/core/progressImage';

import { createExternalStore, createKeyedTransientStore } from '@platform/state/externalStore';

/**
 * The most recent denoising preview image from `invocation_progress` events,
 * as a b64 data URL. Cleared when the run settles so consumers (the editor's
 * Current Image node, progress surfaces) fall back to the last real output.
 */

export type ProgressImageSnapshot = QueueProgressImage;

export interface ProgressImageTarget {
  queueItemId: string;
  itemIndex: number;
}

export type LatestProgressImageSnapshot = ProgressImageSnapshot & { target?: ProgressImageTarget };

const latestSnapshotStore = createExternalStore<{ latestSnapshot: LatestProgressImageSnapshot | null }>({
  latestSnapshot: null,
});
const snapshotsByTarget = createKeyedTransientStore<string, ProgressImageSnapshot>();

const getTargetKey = ({ itemIndex, queueItemId }: ProgressImageTarget): string => `${queueItemId}:${itemIndex}`;

const isLatestTarget = (target: ProgressImageTarget): boolean =>
  latestSnapshotStore.getSnapshot().latestSnapshot?.target?.queueItemId === target.queueItemId &&
  latestSnapshotStore.getSnapshot().latestSnapshot?.target?.itemIndex === target.itemIndex;

export const progressImageStore = {
  clear(target?: ProgressImageTarget): void {
    if (!target) {
      latestSnapshotStore.patchSnapshot({ latestSnapshot: null });
      snapshotsByTarget.clear();

      return;
    }

    const targetKey = getTargetKey(target);
    const didClearLatest = isLatestTarget(target);

    snapshotsByTarget.delete(targetKey);

    if (didClearLatest) {
      latestSnapshotStore.patchSnapshot({ latestSnapshot: null });
    }
  },
  set(image: ProgressImageSnapshot, target?: ProgressImageTarget): void {
    latestSnapshotStore.patchSnapshot({ latestSnapshot: target ? { ...image, target } : image });

    if (target) {
      snapshotsByTarget.set(getTargetKey(target), image);
    }
  },
};

export type ProgressImageSink = typeof progressImageStore;

export const useProgressImage = (): LatestProgressImageSnapshot | null =>
  latestSnapshotStore.useSelector((snapshot) => snapshot.latestSnapshot);

export const useQueueItemProgressImage = (queueItemId: string, itemIndex: number): ProgressImageSnapshot | null =>
  snapshotsByTarget.useValue(getTargetKey({ itemIndex, queueItemId })) ?? null;
