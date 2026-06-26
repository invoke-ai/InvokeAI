import { createExternalStore, createKeyedTransientStore } from '@workbench/externalStore';

/**
 * The most recent denoising preview image from `invocation_progress` events,
 * as a b64 data URL. Cleared when the run settles so consumers (the editor's
 * Current Image node, progress surfaces) fall back to the last real output.
 */

export interface ProgressImageSnapshot {
  dataUrl: string;
  width: number;
  height: number;
}

export interface ProgressImageTarget {
  queueItemId: string;
  itemIndex: number;
}

type LatestProgressImageSnapshot = ProgressImageSnapshot & { target?: ProgressImageTarget };

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

export const useProgressImage = (): ProgressImageSnapshot | null =>
  latestSnapshotStore.useSelector((snapshot) => snapshot.latestSnapshot);

export const useQueueItemProgressImage = (queueItemId: string, itemIndex: number): ProgressImageSnapshot | null =>
  snapshotsByTarget.useValue(getTargetKey({ itemIndex, queueItemId })) ?? null;
