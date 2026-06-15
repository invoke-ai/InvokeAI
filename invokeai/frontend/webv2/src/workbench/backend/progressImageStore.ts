import { useSyncExternalStore } from 'react';

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

let latestSnapshot: LatestProgressImageSnapshot | null = null;
const snapshotsByTarget = new Map<string, ProgressImageSnapshot>();
const listeners = new Set<() => void>();

const getTargetKey = ({ itemIndex, queueItemId }: ProgressImageTarget): string => `${queueItemId}:${itemIndex}`;

const isLatestTarget = (target: ProgressImageTarget): boolean =>
  latestSnapshot?.target?.queueItemId === target.queueItemId && latestSnapshot.target.itemIndex === target.itemIndex;

const emit = (): void => {
  for (const listener of listeners) {
    listener();
  }
};

const subscribe = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};

export const progressImageStore = {
  clear(target?: ProgressImageTarget): void {
    if (!target) {
      if (latestSnapshot !== null || snapshotsByTarget.size > 0) {
        latestSnapshot = null;
        snapshotsByTarget.clear();
        emit();
      }

      return;
    }

    const didDeleteTarget = snapshotsByTarget.delete(getTargetKey(target));
    const didClearLatest = isLatestTarget(target);

    if (didClearLatest) {
      latestSnapshot = null;
    }

    if (didDeleteTarget || didClearLatest) {
      emit();
    }
  },
  set(image: ProgressImageSnapshot, target?: ProgressImageTarget): void {
    latestSnapshot = target ? { ...image, target } : image;

    if (target) {
      snapshotsByTarget.set(getTargetKey(target), image);
    }

    emit();
  },
};

export type ProgressImageSink = typeof progressImageStore;

export const useProgressImage = (): ProgressImageSnapshot | null =>
  useSyncExternalStore(subscribe, () => latestSnapshot);

export const useQueueItemProgressImage = (queueItemId: string, itemIndex: number): ProgressImageSnapshot | null =>
  useSyncExternalStore(subscribe, () => snapshotsByTarget.get(getTargetKey({ itemIndex, queueItemId })) ?? null);
