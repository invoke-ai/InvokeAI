import type { QueueProgressImage } from '@features/queue/core/progressImage';
import type { QueueItemProgressTarget } from '@features/queue/core/types';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore, createKeyedTransientStore } from '@platform/state/externalStore';

/**
 * Hold bounded bridge frames between batch slots and swap frames until image decode. Retain frames across
 * disconnects while HTTP reconciles the continuing run.
 */

export type ProgressImageSnapshot = QueueProgressImage;

export type ProgressImageTarget = QueueItemProgressTarget;

export type LatestProgressImageSnapshot = ProgressImageSnapshot & { target?: ProgressImageTarget };

/** Held frames are latent-grid JPEG data URLs, a few KB each. */
const HELD_FRAME_LIMIT = 8;

/** Allow time for slow fetch/decode without replaying held frames on later visits. */
export const SWAP_FRAME_TTL_MS = 10_000;

const latestSnapshotStore = createExternalStore<{ latestSnapshot: LatestProgressImageSnapshot | null }>({
  latestSnapshot: null,
});
/**
 * A swap frame is bound to the image names its backend item delivered once
 * routing lands (`bindSwapImages`); until then, and for any other image of the
 * same batch, it must not be painted — item 3's denoise frame over item 1.
 */
interface SwapFrame {
  image: ProgressImageSnapshot;
  imageNames: readonly string[] | null;
}

const snapshotsByTarget = createKeyedTransientStore<string, ProgressImageSnapshot>();
/** Insertion order is recency: `set` re-inserts, so the last entry is the most recently updated slot. */
const targetsByKey = new Map<string, ProgressImageTarget>();
const bridgeFrames = createKeyedTransientStore<string, ProgressImageSnapshot>();
const swapFrames = createKeyedTransientStore<string, SwapFrame>();
const swapExpiryTimers = new Map<string, ReturnType<typeof setTimeout>>();

const getTargetKey = ({ itemIndex, queueItemId }: ProgressImageTarget): string => `${queueItemId}:${itemIndex}`;

const isLatestTarget = (target: ProgressImageTarget): boolean =>
  latestSnapshotStore.getSnapshot().latestSnapshot?.target?.queueItemId === target.queueItemId &&
  latestSnapshotStore.getSnapshot().latestSnapshot?.target?.itemIndex === target.itemIndex;

/** Drop the oldest held entries past the cap. Insertion order is age: `hold` re-inserts. */
const evictOldest = <Value>(
  store: { entries: () => Array<[string, Value]> },
  drop: (queueItemId: string) => void
): void => {
  const entries = store.entries();

  for (let index = 0; index < entries.length - HELD_FRAME_LIMIT; index += 1) {
    const entry = entries[index];

    if (entry) {
      drop(entry[0]);
    }
  }
};

const dropBridge = (queueItemId: string): void => {
  bridgeFrames.delete(queueItemId);
};

const dropSwap = (queueItemId: string): void => {
  const timer = swapExpiryTimers.get(queueItemId);

  if (timer !== undefined) {
    clearTimeout(timer);
    swapExpiryTimers.delete(queueItemId);
  }

  swapFrames.delete(queueItemId);
};

const holdBridge = (queueItemId: string, image: ProgressImageSnapshot): void => {
  dropBridge(queueItemId);
  bridgeFrames.set(queueItemId, image);
  evictOldest(bridgeFrames, dropBridge);
};

const holdSwap = (queueItemId: string, image: ProgressImageSnapshot): void => {
  dropSwap(queueItemId);
  swapFrames.set(queueItemId, { image, imageNames: null });
  swapExpiryTimers.set(
    queueItemId,
    setTimeout(() => dropSwap(queueItemId), SWAP_FRAME_TTL_MS)
  );
  evictOldest(swapFrames, dropSwap);
};

export const progressImageStore = {
  clear(target?: ProgressImageTarget): void {
    if (!target) {
      latestSnapshotStore.patchSnapshot({ latestSnapshot: null });
      snapshotsByTarget.clear();
      targetsByKey.clear();
      bridgeFrames.clear();

      for (const timer of swapExpiryTimers.values()) {
        clearTimeout(timer);
      }

      swapExpiryTimers.clear();
      swapFrames.clear();

      return;
    }

    const targetKey = getTargetKey(target);
    const didClearLatest = isLatestTarget(target);

    snapshotsByTarget.delete(targetKey);
    targetsByKey.delete(targetKey);

    if (didClearLatest) {
      // Fall back to another live slot's frame when one clears; concurrent long-running sessions must not go
      // blank.
      latestSnapshotStore.patchSnapshot({ latestSnapshot: getMostRecentSnapshot() });
    }
  },
  /** Routing landed: these are the images the held swap frame may be painted over. */
  bindSwapImages(queueItemId: string, imageNames: readonly string[]): void {
    const entry = swapFrames.get(queueItemId);

    if (entry) {
      // A workflow may save unrelated images in several boards. One preview
      // bridges only to the latest output selected by result routing while it decodes.
      swapFrames.set(queueItemId, { image: entry.image, imageNames: imageNames.slice(-1) });
    }
  },
  /** Forget a queue item's held frames: its run is gone (detached or canceled). */
  clearHeld(queueItemId: string): void {
    dropBridge(queueItemId);
    dropSwap(queueItemId);
  },
  /** Hold the final frame for bridging and swapping; retain the live slot until result fetch completes. */
  hold(target: ProgressImageTarget): void {
    const image = snapshotsByTarget.get(getTargetKey(target));

    if (!image) {
      return;
    }

    holdBridge(target.queueItemId, image);
    holdSwap(target.queueItemId, image);
  },
  set(image: ProgressImageSnapshot, target?: ProgressImageTarget): void {
    latestSnapshotStore.patchSnapshot({ latestSnapshot: target ? { ...image, target } : image });

    if (target) {
      const targetKey = getTargetKey(target);

      snapshotsByTarget.set(targetKey, image);
      targetsByKey.delete(targetKey);
      targetsByKey.set(targetKey, target);
    }
  },
};

function getMostRecentSnapshot(): LatestProgressImageSnapshot | null {
  for (const [targetKey, target] of [...targetsByKey.entries()].reverse()) {
    const image = snapshotsByTarget.get(targetKey);

    if (image) {
      return { ...image, target };
    }
  }

  return null;
}

registerAccountOwnedResource({
  clear: () => progressImageStore.clear(),
  name: 'queue-progress-images',
});

export type ProgressImageSink = typeof progressImageStore;

/** The finished image has decoded on screen; the swap frame has done its job. */
export const consumeQueueItemSwapProgressImage = (queueItemId: string): void => {
  dropSwap(queueItemId);
};

export const getLatestProgressImage = (): LatestProgressImageSnapshot | null =>
  latestSnapshotStore.getSnapshot().latestSnapshot;

export const getQueueItemBridgeProgressImage = (queueItemId: string): ProgressImageSnapshot | null =>
  bridgeFrames.get(queueItemId) ?? null;

const selectSwapProgressImage = (entry: SwapFrame | undefined, imageName: string): ProgressImageSnapshot | null =>
  entry?.imageNames?.includes(imageName) ? entry.image : null;

export const getQueueItemSwapProgressImage = (queueItemId: string, imageName: string): ProgressImageSnapshot | null =>
  selectSwapProgressImage(swapFrames.get(queueItemId), imageName);

export const useProgressImage = (): LatestProgressImageSnapshot | null =>
  latestSnapshotStore.useSelector((snapshot) => snapshot.latestSnapshot);

export const useQueueItemProgressImage = (queueItemId: string, itemIndex: number): ProgressImageSnapshot | null =>
  snapshotsByTarget.useValue(getTargetKey({ itemIndex, queueItemId })) ?? null;

/** The frame to show for a slot of this queue item that has none of its own yet. */
export const useQueueItemBridgeProgressImage = (queueItemId: string): ProgressImageSnapshot | null =>
  bridgeFrames.useValue(queueItemId) ?? null;

/**
 * The frame to show in place of this just-finished image until it has decoded;
 * null once consumed or expired, for images the held frame did not produce, and
 * for images with no local run.
 */
export const useQueueItemSwapProgressImage = (
  queueItemId: string | null | undefined,
  imageName: string
): ProgressImageSnapshot | null => selectSwapProgressImage(swapFrames.useValue(queueItemId ?? ''), imageName);
