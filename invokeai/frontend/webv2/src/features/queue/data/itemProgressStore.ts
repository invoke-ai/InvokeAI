import { mergeQueueProgressImage, type QueueProgressImage } from '@features/queue/core/progressImage';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore, createKeyedTransientStore } from '@platform/state/externalStore';

/**
 * Key progress by backend item ID for server-wide and concurrent GPU work; local submission IDs cannot identify
 * other clients' items.
 */

export interface ItemProgress {
  message: string;
  /** 0..1, or null while indeterminate. */
  percentage: number | null;
  image?: QueueProgressImage | null;
  /** The accelerator running this session, e.g. `cuda:1` or `xpu:1`. Null on unindexed and single-device installs. */
  device?: string | null;
}

const progressByItemId = createKeyedTransientStore<number, ItemProgress>();

/**
 * Keep a stable sorted ID snapshot separate from per-item progress so one session's steps do not rerender other
 * tiles.
 */
const activeItemIdsStore = createExternalStore<{ itemIds: number[] }>({ itemIds: [] });

const syncActiveItemIds = (): void => {
  const itemIds = progressByItemId
    .entries()
    .map(([itemId]) => itemId)
    .sort((left, right) => left - right);
  const current = activeItemIdsStore.getSnapshot().itemIds;

  if (current.length === itemIds.length && current.every((itemId, index) => itemId === itemIds[index])) {
    return;
  }

  activeItemIdsStore.patchSnapshot({ itemIds });
};

export const itemProgressStore = {
  get(itemId: number): ItemProgress | null {
    return progressByItemId.get(itemId) ?? null;
  },
  set(itemId: number, progress: ItemProgress): void {
    const current = progressByItemId.get(itemId);
    const image = mergeQueueProgressImage(current?.image, progress.image);
    const next = image === undefined ? progress : { ...progress, image };

    progressByItemId.set(itemId, next);
    syncActiveItemIds();
  },
  clear(itemId: number): void {
    progressByItemId.delete(itemId);
    syncActiveItemIds();
  },
  clearAll(): void {
    progressByItemId.clear();
    syncActiveItemIds();
  },
};

registerAccountOwnedResource({
  clear: itemProgressStore.clearAll,
  name: 'queue-backend-item-progress',
});

export const useItemProgress = (itemId: number | null | undefined): ItemProgress | null =>
  progressByItemId.useValue(itemId ?? -1) ?? null;

/** Backend item ids with live progress, ascending. Empty when nothing is running. */
export const useActiveProgressItemIds = (): number[] => activeItemIdsStore.useSelector((snapshot) => snapshot.itemIds);

/** Non-reactive read, for imperative callers and tests. */
export const getActiveProgressItemIds = (): number[] => activeItemIdsStore.getSnapshot().itemIds;
