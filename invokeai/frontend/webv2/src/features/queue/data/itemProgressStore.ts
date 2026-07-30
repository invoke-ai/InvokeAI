import { mergeQueueProgressImage, type QueueProgressImage } from '@features/queue/core/progressImage';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore, createKeyedTransientStore } from '@platform/state/externalStore';

/**
 * Live progress for in-flight queue items keyed by the backend `item_id`. The
 * sibling `progressStore` keys by the *local* submission id, which only exists
 * for items this client enqueued; the Queue widget shows the whole server queue,
 * so its NOW & NEXT card needs progress addressable by the backend id carried on
 * the socket's `invocation_started`/`invocation_progress` events.
 *
 * Being keyed by item id is also what makes multi-GPU work: with
 * `generation_devices` (default `auto`) the backend runs one session per GPU, so
 * several items report progress at once and each needs its own entry.
 */

export interface ItemProgress {
  message: string;
  /** 0..1, or null while indeterminate. */
  percentage: number | null;
  image?: QueueProgressImage | null;
  /** The GPU running this session, e.g. `cuda:1`. Null on non-CUDA and single-device installs. */
  device?: string | null;
}

const progressByItemId = createKeyedTransientStore<number, ItemProgress>();

/**
 * The ids with live progress, ascending — a stable top-to-bottom order for stacked
 * progress bars and tiled previews (ascending item id is submission order).
 *
 * Deliberately separate from the keyed store: consumers subscribe to *which* items
 * are running here, and to a single item's progress via `useItemProgress`, so one
 * session's step does not re-render another session's tile. Maintained from the
 * mutation points below rather than derived from the keyed store, whose `entries()`
 * allocates a fresh array per call and so cannot serve as a cached snapshot.
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
