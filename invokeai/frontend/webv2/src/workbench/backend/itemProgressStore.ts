import { createKeyedTransientStore } from '@workbench/externalStore';
import { mergeDenoiseProgressImage, type DenoiseProgressImage } from '@workbench/images/streamingImageSource';

/**
 * Live progress for in-flight queue items keyed by the backend `item_id`. The
 * sibling `progressStore` keys by the *local* submission id, which only exists
 * for items this client enqueued; the Queue widget shows the whole server queue,
 * so its NOW & NEXT card needs progress addressable by the backend id carried on
 * the socket's `invocation_started`/`invocation_progress` events.
 */

export interface ItemProgress {
  message: string;
  /** 0..1, or null while indeterminate. */
  percentage: number | null;
  image?: DenoiseProgressImage | null;
}

const progressByItemId = createKeyedTransientStore<number, ItemProgress>();

export const itemProgressStore = {
  get(itemId: number): ItemProgress | null {
    return progressByItemId.get(itemId) ?? null;
  },
  set(itemId: number, progress: ItemProgress): void {
    const current = progressByItemId.get(itemId);
    const image = mergeDenoiseProgressImage(current?.image, progress.image);
    const next = image === undefined ? progress : { ...progress, image };

    progressByItemId.set(itemId, next);
  },
  clear(itemId: number): void {
    progressByItemId.delete(itemId);
  },
  clearAll(): void {
    progressByItemId.clear();
  },
};

export const useItemProgress = (itemId: number | null | undefined): ItemProgress | null =>
  progressByItemId.useValue(itemId ?? -1) ?? null;
