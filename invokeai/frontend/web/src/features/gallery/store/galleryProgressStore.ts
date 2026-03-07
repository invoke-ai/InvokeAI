import type { BoardId } from 'features/gallery/store/types';
import type { ProgressImage } from 'features/nodes/types/common';
import { atom } from 'nanostores';

type GalleryProgressItemStatus = 'pending' | 'in_progress';

export type GalleryProgressItem = {
  itemId: number;
  batchId: string;
  status: GalleryProgressItemStatus;
  progressImage: ProgressImage | null;
  percentage: number | null;
  message: string | null;
  targetBoardId: BoardId;
  enqueuedAt: number;
};

/**
 * Stores all active (pending/in_progress) queue items that should show in the gallery.
 * Keyed by item ID.
 */
export const $galleryProgressItems = atom<Record<number, GalleryProgressItem>>({});

/**
 * Adds multiple progress items from an enqueue batch result.
 */
export const addGalleryProgressItems = (itemIds: number[], batchId: string, targetBoardId: BoardId) => {
  const current = $galleryProgressItems.get();
  const now = Date.now();
  const newItems: Record<number, GalleryProgressItem> = {};
  for (const itemId of itemIds) {
    newItems[itemId] = {
      itemId,
      batchId,
      status: 'pending',
      progressImage: null,
      percentage: null,
      message: null,
      targetBoardId,
      enqueuedAt: now,
    };
  }
  $galleryProgressItems.set({ ...current, ...newItems });
};

/**
 * Updates a progress item's status to in_progress.
 */
export const setGalleryProgressItemInProgress = (itemId: number) => {
  const current = $galleryProgressItems.get();
  const item = current[itemId];
  if (!item) {
    return;
  }
  $galleryProgressItems.set({
    ...current,
    [itemId]: { ...item, status: 'in_progress' },
  });
};

/**
 * Updates a progress item's image and percentage.
 */
export const updateGalleryProgressItemProgress = (
  itemId: number,
  progressImage: ProgressImage | null,
  percentage: number | null,
  message: string | null
) => {
  const current = $galleryProgressItems.get();
  const item = current[itemId];
  if (!item) {
    return;
  }
  $galleryProgressItems.set({
    ...current,
    [itemId]: { ...item, progressImage, percentage, message },
  });
};

/**
 * Removes a progress item (on completion, failure, or cancellation).
 */
export const removeGalleryProgressItem = (itemId: number) => {
  const current = $galleryProgressItems.get();
  if (!(itemId in current)) {
    return;
  }
  const { [itemId]: _, ...rest } = current;
  $galleryProgressItems.set(rest);
};

/**
 * Clears all progress items (on disconnect/reconnect).
 */
export const clearGalleryProgressItems = () => {
  $galleryProgressItems.set({});
};

/**
 * Returns progress items filtered by board ID, sorted by enqueuedAt ascending.
 */
export const selectFilteredGalleryProgressItems = (
  items: Record<number, GalleryProgressItem>,
  boardId: BoardId
): GalleryProgressItem[] => {
  return Object.values(items)
    .filter((item) => item.targetBoardId === boardId)
    .sort((a, b) => a.enqueuedAt - b.enqueuedAt);
};
