import type { GalleryImage, GeneratedImageContract } from './types';

import {
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  parseGalleryItemKey,
  toGalleryItemKey,
  type GalleryItem,
  type GalleryItemKey,
} from './items';

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object';

const isGeneratedImage = (value: unknown): value is GeneratedImageContract =>
  isRecord(value) && typeof value.imageName === 'string';

const isGalleryItem = (value: unknown): value is GalleryItem => {
  if (
    !isRecord(value) ||
    (value.kind !== 'image' && value.kind !== 'video') ||
    typeof value.name !== 'string' ||
    typeof value.boardId !== 'string' ||
    typeof value.category !== 'string' ||
    typeof value.createdAt !== 'string' ||
    typeof value.fullUrl !== 'string' ||
    typeof value.height !== 'number' ||
    typeof value.isIntermediate !== 'boolean' ||
    typeof value.starred !== 'boolean' ||
    typeof value.thumbnailUrl !== 'string' ||
    typeof value.width !== 'number'
  ) {
    return false;
  }

  return value.kind === 'image' || typeof value.durationSeconds === 'number';
};

const legacyImageToGalleryItem = (
  image: GeneratedImageContract & Partial<GalleryImage>,
  galleryValues: Record<string, unknown>
): GalleryItem =>
  legacyGeneratedImageToGalleryItem({
    ...image,
    boardId:
      image.boardId ?? (typeof galleryValues.selectedBoardId === 'string' ? galleryValues.selectedBoardId : 'none'),
  });

export const getSelectedGalleryItemFromValues = (galleryValues: Record<string, unknown>): GalleryItem | null => {
  if (isGalleryItem(galleryValues.selectedImage)) {
    return galleryValues.selectedImage;
  }

  if (isGeneratedImage(galleryValues.selectedImage)) {
    return legacyImageToGalleryItem(galleryValues.selectedImage, galleryValues);
  }

  const selectedImageName =
    typeof galleryValues.selectedImageName === 'string' ? galleryValues.selectedImageName : null;
  const selectedRef = selectedImageName ? parseGalleryItemKey(selectedImageName) : null;

  if (!selectedRef || selectedRef.kind !== 'image') {
    return null;
  }

  const recentImages = Array.isArray(galleryValues.recentImages) ? galleryValues.recentImages : [];
  const recentImage = recentImages.find(
    (image): image is GeneratedImageContract => isGeneratedImage(image) && image.imageName === selectedRef.name
  );

  return recentImage ? legacyImageToGalleryItem(recentImage, galleryValues) : null;
};

export const getSelectedGalleryImageFromValues = (galleryValues: Record<string, unknown>): GalleryImage | null => {
  const item = getSelectedGalleryItemFromValues(galleryValues);

  return item && isGalleryImageItem(item) ? galleryImageItemToGalleryImage(item) : null;
};

const canonicalizePersistedItemKey = (key: string): GalleryItemKey => toGalleryItemKey(parseGalleryItemKey(key));

export const getPersistedSelectedGalleryItemKeys = (galleryValues: Record<string, unknown>): GalleryItemKey[] => {
  if (Array.isArray(galleryValues.selectedImageNames)) {
    return (galleryValues.selectedImageNames as unknown[])
      .filter((name): name is string => typeof name === 'string')
      .map(canonicalizePersistedItemKey);
  }

  if (typeof galleryValues.selectedImageName === 'string') {
    return [canonicalizePersistedItemKey(galleryValues.selectedImageName)];
  }

  const selectedItem = getSelectedGalleryItemFromValues(galleryValues);

  return selectedItem ? [toGalleryItemKey(selectedItem)] : [];
};

/*
 * Reveal requests: an explicit "scroll this item into view" signal from
 * surfaces outside the grid (the image map's reveal). Deliberately NOT derived
 * from the selection, which also changes when a finished generation
 * auto-selects its image — scrolling on that yanked the grid out from under a
 * browsing user. A reveal is a deliberate gesture, so it gets its own channel,
 * and a token, so repeating the same gesture (re-clicking the same map point
 * after scrolling away) reveals again even though the selection is unchanged.
 *
 * Module-scoped rather than persisted: a reveal is an ephemeral intent for the
 * currently mounted grid, and persisting it would replay a stale scroll in the
 * next session. It lives here, beside the selection helpers it travels with,
 * rather than in a module of its own — a separate one becomes a separate chunk
 * and an extra request in the gallery widget's load, which the performance
 * budgets police.
 */

export interface GalleryRevealRequest {
  itemKey: GalleryItemKey;
  token: number;
}

let currentRequest: GalleryRevealRequest | null = null;
let nextToken = 0;

const listeners = new Set<() => void>();

export const requestGalleryItemReveal = (itemKey: GalleryItemKey): void => {
  nextToken += 1;
  currentRequest = { itemKey, token: nextToken };
  for (const listener of listeners) {
    listener();
  }
};

export const getGalleryRevealRequest = (): GalleryRevealRequest | null => currentRequest;

export const subscribeGalleryRevealRequests = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};
