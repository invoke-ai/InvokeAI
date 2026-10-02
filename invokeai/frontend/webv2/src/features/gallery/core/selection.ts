import type { GalleryImage, GeneratedImageContract } from './types';

import {
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  parseGalleryItemKey,
  toGalleryItemKey,
  type GalleryItem,
  type GalleryItemKey,
  type GalleryItemRef,
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

/**
 * Successor after deleting `primaryKey`: the next eligible entry in display
 * order, else the nearest earlier one — later-first stays out of the starred block.
 */
export const getGalleryDeletionSuccessor = (
  orderedRefs: readonly GalleryItemRef[],
  primaryKey: GalleryItemKey,
  ineligibleKeys: ReadonlySet<GalleryItemKey>
): GalleryItemRef | null => {
  const primaryIndex = orderedRefs.findIndex((ref) => toGalleryItemKey(ref) === primaryKey);

  if (primaryIndex < 0) {
    return null;
  }

  for (let index = primaryIndex + 1; index < orderedRefs.length; index += 1) {
    const candidate = orderedRefs[index];

    if (candidate && !ineligibleKeys.has(toGalleryItemKey(candidate))) {
      return candidate;
    }
  }

  for (let index = primaryIndex - 1; index >= 0; index -= 1) {
    const candidate = orderedRefs[index];

    if (candidate && !ineligibleKeys.has(toGalleryItemKey(candidate))) {
      return candidate;
    }
  }

  return null;
};

/*
 * Explicit, tokenized reveals can repeat without selection changes. Keep them ephemeral to avoid replaying stale
 * scrolls; colocating avoids another gallery chunk.
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

/*
 * Order navigation gestures across mounts and surfaces before async hydration; only the latest may update the
 * shared selection.
 */

let navigationSequence = 0;

/** Claims this navigation's place in the global ordering; every later claim supersedes it. */
export const claimGalleryNavigationSequence = (): number => ++navigationSequence;

/** False once a newer navigation has been claimed, which is when a slow hydrate must stand down. */
export const isGalleryNavigationCurrent = (sequence: number): boolean => sequence === navigationSequence;

/*
 * Grid and Preview share section order; horizontal navigation crosses seams while vertical navigation preserves
 * columns.
 */

export type GalleryNavigationEntry =
  | { kind: 'item'; item: GalleryItem }
  | { kind: 'session'; id: string; navigable: boolean };

export type GalleryNavigationDirection = 'down' | 'left' | 'right' | 'up';

export const getGallerySessionNavigationKey = (sessionId: string): string => `session:${sessionId}`;

const getGalleryNavigationEntryKey = (entry: GalleryNavigationEntry): string =>
  entry.kind === 'item' ? toGalleryItemKey(entry.item) : getGallerySessionNavigationKey(entry.id);

const isNavigable = (entry: GalleryNavigationEntry | undefined): entry is GalleryNavigationEntry =>
  entry !== undefined && (entry.kind === 'item' || entry.navigable);

/**
 * Each section starts its own rows. Vertical navigation preserves columns, selects the nearest navigable cell, and
 * skips empty rows; no cursor starts at the first entry.
 */
export const getGalleryNavigationStep = (
  sections: readonly (readonly GalleryNavigationEntry[])[],
  cursorKey: string | null,
  direction: GalleryNavigationDirection,
  columnCount = 1
): GalleryNavigationEntry | null => {
  const entries = sections.flat();
  const index =
    cursorKey === null ? -1 : entries.findIndex((entry) => getGalleryNavigationEntryKey(entry) === cursorKey);

  if (index === -1) {
    return entries.find(isNavigable) ?? null;
  }

  if (direction === 'left' || direction === 'right') {
    const step = direction === 'right' ? 1 : -1;

    for (let candidate = index + step; candidate >= 0 && candidate < entries.length; candidate += step) {
      if (isNavigable(entries[candidate])) {
        return entries[candidate]!;
      }
    }

    return null;
  }

  const rows: { length: number; start: number }[] = [];
  let sectionStart = 0;

  for (const section of sections) {
    for (let offset = 0; offset < section.length; offset += columnCount) {
      rows.push({ length: Math.min(columnCount, section.length - offset), start: sectionStart + offset });
    }

    sectionStart += section.length;
  }

  const rowIndex = rows.findIndex((row) => index >= row.start && index < row.start + row.length);
  const column = index - rows[rowIndex]!.start;
  const step = direction === 'down' ? 1 : -1;

  for (let target = rowIndex + step; target >= 0 && target < rows.length; target += step) {
    const row = rows[target]!;
    const landing = Math.min(column, row.length - 1);

    // Nearest navigable cell of the row by column distance, the left one on a tie.
    for (let distance = 0; distance < row.length; distance += 1) {
      for (const candidate of [landing - distance, landing + distance]) {
        if (candidate >= 0 && candidate < row.length && isNavigable(entries[row.start + candidate])) {
          return entries[row.start + candidate]!;
        }
      }
    }
  }

  return null;
};
