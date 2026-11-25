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
import { toGalleryColorLabel, toGalleryRating } from './marks';

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

const markedPersistedItems = new WeakMap<object, GalleryItem>();

/** Items persisted before marks existed read as unmarked; memoized so each persisted item keeps one identity. */
const withPersistedMarks = (item: GalleryItem): GalleryItem => {
  const raw = item as Partial<GalleryItem>;

  if (typeof raw.rating === 'number' && typeof raw.rejected === 'boolean' && raw.colorLabel !== undefined) {
    return item;
  }

  const cached = markedPersistedItems.get(item);

  if (cached) {
    return cached;
  }

  const marked: GalleryItem = {
    ...item,
    colorLabel: toGalleryColorLabel(raw.colorLabel),
    rating: toGalleryRating(raw.rating),
    rejected: !item.starred && raw.rejected === true,
  };

  markedPersistedItems.set(item, marked);
  return marked;
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
    return withPersistedMarks(galleryValues.selectedImage);
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

/** Later-first, so the successor stays out of the Bookmarked block. */
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

/** Announced once by a mounted Gallery widget; unset when the user asked. */
export type GalleryRevealAnnouncement = 'external-generation';

export interface GalleryRevealRequest {
  accountSignal: AbortSignal;
  absoluteIndex?: number;
  announcement?: GalleryRevealAnnouncement;
  itemKey: GalleryItemKey;
  token: number;
}

let currentRequest: GalleryRevealRequest | null = null;
let nextToken = 0;

const listeners = new Set<() => void>();

export const requestGalleryItemReveal = (
  itemKey: GalleryItemKey,
  accountSignal: AbortSignal,
  absoluteIndex?: number,
  announcement?: GalleryRevealAnnouncement
): void => {
  nextToken += 1;
  currentRequest = { accountSignal, absoluteIndex, announcement, itemKey, token: nextToken };
  for (const listener of listeners) {
    listener();
  }
};

/** Ignored after its account lifetime changes, including when Gallery mounts later. */
export const getGalleryRevealRequest = (): GalleryRevealRequest | null => {
  const request = currentRequest;

  return request && !request.accountSignal.aborted ? request : null;
};

export const subscribeGalleryRevealRequests = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};

export interface GalleryLocatorRequest {
  signal: AbortSignal;
  release: () => void;
}

const activeGalleryLocatorRequests = new Set<AbortController>();

export const createGalleryLocatorRequest = (): GalleryLocatorRequest => {
  const controller = new AbortController();
  activeGalleryLocatorRequests.add(controller);

  return {
    signal: controller.signal,
    release: () => activeGalleryLocatorRequests.delete(controller),
  };
};

/** A reveal nobody asked for must not supersede one still locating its item. */
export const hasGalleryLocatorRequests = (): boolean => activeGalleryLocatorRequests.size > 0;

export const abortGalleryLocatorRequests = (): void => {
  for (const controller of activeGalleryLocatorRequests) {
    controller.abort();
  }

  activeGalleryLocatorRequests.clear();
};

let navigationSequence = 0;

export const claimGalleryNavigationSequence = (): number => ++navigationSequence;

export const isGalleryNavigationCurrent = (sequence: number): boolean => sequence === navigationSequence;

/** The widget values cover ordinary clicks, which claim no sequence. */
export interface GalleryNavigationSnapshot {
  readonly sequence: number;
  readonly state: string;
}

const NAVIGATION_VALUE_KEYS = [
  'galleryPage',
  'galleryView',
  'mediaFilter',
  'searchTerm',
  'selectedBoardId',
  'selectedImageName',
  'selectedImageNames',
  'semanticImageQuery',
  'semanticSearchText',
  'marksFilter',
] as const;

export const captureGalleryNavigation = (values: Record<string, unknown>): GalleryNavigationSnapshot => ({
  sequence: navigationSequence,
  state: JSON.stringify(NAVIGATION_VALUE_KEYS.map((key) => values[key] ?? null)),
});

export const isGalleryNavigationStateUnchanged = (
  snapshot: GalleryNavigationSnapshot,
  values: Record<string, unknown>
): boolean => captureGalleryNavigation(values).state === snapshot.state;

/** An empty semantic field holds no search. */
export const hasGalleryListingSearch = (values: Record<string, unknown>): boolean =>
  (typeof values.searchTerm === 'string' && values.searchTerm !== '') ||
  (typeof values.semanticSearchText === 'string' && values.semanticSearchText !== '') ||
  (values.semanticImageQuery !== null && values.semanticImageQuery !== undefined);

export type GalleryNavigationEntry =
  | { kind: 'item'; item: GalleryItem }
  | { kind: 'slot'; id: string; navigable: boolean }
  | { kind: 'session'; id: string; navigable: boolean };

export type GalleryNavigationDirection = 'down' | 'left' | 'right' | 'up';

export const getGallerySessionNavigationKey = (sessionId: string): string => `session:${sessionId}`;

const getGalleryNavigationEntryKey = (entry: GalleryNavigationEntry): string =>
  entry.kind === 'item'
    ? toGalleryItemKey(entry.item)
    : entry.kind === 'slot'
      ? entry.id
      : getGallerySessionNavigationKey(entry.id);

/** A cursor candidate the sections do not show is passed over; null means none is shown. */
export const getGalleryNavigationCursor = (
  sections: readonly (readonly GalleryNavigationEntry[])[],
  cursorKeys: readonly (string | null)[]
): string | null => {
  const shownKeys = new Set(sections.flat().map(getGalleryNavigationEntryKey));

  return cursorKeys.find((key): key is string => key !== null && shownKeys.has(key)) ?? null;
};

/** Vertical steps keep columns and skip empty rows; `itemsOnly` steps over sessions without removing their rows. */
export const getGalleryNavigationStep = (
  sections: readonly (readonly GalleryNavigationEntry[])[],
  cursorKeys: readonly (string | null)[],
  direction: GalleryNavigationDirection,
  columnCount = 1,
  { itemsOnly = false }: { itemsOnly?: boolean } = {}
): GalleryNavigationEntry | null => {
  const isNavigable = (entry: GalleryNavigationEntry | undefined): entry is GalleryNavigationEntry =>
    entry !== undefined &&
    (entry.kind === 'item' ||
      (entry.kind === 'slot' && entry.navigable) ||
      (entry.kind === 'session' && !itemsOnly && entry.navigable));
  const entries = sections.flat();
  const cursorKey = getGalleryNavigationCursor(sections, cursorKeys);

  if (cursorKey === null) {
    return entries.find(isNavigable) ?? null;
  }

  const index = entries.findIndex((entry) => getGalleryNavigationEntryKey(entry) === cursorKey);

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

    // The left one on a tie.
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
