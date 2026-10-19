import type { GalleryItem, GalleryItemKey, GalleryItemKind } from '@features/gallery/core/items';

import { toGalleryItemKey } from '@features/gallery/core/items';

export type GalleryPickerAccept = readonly GalleryItemKind[];

export type GalleryPickerSelection =
  | { mode: 'single' }
  | {
      mode: 'multiple';
      addedKeys: ReadonlySet<GalleryItemKey>;
      /** Slots left per kind; a kind absent here is unlimited. */
      remaining: Partial<Record<GalleryItemKind, number>>;
    };

export type GalleryPickerTileState = 'added' | 'full' | 'pickable' | 'unsupported';

export const GALLERY_PICKER_CELL_PX = 72;
export const GALLERY_PICKER_MIN_COLUMNS = 3;
export const GALLERY_PICKER_MAX_COLUMNS = 8;

const getRemaining = (selection: Extract<GalleryPickerSelection, { mode: 'multiple' }>, kind: GalleryItemKind) =>
  selection.remaining[kind] ?? Number.POSITIVE_INFINITY;

export const getGalleryPickerTileState = (
  item: GalleryItem,
  accept: GalleryPickerAccept,
  selection: GalleryPickerSelection
): GalleryPickerTileState => {
  if (!accept.includes(item.kind)) {
    return 'unsupported';
  }

  if (selection.mode === 'single') {
    return 'pickable';
  }

  if (selection.addedKeys.has(toGalleryItemKey(item))) {
    return 'added';
  }

  return getRemaining(selection, item.kind) <= 0 ? 'full' : 'pickable';
};

/** True once no accepted kind has a slot left. */
export const isGalleryPickerFull = (accept: GalleryPickerAccept, selection: GalleryPickerSelection): boolean =>
  selection.mode === 'multiple' && accept.every((kind) => getRemaining(selection, kind) <= 0);

/** The selection as it will stand once `item` is added, so a pick can decide to close without waiting on props. */
export const getGalleryPickerSelectionAfterPick = (
  selection: GalleryPickerSelection,
  item: GalleryItem
): GalleryPickerSelection => {
  if (selection.mode === 'single') {
    return selection;
  }

  const remainingForKind = selection.remaining[item.kind];

  return {
    ...selection,
    addedKeys: new Set(selection.addedKeys).add(toGalleryItemKey(item)),
    remaining:
      remainingForKind === undefined
        ? selection.remaining
        : { ...selection.remaining, [item.kind]: Math.max(0, remainingForKind - 1) },
  };
};

/** Slots left across the accepted kinds; null when any accepted kind is unlimited or in single mode. */
export const getGalleryPickerRemaining = (
  accept: GalleryPickerAccept,
  selection: GalleryPickerSelection
): number | null => {
  if (selection.mode !== 'multiple') {
    return null;
  }

  let total = 0;

  for (const kind of accept) {
    const remaining = selection.remaining[kind];

    if (remaining === undefined) {
      return null;
    }

    total += remaining;
  }

  return total;
};

/** Where the keyboard highlight lands when nothing is active: the first tile Enter could pick, else the first tile. */
export const getGalleryPickerDefaultIndex = (
  items: readonly GalleryItem[],
  accept: GalleryPickerAccept,
  selection: GalleryPickerSelection
): number => {
  if (items.length === 0) {
    return -1;
  }

  const pickable = items.findIndex((item) => getGalleryPickerTileState(item, accept, selection) === 'pickable');

  return pickable >= 0 ? pickable : 0;
};

export type GalleryPickerStatusPart =
  | { count: number; kind: 'boardCount' | 'itemCount' | 'matchCount' | 'remainingCount' | 'windowLimit' }
  | { kind: 'remainingNone' | 'unsupportedImage' | 'unsupportedVideo' };

/** Footer priority: board count, incompatible-highlight explanation, capacity, then window cap or listing size. */
export const getGalleryPickerStatus = ({
  accept,
  activeItem,
  isSearching,
  isWindowTruncated,
  loadedCount,
  pane,
  remaining,
  total,
  visibleBoardCount,
}: {
  accept: GalleryPickerAccept;
  activeItem: GalleryItem | undefined;
  isSearching: boolean;
  isWindowTruncated: boolean;
  loadedCount: number;
  pane: 'boards' | 'items';
  remaining: number | null;
  total: number | null;
  visibleBoardCount: number;
}): GalleryPickerStatusPart[] => {
  if (pane === 'boards') {
    return [{ count: visibleBoardCount, kind: 'boardCount' }];
  }

  if (activeItem && !accept.includes(activeItem.kind)) {
    return [{ kind: activeItem.kind === 'video' ? 'unsupportedVideo' : 'unsupportedImage' }];
  }

  const parts: GalleryPickerStatusPart[] = [];

  if (remaining !== null) {
    parts.push(remaining > 0 ? { count: remaining, kind: 'remainingCount' } : { kind: 'remainingNone' });
  }

  if (isWindowTruncated) {
    parts.push({ count: loadedCount, kind: 'windowLimit' });
  } else if (total !== null) {
    parts.push({ count: total, kind: isSearching ? 'matchCount' : 'itemCount' });
  }

  return parts;
};

export type GalleryPickerNavKey = 'ArrowDown' | 'ArrowLeft' | 'ArrowRight' | 'ArrowUp' | 'End' | 'Home';

/** Left/Right/Home/End belong to the caret while the field has text; Up/Down always move the grid. */
export const isGalleryPickerNavKeyForField = (key: GalleryPickerNavKey, fieldValue: string): boolean =>
  key === 'ArrowUp' || key === 'ArrowDown' || fieldValue === '';

const NAV_KEYS: ReadonlySet<string> = new Set<GalleryPickerNavKey>([
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'End',
  'Home',
]);

export const isGalleryPickerNavKey = (key: string): key is GalleryPickerNavKey => NAV_KEYS.has(key);

/**
 * Keyboard movement over a row-major grid. Left/right walk reading order
 * across row ends; up/down stay in the column and hold still when no tile
 * sits there (a ragged last row). No active tile resolves to an edge.
 */
export const getGalleryPickerNeighborIndex = (
  index: number,
  count: number,
  columnCount: number,
  key: GalleryPickerNavKey
): number => {
  if (count <= 0) {
    return -1;
  }

  if (index < 0 || index >= count) {
    return key === 'End' ? count - 1 : 0;
  }

  switch (key) {
    case 'Home':
      return 0;
    case 'End':
      return count - 1;
    case 'ArrowLeft':
      return Math.max(0, index - 1);
    case 'ArrowRight':
      return Math.min(count - 1, index + 1);
    case 'ArrowUp':
      return index - columnCount >= 0 ? index - columnCount : index;
    case 'ArrowDown':
      return index + columnCount < count ? index + columnCount : index;
  }
};
