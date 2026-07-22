import type { AppGetState } from 'app/store/store';
import { galleryApi } from 'services/api/endpoints/gallery';
import type { GetGalleryItemNamesArgs } from 'services/api/types';

import { selectGetImageNamesQueryArgs } from './gallerySelectors';

/**
 * Returns the names (in display order) of the currently-cached gallery item list.
 *
 * The grid renders via the polymorphic ``getGalleryItemNames`` endpoint, which returns a
 * mixed image+video list. Range-selection click handlers (shift-click for ranges, ctrl-click
 * for discontiguous selection) need that ordered list to compute the items between two
 * clicks.
 *
 * We look up the cache entry whose args match the gallery's current query args. RTK Query
 * keeps recently-used entries warm (60s default ``keepUnusedDataFor``), so after a board
 * switch the cache contains entries for every board the user has visited this session.
 * Falling back to "the first invalidated entry" — the previous behaviour — silently picked
 * a stale board's list, which manifested as shift-click range selection failing at random
 * until the user did anything that invalidated ``GalleryItemNameList`` (move, delete) and
 * forced a refetch.
 */
export const selectCachedGalleryItemNames = (state: ReturnType<AppGetState>): string[] => {
  const args = selectGetImageNamesQueryArgs(state);
  // Exact match: the entry the grid is actively subscribed to. This is the common case.
  const exact = galleryApi.endpoints.getGalleryItemNames.select(args)(state).data;
  if (exact) {
    return exact.items.map((ref) => ref.name);
  }
  // Debounce window: the grid hook debounces its args by ~300ms, so for a moment after the
  // user changes a filter the cache key may not match Redux yet. Best-effort fallback to any
  // cached entry on the same board so range selection still feels responsive — but do not
  // silently fall back to an unrelated board's entry, which was the bug.
  const entries = galleryApi.util.selectInvalidatedBy(state, ['GalleryItemNameList']);
  for (const entry of entries) {
    if (entry.endpointName !== 'getGalleryItemNames') {
      continue;
    }
    const entryArgs = entry.originalArgs as GetGalleryItemNamesArgs | undefined;
    if (!entryArgs || entryArgs.board_id !== args.board_id) {
      continue;
    }
    const data = galleryApi.endpoints.getGalleryItemNames.select(entryArgs)(state).data;
    if (data) {
      return data.items.map((ref) => ref.name);
    }
  }
  return [];
};

/**
 * Given the ordered gallery list, the index of the currently-displayed item in that list,
 * and the set of names being deleted, return the name that should be selected after the
 * deletion completes — preferring the immediate predecessor, falling back to the immediate
 * successor, and returning null if every remaining item was also deleted.
 *
 * Used by the image and video delete flows so that clearing the displayed item from the
 * Viewer lands on an adjacent gallery item instead of the empty-state placeholder.
 */
export const pickSelectionAfterDelete = (
  galleryItemNames: string[],
  deletedIndex: number,
  deletedNames: Set<string>
): string | null => {
  if (deletedIndex < 0) {
    return null;
  }
  for (let i = deletedIndex - 1; i >= 0; i--) {
    const name = galleryItemNames[i];
    if (name && !deletedNames.has(name)) {
      return name;
    }
  }
  for (let i = deletedIndex + 1; i < galleryItemNames.length; i++) {
    const name = galleryItemNames[i];
    if (name && !deletedNames.has(name)) {
      return name;
    }
  }
  return null;
};
