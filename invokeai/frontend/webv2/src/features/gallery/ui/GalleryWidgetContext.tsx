import type { GalleryImageItem, GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryKnownMarks, GalleryMarksFilter } from '@features/gallery/core/marks';
import type { GalleryMediaUsageReport } from '@features/gallery/core/mediaUsage';
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GallerySearchScope, GallerySettings } from '@features/gallery/core/settings';
import type { GalleryGeneralAccess, GalleryView } from '@features/gallery/core/types';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';
import type { GalleryItemMarksPatchEvent } from '@features/gallery/data/queryCache';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { galleryMarksFilterKey, getGalleryItemMarks } from '@features/gallery/core/marks';
import { subscribeGalleryItemMarksPatches } from '@features/gallery/data/queryCache';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useQueryClient } from '@tanstack/react-query';
import { createContext, use, useMemo, useState, useSyncExternalStore } from 'react';

import type { GalleryReadState, GalleryStateView } from './galleryStateView';
import type { GalleryItemActions, GalleryWidgetProps, GalleryWidgetRuntime } from './GalleryUiContext';
import type { GalleryListingState, GallerySparseListing } from './useGalleryData';

/** Shared ImageActions owns cross-widget item operations. */
export interface GalleryActions {
  archiveBoard: (boardId: string, archived: boolean) => Promise<void>;
  clearSelection: () => void;
  /** In the project, or the Library for null; resolves `true` once the board exists and is selected. */
  createBoard: (boardName: string, projectId: string | null) => Promise<boolean>;
  deleteBoard: (boardId: string, includeImages: boolean) => Promise<void>;
  downloadBoard: (boardId: string) => Promise<void>;
  getBoardMediaUsage: (boardId: string, signal: AbortSignal) => Promise<GalleryMediaUsageReport>;
  exportProject: (projectId: string, projectName: string) => void;
  loadMore: () => void;
  /** `destinationLabel` is what the confirmation names. */
  moveBoard: (boardId: string, projectId: string | null, destinationLabel: string) => Promise<void>;
  refresh: () => void;
  renameBoard: (boardId: string, boardName: string) => Promise<void>;
  setBoardGeneralAccess: (boardId: string, generalAccess: GalleryGeneralAccess) => Promise<void>;
  selectBoard: (boardId: string) => void;
  selectItem: (item: GalleryItem, selectionPage?: number) => void;
  selectItemRange: (items: GalleryItemRef[], primaryItem: GalleryItem, selectionPage?: number) => void;
  setCompareItem: (image: GalleryImageItem | null) => void;
  setSearchTerm: (searchTerm: string) => void;
  /** Resets the page like a search. */
  setSearchScope: (searchScope: GallerySearchScope) => void;
  /** Resets the page like a search. */
  setMarksFilter: (marksFilter: GalleryMarksFilter) => void;
  /** Clears text, ranking and semantic mode. */
  clearSearch: () => void;
  /** A no-op once the field has moved on. */
  commitSemanticSearch: (text: string) => void;
  setSemanticImageQuery: (reference: GallerySemanticReference | null) => void;
  /** Keeps the field's text. */
  setSemanticSearchMode: (enabled: boolean) => void;
  /** Ahead of the debounced commit. */
  setSemanticSearchText: (text: string) => void;
  setView: (galleryView: GalleryView) => void;
  toggleItemInSelection: (item: GalleryItem, nextPrimaryItem: GalleryItem | null, selectionPage?: number) => void;
  updateSettings: (settings: Partial<GallerySettings>) => void;
  /** Empty when nothing landed. */
  uploadFiles: (files: File[]) => Promise<GalleryItem[]>;
}

/** The Bookmarked strip; empty whenever it does not apply. */
export interface GalleryStarredStrip {
  items: GalleryItem[];
  state: GalleryReadState;
  /** Per the backend; 0 until known. */
  total: number;
}

export interface GalleryWidgetContextValue {
  gallery: GalleryStateView;
  actions: GalleryActions;
  boardSharing?: boolean;
  /** Its failure is the board panel's to show, never the grid's. */
  boardsState: GalleryReadState;
  /** Shared rather than re-derived so range selection and the list never disagree. */
  filter: GalleryItemsFilter;
  itemActions: GalleryItemActions;
  listing: GalleryListingState;
  /** Strip first, then the listing, without repeats. */
  loadedItems: GalleryItem[];
  /** Main Gallery only; other surfaces use a dense loaded projection. */
  sparseListing?: GallerySparseListing;
  /** Only intersecting page queries stay subscribed. */
  setVisibleRange?: (range: { endIndexExclusive: number; startIndex: number }) => void;
  /** Sparse infinite listings only. */
  pinRevealIndex?: (absoluteIndex: number) => void;
  starredStrip: GalleryStarredStrip;
  projectId: string;
  projectName: string;
  projectNames: ReadonlyMap<string, string>;
  /** Used only to scope cached viewport measurements. */
  region: GalleryWidgetProps['region'];
  runtime: GalleryWidgetRuntime;
}

/** Marks patched onto items no loaded page may hold, bounded: the oldest go first. */
const GALLERY_MARKS_OVERLAY_LIMIT = 10_000;

/** `version` is the patch that last set the entry, so a mark seen on a loaded item can tell which is newer. */
interface MarksEntry {
  marks: GalleryKnownMarks;
  version: number;
}

interface MarksOverlayStore {
  getSnapshot: () => ReadonlyMap<string, MarksEntry>;
  reconcilePatch: (event: GalleryItemMarksPatchEvent) => void;
  subscribe: (listener: () => void) => () => void;
}

const createMarksOverlayStore = (): MarksOverlayStore => {
  let snapshot: ReadonlyMap<string, MarksEntry> = new Map();
  // Only the rollback of the latest patch that set a key restores it.
  const patchedFrom = new Map<string, { before: MarksEntry | undefined; patchId: number }>();
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => snapshot,
    reconcilePatch: (event) => {
      const overlay = new Map(snapshot);

      if (event.kind === 'apply') {
        for (const key of event.itemKeys) {
          const before = overlay.get(key);

          patchedFrom.set(key, { before, patchId: event.patchId });
          overlay.delete(key);
          overlay.set(key, { marks: { ...before?.marks, ...event.patch }, version: event.patchId });
        }
        for (const key of overlay.keys()) {
          if (overlay.size <= GALLERY_MARKS_OVERLAY_LIMIT) {
            break;
          }
          overlay.delete(key);
          patchedFrom.delete(key);
        }
      } else {
        for (const [key, { before, patchId }] of patchedFrom) {
          if (patchId !== event.patchId) {
            continue;
          }

          patchedFrom.delete(key);

          if (before === undefined) {
            overlay.delete(key);
          } else {
            overlay.set(key, before);
          }
        }
      }

      snapshot = overlay;
      listeners.forEach((listener) => listener());
    },
    subscribe: (listener) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
  };
};

interface SeenMarks {
  loadedItems: readonly GalleryItem[];
  /** What selected items showed when a page last held them, for once it no longer does. */
  marks: ReadonlyMap<string, MarksEntry>;
  selectedKeys: readonly string[];
}

const EMPTY_SEEN_MARKS: SeenMarks = { loadedItems: [], marks: new Map(), selectedKeys: [] };

const getSeenMarks = (
  previous: SeenMarks,
  selectedKeys: readonly string[],
  loadedItems: readonly GalleryItem[],
  overlay: ReadonlyMap<string, MarksEntry>
): SeenMarks => {
  const loaded = new Map<string, GalleryItem>(loadedItems.map((item) => [toGalleryItemKey(item), item]));
  const marks = new Map<string, MarksEntry>();

  for (const key of selectedKeys) {
    const item = loaded.get(key);
    const entry = item ? { marks: getGalleryItemMarks(item), version: overlay.get(key)?.version ?? 0 } : undefined;
    const kept = entry ?? previous.marks.get(key);

    if (kept) {
      marks.set(key, kept);
    }
  }

  return { loadedItems, marks, selectedKeys };
};

/**
 * Each selected item's marks: a loaded item's own; else the newer of what a page last showed and what this Gallery's
 * patches set; else what the listing it was selected in implies. A field none of these knows is left out.
 */
export const useGallerySelectionMarks = (
  selectedItems: readonly GalleryItemRef[],
  loadedItems: readonly GalleryItem[]
): GalleryKnownMarks[] => {
  const { gallery } = useGalleryWidget();
  const queryClient = useQueryClient();
  const [store] = useState(createMarksOverlayStore);
  const overlay = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  const selectedKeys = useMemo(() => selectedItems.map(toGalleryItemKey), [selectedItems]);
  const [seen, setSeen] = useState<SeenMarks>(() => getSeenMarks(EMPTY_SEEN_MARKS, selectedKeys, loadedItems, overlay));

  useMountEffect(() => subscribeGalleryItemMarksPatches(queryClient, store.reconcilePatch));

  // Adjusted while rendering, as React documents for state that follows props: only a new listing or selection
  // records what it shows, so a patch that re-renders the same listing stays the newer word.
  if (seen.loadedItems !== loadedItems || seen.selectedKeys !== selectedKeys) {
    setSeen(getSeenMarks(seen, selectedKeys, loadedItems, overlay));
  }

  const loadedMarks = useMemo(
    () => new Map(loadedItems.map((item) => [toGalleryItemKey(item), getGalleryItemMarks(item)])),
    [loadedItems]
  );
  // A selection made in a listing of one flag holds only that flag.
  const [onlyFlag, ...otherFlags] = gallery.marksFilter.flags;
  const impliedFlag =
    onlyFlag !== undefined &&
    otherFlags.length === 0 &&
    galleryMarksFilterKey(gallery.selectionMarksFilter) === galleryMarksFilterKey(gallery.marksFilter)
      ? onlyFlag
      : undefined;

  return useMemo(
    () =>
      selectedKeys.map((key): GalleryKnownMarks => {
        const loaded = loadedMarks.get(key);

        if (loaded) {
          return loaded;
        }

        const patched = overlay.get(key);
        const shown = seen.marks.get(key);
        const implied = impliedFlag ? { flag: impliedFlag } : {};

        return patched && (!shown || patched.version > shown.version)
          ? { ...implied, ...shown?.marks, ...patched.marks }
          : { ...implied, ...shown?.marks };
      }),
    [impliedFlag, loadedMarks, overlay, seen, selectedKeys]
  );
};

export const GalleryWidgetContext = createContext<GalleryWidgetContextValue | null>(null);

export const useGalleryWidget = (): GalleryWidgetContextValue => {
  const value = use(GalleryWidgetContext);

  if (!value) {
    throw new Error('useGalleryWidget must be used within the gallery widget.');
  }

  return value;
};
