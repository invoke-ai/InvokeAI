import type { GalleryImageItem, GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryView } from '@features/gallery/core/types';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { createContext, use, useEffect, useMemo, useSyncExternalStore } from 'react';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryItemActions, GalleryWidgetProps, GalleryWidgetRuntime } from './GalleryUiContext';
import type { GallerySparseListing } from './useGalleryData';

/**
 * The provider maps widget intents to workbench/backend actions; shared ImageActions owns cross-widget item
 * operations.
 */
export interface GalleryActions {
  archiveBoard: (boardId: string, archived: boolean) => Promise<void>;
  createBoard: (boardName: string) => Promise<void>;
  deleteBoard: (boardId: string, includeImages: boolean) => Promise<void>;
  downloadBoard: (boardId: string) => Promise<void>;
  /** Export the project that owns this board as a complete `.invk` archive. */
  exportProject: (projectId: string, projectName: string) => void;
  loadMore: () => void;
  refresh: () => void;
  renameBoard: (boardId: string, boardName: string) => Promise<void>;
  selectBoard: (boardId: string) => void;
  selectItem: (item: GalleryItem, selectionPage?: number) => void;
  selectItemRange: (items: GalleryItemRef[], primaryItem: GalleryItem, selectionPage?: number) => void;
  setCompareItem: (image: GalleryImageItem | null) => void;
  setSearchTerm: (searchTerm: string) => void;
  /** Restricts (or releases) the listing to starred items; resets the page like a search. */
  setStarredOnly: (starredOnly: boolean) => void;
  /** Clears the search field: its text, any ranking, and semantic mode. */
  clearSearch: () => void;
  /** Applies the semantic field's text as the ranking; a no-op once the field has moved on. */
  commitSemanticSearch: (text: string) => void;
  /** Sets (or clears) the image-similarity query shown as a chip in the search field. */
  setSemanticImageQuery: (reference: GallerySemanticReference | null) => void;
  /** Switches the search field between metadata and semantic search, keeping its text. */
  setSemanticSearchMode: (enabled: boolean) => void;
  /** The semantic field's live text, ahead of the debounced commit. */
  setSemanticSearchText: (text: string) => void;
  setView: (galleryView: GalleryView) => void;
  toggleItemInSelection: (item: GalleryItem, nextPrimaryItem: GalleryItem | null) => void;
  updateSettings: (settings: Partial<GallerySettings>) => void;
  /** Resolves with the confirmed uploads; empty when nothing landed. */
  uploadFiles: (files: File[]) => Promise<GalleryItem[]>;
}

/** The bounded starred strip above the listing; empty whenever it does not apply. */
export interface GalleryStarredStrip {
  items: GalleryItem[];
  /** Starred items under the same filter, per the backend; 0 until known. */
  total: number;
}

export interface GalleryWidgetContextValue {
  gallery: GalleryStateView;
  actions: GalleryActions;
  /**
   * The query filter the visible items came from. Shared rather than re-derived
   * so range selection and the item list can never disagree about which query
   * they are operating on.
   */
  filter: GalleryItemsFilter;
  itemActions: GalleryItemActions;
  /** The infinite window is full and the board holds images it cannot reach. */
  isWindowTruncated: boolean;
  /** Everything on hand — strip first, then the listing, without repeats — for lookups by key. */
  loadedItems: GalleryItem[];
  /** Main Gallery's absolute page slots. Other Gallery surfaces continue to use a dense loaded projection. */
  sparseListing?: GallerySparseListing;
  /** Reports the grid's virtual item range so only intersecting page queries stay subscribed. */
  setVisibleRange?: (range: { endIndexExclusive: number; startIndex: number }) => void;
  starredStrip: GalleryStarredStrip;
  projectName: string;
  /** Placement, used only to scope cached viewport measurements. */
  region: GalleryWidgetProps['region'];
  runtime: GalleryWidgetRuntime;
}

interface SelectionStarSnapshot {
  identity: string;
  starredByKey: ReadonlyMap<string, boolean>;
}

interface SelectionStarStore {
  getSnapshot: () => SelectionStarSnapshot;
  subscribe: (listener: () => void) => () => void;
  sync: (identity: string, selectedKeys: readonly string[], loadedItems: readonly GalleryItem[]) => void;
}

const createSelectionStarStore = (): SelectionStarStore => {
  let snapshot: SelectionStarSnapshot = { identity: '', starredByKey: new Map() };
  const listeners = new Set<() => void>();

  return {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
    sync: (identity, selectedKeys, loadedItems) => {
      const selectedKeySet = new Set(selectedKeys);
      const starredByKey = new Map(snapshot.identity === identity ? snapshot.starredByKey : []);

      for (const item of loadedItems) {
        const key = toGalleryItemKey(item);

        if (selectedKeySet.has(key)) {
          starredByKey.set(key, item.starred);
        }
      }

      if (
        snapshot.identity === identity &&
        starredByKey.size === snapshot.starredByKey.size &&
        [...starredByKey].every(([key, starred]) => snapshot.starredByKey.get(key) === starred)
      ) {
        return;
      }

      snapshot = { identity, starredByKey };
      listeners.forEach((listener) => listener());
    },
  };
};

/** Retains star flags for the current selection while sparse listing pages leave the loaded window. */
export const useGallerySelectionStarred = (
  selectedItems: readonly GalleryItemRef[],
  loadedItems: readonly GalleryItem[]
): boolean => {
  const selectedKeys = selectedItems.map(toGalleryItemKey);
  const identity = JSON.stringify(selectedKeys);
  const store = useMemo(() => createSelectionStarStore(), []);
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  useEffect(() => {
    store.sync(identity, selectedKeys, loadedItems);
  }, [identity, loadedItems, selectedKeys, store]);

  const loadedStarred = new Map(loadedItems.map((item) => [toGalleryItemKey(item), item.starred]));
  const knownStarred = snapshot.identity === identity ? snapshot.starredByKey : new Map<string, boolean>();

  return selectedKeys.some((key) => !(loadedStarred.get(key) ?? knownStarred.get(key) ?? false));
};

export const GalleryWidgetContext = createContext<GalleryWidgetContextValue | null>(null);

export const useGalleryWidget = (): GalleryWidgetContextValue => {
  const value = use(GalleryWidgetContext);

  if (!value) {
    throw new Error('useGalleryWidget must be used within the gallery widget.');
  }

  return value;
};
