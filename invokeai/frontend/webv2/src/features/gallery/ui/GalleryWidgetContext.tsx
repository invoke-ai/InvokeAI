import type { GalleryImageItem, GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageReport } from '@features/gallery/core/mediaUsage';
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GallerySearchScope, GallerySettings } from '@features/gallery/core/settings';
import type { GalleryGeneralAccess, GalleryView } from '@features/gallery/core/types';
import type { GalleryItemsFilter } from '@features/gallery/data/queries';
import type { GalleryItemStarPatchEvent } from '@features/gallery/data/queryCache';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { subscribeGalleryItemStarPatches } from '@features/gallery/data/queryCache';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useQueryClient } from '@tanstack/react-query';
import { createContext, use, useEffect, useMemo, useSyncExternalStore } from 'react';

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
  setStarredOnly: (starredOnly: boolean) => void;
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

/** Empty whenever it does not apply. */
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

interface SelectionStarSnapshot {
  identity: string;
  starredByKey: ReadonlyMap<string, boolean>;
}

interface SelectionStarStore {
  getSnapshot: () => SelectionStarSnapshot;
  reconcilePatch: (event: GalleryItemStarPatchEvent) => void;
  subscribe: (listener: () => void) => () => void;
  sync: (identity: string, selectedKeys: readonly string[], loadedItems: readonly GalleryItem[]) => void;
}

const createSelectionStarStore = (): SelectionStarStore => {
  let snapshot: SelectionStarSnapshot = { identity: '', starredByKey: new Map() };
  let selectedKeySet: ReadonlySet<string> = new Set();
  // Only the rollback of the latest patch that set a key restores it.
  let patchedFrom = new Map<string, { patchId: number; starred: boolean | undefined }>();
  const listeners = new Set<() => void>();
  const publish = (identity: string, starredByKey: ReadonlyMap<string, boolean>) => {
    if (
      snapshot.identity === identity &&
      starredByKey.size === snapshot.starredByKey.size &&
      [...starredByKey].every(([key, starred]) => snapshot.starredByKey.get(key) === starred)
    ) {
      return;
    }

    snapshot = { identity, starredByKey };
    listeners.forEach((listener) => listener());
  };

  return {
    getSnapshot: () => snapshot,
    reconcilePatch: (event) => {
      const starredByKey = new Map(snapshot.starredByKey);

      if (event.kind === 'apply') {
        for (const key of event.itemKeys) {
          if (selectedKeySet.has(key)) {
            patchedFrom.set(key, { patchId: event.patchId, starred: starredByKey.get(key) });
            starredByKey.set(key, event.starred);
          }
        }
      } else {
        for (const [key, { patchId, starred }] of patchedFrom) {
          if (patchId !== event.patchId) {
            continue;
          }

          patchedFrom.delete(key);

          if (starred === undefined) {
            starredByKey.delete(key);
          } else {
            starredByKey.set(key, starred);
          }
        }
      }

      publish(snapshot.identity, starredByKey);
    },
    subscribe: (listener) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
    sync: (identity, selectedKeys, loadedItems) => {
      selectedKeySet = new Set(selectedKeys);

      const isStillSelected = ([key]: [string, unknown]) => selectedKeySet.has(key);
      const starredByKey = new Map([...snapshot.starredByKey].filter(isStillSelected));

      if (snapshot.identity !== identity) {
        patchedFrom = new Map([...patchedFrom].filter(isStillSelected));
      }

      for (const item of loadedItems) {
        const key = toGalleryItemKey(item);

        if (selectedKeySet.has(key)) {
          starredByKey.set(key, item.starred);
        }
      }

      publish(identity, starredByKey);
    },
  };
};

/** Follows star patches for selected items no page holds. */
export const useGallerySelectionStarred = (
  selectedItems: readonly GalleryItemRef[],
  loadedItems: readonly GalleryItem[]
): boolean => {
  const { filter, gallery } = useGalleryWidget();
  const selectedKeys = useMemo(() => selectedItems.map(toGalleryItemKey), [selectedItems]);
  const identity = useMemo(() => JSON.stringify(selectedKeys), [selectedKeys]);
  const queryClient = useQueryClient();
  const store = useMemo(() => createSelectionStarStore(), []);
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  useMountEffect(() => subscribeGalleryItemStarPatches(queryClient, store.reconcilePatch));

  useEffect(() => {
    store.sync(identity, selectedKeys, loadedItems);
  }, [identity, loadedItems, selectedKeys, store]);

  const loadedStarred = useMemo(
    () => new Map(loadedItems.map((item) => [toGalleryItemKey(item), item.starred])),
    [loadedItems]
  );
  // Starred-only listings imply starred members only when the selection was made in one.
  const unknownStarred = filter.starred === true && gallery.selectionStarredOnly;

  return selectedKeys.some((key) => !(loadedStarred.get(key) ?? snapshot.starredByKey.get(key) ?? unknownStarred));
};

export const GalleryWidgetContext = createContext<GalleryWidgetContextValue | null>(null);

export const useGalleryWidget = (): GalleryWidgetContextValue => {
  const value = use(GalleryWidgetContext);

  if (!value) {
    throw new Error('useGalleryWidget must be used within the gallery widget.');
  }

  return value;
};
