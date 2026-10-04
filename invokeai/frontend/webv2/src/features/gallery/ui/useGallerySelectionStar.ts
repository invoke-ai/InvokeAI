import type { GalleryItem, GalleryItemRef } from '@features/gallery/core/items';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { useEffect, useMemo, useSyncExternalStore } from 'react';

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
