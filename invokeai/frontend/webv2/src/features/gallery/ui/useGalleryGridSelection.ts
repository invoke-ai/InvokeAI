import type { GalleryItem, GalleryItemRef } from '@features/gallery/core/items';

import {
  isGalleryImageItem,
  parseGalleryItemKey,
  toGalleryItemKey,
  toGalleryItemRef,
} from '@features/gallery/core/items';
import { isDateBoardId, type GalleryItemNames } from '@features/gallery/data/backend';
import { galleryItemNamesOptions } from '@features/gallery/data/queries';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo, useRef, useState, type MouseEvent } from 'react';

import type { GalleryItemContextMenuTarget } from './GalleryUiContext';

import { useGalleryWidget } from './GalleryWidgetContext';

const getGalleryItemRange = (
  orderedRefs: readonly GalleryItemRef[],
  anchorItemKey: string,
  targetItemKey: string
): GalleryItemRef[] | null => {
  const anchorIndex = orderedRefs.findIndex((ref) => toGalleryItemKey(ref) === anchorItemKey);
  const targetIndex = orderedRefs.findIndex((ref) => toGalleryItemKey(ref) === targetItemKey);

  if (anchorIndex === -1 || targetIndex === -1) {
    return null;
  }

  const start = Math.min(anchorIndex, targetIndex);
  const end = Math.max(anchorIndex, targetIndex);

  return orderedRefs.slice(start, end + 1);
};

/**
 * Click semantics for grid tiles — plain select, shift-range, ctrl/cmd-toggle,
 * alt-compare — plus the context-menu target.
 *
 * Range selection can span beyond the loaded window, so it fetches the full
 * ordered name list. That is asynchronous, and the user can keep interacting
 * while it is in flight, hence the captured-context guard: a range only applies
 * if the account, the query filter, and the anchor are all still what they were
 * when the click happened.
 */
export const useGalleryGridSelection = () => {
  const { actions, filter, gallery } = useGalleryWidget();
  const queryClient = useQueryClient();
  const [contextMenuTarget, setContextMenuTarget] = useState<GalleryItemContextMenuTarget | null>(null);

  const selectedItemKeys = useMemo(() => new Set(gallery.selectedItemKeys), [gallery.selectedItemKeys]);
  const selectedItemRefs = useMemo(() => gallery.selectedItemKeys.map(parseGalleryItemKey), [gallery.selectedItemKeys]);

  const filterIdentity = useMemo(() => JSON.stringify(filter), [filter]);
  const rangeInteractionContextRef = useRef({ filterIdentity, selectedItemKey: gallery.selectedItemKey });

  const syncRangeInteractionContext = useCallback(
    (node: HTMLDivElement | null) => {
      if (node) {
        rangeInteractionContextRef.current = { filterIdentity, selectedItemKey: gallery.selectedItemKey };
      }
    },
    [filterIdentity, gallery.selectedItemKey]
  );

  const activeContextMenuTarget = useMemo(() => {
    const primaryTargetItem = contextMenuTarget?.items[0];

    if (!primaryTargetItem) {
      return contextMenuTarget;
    }

    const primaryTargetItemKey = toGalleryItemKey(primaryTargetItem);

    return gallery.items.some((item) => toGalleryItemKey(item) === primaryTargetItemKey) ? contextMenuTarget : null;
  }, [contextMenuTarget, gallery.items]);

  const selectItemRange = useCallback(
    async (item: GalleryItem) => {
      const owner = captureAccountScope();
      const capturedContext = rangeInteractionContextRef.current;
      const anchorItemKey = capturedContext.selectedItemKey;
      const targetItemKey = toGalleryItemKey(item);

      if (!anchorItemKey) {
        actions.selectItem(item);
        return;
      }

      const isInteractionCurrent = () =>
        isAccountScopeCurrent(owner) &&
        rangeInteractionContextRef.current.filterIdentity === capturedContext.filterIdentity &&
        rangeInteractionContextRef.current.selectedItemKey === capturedContext.selectedItemKey;
      const selectFromRefs = (refs: readonly GalleryItemRef[]): boolean => {
        const range = getGalleryItemRange(refs, anchorItemKey, targetItemKey);

        if (!range) {
          return false;
        }

        actions.selectItemRange(range, item);
        return true;
      };
      const materializedRefs = gallery.items.map(toGalleryItemRef);
      const namesOptions = galleryItemNamesOptions(filter);

      try {
        const orderedRefs = isDateBoardId(filter.boardId)
          ? queryClient.getQueryData<GalleryItemNames>(namesOptions.queryKey)?.items
          : (await queryClient.fetchQuery(namesOptions)).items;

        if (!isInteractionCurrent()) {
          return;
        }

        if (orderedRefs && selectFromRefs(orderedRefs)) {
          return;
        }
      } catch {
        if (!isInteractionCurrent()) {
          return;
        }
      }

      if (!selectFromRefs(materializedRefs)) {
        actions.selectItem(item);
      }
    },
    [actions, filter, gallery.items, queryClient]
  );

  const handleThumbnailClick = useCallback(
    (item: GalleryItem, event: MouseEvent) => {
      if (event.shiftKey) {
        void selectItemRange(item);
        return;
      }

      if (event.altKey && isGalleryImageItem(item)) {
        actions.setCompareItem(item);
        return;
      }

      if (event.ctrlKey || event.metaKey) {
        const itemKey = toGalleryItemKey(item);
        const remainingItemKeys = gallery.selectedItemKeys.filter((key) => key !== itemKey);
        const nextPrimaryItem =
          gallery.selectedItemKey === itemKey
            ? (gallery.items.find(
                (candidate) => toGalleryItemKey(candidate) === remainingItemKeys[remainingItemKeys.length - 1]
              ) ?? null)
            : null;

        actions.toggleItemInSelection(item, nextPrimaryItem);
      } else {
        actions.selectItem(item);
      }
    },
    [actions, gallery.items, gallery.selectedItemKey, gallery.selectedItemKeys, selectItemRange]
  );

  const handleThumbnailContextMenu = useCallback(
    (item: GalleryItem, x: number, y: number) => {
      const itemKey = toGalleryItemKey(item);

      if (selectedItemKeys.has(itemKey) && selectedItemKeys.size > 1) {
        const selectionItems = [
          item,
          ...gallery.items.filter(
            (candidate) => toGalleryItemKey(candidate) !== itemKey && selectedItemKeys.has(toGalleryItemKey(candidate))
          ),
        ];

        setContextMenuTarget({ itemRefs: selectedItemRefs, items: selectionItems, x, y });
        return;
      }

      setContextMenuTarget({ itemRefs: [toGalleryItemRef(item)], items: [item], x, y });
    },
    [gallery.items, selectedItemKeys, selectedItemRefs]
  );

  const getDragItems = useCallback(
    (item: GalleryItem): GalleryItemRef[] => {
      const itemKey = toGalleryItemKey(item);

      if (selectedItemKeys.has(itemKey) && selectedItemKeys.size > 1) {
        return selectedItemRefs;
      }

      return [toGalleryItemRef(item)];
    },
    [selectedItemKeys, selectedItemRefs]
  );

  const handleCloseContextMenu = useCallback(() => setContextMenuTarget(null), []);

  /** Falls back to the primary selection so hotkeys work before a multi-select. */
  const actionSelectionRefs = useMemo(
    () =>
      selectedItemRefs.length > 0
        ? selectedItemRefs
        : gallery.selectedItemKey
          ? [parseGalleryItemKey(gallery.selectedItemKey)]
          : [],
    [gallery.selectedItemKey, selectedItemRefs]
  );

  return {
    actionSelectionRefs,
    activeContextMenuTarget,
    getDragItems,
    handleCloseContextMenu,
    handleThumbnailClick,
    handleThumbnailContextMenu,
    selectedItemKeys,
    syncRangeInteractionContext,
  };
};
