import type { GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/core/items';

import {
  isGalleryImageItem,
  parseGalleryItemKey,
  toGalleryItemKey,
  toGalleryItemRef,
} from '@features/gallery/core/items';
import { getGalleryItemByRef, isDateBoardId, type GalleryItemNames } from '@features/gallery/data/backend';
import { galleryItemNamesOptions } from '@features/gallery/data/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
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
 * Async range selection fetches beyond the loaded window; apply only if account, filter, and anchor still match
 * the captured context.
 */
export const useGalleryGridSelection = ({
  getSelectionPage,
}: {
  /** The sparse page stamp for a loaded listing item; selections without one stamp the grid's page. */
  getSelectionPage?: (item: GalleryItem) => number | undefined;
} = {}) => {
  // `loadedItems` includes the strip, whose starred items the listing window
  // may not hold; the context menu and ctrl-toggle must resolve those too.
  const { actions, filter, gallery, loadedItems, starredStrip } = useGalleryWidget();
  const queryClient = useQueryClient();
  const [contextMenuTarget, setContextMenuTarget] = useState<GalleryItemContextMenuTarget | null>(null);

  const selectedItemKeys = useMemo(() => new Set(gallery.selectedItemKeys), [gallery.selectedItemKeys]);
  const selectedItemRefs = useMemo(() => gallery.selectedItemKeys.map(parseGalleryItemKey), [gallery.selectedItemKeys]);

  const filterIdentity = useMemo(() => JSON.stringify(filter), [filter]);
  const selectionIdentity = gallery.selectedItemKeys.join('\n');
  const rangeInteractionContextRef = useRef({
    filterIdentity,
    selectedItemKey: gallery.primarySelectedItemKey,
    selectionIdentity,
  });

  // Next-primary lookups end with the grid; a late result could otherwise toggle a selection changed elsewhere.
  const nextPrimaryLookupsRef = useRef(new Set<AbortController>());
  useMountEffect(() => () => {
    nextPrimaryLookupsRef.current.forEach((controller) => controller.abort());
  });

  const syncRangeInteractionContext = useCallback(
    (node: HTMLDivElement | null) => {
      if (node) {
        rangeInteractionContextRef.current = {
          filterIdentity,
          selectedItemKey: gallery.primarySelectedItemKey,
          selectionIdentity,
        };
      }
    },
    [filterIdentity, gallery.primarySelectedItemKey, selectionIdentity]
  );

  const activeContextMenuTarget = useMemo(() => {
    const primaryTargetItem = contextMenuTarget?.items[0];

    if (!primaryTargetItem) {
      return contextMenuTarget;
    }

    const primaryTargetItemKey = toGalleryItemKey(primaryTargetItem);

    return loadedItems.some((item) => toGalleryItemKey(item) === primaryTargetItemKey) ? contextMenuTarget : null;
  }, [contextMenuTarget, loadedItems]);

  /** Selects from the anchor (the primary selection unless a keyboard range names its own) through `item`. */
  const selectItemRange = useCallback(
    async (
      item: GalleryItem,
      {
        anchorKey,
        isFocusCurrent,
        isNavigationCurrent,
        selectionPage,
      }: {
        anchorKey?: GalleryItemKey | null;
        isFocusCurrent?: () => boolean;
        isNavigationCurrent?: () => boolean;
        selectionPage?: number;
      } = {}
    ) => {
      const owner = captureAccountScope();
      const capturedContext = rangeInteractionContextRef.current;
      const anchorItemKey = anchorKey ?? capturedContext.selectedItemKey;
      const targetItemKey = toGalleryItemKey(item);
      const isInteractionContextCurrent = () =>
        isAccountScopeCurrent(owner) &&
        rangeInteractionContextRef.current.filterIdentity === capturedContext.filterIdentity &&
        rangeInteractionContextRef.current.selectedItemKey === capturedContext.selectedItemKey;
      const isInteractionCurrent = (requireFocusedTarget = false) =>
        isNavigationCurrent?.() !== false &&
        (!requireFocusedTarget || isFocusCurrent?.() !== false) &&
        isInteractionContextCurrent();
      const selectSingleItem = () => {
        if (selectionPage === undefined) {
          actions.selectItem(item);
        } else {
          actions.selectItem(item, selectionPage);
        }
      };

      if (!isInteractionContextCurrent()) {
        return;
      }

      if (!anchorItemKey) {
        selectSingleItem();
        return;
      }

      const selectFromRefs = (refs: readonly GalleryItemRef[]): boolean => {
        const range = getGalleryItemRange(refs, anchorItemKey, targetItemKey);

        if (!range) {
          return false;
        }

        if (selectionPage === undefined) {
          actions.selectItemRange(range, item);
        } else {
          actions.selectItemRange(range, item, selectionPage);
        }
        return true;
      };
      const materializedRefs = gallery.items.map(toGalleryItemRef);
      const namesOptions = galleryItemNamesOptions(filter);
      const usesSynchronousNames = isDateBoardId(filter.boardId);
      let hasAwaitedNames = false;

      try {
        let orderedRefs: readonly GalleryItemRef[] | undefined;
        if (usesSynchronousNames) {
          orderedRefs = queryClient.getQueryData<GalleryItemNames>(namesOptions.queryKey)?.items;
        } else {
          const namesPromise = queryClient.fetchQuery(namesOptions);
          hasAwaitedNames = true;
          orderedRefs = (await namesPromise).items;
        }

        if (!isInteractionCurrent(hasAwaitedNames)) {
          return;
        }

        if (orderedRefs && selectFromRefs(orderedRefs)) {
          return;
        }
      } catch {
        if (!isInteractionCurrent(hasAwaitedNames)) {
          return;
        }
      }

      // The names list describes the listing only; a range inside the strip
      // resolves against the strip's own order.
      if (!isInteractionCurrent(hasAwaitedNames)) {
        return;
      }

      if (!selectFromRefs(materializedRefs) && !selectFromRefs(starredStrip.items.map(toGalleryItemRef))) {
        selectSingleItem();
      }
    },
    [actions, filter, gallery.items, queryClient, starredStrip.items]
  );

  const toggleItem = useCallback(
    (item: GalleryItem, itemSelectionPage = getSelectionPage?.(item)) => {
      const itemKey = toGalleryItemKey(item);
      const remainingItemKeys = gallery.selectedItemKeys.filter((key) => key !== itemKey);
      const isPrimary = gallery.selectedItemKey === itemKey;
      const nextPrimaryKey = isPrimary ? (remainingItemKeys.at(-1) ?? null) : null;
      const loadedNextPrimary =
        nextPrimaryKey === null
          ? null
          : (loadedItems.find((candidate) => toGalleryItemKey(candidate) === nextPrimaryKey) ?? null);
      const toggle = (nextPrimaryItem: GalleryItem | null) => {
        // Stamp whichever item becomes primary where it sits, as a click or range does.
        const selectionPage = isPrimary
          ? nextPrimaryItem
            ? getSelectionPage?.(nextPrimaryItem)
            : undefined
          : itemSelectionPage;

        if (selectionPage === undefined) {
          actions.toggleItemInSelection(item, nextPrimaryItem);
        } else {
          actions.toggleItemInSelection(item, nextPrimaryItem, selectionPage);
        }
      };

      if (nextPrimaryKey === null || loadedNextPrimary) {
        toggle(loadedNextPrimary);
        return;
      }

      // The next primary's page has left the viewport. Without its item the toggle clears the whole selection, so
      // resolve it first. A lookup that fails (the item was deleted elsewhere, or the request failed) still toggles
      // rather than ignoring the click.
      const owner = captureAccountScope();
      const capturedContext = rangeInteractionContextRef.current;
      const controller = new AbortController();
      const signal = AbortSignal.any([owner.signal, controller.signal]);
      const toggleIfCurrent = (nextPrimaryItem: GalleryItem | null) => {
        const current = rangeInteractionContextRef.current;

        if (
          !signal.aborted &&
          isAccountScopeCurrent(owner) &&
          current.filterIdentity === capturedContext.filterIdentity &&
          current.selectedItemKey === capturedContext.selectedItemKey &&
          current.selectionIdentity === capturedContext.selectionIdentity
        ) {
          toggle(nextPrimaryItem);
        }
      };

      nextPrimaryLookupsRef.current.add(controller);
      void getGalleryItemByRef(parseGalleryItemKey(nextPrimaryKey), signal)
        .then(toggleIfCurrent, () => toggleIfCurrent(null))
        .finally(() => nextPrimaryLookupsRef.current.delete(controller));
    },
    [actions, gallery.selectedItemKey, gallery.selectedItemKeys, getSelectionPage, loadedItems]
  );

  const handleThumbnailClick = useCallback(
    (item: GalleryItem, event: MouseEvent, selectionPage?: number) => {
      if (event.shiftKey) {
        void selectItemRange(item, { selectionPage });
        return;
      }

      if (event.altKey && isGalleryImageItem(item)) {
        actions.setCompareItem(item);
        return;
      }

      if (event.ctrlKey || event.metaKey) {
        toggleItem(item, selectionPage);
      } else {
        if (selectionPage === undefined) {
          actions.selectItem(item);
        } else {
          actions.selectItem(item, selectionPage);
        }
      }
    },
    [actions, selectItemRange, toggleItem]
  );

  const handleThumbnailContextMenu = useCallback(
    (item: GalleryItem, x: number, y: number) => {
      const itemKey = toGalleryItemKey(item);

      if (selectedItemKeys.has(itemKey) && selectedItemKeys.size > 1) {
        const selectionItems = [
          item,
          ...loadedItems.filter(
            (candidate) => toGalleryItemKey(candidate) !== itemKey && selectedItemKeys.has(toGalleryItemKey(candidate))
          ),
        ];

        setContextMenuTarget({ itemRefs: selectedItemRefs, items: selectionItems, x, y });
        return;
      }

      setContextMenuTarget({ itemRefs: [toGalleryItemRef(item)], items: [item], x, y });
    },
    [loadedItems, selectedItemKeys, selectedItemRefs]
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
    loadedItems,
    selectedItemKeys,
    selectItemRange,
    syncRangeInteractionContext,
    toggleItem,
  };
};
