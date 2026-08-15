import type { GalleryImageItem, GalleryItem, GalleryItemKey, GalleryView } from '@features/gallery';
import type {
  GalleryQueuePlaceholder,
  GalleryItemsPage,
  getGallerySelectedImageQuery,
} from '@features/gallery/contracts';
import type { QueueItem } from '@features/queue/contracts';
import type { InfiniteData } from '@tanstack/react-query';
import type { KeyboardEvent } from 'react';

import { compareGalleryItems, toGalleryItemKey } from '@features/gallery/contracts';
import {
  flattenGalleryItemsData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryItemsInfiniteOptions,
} from '@features/gallery/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

import type { PreviewNavigationItem } from './previewNavigation';

import {
  getPreviewNavigationCursor,
  getPreviewNavigationSequence,
  getPreviewNavigationTarget,
} from './previewNavigation';

/**
 * Everything behind the preview's left/right stepping, in one place: the board
 * items query, the local/backend merge, the sequence + cursor, the navigate
 * action with its boundary-page fetch, and the neighbor prefetch. The view
 * consumes the result; `previewNavigation.ts` keeps the pure sequence math.
 *
 * Gallery selection and the live-follow preference remain the sources of
 * truth; nothing here stores a cursor.
 */

const EMPTY_PREVIEW_ITEMS: GalleryItem[] = [];

const flattenPreviewItems = (data: InfiniteData<GalleryItemsPage, number> | undefined): GalleryItem[] =>
  flattenGalleryItemsData(data);

const getOrderedPreviewItems = (
  items: GalleryItem[],
  imageOrderDir: 'ASC' | 'DESC',
  starredFirst: boolean,
  inputOrder: 'display' | 'newest-first'
): GalleryItem[] =>
  items
    .map((item, index) => ({ index, item }))
    .sort((a, b) => {
      const canonicalOrder = compareGalleryItems(a.item, b.item, { orderDir: imageOrderDir, starredFirst });

      if (canonicalOrder !== 0) {
        return canonicalOrder;
      }

      return inputOrder === 'newest-first' && imageOrderDir === 'ASC' ? b.index - a.index : a.index - b.index;
    })
    .map(({ item }) => item);

/**
 * Which gallery tab an item belongs to. Mirrors the category split the
 * gallery filters on: `general` is a gallery image, everything else (canvas
 * pixels, control layers, uploads) is an asset.
 */
const getItemGalleryView = (item: GalleryItem): GalleryView => (item.category === 'general' ? 'images' : 'assets');

const getOrderedLocalItems = ({
  boardId,
  galleryView,
  items,
  imageOrderDir,
  starredFirst,
}: {
  boardId: string;
  galleryView: GalleryView;
  items: GalleryItem[];
  imageOrderDir: 'ASC' | 'DESC';
  starredFirst: boolean;
}): GalleryItem[] =>
  getOrderedPreviewItems(
    items.filter((item) => item.boardId === boardId && getItemGalleryView(item) === galleryView),
    imageOrderDir,
    starredFirst,
    'newest-first'
  );

export const mergePreviewBoardItems = (
  backendItems: GalleryItem[],
  localItems: GalleryItem[],
  imageOrderDir: 'ASC' | 'DESC',
  starredFirst: boolean
): GalleryItem[] => {
  const backendKeys = new Set(backendItems.map(toGalleryItemKey));
  const missingLocalItems = localItems.filter((item) => !backendKeys.has(toGalleryItemKey(item)));

  if (missingLocalItems.length === 0) {
    return backendItems.slice(0, GALLERY_MAX_ROWS);
  }

  return getOrderedPreviewItems([...backendItems, ...missingLocalItems], imageOrderDir, starredFirst, 'display').slice(
    0,
    GALLERY_MAX_ROWS
  );
};

export interface PreviewNavigationState {
  boardItems: GalleryItem[];
  handleNavigationKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  isLoadingBoard: boolean;
  navigate: (offset: -1 | 1) => void;
  navigationCursor: number;
  /** Identity of the backing query — the action context's filter identity. */
  navigationQueryKey: string;
  navigationSequence: PreviewNavigationItem<GalleryItem>[];
  selectPreviewItem: (item: GalleryItem) => void;
}

export const usePreviewNavigation = ({
  activePlaceholder,
  enableLiveFollow,
  imageOrderDir,
  isComparing,
  localItems,
  queueItems,
  selectGalleryItem,
  selectedImageQuery,
  selectedItem,
  selectedItemKey,
  shouldFollowLive,
  starredFirst,
}: {
  /** The live slot from getGalleryGenerationSequence, or null. */
  activePlaceholder: GalleryQueuePlaceholder | null;
  /** Turns the live-follow preference back on (stepping onto the placeholder). */
  enableLiveFollow: () => void;
  /** The gallery's own sort settings, used while following live. */
  imageOrderDir: 'ASC' | 'DESC';
  isComparing: boolean;
  /** Recent local generations, already normalized to gallery items. */
  localItems: GalleryImageItem[];
  queueItems: QueueItem[];
  selectGalleryItem: (item: GalleryItem, selectionPage: number) => void;
  selectedImageQuery: ReturnType<typeof getGallerySelectedImageQuery>;
  selectedItem: GalleryItem | null;
  selectedItemKey: GalleryItemKey | null;
  shouldFollowLive: boolean;
  starredFirst: boolean;
}): PreviewNavigationState => {
  const hasSelectedItem = selectedItem !== null;
  const selectedImageSearch = useMemo(
    () => parseDateTokens(selectedImageQuery.searchTerm),
    [selectedImageQuery.searchTerm]
  );
  const navigationBoardId =
    shouldFollowLive && activePlaceholder ? activePlaceholder.boardId : selectedImageQuery.boardId;
  const navigationGalleryView = shouldFollowLive ? 'images' : selectedImageQuery.galleryView;
  const navigationOrderDir = shouldFollowLive ? imageOrderDir : selectedImageQuery.imageOrderDir;
  const navigationStarredFirst = shouldFollowLive ? starredFirst : selectedImageQuery.starredFirst;
  const hasNavigationContext = shouldFollowLive || hasSelectedItem;
  const navigationContextKey = `${shouldFollowLive}:${selectedItemKey ?? ''}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationStarredFirst}:${selectedImageQuery.paginationMode}:${selectedImageQuery.page}:${selectedImageQuery.searchTerm}`;
  const navigationQueryKey = `${shouldFollowLive}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationStarredFirst}:${selectedImageQuery.paginationMode}:${selectedImageQuery.searchTerm}`;

  // Lets a boundary fetch that resolves after the user has moved on compare the
  // context it started in against the one now on screen, and drop its stale
  // result. Written from a LAYOUT effect: layout effects run synchronously
  // inside the commit, so no promise continuation can observe the new UI with
  // the old key — a passive effect leaves a post-paint gap where exactly that
  // interleaving happens. (Render-phase ref writes are rejected by the
  // compiler, and an effect event cannot be called from a promise
  // continuation.)
  const navigationContextKeyRef = useRef(navigationContextKey);

  useLayoutEffect(() => {
    navigationContextKeyRef.current = navigationContextKey;
  }, [navigationContextKey]);

  // A paginated navigation stays anchored to the page the preview opened on,
  // and re-anchors only when the underlying query identity changes. Derived
  // state rather than a ref so the compiler can see the dependency.
  const [navigationAnchor, setNavigationAnchor] = useState({
    page: selectedImageQuery.page,
    queryKey: navigationQueryKey,
  });
  const hasStaleNavigationAnchor = navigationAnchor.queryKey !== navigationQueryKey;

  if (hasStaleNavigationAnchor) {
    setNavigationAnchor({ page: selectedImageQuery.page, queryKey: navigationQueryKey });
  }

  const navigationAnchorPage = hasStaleNavigationAnchor ? selectedImageQuery.page : navigationAnchor.page;

  const {
    data: boardItemsData,
    fetchNextPage: fetchNextBoardItemsPage,
    fetchPreviousPage: fetchPreviousBoardItemsPage,
    hasNextPage: hasNextBoardItemsPage,
    hasPreviousPage: hasPreviousBoardItemsPage,
    isFetching: isFetchingBoardItems,
    isFetchingNextPage: isFetchingNextBoardItemsPage,
    isFetchingPreviousPage: isFetchingPreviousBoardItemsPage,
  } = useInfiniteQuery({
    ...galleryItemsInfiniteOptions(
      {
        boardId: navigationBoardId,
        createdFrom: shouldFollowLive ? undefined : selectedImageSearch.range?.from,
        createdTo: shouldFollowLive ? undefined : selectedImageSearch.range?.to,
        galleryView: navigationGalleryView,
        orderDir: navigationOrderDir,
        searchTerm: shouldFollowLive ? '' : selectedImageSearch.text,
        starredFirst: navigationStarredFirst,
      },
      !shouldFollowLive && selectedImageQuery.paginationMode === 'paginated'
        ? { kind: 'anchor', offset: navigationAnchorPage * GALLERY_PAGE_SIZE }
        : { kind: 'infinite' }
    ),
    enabled: hasNavigationContext,
  });

  const selectPreviewItem = useCallback(
    (item: GalleryItem) => {
      const itemKey = toGalleryItemKey(item);
      const pageIndex = boardItemsData?.pages.findIndex((page) =>
        page.items.some((candidate) => toGalleryItemKey(candidate) === itemKey)
      );
      const pageParam = pageIndex === undefined || pageIndex < 0 ? undefined : boardItemsData?.pageParams[pageIndex];
      const selectionPage =
        typeof pageParam === 'number' ? Math.floor(pageParam / GALLERY_PAGE_SIZE) : selectedImageQuery.page;

      selectGalleryItem(item, selectionPage);
    },
    [boardItemsData, selectGalleryItem, selectedImageQuery.page]
  );

  const optimisticQueueItemIds = useMemo(
    () =>
      new Set(
        queueItems.filter((item) => item.status === 'pending' || item.status === 'running').map((item) => item.id)
      ),
    [queueItems]
  );
  const navigationLocalItems = useMemo(() => {
    // recentImages exists precisely to bridge the gap between "generation
    // finished" and "the backend list has the row" — the refetch is coalesced
    // and takes time, and dropping a completed batch from the merge for that
    // window made arrow keys skip the images just generated. So local items
    // stay in the sequence unconditionally, with two exceptions where the
    // backend window is a *subset* of the board and the dedupe cannot save us:
    // under an active search (the backend list is search-filtered, local items
    // are not) and in paginated mode (the window anchors mid-board, so settled
    // recents from the board's top would splice in permanently — the gallery
    // grid guards this the same way via `shouldOverlayRecentItems`). There,
    // only in-flight work and the selected item itself may merge.
    const hasActiveSearch =
      !shouldFollowLive && (selectedImageSearch.text.trim() !== '' || selectedImageSearch.range !== undefined);
    const isPaginatedWindow = !shouldFollowLive && selectedImageQuery.paginationMode === 'paginated';

    if (!hasActiveSearch && !isPaginatedWindow) {
      return localItems;
    }

    const refreshingSelectedSourceId =
      !shouldFollowLive && isFetchingBoardItems && selectedItem?.kind === 'image'
        ? selectedItem.sourceQueueItemId
        : null;

    return localItems.filter(
      (item) =>
        (item.sourceQueueItemId !== undefined && optimisticQueueItemIds.has(item.sourceQueueItemId)) ||
        item.sourceQueueItemId === refreshingSelectedSourceId
    );
  }, [
    isFetchingBoardItems,
    localItems,
    optimisticQueueItemIds,
    selectedImageQuery.paginationMode,
    selectedImageSearch,
    selectedItem,
    shouldFollowLive,
  ]);
  const localBoardItems = useMemo(
    () =>
      getOrderedLocalItems({
        boardId: navigationBoardId,
        galleryView: navigationGalleryView,
        items: navigationLocalItems,
        imageOrderDir: navigationOrderDir,
        starredFirst: navigationStarredFirst,
      }),
    [navigationBoardId, navigationGalleryView, navigationLocalItems, navigationOrderDir, navigationStarredFirst]
  );
  const previewLocalBoardItems = useMemo(() => {
    if (
      shouldFollowLive ||
      !selectedItem ||
      localBoardItems.some((item) => toGalleryItemKey(item) === selectedItemKey)
    ) {
      return localBoardItems;
    }

    return [selectedItem, ...localBoardItems];
  }, [localBoardItems, selectedItem, selectedItemKey, shouldFollowLive]);
  const backendBoardItems = useMemo(() => flattenPreviewItems(boardItemsData), [boardItemsData]);
  const boardItems = useMemo(
    () =>
      !hasNavigationContext
        ? EMPTY_PREVIEW_ITEMS
        : mergePreviewBoardItems(backendBoardItems, previewLocalBoardItems, navigationOrderDir, navigationStarredFirst),
    [backendBoardItems, hasNavigationContext, navigationOrderDir, navigationStarredFirst, previewLocalBoardItems]
  );
  const isLoadingBoard = hasNavigationContext && isFetchingBoardItems;
  const navigationSequence = useMemo(
    () =>
      getPreviewNavigationSequence({
        activePlaceholder,
        boardId: navigationBoardId,
        boardImages: boardItems,
        galleryView: navigationGalleryView,
        imageOrderDir: navigationOrderDir,
        starredFirst: navigationStarredFirst,
      }),
    [
      activePlaceholder,
      boardItems,
      navigationBoardId,
      navigationGalleryView,
      navigationOrderDir,
      navigationStarredFirst,
    ]
  );
  const navigationCursor = getPreviewNavigationCursor(navigationSequence, {
    isFollowingLive: shouldFollowLive,
    selectedItemKey,
  });

  // One navigation action shared by the arrow keys and the footer buttons.
  // Compare mode stays inert and never exposes the placeholder.
  const navigate = useCallback(
    (offset: -1 | 1) => {
      if (isComparing) {
        return;
      }

      const target = getPreviewNavigationTarget(navigationSequence, navigationCursor, offset);
      const isAtLoadedBackendBoundary =
        selectedItemKey !== null &&
        (offset === 1
          ? backendBoardItems.at(-1) !== undefined &&
            toGalleryItemKey(backendBoardItems.at(-1)!) === selectedItemKey &&
            hasNextBoardItemsPage
          : backendBoardItems[0] !== undefined &&
            toGalleryItemKey(backendBoardItems[0]) === selectedItemKey &&
            hasPreviousBoardItemsPage);

      if (!isAtLoadedBackendBoundary) {
        if (!target) {
          return;
        }

        if (target.kind === 'item') {
          selectPreviewItem(target.item);
        } else {
          enableLiveFollow();
        }
        return;
      }

      if (offset === 1 ? isFetchingNextBoardItemsPage : isFetchingPreviousBoardItemsPage) {
        return;
      }

      const fetchBoundaryPage = offset === 1 ? fetchNextBoardItemsPage : fetchPreviousBoardItemsPage;

      void fetchBoundaryPage().then((result) => {
        if (result.isError || navigationContextKeyRef.current !== navigationContextKey) {
          return;
        }

        const nextBackendBoardItems = flattenPreviewItems(result.data);
        const nextBoardItems = mergePreviewBoardItems(
          nextBackendBoardItems,
          previewLocalBoardItems,
          navigationOrderDir,
          navigationStarredFirst
        );
        const nextNavigationSequence = getPreviewNavigationSequence({
          activePlaceholder,
          boardId: navigationBoardId,
          boardImages: nextBoardItems,
          galleryView: navigationGalleryView,
          imageOrderDir: navigationOrderDir,
          starredFirst: navigationStarredFirst,
        });
        const nextNavigationCursor = getPreviewNavigationCursor(nextNavigationSequence, {
          isFollowingLive: shouldFollowLive,
          selectedItemKey,
        });
        const nextTarget = getPreviewNavigationTarget(nextNavigationSequence, nextNavigationCursor, offset);

        if (nextTarget?.kind === 'item') {
          selectPreviewItem(nextTarget.item);
        } else if (nextTarget?.kind === 'placeholder') {
          enableLiveFollow();
        }
      });
    },
    [
      activePlaceholder,
      backendBoardItems,
      enableLiveFollow,
      fetchNextBoardItemsPage,
      fetchPreviousBoardItemsPage,
      hasNextBoardItemsPage,
      hasPreviousBoardItemsPage,
      isComparing,
      isFetchingNextBoardItemsPage,
      isFetchingPreviousBoardItemsPage,
      navigationBoardId,
      navigationContextKey,
      navigationCursor,
      navigationGalleryView,
      navigationOrderDir,
      navigationSequence,
      navigationStarredFirst,
      previewLocalBoardItems,
      selectedItemKey,
      selectPreviewItem,
      shouldFollowLive,
    ]
  );

  const handleNavigationKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.target instanceof Element && event.target.closest('video')) {
        return;
      }

      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
        return;
      }

      if (isComparing) {
        return;
      }

      // stopPropagation keeps the widget hotkey runtime from handling the same
      // arrow press a second time.
      event.preventDefault();
      event.stopPropagation();
      navigate(event.key === 'ArrowLeft' ? -1 : 1);
    },
    [isComparing, navigate]
  );

  // Warm the browser cache for the sequence neighbors so arrow-key navigation
  // swaps without a decode flash.
  const previousNeighbor = navigationSequence[navigationCursor - 1];
  const nextNeighbor = navigationSequence[navigationCursor + 1];
  const previousNeighborUrl =
    previousNeighbor?.kind === 'item' && previousNeighbor.item.kind === 'image' ? previousNeighbor.item.fullUrl : null;
  const nextNeighborUrl =
    nextNeighbor?.kind === 'item' && nextNeighbor.item.kind === 'image' ? nextNeighbor.item.fullUrl : null;

  useEffect(() => {
    [previousNeighborUrl, nextNeighborUrl].forEach((url) => {
      if (url) {
        new Image().src = url;
      }
    });
  }, [nextNeighborUrl, previousNeighborUrl]);

  return {
    boardItems,
    handleNavigationKeyDown,
    isLoadingBoard,
    navigate,
    navigationCursor,
    navigationQueryKey,
    navigationSequence,
    selectPreviewItem,
  };
};
