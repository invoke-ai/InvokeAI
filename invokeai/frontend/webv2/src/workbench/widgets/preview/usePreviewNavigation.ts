import type { GalleryImageItem, GalleryItem, GalleryItemKey, GalleryView } from '@features/gallery';
import type {
  GalleryItemsPage,
  GalleryNavigationEntry,
  GallerySemanticReference,
  getGallerySelectedImageQuery,
} from '@features/gallery/contracts';
import type { GalleryItemsFilter } from '@features/gallery/queries';
import type { QueueItem, QueueProgressSession } from '@features/queue/contracts';
import type { InfiniteData } from '@tanstack/react-query';
import type { KeyboardEvent } from 'react';

import {
  compareGalleryItems,
  gallerySemanticReferenceKey,
  getGalleryNavigationStep,
  getGallerySessionNavigationKey,
  toGalleryItemKey,
} from '@features/gallery/contracts';
import {
  flattenGalleryItemsData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryItemsInfiniteOptions,
  galleryStarredStripOptions,
} from '@features/gallery/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

/**
 * Own Preview query merging, cursor derivation, boundary fetches, and neighbor prefetch. Follow Gallery's
 * session/starred/list order, but traverse the full bounded starred query beyond the grid's folded rows;
 * selection/follow remain authoritative.
 */

const EMPTY_PREVIEW_ITEMS: GalleryItem[] = [];

const flattenPreviewItems = (data: InfiniteData<GalleryItemsPage, number> | undefined): GalleryItem[] =>
  flattenGalleryItemsData(data);

const getOrderedPreviewItems = (
  items: GalleryItem[],
  imageOrderDir: 'ASC' | 'DESC',
  inputOrder: 'display' | 'newest-first'
): GalleryItem[] =>
  items
    .map((item, index) => ({ index, item }))
    .sort((a, b) => {
      const canonicalOrder = compareGalleryItems(a.item, b.item, { orderDir: imageOrderDir });

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
}: {
  boardId: string;
  galleryView: GalleryView;
  items: GalleryItem[];
  imageOrderDir: 'ASC' | 'DESC';
}): GalleryItem[] =>
  getOrderedPreviewItems(
    items.filter((item) => item.boardId === boardId && getItemGalleryView(item) === galleryView),
    imageOrderDir,
    'newest-first'
  );

export const mergePreviewBoardItems = (
  backendItems: GalleryItem[],
  localItems: GalleryItem[],
  imageOrderDir: 'ASC' | 'DESC',
  { isRanked = false }: { isRanked?: boolean } = {}
): GalleryItem[] => {
  const backendKeys = new Set(backendItems.map(toGalleryItemKey));

  // Preserve ranking order and exclude local generations; retain out-of-ranking selection as a cursor anchor so
  // arrows remain usable.
  if (isRanked) {
    const anchors = localItems.filter((item) => !backendKeys.has(toGalleryItemKey(item)));

    return [...anchors, ...backendItems].slice(0, GALLERY_MAX_ROWS);
  }

  const missingLocalItems = localItems.filter((item) => !backendKeys.has(toGalleryItemKey(item)));

  if (missingLocalItems.length === 0) {
    return backendItems.slice(0, GALLERY_MAX_ROWS);
  }

  return getOrderedPreviewItems([...backendItems, ...missingLocalItems], imageOrderDir, 'display').slice(
    0,
    GALLERY_MAX_ROWS
  );
};

const toItemEntries = (items: readonly GalleryItem[]): GalleryNavigationEntry[] =>
  items.map((item) => ({ item, kind: 'item' }));

/**
 * A step's destination: a saved item, a running session, a page not loaded yet (`more`), or nothing (null).
 */
export type PreviewNeighbor =
  | { kind: 'item'; item: GalleryItem }
  | { kind: 'more' }
  | { kind: 'session'; id: string }
  | null;

export interface PreviewNeighbors {
  next: PreviewNeighbor;
  previous: PreviewNeighbor;
}

const NO_NEIGHBORS: PreviewNeighbors = { next: null, previous: null };

const toNeighbor = (entry: GalleryNavigationEntry | null): PreviewNeighbor =>
  entry === null
    ? null
    : entry.kind === 'item'
      ? { item: entry.item, kind: 'item' }
      : { id: entry.id, kind: 'session' };

export interface PreviewNavigationState {
  /** Every saved item the arrows can reach, in order: the starred strip, then the listing. */
  boardItems: GalleryItem[];
  handleNavigationKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  isLoadingBoard: boolean;
  /** Resolves true once a step was dispatched; false when there was nowhere to go or the step went stale. */
  navigate: (offset: -1 | 1) => Promise<boolean>;
  /** What each step would land on, so a swipe can show it before committing. */
  neighbors: PreviewNeighbors;
  /** The selection's index in `boardItems`; -1 while following live or off the list. */
  navigationCursor: number;
  /** Identity of the backing query — the action context's filter identity. */
  navigationQueryKey: string;
  /** The page a selection of `item` is stamped with — see the action context's `getItemSelectionPage`. */
  getSelectionPage: (item: GalleryItem) => number;
  selectPreviewItem: (item: GalleryItem) => void;
}

export const usePreviewNavigation = ({
  followedSessionId,
  followSession,
  isComparing,
  localItems,
  progressSessions,
  queueItems,
  galleryBoardId,
  galleryPage,
  galleryPaginationMode,
  selectGalleryItem,
  selectedImageQuery,
  selectedItem,
  selectedItemKey,
  semanticQuery,
}: {
  /** The live session on screen, when the preview is following one; the cursor sits on it. */
  followedSessionId: string | null;
  followSession: (sessionId: string) => void;
  /** The board the gallery grid shows; a ranked list ranks within it, as the grid does. */
  galleryBoardId: string;
  /** The page the gallery grid is on; a ranked list mirrors it (see below). */
  galleryPage: number;
  /** The gallery's own pagination mode, likewise mirrored by a ranked list. */
  galleryPaginationMode: 'infinite' | 'paginated';
  isComparing: boolean;
  /** Recent local generations, already normalized to gallery items. */
  localItems: GalleryImageItem[];
  /** The gallery's in-progress tiles, in its order; only running ones can be stepped onto. */
  progressSessions: readonly QueueProgressSession[];
  queueItems: QueueItem[];
  selectGalleryItem: (item: GalleryItem, selectionPage: number) => void;
  selectedImageQuery: ReturnType<typeof getGallerySelectedImageQuery>;
  selectedItem: GalleryItem | null;
  selectedItemKey: GalleryItemKey | null;
  /** The gallery's active similarity search, or null for the board listing. */
  semanticQuery: GallerySemanticReference | null;
}): PreviewNavigationState => {
  const selectedImageSearch = useMemo(
    () => parseDateTokens(selectedImageQuery.searchTerm),
    [selectedImageQuery.searchTerm]
  );
  // A ranked filmstrip follows the gallery's current search, board and paging; a listing follows the selection's.
  const navigationBoardId = semanticQuery ? galleryBoardId : selectedImageQuery.boardId;
  const navigationGalleryView = selectedImageQuery.galleryView;
  const navigationOrderDir = selectedImageQuery.imageOrderDir;
  // The grid partitions: its listing is unstarred-only, with the starred
  // items in the strip above it, unless the starred filter is on.
  const navigationStarredOnly = selectedImageQuery.starredOnly;
  const navigationSemanticQuery = semanticQuery;
  const navigationSemanticKey = gallerySemanticReferenceKey(navigationSemanticQuery);
  // Following live has a cursor too, so the listing loads for the step off it.
  const hasNavigationContext = selectedItem !== null || followedSessionId !== null;
  const navigationContextKey = `${followedSessionId ?? ''}:${selectedItemKey ?? ''}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${selectedImageQuery.paginationMode}:${selectedImageQuery.page}:${selectedImageQuery.searchTerm}:${navigationStarredOnly}:${navigationSemanticKey}`;
  const navigationQueryKey = `${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${selectedImageQuery.paginationMode}:${selectedImageQuery.searchTerm}:${navigationStarredOnly}:${navigationSemanticKey}`;

  // Publish navigation context in layout effect so boundary-fetch continuations cannot observe new UI with a stale
  // fence.
  const navigationContextKeyRef = useRef(navigationContextKey);

  useLayoutEffect(() => {
    navigationContextKeyRef.current = navigationContextKey;
  }, [navigationContextKey]);

  // Keep paginated navigation anchored until query identity changes; expose the dependency as derived state.
  const [navigationAnchor, setNavigationAnchor] = useState({
    page: selectedImageQuery.page,
    queryKey: navigationQueryKey,
  });
  const hasStaleNavigationAnchor = navigationAnchor.queryKey !== navigationQueryKey;

  if (hasStaleNavigationAnchor) {
    setNavigationAnchor({ page: selectedImageQuery.page, queryKey: navigationQueryKey });
  }

  // Only paginated anchors are sticky. Infinite steps preserve their anchor, while external selections must
  // immediately reanchor the window.
  const navigationAnchorPage =
    selectedImageQuery.paginationMode === 'paginated'
      ? hasStaleNavigationAnchor
        ? selectedImageQuery.page
        : navigationAnchor.page
      : selectedImageQuery.page;
  const isPaginatedWindow = selectedImageQuery.paginationMode === 'paginated';
  // Anchor deep navigation at the selection page. Shared infinite windows cannot grow upward without shifting grid
  // content, so retain that anchor and derive mid-board exclusions from it.
  const deepAnchorOffset =
    !isPaginatedWindow &&
    navigationSemanticQuery === null &&
    navigationAnchorPage * GALLERY_PAGE_SIZE >= GALLERY_MAX_ROWS
      ? navigationAnchorPage * GALLERY_PAGE_SIZE
      : 0;

  // Rankings follow grid paging, not stamped board pages, which can address unrelated or empty ranking slices.
  const navigationWindow =
    navigationSemanticQuery !== null
      ? galleryPaginationMode === 'paginated'
        ? ({ kind: 'anchor', offset: galleryPage * GALLERY_PAGE_SIZE } as const)
        : // In infinite mode the grid's page IS its window offset, so mirroring
          // it covers the deep-reveal case below without a separate test.
          ({ kind: 'infinite', offset: galleryPage * GALLERY_PAGE_SIZE } as const)
      : selectedImageQuery.paginationMode === 'paginated'
        ? ({ kind: 'anchor', offset: navigationAnchorPage * GALLERY_PAGE_SIZE } as const)
        : ({ kind: 'infinite', offset: deepAnchorOffset } as const);

  const listingFilter = useMemo(
    (): GalleryItemsFilter => ({
      boardId: navigationBoardId,
      createdFrom: selectedImageSearch.range?.from,
      createdTo: selectedImageSearch.range?.to,
      galleryView: navigationGalleryView,
      orderDir: navigationOrderDir,
      searchTerm: selectedImageSearch.text,
      ...(navigationSemanticQuery ? { semanticQuery: navigationSemanticQuery } : {}),
    }),
    [navigationBoardId, navigationGalleryView, navigationOrderDir, navigationSemanticQuery, selectedImageSearch]
  );

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
    ...galleryItemsInfiniteOptions({ ...listingFilter, starred: navigationStarredOnly }, navigationWindow),
    enabled: hasNavigationContext,
  });

  // Share Gallery's bounded starred strip except for ranked, starred-only, or mid-board windows.
  const hasStrip =
    hasNavigationContext && !navigationStarredOnly && navigationSemanticQuery === null && deepAnchorOffset === 0;
  const { data: stripData } = useQuery({ ...galleryStarredStripOptions(listingFilter), enabled: hasStrip });
  const stripItems = useMemo(() => {
    if (!hasStrip) {
      return EMPTY_PREVIEW_ITEMS;
    }

    const items = stripData?.items ?? EMPTY_PREVIEW_ITEMS;

    // A starred selection beyond the strip's bound (the grid hides it behind
    // "Show all") still belongs to the starred partition: it joins the strip
    // section so the arrows have somewhere to step from.
    return selectedItem?.starred && !items.some((item) => toGalleryItemKey(item) === selectedItemKey)
      ? [...items, selectedItem]
      : items;
  }, [hasStrip, selectedItem, selectedItemKey, stripData]);

  const getSelectionPageIn = useCallback(
    (item: GalleryItem, data: typeof boardItemsData): number => {
      const itemKey = toGalleryItemKey(item);
      const pageIndex = data?.pages.findIndex((page) =>
        page.items.some((candidate) => toGalleryItemKey(candidate) === itemKey)
      );
      const pageParam = pageIndex === undefined || pageIndex < 0 ? undefined : data?.pageParams[pageIndex];
      // Stamp ranked picks with board page zero, never ranking offsets. Infinite picks retain their window anchor
      // to preserve prior pages; items outside the window use the top-of-board context.
      return navigationSemanticQuery !== null
        ? 0
        : selectedImageQuery.paginationMode === 'paginated'
          ? typeof pageParam === 'number'
            ? Math.floor(pageParam / GALLERY_PAGE_SIZE)
            : selectedImageQuery.page
          : typeof pageParam === 'number'
            ? deepAnchorOffset / GALLERY_PAGE_SIZE
            : 0;
    },
    [deepAnchorOffset, navigationSemanticQuery, selectedImageQuery.page, selectedImageQuery.paginationMode]
  );
  const stampSelection = useCallback(
    (item: GalleryItem, data: typeof boardItemsData) => selectGalleryItem(item, getSelectionPageIn(item, data)),
    [getSelectionPageIn, selectGalleryItem]
  );
  const getSelectionPage = useCallback(
    (item: GalleryItem) => getSelectionPageIn(item, boardItemsData),
    [boardItemsData, getSelectionPageIn]
  );
  const selectPreviewItem = useCallback(
    (item: GalleryItem) => stampSelection(item, boardItemsData),
    [boardItemsData, stampSelection]
  );

  const optimisticQueueItemIds = useMemo(
    () =>
      new Set(
        queueItems.filter((item) => item.status === 'pending' || item.status === 'running').map((item) => item.id)
      ),
    [queueItems]
  );
  const navigationLocalItems = useMemo(() => {
    // Keep recent results until listings catch up, except in filtered or mid-board windows where they do not
    // belong. Those windows merge only in-flight work and selection.
    const hasActiveSearch =
      navigationStarredOnly || selectedImageSearch.text.trim() !== '' || selectedImageSearch.range !== undefined;

    if (!hasActiveSearch && !isPaginatedWindow && deepAnchorOffset === 0) {
      return localItems;
    }

    const refreshingSelectedSourceId =
      isFetchingBoardItems && selectedItem?.kind === 'image' ? selectedItem.sourceQueueItemId : null;

    return localItems.filter(
      (item) =>
        (item.sourceQueueItemId !== undefined && optimisticQueueItemIds.has(item.sourceQueueItemId)) ||
        item.sourceQueueItemId === refreshingSelectedSourceId
    );
  }, [
    deepAnchorOffset,
    isFetchingBoardItems,
    isPaginatedWindow,
    localItems,
    navigationStarredOnly,
    optimisticQueueItemIds,
    selectedImageSearch,
    selectedItem,
  ]);
  const localBoardItems = useMemo(
    () =>
      getOrderedLocalItems({
        boardId: navigationBoardId,
        galleryView: navigationGalleryView,
        items: navigationLocalItems,
        imageOrderDir: navigationOrderDir,
      }),
    [navigationBoardId, navigationGalleryView, navigationLocalItems, navigationOrderDir]
  );
  const previewLocalBoardItems = useMemo(() => {
    // A recent starred since it landed has moved to the strip.
    const listingLocalItems = hasStrip ? localBoardItems.filter((item) => !item.starred) : localBoardItems;

    if (
      !selectedItem ||
      (hasStrip && selectedItem.starred) ||
      listingLocalItems.some((item) => toGalleryItemKey(item) === selectedItemKey)
    ) {
      return listingLocalItems;
    }

    return [selectedItem, ...listingLocalItems];
  }, [hasStrip, localBoardItems, selectedItem, selectedItemKey]);
  const backendBoardItems = useMemo(() => flattenPreviewItems(boardItemsData), [boardItemsData]);
  // Rankings accept only selection as cursor anchor, never board recents.
  const previewMergeItems = useMemo(
    () =>
      navigationSemanticQuery === null ? previewLocalBoardItems : selectedItem ? [selectedItem] : EMPTY_PREVIEW_ITEMS,
    [navigationSemanticQuery, previewLocalBoardItems, selectedItem]
  );
  // The listing and the strip refetch independently, so an item just starred
  // can sit on both sides for a moment; the strip keeps it.
  const stripKeys = useMemo(() => new Set(stripItems.map(toGalleryItemKey)), [stripItems]);
  const mergeListingItems = useCallback(
    (backendItems: GalleryItem[]) =>
      mergePreviewBoardItems(backendItems, previewMergeItems, navigationOrderDir, {
        isRanked: navigationSemanticQuery !== null,
      }).filter((item) => !stripKeys.has(toGalleryItemKey(item))),
    [navigationOrderDir, navigationSemanticQuery, previewMergeItems, stripKeys]
  );
  const listingItems = useMemo(
    () => (hasNavigationContext ? mergeListingItems(backendBoardItems) : EMPTY_PREVIEW_ITEMS),
    [backendBoardItems, hasNavigationContext, mergeListingItems]
  );
  const boardItems = useMemo(
    () => (stripItems.length === 0 ? listingItems : [...stripItems, ...listingItems]),
    [listingItems, stripItems]
  );
  const isLoadingBoard = hasNavigationContext && isFetchingBoardItems;
  const sessionEntries = useMemo(
    (): GalleryNavigationEntry[] =>
      progressSessions.map((session) => ({ id: session.id, kind: 'session', navigable: session.state === 'running' })),
    [progressSessions]
  );
  const stripEntries = useMemo(() => toItemEntries(stripItems), [stripItems]);
  const navigationSections = useMemo(
    () => [sessionEntries, stripEntries, toItemEntries(listingItems)],
    [listingItems, sessionEntries, stripEntries]
  );
  const cursorKey = followedSessionId !== null ? getGallerySessionNavigationKey(followedSessionId) : selectedItemKey;
  const navigationCursor =
    followedSessionId !== null || selectedItemKey === null
      ? -1
      : boardItems.findIndex((item) => toGalleryItemKey(item) === selectedItemKey);

  // Preview is the only surface that walks a paginated listing across its pages, so at a loaded edge the next page
  // wins over the strip seam; the strip is reached from the listing's first page.
  const isAtLoadedBackendBoundary = useCallback(
    (offset: -1 | 1): boolean =>
      followedSessionId === null &&
      selectedItemKey !== null &&
      (offset === 1
        ? backendBoardItems.at(-1) !== undefined &&
          toGalleryItemKey(backendBoardItems.at(-1)!) === selectedItemKey &&
          hasNextBoardItemsPage
        : backendBoardItems[0] !== undefined &&
          toGalleryItemKey(backendBoardItems[0]) === selectedItemKey &&
          hasPreviousBoardItemsPage),
    [backendBoardItems, followedSessionId, hasNextBoardItemsPage, hasPreviousBoardItemsPage, selectedItemKey]
  );

  // Share navigation between keyboard, footer, and swipe; comparison does not step saved images.
  const navigate = useCallback(
    (offset: -1 | 1): Promise<boolean> => {
      if (isComparing) {
        return Promise.resolve(false);
      }

      const direction = offset === 1 ? 'right' : 'left';
      const stepTo = (entry: GalleryNavigationEntry | null, data: typeof boardItemsData): boolean => {
        if (entry?.kind === 'session') {
          followSession(entry.id);
        } else if (entry) {
          stampSelection(entry.item, data);
        }

        return entry !== null;
      };

      if (!isAtLoadedBackendBoundary(offset)) {
        return Promise.resolve(
          stepTo(getGalleryNavigationStep(navigationSections, cursorKey, direction), boardItemsData)
        );
      }

      if (offset === 1 ? isFetchingNextBoardItemsPage : isFetchingPreviousBoardItemsPage) {
        return Promise.resolve(false);
      }

      const fetchBoundaryPage = offset === 1 ? fetchNextBoardItemsPage : fetchPreviousBoardItemsPage;

      return fetchBoundaryPage().then((result) => {
        if (result.isError || navigationContextKeyRef.current !== navigationContextKey) {
          return false;
        }

        // Against the data just fetched: the item is not in the pages this
        // render closed over, and a lookup there would read it as an item
        // the window does not hold.
        const nextSections = [
          sessionEntries,
          stripEntries,
          toItemEntries(mergeListingItems(flattenPreviewItems(result.data))),
        ];

        return stepTo(getGalleryNavigationStep(nextSections, cursorKey, direction), result.data);
      });
    },
    [
      boardItemsData,
      cursorKey,
      fetchNextBoardItemsPage,
      fetchPreviousBoardItemsPage,
      followSession,
      isAtLoadedBackendBoundary,
      isComparing,
      isFetchingNextBoardItemsPage,
      isFetchingPreviousBoardItemsPage,
      mergeListingItems,
      navigationContextKey,
      navigationSections,
      sessionEntries,
      stampSelection,
      stripEntries,
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
      void navigate(event.key === 'ArrowLeft' ? -1 : 1);
    },
    [isComparing, navigate]
  );

  // The same resolution navigate() makes, so a swipe reveals what committing it will select.
  const neighbors = useMemo((): PreviewNeighbors => {
    if (isComparing) {
      return NO_NEIGHBORS;
    }

    const resolve = (offset: -1 | 1): PreviewNeighbor =>
      isAtLoadedBackendBoundary(offset)
        ? { kind: 'more' }
        : toNeighbor(getGalleryNavigationStep(navigationSections, cursorKey, offset === 1 ? 'right' : 'left'));

    return { next: resolve(1), previous: resolve(-1) };
  }, [cursorKey, isAtLoadedBackendBoundary, isComparing, navigationSections]);

  // Prefetch the images a step would land on to avoid decode flashes during navigation.
  const previousNeighborUrl =
    neighbors.previous?.kind === 'item' && neighbors.previous.item.kind === 'image'
      ? neighbors.previous.item.fullUrl
      : null;
  const nextNeighborUrl =
    neighbors.next?.kind === 'item' && neighbors.next.item.kind === 'image' ? neighbors.next.item.fullUrl : null;

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
    neighbors,
    navigationCursor,
    navigationQueryKey,
    getSelectionPage,
    selectPreviewItem,
  };
};
