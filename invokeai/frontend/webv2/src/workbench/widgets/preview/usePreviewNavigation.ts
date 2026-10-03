import type { GalleryImageItem, GalleryItem, GalleryItemKey, GalleryItemRef, GalleryView } from '@features/gallery';
import type {
  GalleryItemsPage,
  GalleryNavigationEntry,
  GallerySemanticReference,
  getGallerySelectedImageQuery,
} from '@features/gallery/contracts';
import type { GalleryItemsFilter } from '@features/gallery/queries';
import type { QueueItem, QueueProgressSession } from '@features/queue/contracts';
import type { KeyboardEvent } from 'react';

import {
  compareGalleryItems,
  gallerySemanticReferenceKey,
  getGalleryNavigationStep,
  getGallerySessionNavigationKey,
  toGalleryItemKey,
  toGalleryItemRef,
} from '@features/gallery/contracts';
import {
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  fetchGalleryItemsPage,
  galleryItemNamesOptions,
  galleryItemsPageOptions,
  galleryStarredStripOptions,
} from '@features/gallery/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from 'react';

/**
 * Own Preview query merging, cursor derivation, boundary fetches, and neighbor prefetch. Shared absolute pages let
 * Preview reach deep selections without replaying every earlier page.
 */

const EMPTY_PREVIEW_ITEMS: GalleryItem[] = [];

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
  /** Lazily fetch the full ordered listing when an action needs an item beyond the sparse loaded pages. */
  loadOrderedRefs: (signal: AbortSignal) => Promise<GalleryItemRef[]>;
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
  const navigationPaginationMode =
    navigationSemanticQuery === null ? selectedImageQuery.paginationMode : galleryPaginationMode;
  // Semantic pages follow Gallery's current ranking page; ordinary pages follow the selected item's page stamp.
  const selectedPage = navigationSemanticQuery === null ? selectedImageQuery.page : galleryPage;
  // Following live has a cursor too, so the listing loads for the step off it.
  const hasNavigationContext = selectedItem !== null || followedSessionId !== null;
  const navigationContextKey = `${followedSessionId ?? ''}:${selectedItemKey ?? ''}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationPaginationMode}:${selectedPage}:${selectedImageQuery.searchTerm}:${navigationStarredOnly}:${navigationSemanticKey}`;
  const navigationQueryKey = `${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationPaginationMode}:${selectedImageQuery.searchTerm}:${navigationStarredOnly}:${navigationSemanticKey}`;
  const queryClient = useQueryClient();

  // Publish navigation context in layout effect so boundary-fetch continuations cannot observe new UI with a stale
  // fence.
  const navigationContextKeyRef = useRef(navigationContextKey);
  const pendingNavigationContextRef = useRef<string | null>(null);
  const pageFetchRequestRef = useRef<{ contextKey: string; controller: AbortController } | null>(null);

  useLayoutEffect(() => {
    navigationContextKeyRef.current = navigationContextKey;
    const pageFetchRequest = pageFetchRequestRef.current;

    if (pageFetchRequest && pageFetchRequest.contextKey !== navigationContextKey) {
      pageFetchRequest.controller.abort();
      pageFetchRequestRef.current = null;
    }

    if (pendingNavigationContextRef.current !== navigationContextKey) {
      pendingNavigationContextRef.current = null;
    }
  }, [navigationContextKey]);

  useEffect(
    () => () => {
      pageFetchRequestRef.current?.controller.abort();
    },
    []
  );

  const isPaginatedWindow = navigationPaginationMode === 'paginated';
  const selectedPageOffset = Math.max(0, selectedPage) * GALLERY_PAGE_SIZE;
  const isDeepPageWindow =
    !isPaginatedWindow && navigationSemanticQuery === null && selectedPageOffset >= GALLERY_MAX_ROWS;

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

  const listingFilterWithStarred = useMemo(
    () => ({ ...listingFilter, starred: navigationStarredOnly }),
    [listingFilter, navigationStarredOnly]
  );
  const selectedPageQuery = useQuery({
    ...galleryItemsPageOptions(listingFilterWithStarred, selectedPageOffset),
    enabled: hasNavigationContext,
  });
  const adjacentPageOffsets = useMemo(() => {
    const offsets: number[] = [];

    if (selectedPageOffset >= GALLERY_PAGE_SIZE) {
      offsets.push(selectedPageOffset - GALLERY_PAGE_SIZE);
    }

    if (selectedPageQuery.data === undefined || selectedPageOffset + GALLERY_PAGE_SIZE < selectedPageQuery.data.total) {
      offsets.push(selectedPageOffset + GALLERY_PAGE_SIZE);
    }

    return offsets;
  }, [selectedPageOffset, selectedPageQuery.data]);
  const adjacentPageQueries = useQueries({
    queries: adjacentPageOffsets.map((offset) => ({
      ...galleryItemsPageOptions(listingFilterWithStarred, offset),
      enabled: hasNavigationContext,
    })),
  });
  const boardPageResults = useMemo(
    () =>
      [
        { offset: selectedPageOffset, data: selectedPageQuery.data },
        ...adjacentPageOffsets.map((offset, index) => ({
          offset,
          data: adjacentPageQueries[index]?.data,
        })),
      ]
        .filter((page): page is { offset: number; data: GalleryItemsPage } => page.data !== undefined)
        .sort((a, b) => a.offset - b.offset),
    [adjacentPageOffsets, adjacentPageQueries, selectedPageOffset, selectedPageQuery.data]
  );
  const backendBoardItems = useMemo(() => boardPageResults.flatMap(({ data }) => data.items), [boardPageResults]);
  const listingTotal = selectedPageQuery.data?.total ?? boardPageResults[0]?.data.total;
  const isFetchingBoardItems = selectedPageQuery.isFetching || adjacentPageQueries.some((query) => query.isFetching);

  // Share Gallery's bounded starred strip except for ranked, starred-only, or deep windows.
  const hasStrip =
    hasNavigationContext && !navigationStarredOnly && navigationSemanticQuery === null && !isDeepPageWindow;
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
    (item: GalleryItem, pages: typeof boardPageResults): number => {
      const itemKey = toGalleryItemKey(item);
      const page = pages.find(({ data }) => data.items.some((candidate) => toGalleryItemKey(candidate) === itemKey));

      // Ranked picks retain their board-top stamp. Ordinary picks use their exact shared page; local anchors fall
      // back to the selected page in paginated mode or the board top for recent items outside this page set.
      if (navigationSemanticQuery !== null) {
        return 0;
      }

      return page === undefined
        ? isPaginatedWindow && !isDeepPageWindow
          ? selectedPage
          : 0
        : page.offset / GALLERY_PAGE_SIZE;
    },
    [isDeepPageWindow, isPaginatedWindow, navigationSemanticQuery, selectedPage]
  );
  const stampSelection = useCallback(
    (item: GalleryItem, pages: typeof boardPageResults) => selectGalleryItem(item, getSelectionPageIn(item, pages)),
    [getSelectionPageIn, selectGalleryItem]
  );
  const getSelectionPage = useCallback(
    (item: GalleryItem) => getSelectionPageIn(item, boardPageResults),
    [boardPageResults, getSelectionPageIn]
  );
  const selectPreviewItem = useCallback(
    (item: GalleryItem) => stampSelection(item, boardPageResults),
    [boardPageResults, stampSelection]
  );

  const optimisticQueueItemIds = useMemo(
    () =>
      new Set(
        queueItems.filter((item) => item.status === 'pending' || item.status === 'running').map((item) => item.id)
      ),
    [queueItems]
  );
  const navigationLocalItems = useMemo(() => {
    // Keep recent results until listings catch up, except in filtered or deep windows where they do not
    // belong. Those windows merge only in-flight work and selection.
    const hasActiveSearch =
      navigationStarredOnly || selectedImageSearch.text.trim() !== '' || selectedImageSearch.range !== undefined;

    if (!hasActiveSearch && !isPaginatedWindow && !isDeepPageWindow) {
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
    isDeepPageWindow,
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
  const loadOrderedRefs = useCallback(
    async (signal: AbortSignal): Promise<GalleryItemRef[]> => {
      signal.throwIfAborted();
      const namesPromise = queryClient.fetchQuery(galleryItemNamesOptions(listingFilterWithStarred));
      let abortListener: (() => void) | undefined;

      try {
        const names = await Promise.race([
          namesPromise,
          new Promise<never>((_resolve, reject) => {
            abortListener = () => reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
            signal.addEventListener('abort', abortListener, { once: true });
            if (signal.aborted) {
              abortListener();
            }
          }),
        ]);

        signal.throwIfAborted();
        if (navigationContextKeyRef.current !== navigationContextKey) {
          throw new DOMException('The Preview listing changed.', 'AbortError');
        }

        return [...stripItems.map(toGalleryItemRef), ...names.items];
      } finally {
        if (abortListener) {
          signal.removeEventListener('abort', abortListener);
        }
      }
    },
    [listingFilterWithStarred, navigationContextKey, queryClient, stripItems]
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
  const selectedItemPageOffset = useMemo(() => {
    if (selectedItemKey === null) {
      return selectedPageOffset;
    }

    return (
      boardPageResults.find(({ data }) => data.items.some((item) => toGalleryItemKey(item) === selectedItemKey))
        ?.offset ?? selectedPageOffset
    );
  }, [boardPageResults, selectedItemKey, selectedPageOffset]);

  // Share navigation between keyboard, footer, and swipe; comparison does not step saved images.
  const navigate = useCallback(
    (offset: -1 | 1): Promise<boolean> => {
      if (isComparing) {
        return Promise.resolve(false);
      }

      const direction = offset === 1 ? 'right' : 'left';
      const stepTo = (entry: GalleryNavigationEntry | null, pages: typeof boardPageResults): boolean => {
        if (entry?.kind === 'session') {
          followSession(entry.id);
        } else if (entry) {
          stampSelection(entry.item, pages);
        }

        return entry !== null;
      };

      const loadedEntry = getGalleryNavigationStep(navigationSections, cursorKey, direction);

      if (loadedEntry !== null) {
        pageFetchRequestRef.current?.controller.abort();
        pageFetchRequestRef.current = null;
        pendingNavigationContextRef.current = null;
        return Promise.resolve(stepTo(loadedEntry, boardPageResults));
      }

      if (pendingNavigationContextRef.current === navigationContextKey) {
        return Promise.resolve(false);
      }

      pendingNavigationContextRef.current = navigationContextKey;
      const controller = new AbortController();
      pageFetchRequestRef.current = { contextKey: navigationContextKey, controller };

      const pages = [...boardPageResults];
      let pageOffset = selectedItemPageOffset;
      let total = listingTotal ?? pages[0]?.data.total;

      return (async () => {
        try {
          if (total === undefined) {
            const currentPage = await fetchGalleryItemsPage(queryClient, listingFilterWithStarred, selectedPageOffset, {
              signal: controller.signal,
            });

            if (navigationContextKeyRef.current !== navigationContextKey) {
              return false;
            }

            pages.push({ offset: selectedPageOffset, data: currentPage });
            total = currentPage.total;

            const currentSections = [
              sessionEntries,
              stripEntries,
              toItemEntries(
                mergePreviewBoardItems(
                  pages.flatMap(({ data }) => data.items),
                  previewMergeItems,
                  navigationOrderDir,
                  { isRanked: navigationSemanticQuery !== null }
                ).filter((item) => !stripKeys.has(toGalleryItemKey(item)))
              ),
            ];
            const currentEntry = getGalleryNavigationStep(currentSections, cursorKey, direction);

            if (currentEntry !== null) {
              return stepTo(currentEntry, pages);
            }
          }

          while (true) {
            pageOffset += offset * GALLERY_PAGE_SIZE;

            if (
              pageOffset < 0 ||
              pageOffset >= (total ?? 0) ||
              navigationContextKeyRef.current !== navigationContextKey
            ) {
              return false;
            }

            try {
              // Query deduplicates with the adjacent observer if it is already loading this page.
              const page = await fetchGalleryItemsPage(queryClient, listingFilterWithStarred, pageOffset, {
                signal: controller.signal,
              });

              if (navigationContextKeyRef.current !== navigationContextKey) {
                return false;
              }

              pages.push({ offset: pageOffset, data: page });
              pages.sort((a, b) => a.offset - b.offset);
              total = page.total;
              const pageItems = pages.flatMap(({ data }) => data.items);
              const sections = [
                sessionEntries,
                stripEntries,
                toItemEntries(
                  mergePreviewBoardItems(pageItems, previewMergeItems, navigationOrderDir, {
                    isRanked: navigationSemanticQuery !== null,
                  }).filter((item) => !stripKeys.has(toGalleryItemKey(item)))
                ),
              ];
              const entry = getGalleryNavigationStep(sections, cursorKey, direction);

              if (entry !== null) {
                return stepTo(entry, pages);
              }
            } catch {
              return false;
            }
          }
        } catch {
          return false;
        } finally {
          if (pageFetchRequestRef.current?.controller === controller) {
            pageFetchRequestRef.current = null;
            if (pendingNavigationContextRef.current === navigationContextKey) {
              pendingNavigationContextRef.current = null;
            }
          }
        }
      })();
    },
    [
      boardPageResults,
      cursorKey,
      followSession,
      isComparing,
      listingFilterWithStarred,
      listingTotal,
      navigationContextKey,
      navigationSections,
      navigationOrderDir,
      selectedPageOffset,
      navigationSemanticQuery,
      pendingNavigationContextRef,
      pageFetchRequestRef,
      previewMergeItems,
      queryClient,
      selectedItemPageOffset,
      sessionEntries,
      stripKeys,
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

    const resolve = (offset: -1 | 1): PreviewNeighbor => {
      const neighbor = getGalleryNavigationStep(navigationSections, cursorKey, offset === 1 ? 'right' : 'left');

      if (neighbor !== null) {
        return toNeighbor(neighbor);
      }

      const targetOffset = selectedItemPageOffset + offset * GALLERY_PAGE_SIZE;

      return targetOffset >= 0 && (listingTotal === undefined || targetOffset < listingTotal) ? { kind: 'more' } : null;
    };

    return { next: resolve(1), previous: resolve(-1) };
  }, [cursorKey, isComparing, listingTotal, navigationSections, selectedItemPageOffset]);

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
    loadOrderedRefs,
    selectPreviewItem,
  };
};
