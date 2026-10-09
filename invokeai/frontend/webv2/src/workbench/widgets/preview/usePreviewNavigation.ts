import type { GalleryImageItem, GalleryItem, GalleryItemKey, GalleryItemRef, GalleryView } from '@features/gallery';
import type {
  GalleryItemsPage,
  GalleryNavigationEntry,
  GallerySemanticReference,
  getGallerySelectedImageQuery,
} from '@features/gallery/contracts';
import type { GalleryItemsFilter } from '@features/gallery/queries';
import type { QueueItem, QueueProgressSession } from '@features/queue/contracts';

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
  fetchVerifiedGalleryItemPage,
  isDateBoardId,
  galleryItemNamesOptions,
  galleryItemsPageOptions,
  galleryStarredStripOptions,
} from '@features/gallery/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
import { parseDateTokens } from '@platform/search/dateTokens';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from 'react';

/**
 * Own Preview query merging, cursor derivation, boundary fetches, and neighbor prefetch. Shared absolute pages let
 * Preview reach deep selections without replaying every earlier page.
 */

const EMPTY_PREVIEW_ITEMS: GalleryItem[] = [];

export const getPreviewSelectedPage = ({
  galleryPage,
  navigationBoardId,
  navigationSemanticKey,
  selectedImageQuery,
}: {
  galleryPage: number;
  navigationBoardId: string;
  navigationSemanticKey: string;
  selectedImageQuery: ReturnType<typeof getGallerySelectedImageQuery>;
}): number =>
  navigationSemanticKey === ''
    ? selectedImageQuery.semanticKey === null
      ? selectedImageQuery.page
      : 0
    : selectedImageQuery.boardId === navigationBoardId && selectedImageQuery.semanticKey === navigationSemanticKey
      ? selectedImageQuery.page
      : galleryPage;

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
      : entry.kind === 'session'
        ? { id: entry.id, kind: 'session' }
        : null;

export interface PreviewNavigationState {
  /** Every saved item the arrows can reach, in order: the starred strip, then the listing. */
  boardItems: GalleryItem[];
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
  /** How many of `boardItems` lead as the starred strip, ahead of the in-progress sessions. */
  stripItemCount: number;
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
  selectGalleryItem: (item: GalleryItem, selectionPage: number, absoluteIndex?: number) => void;
  selectedImageQuery: ReturnType<typeof getGallerySelectedImageQuery>;
  selectedItem: GalleryItem | null;
  selectedItemKey: GalleryItemKey | null;
  /** The gallery's active similarity search, or null for the board listing. */
  semanticQuery: GallerySemanticReference | null;
}): PreviewNavigationState => {
  const accountScope = captureAccountScope();
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
  const selectedPage = getPreviewSelectedPage({
    galleryPage,
    navigationBoardId,
    navigationSemanticKey,
    selectedImageQuery,
  });
  // Following live has a cursor too, so the listing loads for the step off it.
  const hasNavigationContext = selectedItem !== null || followedSessionId !== null;
  const navigationContextKey = `${accountScope.epoch}:${followedSessionId ?? ''}:${selectedItemKey ?? ''}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationPaginationMode}:${selectedPage}:${selectedImageQuery.searchTerm}:${navigationStarredOnly}:${navigationSemanticKey}`;
  const navigationQueryKey = `${accountScope.epoch}:${navigationBoardId}:${navigationGalleryView}:${navigationOrderDir}:${navigationPaginationMode}:${selectedImageQuery.searchTerm}:${navigationStarredOnly}:${navigationSemanticKey}`;
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

  useMountEffect(() => () => pageFetchRequestRef.current?.controller.abort());

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

      // Ranked picks retain their exact result page. Ordinary picks use their exact shared page; local anchors fall
      // back to the selected page in paginated mode or the board top for recent items outside this page set.
      if (navigationSemanticQuery !== null) {
        return page === undefined ? selectedPage : page.offset / GALLERY_PAGE_SIZE;
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
    (item: GalleryItem, pages: typeof boardPageResults) => {
      if (isAccountScopeCurrent(accountScope)) {
        const page = pages.find(({ data }) =>
          data.items.some((candidate) => toGalleryItemKey(candidate) === toGalleryItemKey(item))
        );
        const itemIndex = page?.data.items.findIndex(
          (candidate) => toGalleryItemKey(candidate) === toGalleryItemKey(item)
        );
        const pageItemIndex = itemIndex !== undefined && itemIndex >= 0 ? itemIndex : undefined;
        const candidateAbsoluteIndex =
          page && pageItemIndex !== undefined
            ? (page.data.itemIndices?.[pageItemIndex] ?? page.offset + pageItemIndex)
            : undefined;
        const absoluteIndex =
          candidateAbsoluteIndex !== undefined &&
          Number.isInteger(candidateAbsoluteIndex) &&
          candidateAbsoluteIndex >= 0
            ? candidateAbsoluteIndex
            : undefined;

        selectGalleryItem(item, getSelectionPageIn(item, pages), absoluteIndex);
      }
    },
    [accountScope, getSelectionPageIn, selectGalleryItem]
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
      const requestSignal = AbortSignal.any([signal, accountScope.signal]);
      requestSignal.throwIfAborted();
      const namesPromise = queryClient.fetchQuery(galleryItemNamesOptions(listingFilterWithStarred));
      let abortListener: (() => void) | undefined;

      try {
        const names = await Promise.race([
          namesPromise,
          new Promise<never>((_resolve, reject) => {
            abortListener = () =>
              reject(requestSignal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
            requestSignal.addEventListener('abort', abortListener, { once: true });
            if (requestSignal.aborted) {
              abortListener();
            }
          }),
        ]);

        requestSignal.throwIfAborted();
        if (!isAccountScopeCurrent(accountScope) || navigationContextKeyRef.current !== navigationContextKey) {
          throw new DOMException('The Preview listing changed.', 'AbortError');
        }

        return [...stripItems.map(toGalleryItemRef), ...names.items];
      } finally {
        if (abortListener) {
          requestSignal.removeEventListener('abort', abortListener);
        }
      }
    },
    [accountScope, listingFilterWithStarred, navigationContextKey, queryClient, stripItems]
  );
  const isLoadingBoard = hasNavigationContext && isFetchingBoardItems;
  const sessionEntries = useMemo(
    (): GalleryNavigationEntry[] =>
      progressSessions.map((session) => ({ id: session.id, kind: 'session', navigable: session.state === 'running' })),
    [progressSessions]
  );
  const stripEntries = useMemo(() => toItemEntries(stripItems), [stripItems]);
  // The grid's order: the starred strip it pins at the top, its In progress section, then the listing.
  const navigationSections = useMemo(
    () => [stripEntries, sessionEntries, toItemEntries(listingItems)],
    [listingItems, sessionEntries, stripEntries]
  );
  // The live session while following it, else the selection; a session that has just gone gives way to the selection.
  const cursorKeys = useMemo(
    () => [followedSessionId === null ? null : getGallerySessionNavigationKey(followedSessionId), selectedItemKey],
    [followedSessionId, selectedItemKey]
  );
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
  const selectedListingPage = boardPageResults.find(({ data }) =>
    data.items.some((item) => toGalleryItemKey(item) === selectedItemKey)
  );
  const selectedListingPageItems = selectedListingPage?.data.items ?? EMPTY_PREVIEW_ITEMS;
  const isSelectedListingItem = (item: GalleryItem | undefined): boolean =>
    item !== undefined && toGalleryItemKey(item) === selectedItemKey;
  const selectedItemStartsPage = isSelectedListingItem(selectedListingPageItems[0]);
  const selectedItemEndsPage = isSelectedListingItem(selectedListingPageItems.at(-1));
  // A recent local item or another section can sit just past a page edge; while the neighboring page is missing,
  // loading, or failed, a loaded step would skip that page's items.
  const isListingPageUnresolved = (pageOffset: number): boolean => {
    if (pageOffset < 0 || (listingTotal !== undefined && pageOffset >= listingTotal)) {
      return false;
    }

    const pageQuery =
      pageOffset === selectedPageOffset
        ? selectedPageQuery
        : adjacentPageQueries[adjacentPageOffsets.indexOf(pageOffset)];

    return (
      !boardPageResults.some(({ offset }) => offset === pageOffset) ||
      pageQuery?.isFetching === true ||
      pageQuery?.isError === true
    );
  };
  const hasUnresolvedPreviousListingPage =
    selectedItemStartsPage && isListingPageUnresolved(selectedItemPageOffset - GALLERY_PAGE_SIZE);
  const hasUnresolvedNextListingPage =
    selectedItemEndsPage && isListingPageUnresolved(selectedItemPageOffset + GALLERY_PAGE_SIZE);
  const selectedItemIsMissingFromStampedPage =
    selectedItem !== null &&
    selectedItemKey !== null &&
    navigationSemanticQuery === null &&
    !isDateBoardId(navigationBoardId) &&
    selectedPageQuery.data !== undefined &&
    !selectedPageQuery.isFetching &&
    !stripItems.some((item) => toGalleryItemKey(item) === selectedItemKey) &&
    !navigationLocalItems.some((item) => toGalleryItemKey(item) === selectedItemKey) &&
    !boardPageResults
      .find(({ offset }) => offset === selectedPageOffset)
      ?.data.items.some((item) => toGalleryItemKey(item) === selectedItemKey);
  const selectedItemNeedsLocation = selectedItemIsMissingFromStampedPage;

  // Share navigation between keyboard, footer, and swipe; comparison does not step saved images.
  const navigate = useCallback(
    (offset: -1 | 1): Promise<boolean> => {
      if (isComparing || !isAccountScopeCurrent(accountScope)) {
        return Promise.resolve(false);
      }

      const direction = offset === 1 ? 'right' : 'left';
      const isCurrentNavigation = (): boolean =>
        isAccountScopeCurrent(accountScope) && navigationContextKeyRef.current === navigationContextKey;
      const stepTo = (entry: GalleryNavigationEntry | null, pages: typeof boardPageResults): boolean => {
        if (entry?.kind === 'session') {
          followSession(entry.id);
        } else if (entry?.kind === 'item') {
          stampSelection(entry.item, pages);
        } else {
          return false;
        }

        return true;
      };

      const loadedEntry = getGalleryNavigationStep(navigationSections, cursorKeys, direction);

      if (
        loadedEntry !== null &&
        !selectedItemNeedsLocation &&
        !(offset === 1 ? hasUnresolvedNextListingPage : hasUnresolvedPreviousListingPage)
      ) {
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
      const requestSignal = AbortSignal.any([controller.signal, accountScope.signal]);
      pageFetchRequestRef.current = { contextKey: navigationContextKey, controller };

      const pages = [...boardPageResults];
      let pageOffset = selectedItemPageOffset;
      let total = listingTotal ?? pages[0]?.data.total;
      const updatePage = (offset: number, data: GalleryItemsPage) => {
        const index = pages.findIndex((page) => page.offset === offset);
        const page = { offset, data };

        if (index === -1) {
          pages.push(page);
        } else {
          pages[index] = page;
        }

        pages.sort((a, b) => a.offset - b.offset);
      };

      return (async () => {
        try {
          if (selectedItemNeedsLocation && selectedItem) {
            const located = await fetchVerifiedGalleryItemPage(
              queryClient,
              listingFilterWithStarred,
              toGalleryItemRef(selectedItem),
              accountScope,
              requestSignal
            );

            if (!isCurrentNavigation()) {
              return false;
            }

            if (located !== null) {
              pages.splice(0, pages.length, { offset: located.offset, data: located.page });
              pageOffset = located.offset;
              total = located.total;

              if (direction === 'left' && located.index === located.offset && located.offset > 0) {
                const previousOffset = located.offset - GALLERY_PAGE_SIZE;
                const previousPage = await fetchGalleryItemsPage(
                  queryClient,
                  listingFilterWithStarred,
                  previousOffset,
                  { signal: requestSignal, staleTime: 0 }
                );

                if (!isCurrentNavigation()) {
                  return false;
                }

                updatePage(previousOffset, previousPage);
              }

              const sections = [
                stripEntries,
                sessionEntries,
                toItemEntries(
                  mergePreviewBoardItems(
                    pages.flatMap(({ data }) => data.items),
                    previewMergeItems,
                    navigationOrderDir,
                    { isRanked: navigationSemanticQuery !== null }
                  ).filter((item) => !stripKeys.has(toGalleryItemKey(item)))
                ),
              ];
              const entry = getGalleryNavigationStep(sections, cursorKeys, direction);

              if (entry !== null) {
                return stepTo(entry, pages);
              }
            } else if (selectedPageOffset >= (listingTotal ?? total ?? 0)) {
              return false;
            } else if (loadedEntry !== null) {
              return stepTo(loadedEntry, boardPageResults);
            }
          }

          if (total === undefined) {
            const currentPage = await fetchGalleryItemsPage(queryClient, listingFilterWithStarred, selectedPageOffset, {
              signal: requestSignal,
            });

            if (!isCurrentNavigation()) {
              return false;
            }

            updatePage(selectedPageOffset, currentPage);
            total = currentPage.total;

            const currentSections = [
              stripEntries,
              sessionEntries,
              toItemEntries(
                mergePreviewBoardItems(
                  pages.flatMap(({ data }) => data.items),
                  previewMergeItems,
                  navigationOrderDir,
                  { isRanked: navigationSemanticQuery !== null }
                ).filter((item) => !stripKeys.has(toGalleryItemKey(item)))
              ),
            ];
            const currentEntry = getGalleryNavigationStep(currentSections, cursorKeys, direction);

            if (currentEntry !== null) {
              return stepTo(currentEntry, pages);
            }
          }

          while (true) {
            pageOffset += offset * GALLERY_PAGE_SIZE;

            if (pageOffset < 0 || pageOffset >= (total ?? 0) || !isCurrentNavigation()) {
              return false;
            }

            try {
              // Query deduplicates with the adjacent observer if it is already loading this page.
              const page = await fetchGalleryItemsPage(queryClient, listingFilterWithStarred, pageOffset, {
                signal: requestSignal,
              });

              if (!isCurrentNavigation()) {
                return false;
              }

              updatePage(pageOffset, page);
              total = page.total;
              const pageItems = pages.flatMap(({ data }) => data.items);
              const sections = [
                stripEntries,
                sessionEntries,
                toItemEntries(
                  mergePreviewBoardItems(pageItems, previewMergeItems, navigationOrderDir, {
                    isRanked: navigationSemanticQuery !== null,
                  }).filter((item) => !stripKeys.has(toGalleryItemKey(item)))
                ),
              ];
              const entry = getGalleryNavigationStep(sections, cursorKeys, direction);

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
      cursorKeys,
      followSession,
      isComparing,
      accountScope,
      listingFilterWithStarred,
      listingTotal,
      navigationContextKey,
      navigationSections,
      navigationOrderDir,
      selectedPageOffset,
      selectedItem,
      selectedItemNeedsLocation,
      hasUnresolvedNextListingPage,
      hasUnresolvedPreviousListingPage,
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

  // The same resolution navigate() makes, so a swipe reveals what committing it will select.
  const neighbors = useMemo((): PreviewNeighbors => {
    if (isComparing) {
      return NO_NEIGHBORS;
    }

    const resolve = (offset: -1 | 1): PreviewNeighbor => {
      if (
        !selectedItemNeedsLocation &&
        (offset === 1 ? hasUnresolvedNextListingPage : hasUnresolvedPreviousListingPage)
      ) {
        return { kind: 'more' };
      }

      const neighbor = getGalleryNavigationStep(navigationSections, cursorKeys, offset === 1 ? 'right' : 'left');

      if (neighbor !== null) {
        return toNeighbor(neighbor);
      }

      const targetOffset = selectedItemPageOffset + offset * GALLERY_PAGE_SIZE;

      return targetOffset >= 0 && (listingTotal === undefined || targetOffset < listingTotal) ? { kind: 'more' } : null;
    };

    return { next: resolve(1), previous: resolve(-1) };
  }, [
    cursorKeys,
    hasUnresolvedNextListingPage,
    hasUnresolvedPreviousListingPage,
    isComparing,
    listingTotal,
    navigationSections,
    selectedItemNeedsLocation,
    selectedItemPageOffset,
  ]);

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
    isLoadingBoard,
    navigate,
    neighbors,
    navigationCursor,
    navigationQueryKey,
    getSelectionPage,
    loadOrderedRefs,
    selectPreviewItem,
    stripItemCount: stripItems.length,
  };
};
