import type { GalleryItem } from '@features/gallery/core/items';
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryBoard, GalleryView, GeneratedImageContract } from '@features/gallery/core/types';

import { compareGalleryItems, legacyGeneratedImageToGalleryItem, toGalleryItemKey } from '@features/gallery/core/items';
import { GALLERY_RECENT_IMAGE_LIMIT } from '@features/gallery/core/recentImages';
import { ALL_READABLE_BOARDS_ID, isDateBoardId } from '@features/gallery/data/backend';
import {
  flattenGalleryItemsData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryItemsInfiniteOptions,
  galleryItemsPageOptions,
  galleryItemsTotalOptions,
  getGalleryListingBoardsQuery,
  type GalleryItemsFilter,
} from '@features/gallery/data/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import {
  hashKey,
  keepPreviousData,
  useInfiniteQuery,
  useQueries,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { planGalleryPageOffsets } from './galleryGridLayout';
import {
  getGalleryListing,
  getGalleryReadStatus,
  resolveGallerySelectedBoardId,
  type GalleryReadState,
} from './galleryStateView';

/** The listing's read state, plus whether a further page is on its way. */
export interface GalleryListingState extends GalleryReadState {
  isFetchingMore: boolean;
}

export interface GalleryData {
  boards: GalleryBoard[];
  boardsState: GalleryReadState;
  filter: GalleryItemsFilter;
  hasMore: boolean;
  isLoadingItems: boolean;
  /** The resolved board the items were fetched for. */
  selectedBoardId: string;
  /** Distinguish reaching the window cap from reaching the board end; only truncation needs an explanation. */
  isWindowTruncated: boolean;
  /** The current scope's items; null until it has something to show, and whenever it failed without data. */
  items: GalleryItem[] | null;
  /** The current query's failure, or null while it is healthy. */
  queryError: Error | null;
  listing: GalleryListingState;
  loadMore: () => void;
  /**
   * The previous scope's items while the current one loads, for consumers that keep them on screen (dimmed and
   * busy) across a scope change. Never set once the current scope has data or has failed.
   */
  previousScopeItems: GalleryItem[] | null;
  total: number | null;
  /** Absolute backend positions for the currently subscribed pages in the main Gallery and picker. */
  sparseListing?: GallerySparseListing;
  setVisibleRange?: (range: { endIndexExclusive: number; startIndex: number }) => void;
  /**
   * Keep a locator-verified index's page subscribed even past a stale total, so that page's fresher total reconciles
   * the listing. Lasts until the listing changes or another reveal replaces it.
   */
  pinRevealIndex?: (absoluteIndex: number) => void;
}

export interface GallerySparsePageState {
  error: Error | null;
  isLoading: boolean;
  retry: () => Promise<unknown>;
}

export interface GallerySparseListing {
  itemSlots: ReadonlyMap<number, GalleryItem>;
  pageStates: ReadonlyMap<number, GallerySparsePageState>;
  /** Project-local outputs stay in separate virtual rows and never change backend item indexes. */
  recentItems: GalleryItem[];
  total: number | null;
}

/** Rebuild absolute slots without filling hydration gaps or assuming returned-item position equals requested index. */
export const mapGalleryItemPageSlots = ({
  pageLocalOffset,
  pageOffsets,
  pages,
}: {
  pageLocalOffset?: number;
  pageOffsets: readonly number[];
  pages: readonly (
    | {
        itemIndices?: readonly number[];
        items: readonly GalleryItem[];
        offset?: number;
      }
    | undefined
  )[];
}): Map<number, GalleryItem> => {
  const slots = new Map<number, GalleryItem>();

  pages.forEach((page, pageIndex) => {
    if (!page) {
      return;
    }

    const pageOffset = page.offset ?? pageOffsets[pageIndex] ?? 0;

    page.items.forEach((item, itemIndex) => {
      const absoluteIndex = page.itemIndices?.[itemIndex] ?? pageOffset + itemIndex;
      const slotIndex = pageLocalOffset === undefined ? absoluteIndex : absoluteIndex - pageLocalOffset;

      if (Number.isInteger(slotIndex) && slotIndex >= 0) {
        slots.set(slotIndex, item);
      }
    });
  });

  return slots;
};

const EMPTY_BOARDS: GalleryBoard[] = [];

const useGalleryBoards = ({ settings }: { settings: GallerySettings }) => {
  const { data, error, errorUpdateCount, isError, isFetching, refetch } = useQuery(
    galleryBoardsOptions(getGalleryListingBoardsQuery(settings))
  );
  const boards = data ?? EMPTY_BOARDS;
  const status = getGalleryReadStatus({ errorUpdateCount, hasData: data !== undefined, isError }, boards.length);
  const isFailed = status === 'error' || status === 'stale-error';
  const retry = useCallback(async () => {
    await refetch();
  }, [refetch]);
  const boardsState = useMemo<GalleryReadState>(
    () => ({ error, isRetrying: isFailed && isFetching, retry, status }),
    [error, isFailed, isFetching, retry, status]
  );

  return { boards, boardsState };
};

const isRecentItemVisible = (item: GalleryItem, filter: GalleryItemsFilter): boolean => {
  // Overlay recents only in the unstarred listing; newly starred recents belong to the strip.
  if (
    filter.searchTerm !== '' ||
    filter.createdFrom !== undefined ||
    filter.createdTo !== undefined ||
    filter.starred === true ||
    (filter.starred === false && item.starred) ||
    Boolean(filter.semanticQuery) ||
    isDateBoardId(filter.boardId)
  ) {
    return false;
  }

  const hasMatchingBoard = filter.boardId === ALL_READABLE_BOARDS_ID || filter.boardId === item.boardId;
  const hasMatchingCategory =
    filter.galleryView === 'images'
      ? item.category === 'general'
      : item.kind === 'image' && item.category !== 'general';

  return hasMatchingBoard && hasMatchingCategory;
};

export const mergeGalleryItemWindow = ({
  backendItems,
  filter,
  knownBackendItemKeys,
  maxRows,
  recentImages,
}: {
  backendItems: readonly GalleryItem[];
  filter: GalleryItemsFilter;
  knownBackendItemKeys?: ReadonlySet<ReturnType<typeof toGalleryItemKey>>;
  maxRows: number;
  recentImages: readonly GeneratedImageContract[];
}): GalleryItem[] => {
  const missingRecentItems = getGalleryRecentItems({ backendItems, filter, knownBackendItemKeys, recentImages });
  const seenItemKeys = new Set<string>();

  const mergedItems = [...missingRecentItems, ...backendItems].filter((item) => {
    const key = toGalleryItemKey(item);

    if (seenItemKeys.has(key)) {
      return false;
    }

    seenItemKeys.add(key);
    return true;
  });

  // Semantic results arrive in relevance order, which a date re-sort would
  // destroy; the backend order is the meaning of the list. (No recent items
  // are overlaid in that mode, so the merge is the backend window itself.)
  if (!filter.semanticQuery) {
    mergedItems.sort((a, b) => compareGalleryItems(a, b, { orderDir: filter.orderDir }));
  }

  return mergedItems.slice(0, maxRows);
};

/** Returns the bounded local recent overlay without assigning it a backend position. */
export const getGalleryRecentItems = ({
  backendItems,
  knownBackendItemKeys,
  filter,
  recentImages,
}: {
  backendItems: readonly GalleryItem[];
  knownBackendItemKeys?: ReadonlySet<ReturnType<typeof toGalleryItemKey>>;
  filter: GalleryItemsFilter;
  recentImages: readonly GeneratedImageContract[];
}): GalleryItem[] => {
  const backendItemKeys = new Set(backendItems.map(toGalleryItemKey));
  const recentItems = recentImages
    .slice(0, GALLERY_RECENT_IMAGE_LIMIT)
    .map(legacyGeneratedImageToGalleryItem)
    .filter(
      (item) =>
        !backendItemKeys.has(toGalleryItemKey(item)) &&
        !knownBackendItemKeys?.has(toGalleryItemKey(item)) &&
        isRecentItemVisible(item, filter)
    );

  if (!filter.semanticQuery) {
    recentItems.sort((a, b) => compareGalleryItems(a, b, { orderDir: filter.orderDir }));
  }

  return recentItems;
};

/** A false hasNextPage can mean either completion or truncation. Paginated mode remains fully reachable. */
export const isGalleryWindowTruncated = ({
  hasNextPage,
  isPaginated,
  loadedRowCount,
  maxRows,
  total,
}: {
  hasNextPage: boolean;
  isPaginated: boolean;
  loadedRowCount: number;
  maxRows: number;
  total: number | null;
}): boolean => !isPaginated && !hasNextPage && total !== null && loadedRowCount >= maxRows && total > loadedRowCount;

export const useGalleryData = ({
  galleryView,
  keepPreviousScope = false,
  page,
  projectBoardId,
  recentImages,
  searchTerm,
  selectedBoardId,
  semanticQuery = null,
  settings,
  starred,
  sparseViewport = false,
}: {
  galleryView: GalleryView;
  /** Hold the previous scope's items as `previousScopeItems` while a new scope loads. */
  keepPreviousScope?: boolean;
  page: number;
  projectBoardId: string | null;
  recentImages: readonly GeneratedImageContract[];
  searchTerm: string;
  selectedBoardId: string | null;
  /** When set, items come from semantic search (similarity order) instead of the board listing. */
  semanticQuery?: GallerySemanticReference | null;
  settings: GallerySettings;
  /**
   * Partition to list: the grid asks for unstarred (`false`) or, under its
   * starred filter, starred (`true`); consumers like the picker omit it and
   * see everything.
   */
  starred?: boolean;
  /** Main Gallery and picker opt into page-sized viewport subscriptions; other consumers may keep the bounded window. */
  sparseViewport?: boolean;
}): GalleryData => {
  const { boards, boardsState } = useGalleryBoards({ settings });
  const boardId = resolveGallerySelectedBoardId({ projectBoardId, selectedBoardId }, boards);
  const queryClient = useQueryClient();
  const isPaginated = settings.paginationMode === 'paginated';
  const dateParse = useMemo(() => parseDateTokens(searchTerm), [searchTerm]);
  const filter = useMemo<GalleryItemsFilter>(
    () => ({
      boardId,
      createdFrom: dateParse.range?.from,
      createdTo: dateParse.range?.to,
      galleryView,
      orderDir: settings.imageOrderDir,
      searchTerm: dateParse.text,
      ...(semanticQuery ? { semanticQuery } : {}),
      ...(starred !== undefined ? { starred } : {}),
    }),
    [
      boardId,
      dateParse.range?.from,
      dateParse.range?.to,
      dateParse.text,
      galleryView,
      semanticQuery,
      settings.imageOrderDir,
      starred,
    ]
  );
  // Query keys include the captured account epoch, so range and total state cannot survive an account switch.
  const firstPageOptions = galleryItemsPageOptions(filter, 0);
  const totalOptions = galleryItemsTotalOptions(filter);
  const { accountId, epoch } = firstPageOptions.queryKey[3];
  const filterIdentity = JSON.stringify([accountId, epoch, filter]);
  const filterIdentityRef = useRef(filterIdentity);
  useLayoutEffect(() => {
    filterIdentityRef.current = filterIdentity;
  }, [filterIdentity]);
  const [requestedRange, setRequestedRange] = useState<{
    endIndexExclusive: number;
    filterIdentity: string;
    startIndex: number;
  } | null>(null);
  const hasRequestedRange = requestedRange?.filterIdentity === filterIdentity;
  const visibleStartIndex = hasRequestedRange ? requestedRange.startIndex : 0;
  const visibleEndIndexExclusive = hasRequestedRange ? requestedRange.endIndexExclusive : GALLERY_PAGE_SIZE;
  const setVisibleRange = useCallback(
    ({ endIndexExclusive, startIndex }: { endIndexExclusive: number; startIndex: number }) => {
      const activeFilterIdentity = filterIdentityRef.current;
      const nextStart = Math.max(0, Math.floor(startIndex));
      const nextEnd = Math.max(nextStart, Math.ceil(endIndexExclusive));

      setRequestedRange((current) =>
        current?.filterIdentity === activeFilterIdentity &&
        current.startIndex === nextStart &&
        current.endIndexExclusive === nextEnd
          ? current
          : { endIndexExclusive: nextEnd, filterIdentity: activeFilterIdentity, startIndex: nextStart }
      );
    },
    []
  );
  const [revealPin, setRevealPin] = useState<{ filterIdentity: string; offset: number } | null>(null);
  const revealPinOffset = revealPin?.filterIdentity === filterIdentity ? revealPin.offset : null;
  const pinRevealIndex = useCallback((absoluteIndex: number) => {
    const offset = Math.floor(Math.max(0, absoluteIndex) / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
    const activeFilterIdentity = filterIdentityRef.current;

    setRevealPin((current) =>
      current?.filterIdentity === activeFilterIdentity && current.offset === offset
        ? current
        : { filterIdentity: activeFilterIdentity, offset }
    );
  }, []);
  const [knownTotalSnapshot, setKnownTotalSnapshot] = useState<{ filterIdentity: string; total: number } | null>(null);
  const retainedTotal = knownTotalSnapshot?.filterIdentity === filterIdentity ? knownTotalSnapshot.total : null;
  const cachedFirstPage = queryClient.getQueryData<{ items: GalleryItem[]; total: number }>(firstPageOptions.queryKey);
  const {
    data: queriedTotal,
    error: totalError,
    isFetching: isFetchingTotal,
    refetch: refetchTotal,
  } = useQuery({
    ...totalOptions,
    enabled: sparseViewport && isPaginated && page > 0 && retainedTotal === null && cachedFirstPage === undefined,
  });
  const hasUnresolvedTotalError = totalError !== null && retainedTotal === null && cachedFirstPage === undefined;
  const isInitialTotalDiscovery = retainedTotal === null && cachedFirstPage === undefined;
  const stableTotal =
    retainedTotal ?? cachedFirstPage?.total ?? (isInitialTotalDiscovery ? queriedTotal : undefined) ?? null;
  const knownTotal = stableTotal;
  const maxPaginatedPage =
    knownTotal === null || !Number.isFinite(knownTotal)
      ? null
      : Math.max(0, Math.ceil(Math.max(0, knownTotal) / GALLERY_PAGE_SIZE) - 1);
  const selectedPageOffset =
    (maxPaginatedPage === null ? Math.max(0, page) : Math.min(Math.max(0, page), maxPaginatedPage)) * GALLERY_PAGE_SIZE;
  const pageOffsets = useMemo(() => {
    if (!sparseViewport) {
      return [];
    }

    if (isPaginated) {
      if (knownTotal === 0) {
        // Keep page zero observed so gallery invalidation can discover items added after an empty result.
        return [0];
      }

      if (knownTotal === null || isFetchingTotal || hasUnresolvedTotalError) {
        return page === 0 ? [0] : [];
      }

      return [selectedPageOffset];
    }

    // Infinite listings learn their total from page zero. Once known, subscriptions follow only the virtual range
    // and its virtualizer overscan, even when it is far from the start of the listing.
    const plannedOffsets =
      knownTotal === 0 || (knownTotal === null && !hasRequestedRange)
        ? [0]
        : planGalleryPageOffsets({
            endIndexExclusive: visibleEndIndexExclusive,
            startIndex: visibleStartIndex,
            total: knownTotal,
          });

    // A revealed item may sit past a total counted before another client added it.
    return revealPinOffset === null || plannedOffsets.includes(revealPinOffset)
      ? plannedOffsets
      : [...plannedOffsets, revealPinOffset].sort((left, right) => left - right);
  }, [
    hasRequestedRange,
    isPaginated,
    knownTotal,
    isFetchingTotal,
    hasUnresolvedTotalError,
    page,
    revealPinOffset,
    selectedPageOffset,
    sparseViewport,
    visibleEndIndexExclusive,
    visibleStartIndex,
  ]);
  const [retryingSparseErrors, setRetryingSparseErrors] = useState<Map<string, Error>>(() => new Map());
  const getSparsePageErrorKey = useCallback((offset: number) => `${filterIdentity}:${offset}`, [filterIdentity]);
  const retrySparsePage = useCallback(
    async (offset: number, refetchPage: () => Promise<unknown>, error: Error | null) => {
      const errorKey = `${filterIdentity}:${offset}`;

      if (error) {
        setRetryingSparseErrors((current) => new Map(current).set(errorKey, error));
      }

      try {
        await refetchPage();
      } finally {
        setRetryingSparseErrors((current) => {
          if (!current.has(errorKey)) {
            return current;
          }

          const next = new Map(current);
          next.delete(errorKey);
          return next;
        });
      }
    },
    [filterIdentity]
  );
  const pageOptions = sparseViewport
    ? pageOffsets.map((offset) => {
        const options = galleryItemsPageOptions(filter, offset);

        // Failed pages wait for their own Retry; broad invalidation must not silently retry visible failures.
        return { ...options, enabled: queryClient.getQueryState(options.queryKey)?.status !== 'error' };
      })
    : [];
  const pageResults = useQueries({ queries: pageOptions });
  const loadedPageTotals = pageResults.flatMap((result) => (result.data ? [result.data.total] : []));
  const hasConflictingPageTotals = new Set(loadedPageTotals).size > 1;
  const pageResultsSettled = pageResults.every((result) => !result.isFetching);
  const observedTotal =
    sparseViewport && pageResultsSettled && !hasConflictingPageTotals ? loadedPageTotals[0] : undefined;
  const [pageTotalReconciliation, setPageTotalReconciliation] = useState({
    conflictObserved: false,
    filterIdentity,
    generation: 0,
  });
  const currentPageTotalReconciliation =
    pageTotalReconciliation.filterIdentity === filterIdentity
      ? pageTotalReconciliation
      : { conflictObserved: false, filterIdentity, generation: 0 };

  // A settled agreement ends the current conflict generation. The next conflict can then reconcile even when its
  // filter, total, and active page offsets match an earlier generation.
  if (sparseViewport && !isPaginated && pageResultsSettled && loadedPageTotals.length > 0) {
    if (hasConflictingPageTotals && !currentPageTotalReconciliation.conflictObserved) {
      setPageTotalReconciliation({ ...currentPageTotalReconciliation, conflictObserved: true });
    } else if (!hasConflictingPageTotals && currentPageTotalReconciliation.conflictObserved) {
      setPageTotalReconciliation({
        ...currentPageTotalReconciliation,
        conflictObserved: false,
        generation: currentPageTotalReconciliation.generation + 1,
      });
    }
  }

  useQuery({
    enabled: sparseViewport && !isPaginated && pageResultsSettled && hasConflictingPageTotals,
    gcTime: 0,
    queryFn: async ({ client }) => {
      await Promise.all(pageOptions.map(({ queryKey }) => client.invalidateQueries({ exact: true, queryKey })));

      return true;
    },
    // One active-range reconciliation per listing, retained total, and range. A persistent disagreement therefore
    // cannot trigger an invalidation loop. Agreement advances the generation so a later same-total conflict can retry.
    queryKey: [
      'gallery',
      'items',
      'page-total-reconciliation',
      filterIdentity,
      stableTotal,
      currentPageTotalReconciliation.generation,
      pageOffsets,
    ],
    staleTime: Infinity,
  });

  // Keep this listing's total for the hook lifetime after its count/page Query data leaves cache. Active page totals
  // replace it only after loaded pages settle and agree.
  const currentTotal =
    observedTotal ?? (isPaginated && isInitialTotalDiscovery && !isFetchingTotal ? queriedTotal : undefined);
  if (
    sparseViewport &&
    currentTotal !== undefined &&
    (knownTotalSnapshot?.filterIdentity !== filterIdentity || knownTotalSnapshot.total !== currentTotal)
  ) {
    setKnownTotalSnapshot({ filterIdentity, total: currentTotal });
  }
  const itemsOptions = galleryItemsInfiniteOptions(
    filter,
    // Infinite page values anchor deep reveals; board/search/view changes reset them to zero.
    isPaginated
      ? { kind: 'anchor', offset: page * GALLERY_PAGE_SIZE }
      : { kind: 'infinite', offset: page * GALLERY_PAGE_SIZE }
  );
  const scopeHash = hashKey(itemsOptions.queryKey);
  const {
    data,
    error,
    errorUpdateCount,
    fetchNextPage,
    hasNextPage,
    isError,
    isFetching,
    isFetchingNextPage,
    isFetchNextPageError,
    isPlaceholderData,
    refetch,
  } = useInfiniteQuery({
    ...itemsOptions,
    enabled: !sparseViewport,
    ...(keepPreviousScope && !sparseViewport ? { placeholderData: keepPreviousData } : {}),
  });
  const pageItemsByOffset = useMemo(
    () => new Map(pageOffsets.map((offset, index) => [offset, pageResults[index]?.data?.items ?? []])),
    [pageOffsets, pageResults]
  );
  // Query forgets which fetch failed as soon as any other starts, so an unrelated refetch (an invalidation after a
  // generation) would turn a failed next page into a "refresh" failure and let scrolling silently retry it. Keep
  // the page count the failure left this scope at: the next page stays failed until a Retry, a page beyond it, or
  // another scope. Recorded from the fetch's own result, in the handler that started it.
  const [failedNextPage, setFailedNextPage] = useState<{ pageCount: number; scopeHash: string } | null>(null);
  // Everything below describes this scope; another scope's placeholder is only ever `previousScopeItems`.
  const queryData = sparseViewport || isPlaceholderData ? undefined : data;
  const backendItems = useMemo(() => {
    if (!sparseViewport) {
      return flattenGalleryItemsData(queryData);
    }

    if (isPaginated) {
      return pageItemsByOffset.get(selectedPageOffset) ?? [];
    }

    return [...pageItemsByOffset.entries()].sort(([left], [right]) => left - right).flatMap(([, items]) => items);
  }, [isPaginated, pageItemsByOffset, queryData, selectedPageOffset, sparseViewport]);
  const backendItemKeys = useMemo(() => new Set(backendItems.map(toGalleryItemKey)), [backendItems]);
  const eligibleRecentKeys = useMemo(
    () =>
      new Set<ReturnType<typeof toGalleryItemKey>>(
        recentImages
          .slice(0, GALLERY_RECENT_IMAGE_LIMIT)
          .map(legacyGeneratedImageToGalleryItem)
          .filter((item) => isRecentItemVisible(item, filter))
          .map(toGalleryItemKey)
      ),
    [filter, recentImages]
  );
  const [authoritativeRecentSnapshot, setAuthoritativeRecentSnapshot] = useState<{
    filterIdentity: string;
    keys: Set<ReturnType<typeof toGalleryItemKey>>;
  }>({ filterIdentity, keys: new Set() });
  const reconciledBackendRecentKeys = useMemo(() => {
    const knownKeys =
      authoritativeRecentSnapshot.filterIdentity === filterIdentity
        ? authoritativeRecentSnapshot.keys
        : new Set<ReturnType<typeof toGalleryItemKey>>();
    const keys = new Set([...knownKeys].filter((key) => eligibleRecentKeys.has(key)));

    for (const key of eligibleRecentKeys) {
      if (backendItemKeys.has(key)) {
        keys.add(key);
      }
    }

    return keys;
  }, [authoritativeRecentSnapshot, backendItemKeys, eligibleRecentKeys, filterIdentity]);
  if (
    authoritativeRecentSnapshot.filterIdentity !== filterIdentity ||
    reconciledBackendRecentKeys.size !== authoritativeRecentSnapshot.keys.size ||
    [...reconciledBackendRecentKeys].some((key) => !authoritativeRecentSnapshot.keys.has(key))
  ) {
    setAuthoritativeRecentSnapshot({ filterIdentity, keys: reconciledBackendRecentKeys });
  }
  const itemSlots = useMemo(
    () =>
      sparseViewport
        ? mapGalleryItemPageSlots({
            pageLocalOffset: isPaginated ? selectedPageOffset : undefined,
            pageOffsets,
            pages: pageResults.map((result) => result.data),
          })
        : new Map<number, GalleryItem>(),
    [isPaginated, pageOffsets, pageResults, selectedPageOffset, sparseViewport]
  );
  // Recents belong at the top of the listing; overlaying them onto a window
  // anchored mid-board would sort them into a part of the list they are
  // nowhere near.
  const shouldOverlayRecentItems = !isPaginated && page === 0;
  const maxRows = isPaginated ? GALLERY_PAGE_SIZE : GALLERY_MAX_ROWS;
  const recentItems = useMemo(
    () =>
      sparseViewport && shouldOverlayRecentItems
        ? getGalleryRecentItems({
            backendItems,
            filter,
            knownBackendItemKeys: reconciledBackendRecentKeys,
            recentImages,
          })
        : [],
    [backendItems, filter, recentImages, reconciledBackendRecentKeys, shouldOverlayRecentItems, sparseViewport]
  );
  const sparseItems = useMemo(
    () =>
      sparseViewport
        ? pageResults.some((result) => result.data) || recentItems.length > 0 || knownTotal === 0
          ? mergeGalleryItemWindow({
              backendItems,
              filter,
              knownBackendItemKeys: reconciledBackendRecentKeys,
              maxRows: Math.max(GALLERY_MAX_ROWS, backendItems.length + GALLERY_RECENT_IMAGE_LIMIT),
              recentImages: shouldOverlayRecentItems ? recentImages : [],
            })
          : null
        : queryData || (shouldOverlayRecentItems && recentImages.length > 0)
          ? mergeGalleryItemWindow({
              backendItems,
              filter,
              maxRows,
              recentImages: shouldOverlayRecentItems ? recentImages : [],
            })
          : null,
    [
      backendItems,
      filter,
      knownTotal,
      maxRows,
      pageResults,
      queryData,
      recentItems.length,
      recentImages,
      reconciledBackendRecentKeys,
      shouldOverlayRecentItems,
      sparseViewport,
    ]
  );
  const scopedItems = useMemo(
    () =>
      mergeGalleryItemWindow({
        backendItems,
        filter,
        maxRows,
        recentImages: shouldOverlayRecentItems ? recentImages : [],
      }),
    [backendItems, filter, maxRows, recentImages, shouldOverlayRecentItems]
  );
  const isNextPageFailed =
    isFetchNextPageError ||
    (failedNextPage?.scopeHash === scopeHash && failedNextPage.pageCount === (queryData?.pages.length ?? 0));
  const { items: denseItems, status: denseStatus } = useMemo(
    () =>
      getGalleryListing(
        { errorUpdateCount, hasData: queryData !== undefined, isError, isFetchNextPageError: isNextPageFailed },
        scopedItems
      ),
    [errorUpdateCount, isError, isNextPageFailed, queryData, scopedItems]
  );
  const previousScopeItems = useMemo(
    () =>
      denseStatus === 'loading' && isPlaceholderData && data ? flattenGalleryItemsData(data).slice(0, maxRows) : null,
    [data, denseStatus, isPlaceholderData, maxRows]
  );
  const total = sparseViewport ? (observedTotal ?? stableTotal) : (queryData?.pages[0]?.total ?? null);
  const hasMore = !sparseViewport && !isPaginated && queryData !== undefined && Boolean(hasNextPage);
  const isWindowTruncated =
    !sparseViewport &&
    isGalleryWindowTruncated({
      hasNextPage: Boolean(hasNextPage),
      isPaginated,
      loadedRowCount: backendItems.length,
      maxRows,
      total,
    });
  const fetchMore = useCallback(async () => {
    const result = await fetchNextPage();

    if (result.isFetchNextPageError) {
      setFailedNextPage({ pageCount: result.data?.pages.length ?? 0, scopeHash });
    }
  }, [fetchNextPage, scopeHash]);
  // A failed page waits for an explicit Retry; scrolling near the end must not hammer it.
  const items = sparseViewport ? sparseItems : denseItems;
  const loadMore = useCallback(() => {
    if (!hasMore || isFetchingNextPage || isNextPageFailed) {
      return;
    }

    void fetchMore();
  }, [fetchMore, hasMore, isFetchingNextPage, isNextPageFailed]);
  const sparseListing = useMemo<GallerySparseListing | undefined>(() => {
    if (!sparseViewport) {
      return undefined;
    }

    const pageStates = new Map<number, GallerySparsePageState>();

    pageOffsets.forEach((offset, index) => {
      const result = pageResults[index];

      if (result) {
        pageStates.set(offset, {
          error: result.error ?? retryingSparseErrors.get(getSparsePageErrorKey(offset)) ?? null,
          isLoading: result.isFetching,
          retry: () =>
            retrySparsePage(
              offset,
              () => result.refetch(),
              result.error ?? retryingSparseErrors.get(getSparsePageErrorKey(offset)) ?? null
            ),
        });
      }
    });

    if (hasUnresolvedTotalError) {
      pageStates.set(0, {
        error: totalError,
        isLoading: isFetchingTotal,
        retry: () => refetchTotal(),
      });
    }

    return { itemSlots, pageStates, recentItems, total };
  }, [
    itemSlots,
    hasUnresolvedTotalError,
    pageOffsets,
    pageResults,
    getSparsePageErrorKey,
    recentItems,
    retrySparsePage,
    retryingSparseErrors,
    sparseViewport,
    total,
    totalError,
    isFetchingTotal,
    refetchTotal,
  ]);
  const firstPageError = sparseListing?.pageStates.get(0)?.error ?? null;
  const pageErrors = pageResults.flatMap((result, index) => {
    const offset = pageOffsets[index];

    if (offset === undefined) {
      return [];
    }

    const error = result.error ?? retryingSparseErrors.get(getSparsePageErrorKey(offset)) ?? null;

    return error ? [{ error, offset, result }] : [];
  });
  const pageError = pageErrors[0]?.error ?? null;
  const activePageLoading = pageResults.some((result) => result.isFetching) || isFetchingTotal;
  // The observer re-runs whatever its current key needs, so a retry started in one scope never lands in another.
  const denseRetry = useCallback(async () => {
    if (denseStatus !== 'more-error') {
      await refetch();
      return;
    }

    setFailedNextPage(null);
    await fetchMore();
  }, [denseStatus, fetchMore, refetch]);
  const isDenseFailed = denseStatus === 'error' || denseStatus === 'stale-error';
  const denseListing = useMemo<GalleryListingState>(
    () => ({
      error,
      isFetchingMore: isFetchingNextPage,
      // An unrelated refetch is not a retry of the failed page.
      isRetrying: denseStatus === 'more-error' ? isFetchingNextPage : isDenseFailed && isFetching,
      retry: denseRetry,
      status: denseStatus,
    }),
    [denseStatus, denseRetry, error, isDenseFailed, isFetching, isFetchingNextPage]
  );
  const sparseError = hasUnresolvedTotalError ? totalError : pageError;
  const sparseHasData = pageResults.some((result) => result.data !== undefined) || knownTotal !== null;
  const sparseErrorPageOffset = hasUnresolvedTotalError ? 0 : (pageErrors[0]?.offset ?? 0);
  const sparseReadStatus = getGalleryReadStatus(
    { errorUpdateCount: sparseError ? 1 : 0, hasData: sparseHasData, isError: sparseError !== null },
    sparseItems?.length ?? 0
  );
  // A failed page beyond the first page is a local continuation error. The grid owns its retry control; the shared
  // status still announces the failure as "more-error" without suggesting that earlier results are stale.
  const sparseStatus =
    sparseError !== null && sparseHasData && sparseErrorPageOffset > 0 ? 'more-error' : sparseReadStatus;
  const retrySparse = useCallback(async () => {
    if (hasUnresolvedTotalError) {
      await refetchTotal();
      return;
    }

    const failedPage = pageErrors[0];

    if (failedPage) {
      await retrySparsePage(failedPage.offset, () => failedPage.result.refetch(), failedPage.error);
    }
  }, [hasUnresolvedTotalError, pageErrors, refetchTotal, retrySparsePage]);
  const sparseListingState = useMemo<GalleryListingState>(
    () => ({
      error: sparseError,
      isFetchingMore: pageResults.some((result) => result.isFetching),
      isRetrying:
        sparseError !== null &&
        (hasUnresolvedTotalError ? isFetchingTotal : pageErrors.some(({ result }) => result.isFetching)),
      retry: retrySparse,
      status: sparseStatus,
    }),
    [hasUnresolvedTotalError, isFetchingTotal, pageErrors, pageResults, retrySparse, sparseError, sparseStatus]
  );
  const listing = sparseViewport ? sparseListingState : denseListing;

  return {
    boards,
    boardsState,
    filter,
    hasMore,
    isLoadingItems: sparseViewport ? activePageLoading : isFetching,
    isWindowTruncated,
    items,
    listing,
    loadMore,
    queryError: sparseViewport ? (pageError ?? firstPageError ?? (hasUnresolvedTotalError ? totalError : null)) : error,
    previousScopeItems,
    selectedBoardId: boardId,
    pinRevealIndex: sparseViewport && !isPaginated ? pinRevealIndex : undefined,
    setVisibleRange: sparseViewport ? setVisibleRange : undefined,
    sparseListing,
    total,
  };
};
