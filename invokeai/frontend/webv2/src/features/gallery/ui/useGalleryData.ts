import type { GalleryItem, GalleryItemKind } from '@features/gallery/core/items';
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GallerySearchScope, GallerySettings } from '@features/gallery/core/settings';
import type { GalleryBoard, GalleryView, GeneratedImageContract } from '@features/gallery/core/types';

import {
  compareCreatedAt,
  compareGalleryItems,
  getGalleryItemView,
  legacyGeneratedImageToGalleryItem,
  toGalleryItemKey,
} from '@features/gallery/core/items';
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
  getFreshGalleryItemsPage,
  isGalleryItemsPageAwaitingRefresh,
  type GalleryItemsFilter,
} from '@features/gallery/data/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import {
  hashKey,
  keepPreviousData,
  notifyManager,
  useInfiniteQuery,
  useQueries,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { useCallback, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';

import { planGalleryPageOffsets } from './galleryGridLayout';
import {
  getGalleryListing,
  getGalleryReadStatus,
  resolveGallerySelectedBoardId,
  type GalleryReadState,
} from './galleryStateView';
import { useHostGalleryBoards } from './GalleryUiContext';

export interface GalleryListingState extends GalleryReadState {
  isFetchingMore: boolean;
}

export interface GalleryData {
  boards: GalleryBoard[];
  boardsState: GalleryReadState;
  filter: GalleryItemsFilter;
  hasMore: boolean;
  isLoadingItems: boolean;
  /** The items came from it unless the search looks in every board. */
  selectedBoardId: string;
  /** Null until it has something to show, and whenever it failed without data. */
  items: GalleryItem[] | null;
  queryError: Error | null;
  listing: GalleryListingState;
  loadMore: () => void;
  /** Never set once the current scope has data or has failed. */
  previousScopeItems: GalleryItem[] | null;
  total: number | null;
  sparseListing?: GallerySparseListing;
  setVisibleRange?: (range: { endIndexExclusive: number; startIndex: number }) => void;
  /** Keeps a verified index's page subscribed past a stale total until the listing changes or another reveal. */
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
  /** Rendered as separate rows beside the newest end; they never change backend item indexes. */
  recentItems: GalleryItem[];
  total: number | null;
}

/** Never fills hydration gaps or assumes a returned item sits at its requested index. */
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

const useGalleryBoards = ({ settings }: { settings: GallerySettings }) => {
  const { data, error, errorUpdateCount, isError, isFetching, refetch } = useQuery(
    galleryBoardsOptions(getGalleryListingBoardsQuery(settings))
  );
  const boards = useHostGalleryBoards(data);
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
  // Newly starred recents belong to the strip.
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

  if (filter.mediaKind !== undefined && item.kind !== filter.mediaKind) {
    return false;
  }

  const hasMatchingBoard = filter.boardId === ALL_READABLE_BOARDS_ID || filter.boardId === item.boardId;

  return hasMatchingBoard && getGalleryItemView(item) === filter.galleryView;
};

const mergeRecentItemsIntoWindow = ({
  backendItems,
  filter,
  maxRows,
  recentItems,
}: {
  backendItems: readonly GalleryItem[];
  filter: GalleryItemsFilter;
  maxRows: number;
  recentItems: readonly GalleryItem[];
}): GalleryItem[] => {
  const seenItemKeys = new Set<string>();

  const mergedItems = [...recentItems, ...backendItems].filter((item) => {
    const key = toGalleryItemKey(item);

    if (seenItemKeys.has(key)) {
      return false;
    }

    seenItemKeys.add(key);
    return true;
  });

  // Semantic results arrive in relevance order, which a date re-sort would destroy.
  if (!filter.semanticQuery) {
    mergedItems.sort((a, b) => compareGalleryItems(a, b, { orderDir: filter.orderDir }));
  }

  return mergedItems.slice(0, maxRows);
};

export const mergeGalleryItemWindow = ({
  backendItems,
  filter,
  maxRows,
  recentImages,
}: {
  backendItems: readonly GalleryItem[];
  filter: GalleryItemsFilter;
  maxRows: number;
  recentImages: readonly GeneratedImageContract[];
}): GalleryItem[] =>
  mergeRecentItemsIntoWindow({
    backendItems,
    filter,
    maxRows,
    recentItems: getGalleryRecentItems({ backendItems, filter, recentImages }),
  });

const getNewestCreatedAt = (items: readonly GalleryItem[]): string | undefined =>
  items.reduce<string | undefined>(
    (newest, { createdAt }) => (newest === undefined || compareCreatedAt(createdAt, newest) > 0 ? createdAt : newest),
    undefined
  );

export const getGalleryRecentItems = ({
  backendItems,
  knownBackendItemKeys,
  filter,
  newerThan,
  recentImages,
}: {
  backendItems: readonly GalleryItem[];
  knownBackendItemKeys?: ReadonlySet<ReturnType<typeof toGalleryItemKey>>;
  filter: GalleryItemsFilter;
  /** Sparse viewports cannot interleave recents, so ones no newer than the listing's newest are left to it. */
  newerThan?: string;
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
        (newerThan === undefined || compareCreatedAt(item.createdAt, newerThan) > 0) &&
        isRecentItemVisible(item, filter)
    );

  if (!filter.semanticQuery) {
    recentItems.sort((a, b) => compareGalleryItems(a, b, { orderDir: filter.orderDir }));
  }

  return recentItems;
};

export const useGalleryData = ({
  galleryView,
  keepPreviousScope = false,
  mediaKind,
  page,
  projectBoardId,
  recentImages,
  searchScope = 'board',
  searchTerm,
  selectedBoardId,
  semanticQuery = null,
  settings,
  starred,
  sparseViewport = false,
}: {
  galleryView: GalleryView;
  keepPreviousScope?: boolean;
  /** Explicit, not read from `settings`: the picker lists both kinds whatever the Gallery widget shows. */
  mediaKind?: GalleryItemKind;
  page: number;
  projectBoardId: string | null;
  recentImages: readonly GeneratedImageContract[];
  searchScope?: GallerySearchScope;
  searchTerm: string;
  selectedBoardId: string | null;
  semanticQuery?: GallerySemanticReference | null;
  settings: GallerySettings;
  /** The grid passes `false`, its starred filter `true`; the picker omits it to see all. */
  starred?: boolean;
  sparseViewport?: boolean;
}): GalleryData => {
  const { boards, boardsState } = useGalleryBoards({ settings });
  const resolvedSelectedBoardId = resolveGallerySelectedBoardId({ projectBoardId, selectedBoardId }, boards);
  const boardId = searchScope === 'all' ? ALL_READABLE_BOARDS_ID : resolvedSelectedBoardId;
  const queryClient = useQueryClient();
  const isPaginated = settings.paginationMode === 'paginated';
  const dateParse = useMemo(() => parseDateTokens(searchTerm), [searchTerm]);
  const filter = useMemo<GalleryItemsFilter>(
    () => ({
      boardId,
      createdFrom: dateParse.range?.from,
      createdTo: dateParse.range?.to,
      galleryView,
      ...(mediaKind ? { mediaKind } : {}),
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
      mediaKind,
      semanticQuery,
      settings.imageOrderDir,
      starred,
    ]
  );
  // Query keys include the account epoch, so range and total state cannot survive an account switch.
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
  const requestedPageOffset = Math.max(0, page) * GALLERY_PAGE_SIZE;
  const getCachedPageTotal = (offset: number) =>
    queryClient.getQueryData<{ total: number }>(galleryItemsPageOptions(filter, offset).queryKey)?.total;
  // A requested page's own total is fresher than the retained one right after Find in Gallery verified it.
  const requestedPageTotal = sparseViewport && isPaginated ? getCachedPageTotal(requestedPageOffset) : undefined;
  const paginatedTotal = requestedPageTotal ?? knownTotal;
  const clampPaginatedOffset = (total: number | null) =>
    total === null || !Number.isFinite(total)
      ? requestedPageOffset
      : Math.min(
          requestedPageOffset,
          Math.max(0, Math.ceil(Math.max(0, total) / GALLERY_PAGE_SIZE) - 1) * GALLERY_PAGE_SIZE
        );
  const selectedPageOffset = clampPaginatedOffset(paginatedTotal);
  const pageOffsets = useMemo(() => {
    if (!sparseViewport) {
      return [];
    }

    if (isPaginated) {
      if (paginatedTotal === 0) {
        // Keep page zero observed so invalidation can discover items added after an empty result.
        return [0];
      }

      if (knownTotal === null || isFetchingTotal || hasUnresolvedTotalError) {
        return page === 0 ? [0] : [];
      }

      return [selectedPageOffset];
    }

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
    paginatedTotal,
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
  // Observed page totals must agree before the retained total moves, so they reconcile through one refetch.
  const totalCheckOffsets = new Set<number>();

  if (sparseViewport && isPaginated && pageOffsets[0] === selectedPageOffset) {
    const selectedPageTotal = getCachedPageTotal(selectedPageOffset);

    if (requestedPageTotal !== undefined) {
      totalCheckOffsets.add(requestedPageOffset);
    }
    if (selectedPageTotal !== undefined) {
      totalCheckOffsets.add(clampPaginatedOffset(selectedPageTotal));
    }
    totalCheckOffsets.delete(selectedPageOffset);
  }

  const shouldOverlayRecentItems = !isPaginated && page === 0;
  const newestPageOffset =
    filter.orderDir === 'ASC'
      ? Math.floor(Math.max(0, (knownTotal ?? 0) - 1) / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE
      : 0;
  const toPageOptions = (offset: number) => {
    const options = galleryItemsPageOptions(filter, offset);

    // Failed pages wait for their own Retry; broad invalidation must not silently retry visible failures.
    return {
      ...options,
      enabled: queryClient.getQueryState(options.queryKey)?.status !== 'error',
      // Manual patches reset Query's freshness clock: refresh suspended pages without retrying a mounted failed one.
      refetchOnMount:
        shouldOverlayRecentItems &&
        offset === newestPageOffset &&
        isGalleryItemsPageAwaitingRefresh(queryClient, hashKey(options.queryKey))
          ? ('always' as const)
          : true,
    };
  };
  const pageOptions = sparseViewport ? pageOffsets.map(toPageOptions) : [];
  const observedPageOptions = [...pageOptions, ...[...totalCheckOffsets].map(toPageOptions)];
  const observedPageResults = useQueries({ queries: observedPageOptions });
  const pageResults = observedPageResults.slice(0, pageOptions.length);
  const loadedPageTotals = observedPageResults.flatMap((result) => (result.data ? [result.data.total] : []));
  const hasConflictingPageTotals = new Set(loadedPageTotals).size > 1;
  // A page still loading its first data has no total to disagree with.
  const pageResultsSettled = observedPageResults.every(
    (result) => !result.isFetching || (isPaginated && result.data === undefined)
  );
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

  // A settled agreement ends the conflict generation, so a later identical conflict can reconcile again.
  if (sparseViewport && pageResultsSettled && loadedPageTotals.length > 0) {
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
    enabled: sparseViewport && pageResultsSettled && hasConflictingPageTotals,
    gcTime: 0,
    queryFn: async ({ client }) => {
      await Promise.all(observedPageOptions.map(({ queryKey }) => client.invalidateQueries({ exact: true, queryKey })));

      return true;
    },
    // One reconciliation per listing, total and range, so a persistent disagreement cannot loop invalidations.
    queryKey: [
      'gallery',
      'items',
      'page-total-reconciliation',
      filterIdentity,
      stableTotal,
      currentPageTotalReconciliation.generation,
      observedPageOptions.map(({ queryKey }) => queryKey[6]),
    ],
    staleTime: Infinity,
  });

  // Keeps the total after its Query data leaves cache; active totals replace it only once loaded pages agree.
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
  // Query forgets which fetch failed once another starts, so an unrelated refetch would let scrolling silently retry a
  // failed next page. Record it from the fetch's own result.
  const [failedNextPage, setFailedNextPage] = useState<{ pageCount: number; scopeHash: string } | null>(null);
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
  const maxRows = isPaginated ? GALLERY_PAGE_SIZE : GALLERY_MAX_ROWS;
  // Read without observing so scrolling may evict it; its cached state still gates the overlay cutoff.
  const newestPageHash =
    sparseViewport && shouldOverlayRecentItems
      ? hashKey(galleryItemsPageOptions(filter, newestPageOffset).queryKey)
      : null;
  const queryCache = queryClient.getQueryCache();
  const subscribeNewestPage = useCallback(
    (onStoreChange: () => void) => {
      if (newestPageHash === null) {
        return () => undefined;
      }
      // Read after all cache listeners have applied the page lifecycle's fetch evidence.
      const notify = notifyManager.batchCalls(onStoreChange);
      return queryCache.subscribe((event) => {
        if (event.query.queryHash === newestPageHash) {
          notify();
        }
      });
    },
    [newestPageHash, queryCache]
  );
  const getNewestPageSnapshot = useCallback(
    () => (newestPageHash ? getFreshGalleryItemsPage(queryClient, newestPageHash) : undefined),
    [newestPageHash, queryClient]
  );
  const newestPage = useSyncExternalStore(subscribeNewestPage, getNewestPageSnapshot);
  const hasListingNewest =
    newestPage !== undefined &&
    (filter.orderDir !== 'ASC' ||
      newestPageOffset === Math.floor(Math.max(0, newestPage.total - 1) / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE);
  const listingNewestCreatedAt = hasListingNewest ? getNewestCreatedAt(newestPage.items) : undefined;
  const recentItems = useMemo(
    () =>
      hasListingNewest
        ? getGalleryRecentItems({
            backendItems,
            filter,
            knownBackendItemKeys: reconciledBackendRecentKeys,
            newerThan: listingNewestCreatedAt,
            recentImages,
          })
        : [],
    [backendItems, filter, hasListingNewest, listingNewestCreatedAt, recentImages, reconciledBackendRecentKeys]
  );
  const sparseItems = useMemo(
    () =>
      sparseViewport && (pageResults.some((result) => result.data) || recentItems.length > 0 || knownTotal === 0)
        ? mergeRecentItemsIntoWindow({
            backendItems,
            filter,
            maxRows: Math.max(GALLERY_MAX_ROWS, backendItems.length + GALLERY_RECENT_IMAGE_LIMIT),
            recentItems,
          })
        : null,
    [backendItems, filter, knownTotal, pageResults, recentItems, sparseViewport]
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
  // A retry started in one scope never lands in another: the observer re-runs its current key.
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
  // A failed later page is a continuation error, not stale earlier results; the grid owns its retry.
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
    items,
    listing,
    loadMore,
    queryError: sparseViewport ? (pageError ?? firstPageError ?? (hasUnresolvedTotalError ? totalError : null)) : error,
    previousScopeItems,
    selectedBoardId: resolvedSelectedBoardId,
    pinRevealIndex: sparseViewport && !isPaginated ? pinRevealIndex : undefined,
    setVisibleRange: sparseViewport ? setVisibleRange : undefined,
    sparseListing,
    total,
  };
};
