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
import { useInfiniteQuery, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { planGalleryPageOffsets } from './galleryGridLayout';
import { resolveGallerySelectedBoardId } from './galleryStateView';

export interface GalleryData {
  boards: GalleryBoard[];
  filter: GalleryItemsFilter;
  hasMore: boolean;
  isLoadingItems: boolean;
  /** The resolved board the items were fetched for. */
  selectedBoardId: string;
  /** Distinguish reaching the window cap from reaching the board end; only truncation needs an explanation. */
  isWindowTruncated: boolean;
  items: GalleryItem[] | null;
  loadMore: () => void;
  /** The current query's failure, or null while it is healthy. */
  queryError: Error | null;
  total: number | null;
  /** Absolute backend positions for the currently subscribed pages in the main Gallery and picker. */
  sparseListing?: GallerySparseListing;
  setVisibleRange?: (range: { endIndexExclusive: number; startIndex: number }) => void;
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
  const query = useQuery(galleryBoardsOptions(getGalleryListingBoardsQuery(settings)));

  return { boards: query.data ?? EMPTY_BOARDS };
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
  const { boards } = useGalleryBoards({ settings });
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
    if (knownTotal === 0) {
      return [0];
    }

    if (knownTotal === null && !hasRequestedRange) {
      return [0];
    }

    return planGalleryPageOffsets({
      endIndexExclusive: visibleEndIndexExclusive,
      startIndex: visibleStartIndex,
      total: knownTotal,
    });
  }, [
    hasRequestedRange,
    isPaginated,
    knownTotal,
    isFetchingTotal,
    hasUnresolvedTotalError,
    page,
    selectedPageOffset,
    sparseViewport,
    visibleEndIndexExclusive,
    visibleStartIndex,
  ]);
  const pageOptions = sparseViewport ? pageOffsets.map((offset) => galleryItemsPageOptions(filter, offset)) : [];
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
  const {
    data: queryData,
    error: queryError,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isFetchingNextPage,
  } = useInfiniteQuery({
    ...galleryItemsInfiniteOptions(
      filter,
      // The picker still uses this bounded window. Main Gallery uses independent sparse page queries.
      isPaginated
        ? { kind: 'anchor', offset: page * GALLERY_PAGE_SIZE }
        : { kind: 'infinite', offset: page * GALLERY_PAGE_SIZE }
    ),
    enabled: !sparseViewport,
  });
  const pageItemsByOffset = useMemo(
    () => new Map(pageOffsets.map((offset, index) => [offset, pageResults[index]?.data?.items ?? []])),
    [pageOffsets, pageResults]
  );
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
  const items = useMemo(
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
  const total = sparseViewport ? (observedTotal ?? stableTotal) : (queryData?.pages[0]?.total ?? null);
  const hasMore = !sparseViewport && !isPaginated && Boolean(hasNextPage);
  const isWindowTruncated =
    !sparseViewport &&
    isGalleryWindowTruncated({
      hasNextPage: Boolean(hasNextPage),
      isPaginated,
      loadedRowCount: backendItems.length,
      maxRows,
      total,
    });
  const loadMore = useCallback(() => {
    if (!hasMore || isFetchingNextPage) {
      return;
    }

    void fetchNextPage();
  }, [fetchNextPage, hasMore, isFetchingNextPage]);
  const sparseListing = useMemo<GallerySparseListing | undefined>(() => {
    if (!sparseViewport) {
      return undefined;
    }

    const pageStates = new Map<number, GallerySparsePageState>();

    pageOffsets.forEach((offset, index) => {
      const result = pageResults[index];

      if (result) {
        pageStates.set(offset, {
          error: result.error,
          isLoading: result.isFetching,
          retry: () => result.refetch(),
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
    recentItems,
    sparseViewport,
    total,
    totalError,
    isFetchingTotal,
    refetchTotal,
  ]);
  const firstPageError = sparseListing?.pageStates.get(0)?.error ?? null;
  const pageError = pageResults.find((result) => result.error)?.error ?? null;
  const activePageLoading = pageResults.some((result) => result.isFetching) || isFetchingTotal;

  return {
    boards,
    filter,
    hasMore,
    isLoadingItems: sparseViewport ? activePageLoading : isFetching,
    isWindowTruncated,
    items,
    loadMore,
    queryError: sparseViewport
      ? (pageError ?? firstPageError ?? (hasUnresolvedTotalError ? totalError : null))
      : queryError,
    selectedBoardId: boardId,
    setVisibleRange: sparseViewport ? setVisibleRange : undefined,
    sparseListing,
    total,
  };
};
