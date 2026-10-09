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
  getGalleryListingBoardsQuery,
  type GalleryItemsFilter,
} from '@features/gallery/data/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import { hashKey, keepPreviousData, useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';

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
  /** The resolved board the items were fetched for. */
  selectedBoardId: string;
  /** Distinguish reaching the window cap from reaching the board end; only truncation needs an explanation. */
  isWindowTruncated: boolean;
  /** The current scope's items; null until it has something to show, and whenever it failed without data. */
  items: GalleryItem[] | null;
  listing: GalleryListingState;
  loadMore: () => void;
  /**
   * The previous scope's items while the current one loads, for consumers that keep them on screen (dimmed and
   * busy) across a scope change. Never set once the current scope has data or has failed.
   */
  previousScopeItems: GalleryItem[] | null;
  total: number | null;
}

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
  maxRows,
  recentImages,
}: {
  backendItems: readonly GalleryItem[];
  filter: GalleryItemsFilter;
  maxRows: number;
  recentImages: readonly GeneratedImageContract[];
}): GalleryItem[] => {
  const backendItemKeys = new Set(backendItems.map(toGalleryItemKey));
  const missingRecentItems = recentImages
    .slice(0, GALLERY_RECENT_IMAGE_LIMIT)
    .map(legacyGeneratedImageToGalleryItem)
    .filter((item) => !backendItemKeys.has(toGalleryItemKey(item)) && isRecentItemVisible(item, filter));
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
}): GalleryData => {
  const { boards, boardsState } = useGalleryBoards({ settings });
  const boardId = resolveGallerySelectedBoardId({ projectBoardId, selectedBoardId }, boards);
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
    ...(keepPreviousScope ? { placeholderData: keepPreviousData } : {}),
  });
  // Query forgets which fetch failed as soon as any other starts, so an unrelated refetch (an invalidation after a
  // generation) would turn a failed next page into a "refresh" failure and let scrolling silently retry it. Keep
  // the page count the failure left this scope at: the next page stays failed until a Retry, a page beyond it, or
  // another scope. Recorded from the fetch's own result, in the handler that started it.
  const [failedNextPage, setFailedNextPage] = useState<{ pageCount: number; scopeHash: string } | null>(null);
  // Everything below describes this scope; another scope's placeholder is only ever `previousScopeItems`.
  const queryData = isPlaceholderData ? undefined : data;
  const backendItems = useMemo(() => {
    if (!isPaginated) {
      return flattenGalleryItemsData(queryData);
    }

    const pageOffset = page * GALLERY_PAGE_SIZE;
    const pageIndex = queryData?.pageParams.indexOf(pageOffset) ?? -1;

    return pageIndex === -1 ? [] : (queryData?.pages[pageIndex]?.items ?? []).slice(0, GALLERY_PAGE_SIZE);
  }, [isPaginated, page, queryData]);
  // Recents belong at the top of the listing; overlaying them onto a window
  // anchored mid-board would sort them into a part of the list they are
  // nowhere near.
  const shouldOverlayRecentItems = !isPaginated && page === 0;
  const maxRows = isPaginated ? GALLERY_PAGE_SIZE : GALLERY_MAX_ROWS;
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
  const { items, status } = useMemo(
    () =>
      getGalleryListing(
        { errorUpdateCount, hasData: queryData !== undefined, isError, isFetchNextPageError: isNextPageFailed },
        scopedItems
      ),
    [errorUpdateCount, isError, isNextPageFailed, queryData, scopedItems]
  );
  const previousScopeItems = useMemo(
    () => (status === 'loading' && isPlaceholderData && data ? flattenGalleryItemsData(data).slice(0, maxRows) : null),
    [data, isPlaceholderData, maxRows, status]
  );
  const total = queryData?.pages[0]?.total ?? null;
  const hasMore = !isPaginated && queryData !== undefined && Boolean(hasNextPage);
  const isWindowTruncated = isGalleryWindowTruncated({
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
  const loadMore = useCallback(() => {
    if (!hasMore || isFetchingNextPage || isNextPageFailed) {
      return;
    }

    void fetchMore();
  }, [fetchMore, hasMore, isFetchingNextPage, isNextPageFailed]);
  // The observer re-runs whatever its current key needs, so a retry started in one scope never lands in another.
  const retry = useCallback(async () => {
    if (status !== 'more-error') {
      await refetch();
      return;
    }

    setFailedNextPage(null);
    await fetchMore();
  }, [fetchMore, refetch, status]);
  const isFailed = status === 'error' || status === 'stale-error';
  const listing = useMemo<GalleryListingState>(
    () => ({
      error,
      isFetchingMore: isFetchingNextPage,
      // An unrelated refetch is not a retry of the failed page.
      isRetrying: status === 'more-error' ? isFetchingNextPage : isFailed && isFetching,
      retry,
      status,
    }),
    [error, isFailed, isFetching, isFetchingNextPage, retry, status]
  );

  return {
    boards,
    boardsState,
    filter,
    hasMore,
    isWindowTruncated,
    items,
    listing,
    loadMore,
    previousScopeItems,
    selectedBoardId: boardId,
    total,
  };
};
