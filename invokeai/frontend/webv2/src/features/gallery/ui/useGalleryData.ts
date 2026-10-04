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
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

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
}

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
  const { boards } = useGalleryBoards({ settings });
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
  const {
    data: queryData,
    error: queryError,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isFetchingNextPage,
  } = useInfiniteQuery(
    galleryItemsInfiniteOptions(
      filter,
      // Infinite page values anchor deep reveals; board/search/view changes reset them to zero.
      isPaginated
        ? { kind: 'anchor', offset: page * GALLERY_PAGE_SIZE }
        : { kind: 'infinite', offset: page * GALLERY_PAGE_SIZE }
    )
  );
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
  const items = useMemo(
    () =>
      queryData || (shouldOverlayRecentItems && recentImages.length > 0)
        ? mergeGalleryItemWindow({
            backendItems,
            filter,
            maxRows,
            recentImages: shouldOverlayRecentItems ? recentImages : [],
          })
        : null,
    [backendItems, filter, maxRows, queryData, recentImages, shouldOverlayRecentItems]
  );
  const total = queryData?.pages[0]?.total ?? null;
  const hasMore = !isPaginated && Boolean(hasNextPage);
  const isWindowTruncated = isGalleryWindowTruncated({
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

  return {
    boards,
    filter,
    hasMore,
    isLoadingItems: isFetching,
    isWindowTruncated,
    items,
    loadMore,
    queryError,
    selectedBoardId: boardId,
    total,
  };
};
