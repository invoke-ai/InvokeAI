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
  type GalleryItemsFilter,
} from '@features/gallery/data/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

export interface GalleryData {
  boards: GalleryBoard[];
  filter: GalleryItemsFilter;
  hasMore: boolean;
  isLoadingItems: boolean;
  /**
   * True when the infinite window is full *and* the board holds more images
   * than it can reach. `hasMore` is false in that case exactly as it is at the
   * true end of a board, so the two must be distinguishable: stopping at the
   * end is complete, stopping at the cap is not, and only one of them owes the
   * user an explanation.
   */
  isWindowTruncated: boolean;
  items: GalleryItem[] | null;
  loadMore: () => void;
  /** The current query's failure, or null while it is healthy. */
  queryError: Error | null;
  total: number | null;
}

const useGalleryBoards = ({ settings }: { settings: GallerySettings }) => {
  const query = useQuery(
    galleryBoardsOptions({
      includeArchived: settings.showArchivedBoards,
      includeDateBoards: settings.showDateBoards,
      orderBy: settings.boardOrderBy,
      orderDir: settings.boardOrderDir,
    })
  );

  return { boards: query.data ?? [] };
};

const isRecentItemVisible = (item: GalleryItem, filter: GalleryItemsFilter): boolean => {
  if (
    filter.searchTerm !== '' ||
    filter.createdFrom !== undefined ||
    filter.createdTo !== undefined ||
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
    mergedItems.sort((a, b) => compareGalleryItems(a, b, filter));
  }

  return mergedItems.slice(0, maxRows);
};

/**
 * Distinguishes "this board has no more images" from "this board has more
 * images than the infinite window can reach".
 *
 * Both leave `hasNextPage` false, but only the second one is incomplete, and a
 * gallery that silently stops scrolling with a thousand images on the server
 * reads as a bug. Paginated mode is never truncated: every page is reachable
 * by asking for it.
 */
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
  recentImages,
  searchTerm,
  selectedBoardId,
  semanticQuery = null,
  settings,
}: {
  galleryView: GalleryView;
  page: number;
  recentImages: readonly GeneratedImageContract[];
  searchTerm: string;
  selectedBoardId: string;
  /** When set, items come from semantic search (similarity order) instead of the board listing. */
  semanticQuery?: GallerySemanticReference | null;
  settings: GallerySettings;
}): GalleryData => {
  const { boards } = useGalleryBoards({ settings });
  const boardId =
    boards.length === 0 || boards.some((board) => board.id === selectedBoardId) ? selectedBoardId : 'none';
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
      starredFirst: settings.starredFirst,
    }),
    [
      boardId,
      dateParse.range?.from,
      dateParse.range?.to,
      dateParse.text,
      galleryView,
      semanticQuery,
      settings.imageOrderDir,
      settings.starredFirst,
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
      // In infinite mode `page` anchors where the window starts — 0 in normal
      // browsing (every board/search/view change resets it). A reveal to an
      // image deeper than the base window's reach sets it, so the grid can
      // show that part of the listing at all.
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

  return { boards, filter, hasMore, isLoadingItems: isFetching, isWindowTruncated, items, loadMore, queryError, total };
};
