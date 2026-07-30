import type { GalleryItem } from '@features/gallery/core/items';
import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryBoard, GalleryImage, GalleryView, GeneratedImageContract } from '@features/gallery/core/types';

import { normalizeGalleryImage } from '@features/gallery/core/image';
import { legacyGeneratedImageToGalleryItem, toGalleryItemKey } from '@features/gallery/core/items';
import { GALLERY_RECENT_IMAGE_LIMIT } from '@features/gallery/core/recentImages';
import { ALL_READABLE_BOARDS_ID, isDateBoardId } from '@features/gallery/data/backend';
import {
  flattenGalleryImagesData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryImagesInfiniteOptions,
  type GalleryImagesFilter,
} from '@features/gallery/data/queries';
import { parseDateTokens } from '@platform/search/dateTokens';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

export interface GalleryData {
  boards: GalleryBoard[];
  hasMore: boolean;
  images: GalleryImage[] | null;
  isLoadingImages: boolean;
  /**
   * True when the infinite window is full *and* the board holds more images
   * than it can reach. `hasMore` is false in that case exactly as it is at the
   * true end of a board, so the two must be distinguishable: stopping at the
   * end is complete, stopping at the cap is not, and only one of them owes the
   * user an explanation.
   */
  isWindowTruncated: boolean;
  loadMore: () => void;
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

const isRecentImageVisible = (image: GalleryImage, filter: GalleryImagesFilter): boolean => {
  if (
    filter.searchTerm !== '' ||
    filter.createdFrom !== undefined ||
    filter.createdTo !== undefined ||
    isDateBoardId(filter.boardId)
  ) {
    return false;
  }

  const hasMatchingBoard = filter.boardId === ALL_READABLE_BOARDS_ID || filter.boardId === image.boardId;
  const hasMatchingCategory =
    filter.galleryView === 'images' ? image.imageCategory === 'general' : image.imageCategory !== 'general';

  return hasMatchingBoard && hasMatchingCategory;
};

const compareGalleryImages = (
  a: GalleryImage,
  b: GalleryImage,
  { orderDir = 'DESC', starredFirst = false }: GalleryImagesFilter
): number => {
  if (starredFirst && a.starred !== b.starred) {
    return a.starred ? -1 : 1;
  }

  const chronologicalOrder = a.queuedAt.localeCompare(b.queuedAt);

  return orderDir === 'ASC' ? chronologicalOrder : -chronologicalOrder;
};

const isRecentItemVisible = (item: GalleryItem, filter: GalleryImagesFilter): boolean => {
  if (
    filter.searchTerm !== '' ||
    filter.createdFrom !== undefined ||
    filter.createdTo !== undefined ||
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

const compareGalleryItems = (
  a: GalleryItem,
  b: GalleryItem,
  { orderDir = 'DESC', starredFirst = false }: GalleryImagesFilter
): number => {
  if (starredFirst && a.starred !== b.starred) {
    return a.starred ? -1 : 1;
  }

  const direction = orderDir === 'ASC' ? 1 : -1;
  const chronologicalOrder = a.createdAt.localeCompare(b.createdAt);

  if (chronologicalOrder !== 0) {
    return direction * chronologicalOrder;
  }

  const kindOrder = a.kind.localeCompare(b.kind);

  if (kindOrder !== 0) {
    return direction * kindOrder;
  }

  return direction * a.name.localeCompare(b.name);
};

export const mergeGalleryItemWindow = ({
  backendItems,
  filter,
  maxRows,
  recentImages,
}: {
  backendItems: readonly GalleryItem[];
  filter: GalleryImagesFilter;
  maxRows: number;
  recentImages: readonly GeneratedImageContract[];
}): GalleryItem[] => {
  const backendItemKeys = new Set(backendItems.map(toGalleryItemKey));
  const missingRecentItems = recentImages
    .slice(0, GALLERY_RECENT_IMAGE_LIMIT)
    .map(legacyGeneratedImageToGalleryItem)
    .filter((item) => !backendItemKeys.has(toGalleryItemKey(item)) && isRecentItemVisible(item, filter));
  const seenItemKeys = new Set<string>();

  return [...missingRecentItems, ...backendItems]
    .filter((item) => {
      const key = toGalleryItemKey(item);

      if (seenItemKeys.has(key)) {
        return false;
      }

      seenItemKeys.add(key);
      return true;
    })
    .sort((a, b) => compareGalleryItems(a, b, filter))
    .slice(0, maxRows);
};

/** TODO(Task 5): Remove after the Gallery projection consumes GalleryItem. */
export const mergeGalleryImageWindow = ({
  backendImages,
  filter,
  maxRows,
  recentImages,
}: {
  backendImages: readonly GalleryImage[];
  filter: GalleryImagesFilter;
  maxRows: number;
  recentImages: readonly GeneratedImageContract[];
}): GalleryImage[] => {
  const backendImageNames = new Set(backendImages.map((image) => image.imageName));
  const missingRecentImages = recentImages
    .slice(0, GALLERY_RECENT_IMAGE_LIMIT)
    .map((image) => normalizeGalleryImage(image))
    .filter((image) => !backendImageNames.has(image.imageName) && isRecentImageVisible(image, filter));

  if (missingRecentImages.length === 0) {
    return backendImages.length <= maxRows ? (backendImages as GalleryImage[]) : backendImages.slice(0, maxRows);
  }

  return [...missingRecentImages, ...backendImages]
    .sort((a, b) => compareGalleryImages(a, b, filter))
    .slice(0, maxRows);
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
  settings,
}: {
  galleryView: GalleryView;
  page: number;
  recentImages: readonly GeneratedImageContract[];
  searchTerm: string;
  selectedBoardId: string;
  settings: GallerySettings;
}): GalleryData => {
  const { boards } = useGalleryBoards({ settings });
  const boardId =
    boards.length === 0 || boards.some((board) => board.id === selectedBoardId) ? selectedBoardId : 'none';
  const isPaginated = settings.paginationMode === 'paginated';
  const dateParse = useMemo(() => parseDateTokens(searchTerm), [searchTerm]);
  const filter = useMemo<GalleryImagesFilter>(
    () => ({
      boardId,
      createdFrom: dateParse.range?.from,
      createdTo: dateParse.range?.to,
      galleryView,
      orderDir: settings.imageOrderDir,
      searchTerm: dateParse.text,
      starredFirst: settings.starredFirst,
    }),
    [
      boardId,
      dateParse.range?.from,
      dateParse.range?.to,
      dateParse.text,
      galleryView,
      settings.imageOrderDir,
      settings.starredFirst,
    ]
  );
  const {
    data: queryData,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isFetchingNextPage,
  } = useInfiniteQuery(
    galleryImagesInfiniteOptions(
      filter,
      isPaginated ? { kind: 'anchor', offset: page * GALLERY_PAGE_SIZE } : { kind: 'infinite' }
    )
  );
  const backendImages = useMemo(() => {
    if (!isPaginated) {
      return flattenGalleryImagesData(queryData);
    }

    const pageOffset = page * GALLERY_PAGE_SIZE;
    const pageIndex = queryData?.pageParams.indexOf(pageOffset) ?? -1;

    return pageIndex === -1 ? [] : (queryData?.pages[pageIndex]?.images ?? []).slice(0, GALLERY_PAGE_SIZE);
  }, [isPaginated, page, queryData]);
  const shouldOverlayRecentImages = !isPaginated;
  const maxRows = isPaginated ? GALLERY_PAGE_SIZE : GALLERY_MAX_ROWS;
  const images = useMemo(
    () =>
      queryData || (shouldOverlayRecentImages && recentImages.length > 0)
        ? mergeGalleryImageWindow({
            backendImages,
            filter,
            maxRows,
            recentImages: shouldOverlayRecentImages ? recentImages : [],
          })
        : null,
    [backendImages, filter, maxRows, queryData, recentImages, shouldOverlayRecentImages]
  );
  const total = queryData?.pages[0]?.total ?? null;
  const hasMore = !isPaginated && Boolean(hasNextPage);
  const isWindowTruncated = isGalleryWindowTruncated({
    hasNextPage: Boolean(hasNextPage),
    isPaginated,
    loadedRowCount: backendImages.length,
    maxRows,
    total,
  });
  const loadMore = useCallback(() => {
    if (!hasMore || isFetchingNextPage) {
      return;
    }

    void fetchNextPage();
  }, [fetchNextPage, hasMore, isFetchingNextPage]);

  return { boards, hasMore, images, isLoadingImages: isFetching, isWindowTruncated, loadMore, total };
};
