import { useCallback, useEffect, useRef, useState } from 'react';

import {
  listGalleryBoards,
  listGalleryDateBoards,
  listGalleryImages,
  type GalleryBoard,
  type GalleryImage,
  type GalleryView,
} from '@workbench/gallery/api';
import type { GallerySettings } from '@workbench/gallery/settings';

export const GALLERY_PAGE_SIZE = 60;

const GALLERY_CACHE_TTL_MS = 60_000;
const GALLERY_BOARDS_CACHE_LIMIT = 8;
const GALLERY_IMAGES_CACHE_LIMIT = 24;
const GALLERY_INFINITE_WINDOW_CACHE_LIMIT = 24;

interface CacheEntry<T> {
  cachedAt: number;
  value: T;
}

interface BoardsResult {
  boards: GalleryBoard[];
  key: string;
}

interface BackendImagesResult {
  /** Query identity without the pagination window — results stay visible while the window changes. */
  baseKey: string;
  /** Full identity including the pagination window and refresh token. */
  fetchKey: string;
  images: GalleryImage[];
  total: number;
}

const boardsCache = new Map<string, CacheEntry<GalleryBoard[]>>();
const imageResultByFetchKeyCache = new Map<string, CacheEntry<BackendImagesResult>>();
const imageResultByBaseKeyCache = new Map<string, CacheEntry<BackendImagesResult>>();
const infiniteWindowCache = new Map<string, number>();

const isCacheEntryFresh = <T>(entry: CacheEntry<T> | undefined): boolean =>
  entry !== undefined && Date.now() - entry.cachedAt < GALLERY_CACHE_TTL_MS;

const setLimitedMapEntry = <T>(cache: Map<string, T>, key: string, value: T, limit: number): void => {
  cache.delete(key);
  cache.set(key, value);

  while (cache.size > limit) {
    const oldestKey = cache.keys().next().value;

    if (oldestKey === undefined) {
      return;
    }

    cache.delete(oldestKey);
  }
};

const cacheBoards = (key: string, boards: GalleryBoard[]): void => {
  setLimitedMapEntry(boardsCache, key, { cachedAt: Date.now(), value: boards }, GALLERY_BOARDS_CACHE_LIMIT);
};

const cacheImageResult = (result: BackendImagesResult): void => {
  const entry = { cachedAt: Date.now(), value: result };

  setLimitedMapEntry(imageResultByFetchKeyCache, result.fetchKey, entry, GALLERY_IMAGES_CACHE_LIMIT);
  setLimitedMapEntry(imageResultByBaseKeyCache, result.baseKey, entry, GALLERY_IMAGES_CACHE_LIMIT);
};

const getCachedPageCount = (baseKey: string): number => Math.max(1, infiniteWindowCache.get(baseKey) ?? 1);

const cachePageCount = (baseKey: string, pageCount: number): void => {
  setLimitedMapEntry(infiniteWindowCache, baseKey, Math.max(1, pageCount), GALLERY_INFINITE_WINDOW_CACHE_LIMIT);
};

export interface GalleryData {
  boards: GalleryBoard[];
  /** True while more items exist beyond the loaded infinite-scroll window. */
  hasMore: boolean;
  images: GalleryImage[] | null;
  isLoadingImages: boolean;
  /** Grow the infinite-scroll window by one page. No-op while fetching, when exhausted, or in paginated mode. */
  loadMore: () => void;
  /** Optimistically patch the currently loaded image list (e.g. star toggles). */
  patchImages: (getImages: (images: GalleryImage[]) => GalleryImage[]) => void;
  /** Backend total for the current query, or null while unknown. */
  total: number | null;
}

export const useGalleryData = ({
  galleryView,
  onError,
  page,
  recentImagesKey,
  refreshToken,
  searchTerm,
  selectedBoardId,
  settings,
}: {
  galleryView: GalleryView;
  onError: (message: string) => void;
  page: number;
  recentImagesKey: string;
  refreshToken: string;
  searchTerm: string;
  selectedBoardId: string;
  settings: GallerySettings;
}): GalleryData => {
  const {
    boardOrderBy,
    boardOrderDir,
    imageOrderDir,
    paginationMode,
    showArchivedBoards,
    showDateBoards,
    starredFirst,
  } = settings;
  const boardsKey = [
    String(showArchivedBoards),
    boardOrderBy,
    boardOrderDir,
    String(showDateBoards),
    refreshToken,
    recentImagesKey,
  ].join('\0');
  const cachedBoardsEntry = boardsCache.get(boardsKey);
  const isBoardsCacheFresh = isCacheEntryFresh(cachedBoardsEntry);
  const [boardsResult, setBoardsResult] = useState<BoardsResult | null>(() =>
    cachedBoardsEntry ? { boards: cachedBoardsEntry.value, key: boardsKey } : null
  );
  const boards = boardsResult?.key === boardsKey ? boardsResult.boards : (cachedBoardsEntry?.value ?? []);
  const boardId =
    boards.length === 0 || boards.some((board) => board.id === selectedBoardId) ? selectedBoardId : 'none';
  const baseKey = [galleryView, boardId, searchTerm.trim(), imageOrderDir, String(starredFirst)].join('\0');
  const cachedPageCount = getCachedPageCount(baseKey);
  const [infiniteWindow, setInfiniteWindow] = useState<{ baseKey: string; pageCount: number }>(() => ({
    baseKey,
    pageCount: cachedPageCount,
  }));
  const isPaginated = paginationMode === 'paginated';
  const pageCount = infiniteWindow.baseKey === baseKey ? infiniteWindow.pageCount : cachedPageCount;
  const offset = isPaginated ? page * GALLERY_PAGE_SIZE : 0;
  const limit = isPaginated ? GALLERY_PAGE_SIZE : pageCount * GALLERY_PAGE_SIZE;
  const fetchKey = [baseKey, String(offset), String(limit), refreshToken, recentImagesKey].join('\0');
  const exactCachedImageResultEntry = imageResultByFetchKeyCache.get(fetchKey);
  const baseCachedImageResultEntry = imageResultByBaseKeyCache.get(baseKey);
  const cachedImageResultEntry = exactCachedImageResultEntry ?? baseCachedImageResultEntry;
  const isExactImageCacheFresh = isCacheEntryFresh(exactCachedImageResultEntry);
  const [result, setResult] = useState<BackendImagesResult | null>(() => cachedImageResultEntry?.value ?? null);
  const [loadingFetchKey, setLoadingFetchKey] = useState<string | null>(null);
  const stateResult = result !== null && result.baseKey === baseKey ? result : null;
  const visibleResult = exactCachedImageResultEntry?.value ?? stateResult ?? baseCachedImageResultEntry?.value ?? null;
  const images = visibleResult !== null && visibleResult.baseKey === baseKey ? visibleResult.images : null;
  const total = visibleResult !== null && visibleResult.baseKey === baseKey ? visibleResult.total : null;
  const isLoadingImages = loadingFetchKey === fetchKey || (!isExactImageCacheFresh && result?.fetchKey !== fetchKey);
  const hasMore = !isPaginated && total !== null && images !== null && images.length < total;
  const hasMoreRef = useRef(hasMore);
  const isLoadingRef = useRef(isLoadingImages);
  const visibleResultRef = useRef(visibleResult);

  hasMoreRef.current = hasMore;
  isLoadingRef.current = isLoadingImages;
  visibleResultRef.current = visibleResult;

  useEffect(() => {
    if (isBoardsCacheFresh) {
      return;
    }

    let isStale = false;

    Promise.all([
      listGalleryBoards({ includeArchived: showArchivedBoards, orderBy: boardOrderBy, orderDir: boardOrderDir }),
      showDateBoards ? listGalleryDateBoards() : Promise.resolve([]),
    ])
      .then(([backendBoards, dateBoards]) => {
        if (!isStale) {
          const nextBoards = [...backendBoards, ...dateBoards];

          cacheBoards(boardsKey, nextBoards);
          setBoardsResult({ boards: nextBoards, key: boardsKey });
        }
      })
      .catch((error: unknown) => {
        onError(error instanceof Error ? error.message : String(error));
      });

    return () => {
      isStale = true;
    };
  }, [boardOrderBy, boardOrderDir, boardsKey, isBoardsCacheFresh, onError, showArchivedBoards, showDateBoards]);

  useEffect(() => {
    if (isExactImageCacheFresh) {
      return;
    }

    let isStale = false;

    setLoadingFetchKey(fetchKey);
    listGalleryImages({
      boardId,
      galleryView,
      limit,
      offset,
      orderDir: imageOrderDir,
      searchTerm,
      starredFirst,
    })
      .then((backendPage) => {
        if (!isStale) {
          const nextResult = { baseKey, fetchKey, images: backendPage.images, total: backendPage.total };

          cacheImageResult(nextResult);
          setResult(nextResult);
        }
      })
      .catch((error: unknown) => {
        if (!isStale) {
          const fallbackResult = visibleResultRef.current?.baseKey === baseKey ? visibleResultRef.current : null;

          setResult(fallbackResult ? { ...fallbackResult, fetchKey } : { baseKey, fetchKey, images: [], total: 0 });
          onError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        if (!isStale) {
          setLoadingFetchKey((currentFetchKey) => (currentFetchKey === fetchKey ? null : currentFetchKey));
        }
      });

    return () => {
      isStale = true;
    };
  }, [
    baseKey,
    boardId,
    fetchKey,
    galleryView,
    imageOrderDir,
    isExactImageCacheFresh,
    limit,
    offset,
    onError,
    searchTerm,
    starredFirst,
  ]);

  const patchImages = useCallback((getImages: (images: GalleryImage[]) => GalleryImage[]) => {
    const currentResult = visibleResultRef.current;

    if (currentResult === null) {
      return;
    }

    const nextResult = { ...currentResult, images: getImages(currentResult.images) };

    cacheImageResult(nextResult);
    setResult(nextResult);
  }, []);

  const loadMore = useCallback(() => {
    if (!hasMoreRef.current || isLoadingRef.current) {
      return;
    }

    setInfiniteWindow((currentWindow) => {
      const nextPageCount =
        currentWindow.baseKey === baseKey ? currentWindow.pageCount + 1 : getCachedPageCount(baseKey) + 1;

      cachePageCount(baseKey, nextPageCount);

      return { baseKey, pageCount: nextPageCount };
    });
  }, [baseKey]);

  return { boards, hasMore, images, isLoadingImages, loadMore, patchImages, total };
};
