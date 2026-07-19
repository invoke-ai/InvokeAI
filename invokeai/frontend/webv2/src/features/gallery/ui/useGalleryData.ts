import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryBoard, GalleryImage, GalleryView } from '@features/gallery/core/types';

import { galleryBoardsOptions, galleryImagesOptions } from '@features/gallery/data/queries';
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useState } from 'react';

export const GALLERY_PAGE_SIZE = 60;

export interface GalleryData {
  boards: GalleryBoard[];
  hasMore: boolean;
  images: GalleryImage[] | null;
  isLoadingImages: boolean;
  loadMore: () => void;
  patchImages: (getImages: (images: GalleryImage[]) => GalleryImage[]) => void;
  total: number | null;
}

const useGalleryBoards = ({ refreshToken, settings }: { refreshToken: string; settings: GallerySettings }) => {
  const query = useQuery(
    galleryBoardsOptions({
      includeArchived: settings.showArchivedBoards,
      includeDateBoards: settings.showDateBoards,
      orderBy: settings.boardOrderBy,
      orderDir: settings.boardOrderDir,
      revision: refreshToken,
    })
  );

  return { boards: query.data ?? [] };
};

export const useGalleryData = ({
  galleryView,
  imageRefreshToken,
  page,
  recentImagesKey,
  refreshToken,
  searchTerm,
  selectedBoardId,
  settings,
}: {
  galleryView: GalleryView;
  imageRefreshToken: string;
  page: number;
  recentImagesKey: string;
  refreshToken: string;
  searchTerm: string;
  selectedBoardId: string;
  settings: GallerySettings;
}): GalleryData => {
  const queryClient = useQueryClient();
  const { boards } = useGalleryBoards({ refreshToken, settings });
  const boardId =
    boards.length === 0 || boards.some((board) => board.id === selectedBoardId) ? selectedBoardId : 'none';
  const baseKey = [galleryView, boardId, searchTerm.trim(), settings.imageOrderDir, String(settings.starredFirst)].join(
    '\0'
  );
  const [infiniteWindow, setInfiniteWindow] = useState({ baseKey, pageCount: 1 });
  const isPaginated = settings.paginationMode === 'paginated';
  const pageCount = infiniteWindow.baseKey === baseKey ? infiniteWindow.pageCount : 1;
  const offset = isPaginated ? page * GALLERY_PAGE_SIZE : 0;
  const limit = isPaginated ? GALLERY_PAGE_SIZE : pageCount * GALLERY_PAGE_SIZE;
  const options = galleryImagesOptions({
    boardId,
    galleryView,
    limit,
    offset,
    orderDir: settings.imageOrderDir,
    revision: `${imageRefreshToken}:${recentImagesKey}`,
    searchTerm,
    starredFirst: settings.starredFirst,
  });
  const query = useQuery({ ...options, placeholderData: keepPreviousData });
  const images = query.data?.images ?? null;
  const total = query.data?.total ?? null;
  const hasMore = !isPaginated && total !== null && images !== null && images.length < total;
  const loadMore = useCallback(() => {
    if (!hasMore || query.isFetching) {
      return;
    }

    setInfiniteWindow((current) => ({
      baseKey,
      pageCount: current.baseKey === baseKey ? current.pageCount + 1 : 2,
    }));
  }, [baseKey, hasMore, query.isFetching]);

  const patchImages = useCallback(
    (getImages: (images: GalleryImage[]) => GalleryImage[]) => {
      queryClient.setQueryData(options.queryKey, (current) =>
        current ? { ...current, images: getImages(current.images) } : current
      );
    },
    [options.queryKey, queryClient]
  );

  return { boards, hasMore, images, isLoadingImages: query.isFetching, loadMore, patchImages, total };
};
