import type { GalleryBoardOrderBy, GalleryOrderDir, GalleryView } from '@features/gallery/core/types';

import { queryOptions } from '@tanstack/react-query';

import { listGalleryBoards, listGalleryDateBoards, listGalleryImages } from './backend';

export interface GalleryBoardsQuery {
  includeArchived?: boolean;
  includeDateBoards?: boolean;
  orderBy?: GalleryBoardOrderBy;
  orderDir?: GalleryOrderDir;
  revision?: string;
}

export interface GalleryImagesQuery {
  boardId: string;
  /** Inclusive lower-bound calendar day (YYYY-MM-DD) on created_at. */
  createdFrom?: string;
  /** Inclusive upper-bound calendar day (YYYY-MM-DD) on created_at. */
  createdTo?: string;
  galleryView: GalleryView;
  limit?: number;
  offset?: number;
  orderDir?: GalleryOrderDir;
  revision?: string;
  searchTerm: string;
  starredFirst?: boolean;
}

export const galleryKeys = {
  all: ['gallery'] as const,
  boards: (query: GalleryBoardsQuery) => [...galleryKeys.all, 'boards', query] as const,
  images: (query: GalleryImagesQuery) => [...galleryKeys.all, 'images', query] as const,
};

export const galleryBoardsOptions = (query: GalleryBoardsQuery = {}) =>
  queryOptions({
    queryFn: async () => {
      const [boards, dateBoards] = await Promise.all([
        listGalleryBoards(query),
        query.includeDateBoards ? listGalleryDateBoards() : Promise.resolve([]),
      ]);

      return [...boards, ...dateBoards];
    },
    queryKey: galleryKeys.boards(query),
    staleTime: 60_000,
  });

export const galleryImagesOptions = (query: GalleryImagesQuery) =>
  queryOptions({
    queryFn: () => listGalleryImages(query),
    queryKey: galleryKeys.images(query),
    staleTime: 60_000,
  });
