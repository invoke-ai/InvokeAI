import type { GalleryBoardOrderBy, GalleryOrderDir, GalleryView } from '@features/gallery/core/types';

import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
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

export const galleryBoardsOptions = (query: GalleryBoardsQuery = {}) => {
  const owner = captureAccountScope();

  return queryOptions({
    queryFn: async ({ signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const [boards, dateBoards] = await Promise.all([
        listGalleryBoards({ ...query, signal: requestSignal }),
        query.includeDateBoards ? listGalleryDateBoards(requestSignal) : Promise.resolve([]),
      ]);

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return [...boards, ...dateBoards];
    },
    queryKey: galleryKeys.boards(query),
    staleTime: 60_000,
  });
};

export const galleryImagesOptions = (query: GalleryImagesQuery) => {
  const owner = captureAccountScope();

  return queryOptions({
    queryFn: async ({ signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const result = await listGalleryImages({ ...query, signal: requestSignal });

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return result;
    },
    queryKey: galleryKeys.images(query),
    staleTime: 60_000,
  });
};
