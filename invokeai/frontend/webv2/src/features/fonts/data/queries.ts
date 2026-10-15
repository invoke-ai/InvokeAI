import type { FontCatalogPage } from '@features/fonts/core/types';

import { infiniteQueryOptions, queryOptions, type InfiniteData } from '@tanstack/react-query';

import type { FontListParams } from './keys';

import { listFonts } from './api';
import { fontKeys } from './keys';

export const FONT_PAGE_SIZE = 100;

export { fontKeys } from './keys';

export const fontsQueryOptions = (params: FontListParams = {}) =>
  queryOptions({
    queryFn: ({ signal }) => listFonts(params, signal),
    queryKey: fontKeys.catalog(params),
    staleTime: 30_000,
  });

/** Infinite-query catalog pagination uses bounded offset requests and stops at the server total. */
export const fontsInfiniteQueryOptions = (params: FontListParams = {}) => {
  const { offset: initialOffset = 0, ...filters } = params;
  const limit = params.limit ?? FONT_PAGE_SIZE;
  const queryParams = { ...filters, limit, ...(initialOffset > 0 ? { offset: initialOffset } : {}) };

  return infiniteQueryOptions<
    FontCatalogPage,
    Error,
    InfiniteData<FontCatalogPage, number>,
    ReturnType<typeof fontKeys.infiniteCatalog>,
    number
  >({
    getNextPageParam: (lastPage, _allPages, lastPageParam) => {
      if (lastPage.items.length === 0) {
        return undefined;
      }

      const pageSize = lastPage.limit > 0 ? lastPage.limit : limit;
      const nextOffset = lastPageParam + pageSize;
      return nextOffset < lastPage.total ? nextOffset : undefined;
    },
    initialPageParam: initialOffset,
    queryFn: ({ pageParam, signal }) => listFonts({ ...filters, limit, offset: pageParam }, signal),
    queryKey: fontKeys.infiniteCatalog(queryParams),
    staleTime: 30_000,
  });
};
