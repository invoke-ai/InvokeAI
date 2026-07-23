import type {
  GetGalleryItemNamesArgs,
  GetGalleryItemNamesResult,
  ListGalleryItemsArgs,
  ListGalleryItemsResponse,
} from 'services/api/types';
import { getListGalleryItemsUrl } from 'services/api/util';
import stableHash from 'stable-hash';

import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the gallery router.
 * @example
 * buildGalleryUrl('items/') // 'api/v1/gallery/items/'
 */
const buildGalleryUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`gallery/${path}`, query);

export const galleryApi = api.injectEndpoints({
  endpoints: (build) => ({
    /** Paginated polymorphic stream of images + videos, sorted by created_at. */
    listGalleryItems: build.query<ListGalleryItemsResponse, ListGalleryItemsArgs>({
      query: (queryArgs) => ({
        url: getListGalleryItemsUrl(queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        'GalleryItemList',
        'FetchOnReconnect',
        { type: 'GalleryItemList', id: stableHash(queryArgs) },
        { type: 'Board', id: queryArgs.board_id ?? 'none' },
      ],
    }),

    /**
     * Ordered (kind, name) refs for virtualized selection. The gallery grid's name list and
     * keyboard navigation use this — the flat string list is derived by mapping items to `name`.
     */
    getGalleryItemNames: build.query<GetGalleryItemNamesResult, GetGalleryItemNamesArgs>({
      query: (queryArgs) => ({
        url: buildGalleryUrl('items/names', queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        'GalleryItemNameList',
        'FetchOnReconnect',
        { type: 'GalleryItemNameList', id: stableHash(queryArgs) },
      ],
    }),
  }),
});

// useGetGalleryItemNamesQuery is consumed by use-gallery-image-names.ts.
export const { useGetGalleryItemNamesQuery } = galleryApi;

/** @knipignore Lands with the paged gallery view / future bulk-DTO consumers; not used today. */
export const { useListGalleryItemsQuery } = galleryApi;
