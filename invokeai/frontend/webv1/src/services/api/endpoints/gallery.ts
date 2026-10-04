import type {
  ListGalleryItemNamesArgs,
  ListGalleryItemNamesResult,
  ListGalleryItemsArgs,
  ListGalleryItemsResponse,
} from 'services/api/types';
import { getListGalleryItemsUrl } from 'services/api/util';
import stableHash from 'stable-hash';

import type { ApiTagDescription } from '..';
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
     * Ordered flat name list for virtualized selection — the gallery grid and keyboard
     * navigation run off this. A name ending in `.mp4` is a video.
     *
     * `created_date` selects a date-based virtual board; without it the usual board/category
     * filters apply. Both cases go through this one endpoint.
     */
    listGalleryItemNames: build.query<ListGalleryItemNamesResult, ListGalleryItemNamesArgs>({
      query: (queryArgs) => ({
        url: buildGalleryUrl('item_names', queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => {
        const tags: ApiTagDescription[] = [
          'GalleryItemNameList',
          'FetchOnReconnect',
          { type: 'GalleryItemNameList', id: stableHash(queryArgs) },
        ];
        if (queryArgs.created_date) {
          // Image and video mutations both have to refetch a virtual date's contents, so a
          // date-scoped request also carries each kind's name-list tag.
          tags.push({ type: 'ImageNameList', id: `virtual_${queryArgs.created_date}` });
          tags.push({ type: 'VideoNameList', id: `virtual_${queryArgs.created_date}` });
        }
        return tags;
      },
    }),
  }),
});

// useListGalleryItemNamesQuery is consumed by use-gallery-image-names.ts.
export const { useListGalleryItemNamesQuery } = galleryApi;

/** @knipignore Lands with the paged gallery view / future bulk-DTO consumers; not used today. */
export const { useListGalleryItemsQuery } = galleryApi;
