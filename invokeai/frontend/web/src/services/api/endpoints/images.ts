import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { ApiFullTagDescription, api } from '..';
import { components, paths } from '../schema';
import { ImageDTO, OffsetPaginatedResults_ImageDTO_ } from '../types';
import { dateComparator } from 'common/util/dateComparator';
import queryString from 'query-string';

type ListImagesArgs = NonNullable<
  paths['/api/v1/images/']['get']['parameters']['query']
>;

/**
 * This is an unsafe type; the object inside is not guaranteed to be valid.
 */
export type UnsafeImageMetadata = {
  metadata: components['schemas']['CoreMetadata'];
  graph: NonNullable<components['schemas']['Graph']>;
};

// The adapter is not actually the data store - it just provides helper functions to interact
// with some other store of data. We will use the RTK Query cache as that store.
export const imagesAdapter = createEntityAdapter<ImageDTO>({
  selectId: (image) => image.image_name,
  sortComparer: (a, b) => dateComparator(b.updated_at, a.updated_at),
});

// We want to also store the images total in the cache. When we initialize the cache state,
// we will provide this type arg so the adapter knows we want the total.
type AdditionalImagesAdapterState = { total: number };

// Create selectors for the adapter.
const imagesSelectors = imagesAdapter.getSelectors();

// Helper to create the url for the listImages endpoint. Also we use it to create the cache key.
export const getListImagesUrl = (queryArgs: ListImagesArgs) =>
  `images/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`;

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    listImages: build.query<
      EntityState<ImageDTO> & { total: number },
      ListImagesArgs
    >({
      query: (queryArgs) => ({
        // Use the helper to create the URL.
        url: getListImagesUrl(queryArgs),
        method: 'GET',
      }),
      serializeQueryArgs: ({ queryArgs }) => {
        // Create cache & key based on board_id and categories - skip the other args.
        // Offset is the size of the cache, and limit is always the same. Both are provided by
        // the consumer of the query.
        const { board_id, categories } = queryArgs;

        // Just use the same fn used to create the url; it makes an understandable cache key.
        // This cache key is the same for any combo of board_id and categories, doesn't change
        // when offset & limit change.
        const cacheKey = getListImagesUrl({ board_id, categories });
        return cacheKey;
      },
      transformResponse(response: OffsetPaginatedResults_ImageDTO_) {
        const { total, items: images } = response;
        // Use the adapter to convert the response to the right shape, and adding the new total.
        // The trick is to just provide an empty state and add the images array to it. This returns
        // a properly shaped EntityState.
        return imagesAdapter.addMany(
          imagesAdapter.getInitialState<AdditionalImagesAdapterState>({
            total,
          }),
          images
        );
      },
      merge: (cache, response) => {
        // Here we actually update the cache. `response` here is the output of `transformResponse`
        // above. In a similar vein to `transformResponse`, we can use the imagesAdapter to get
        // things in the right shape. Also update the total image count.
        imagesAdapter.addMany(cache, imagesSelectors.selectAll(response));
        cache.total = response.total;
      },
      forceRefetch({ currentArg, previousArg }) {
        // Refetch when the offset changes (which means we are on a new page).
        return currentArg?.offset !== previousArg?.offset;
      },
      // 24 hours - reducing this to a few minutes would reduce memory usage.
      keepUnusedDataFor: 86400,
    }),
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: `images/${image_name}` }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [{ type: 'Image', id: arg }];
        if (result?.board_id) {
          tags.push({ type: 'Board', id: result.board_id });
        }
        return tags;
      },
      keepUnusedDataFor: 86400, // 24 hours
    }),
    getImageMetadata: build.query<UnsafeImageMetadata, string>({
      query: (image_name) => ({ url: `images/${image_name}/metadata` }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [
          { type: 'ImageMetadata', id: arg },
        ];
        return tags;
      },
      keepUnusedDataFor: 86400, // 24 hours
    }),
    clearIntermediates: build.mutation({
      query: () => ({ url: `images/clear-intermediates`, method: 'POST' }),
    }),
  }),
});

export const {
  useListImagesQuery,
  useLazyListImagesQuery,
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
  useClearIntermediatesMutation,
} = imagesApi;
