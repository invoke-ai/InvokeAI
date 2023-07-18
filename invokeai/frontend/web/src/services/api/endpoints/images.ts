import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { PatchCollection } from '@reduxjs/toolkit/dist/query/core/buildThunks';
import { dateComparator } from 'common/util/dateComparator';
import {
  ASSETS_CATEGORIES,
  BoardId,
  IMAGE_CATEGORIES,
} from 'features/gallery/store/gallerySlice';
import { forEach } from 'lodash-es';
import queryString from 'query-string';
import { ApiFullTagDescription, api } from '..';
import { components, paths } from '../schema';
import {
  ImageCategory,
  ImageChanges,
  ImageDTO,
  OffsetPaginatedResults_ImageDTO_,
  PostUploadAction,
} from '../types';
import { getIsImageInDateRange } from './util';

export type ListImagesArgs = NonNullable<
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
export type AdditionalImagesAdapterState = { total: number };

// Create selectors for the adapter.
export const imagesSelectors = imagesAdapter.getSelectors();

// Helper to create the url for the listImages endpoint. Also we use it to create the cache key.
export const getListImagesUrl = (queryArgs: ListImagesArgs) =>
  `images/?${queryString.stringify(queryArgs, { arrayFormat: 'none' })}`;

export const SYSTEM_BOARDS = ['all', 'none', 'batch'];

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
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        try {
          const { data } = await queryFulfilled;
          forEach(data.entities, (imageDTO) => {
            if (!imageDTO) {
              return;
            }

            dispatch(
              imagesApi.util.upsertQueryData(
                'getImageDTO',
                imageDTO.image_name,
                imageDTO
              )
            );
          });
        } catch {
          // no-op
        }
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
    deleteImage: build.mutation<void, ImageDTO>({
      query: ({ image_name }) => ({
        url: `images/${image_name}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Image', id: arg.image_name },
      ],
      async onQueryStarted(imageDTO, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for deleteImage:
         * - Remove from "All Images"
         * - Remove from image's `board_id` if it has one, or "No Board" if not
         * - Remove from "Batch"
         */

        const { image_name, board_id, image_category } = imageDTO;

        // Figure out the `listImages` caches that we need to update
        // That means constructing the possible query args that are serialized into the cache key...

        const removeFromCacheKeys: ListImagesArgs[] = [];
        const categories = IMAGE_CATEGORIES.includes(image_category)
          ? IMAGE_CATEGORIES
          : ASSETS_CATEGORIES;

        // All Images board (e.g. no board)
        removeFromCacheKeys.push({ categories });

        // Board specific
        if (board_id) {
          removeFromCacheKeys.push({ board_id, categories });
        } else {
          // TODO: No Board
          // cacheKeys.push({ board_id: 'none', categories });
        }

        // TODO: Batch - do we want to artificially create an RTK query cache for batch?
        // cacheKeys.push({ board_id: 'batch', categories });

        const patches: PatchCollection[] = [];
        removeFromCacheKeys.forEach((cacheKey) => {
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                cacheKey,
                (draft) => {
                  imagesAdapter.removeOne(draft, image_name);
                  draft.total -= 1;
                }
              )
            )
          );
        });

        try {
          await queryFulfilled;
        } catch {
          patches.forEach((patchResult) => patchResult.undo());
        }
      },
    }),
    updateImage: build.mutation<
      void,
      { image_name: string; changes: ImageChanges }
    >({
      query: ({ image_name, changes }) => ({
        url: `images/${image_name}`,
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (result, error, { image_name }) => [
        { type: 'Image', id: image_name },
      ],
      async onQueryStarted(
        { image_name, changes },
        { dispatch, queryFulfilled, getState }
      ) {
        // TODO: Update caches for updateImage
        // TODO: Should we handle changes to boards in this also?
      },
    }),
    uploadImage: build.mutation<
      ImageDTO,
      {
        file: File;
        image_category: ImageCategory;
        is_intermediate: boolean;
        postUploadAction?: PostUploadAction;
        session_id?: string;
      }
    >({
      query: ({ file, image_category, is_intermediate, session_id }) => {
        const formData = new FormData();
        formData.append('file', file);
        return {
          url: `images/`,
          method: 'POST',
          body: formData,
          params: {
            image_category,
            is_intermediate,
            session_id,
          },
        };
      },
    }),
    addImageToBoard: build.mutation<
      void,
      { board_id: BoardId; imageDTO: ImageDTO }
    >({
      query: ({ board_id, imageDTO }) => {
        const { image_name } = imageDTO;
        return {
          url: `board_images/`,
          method: 'POST',
          body: { board_id, image_name },
        };
      },
      invalidatesTags: (result, error, arg) => [
        { type: 'BoardImage' },
        { type: 'Board', id: arg.board_id },
      ],
      async onQueryStarted(
        { board_id, imageDTO: oldImageDTO },
        { dispatch, queryFulfilled, getState }
      ) {
        /**
         * Cache changes for addImageToBoard:
         * - Remove from "No Board"
         * - Remove from `old_board_id` if it has one
         * - Add to new `board_id`
         *    - IF the image's `created_at` is within the range of the board's cached images
         *    - OR the board cache has length of 0 or 1
         * - Update the `total` for each board whose cache is updated
         * - Update the ImageDTO
         *
         * TODO: maybe total should just be updated in the boards endpoints?
         */

        const {
          image_name,
          image_category,
          board_id: old_board_id,
        } = oldImageDTO;

        // Figure out the `listImages` caches that we need to update
        const removeFromQueryArgs: ListImagesArgs[] = [];
        const categories = IMAGE_CATEGORIES.includes(image_category)
          ? IMAGE_CATEGORIES
          : ASSETS_CATEGORIES;

        // TODO: No Board
        // removeFromCacheKeys.push({ board_id: 'none', categories });

        // TODO: Batch - do we want to artificially create an RTK query cache for batch?
        // cacheKeys.push({ board_id: 'batch', categories });

        // Remove from old board
        if (old_board_id) {
          removeFromQueryArgs.push({ board_id: old_board_id, categories });
        }

        // Store all patch results in case we need to roll back
        const patches: PatchCollection[] = [];

        // Updated imageDTO with new board_id
        const newImageDTO = { ...oldImageDTO, board_id };

        // Update getImageDTO cache
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'getImageDTO',
              image_name,
              (draft) => {
                Object.assign(draft, newImageDTO);
              }
            )
          )
        );

        // Do the "Remove from" cache updates
        removeFromQueryArgs.forEach((queryArgs) => {
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                queryArgs,
                (draft) => {
                  imagesAdapter.removeOne(draft, image_name);
                  draft.total -= 1;
                }
              )
            )
          );
        });

        // We only need to add to the cache if the board is not a system board
        if (!SYSTEM_BOARDS.includes(board_id)) {
          const { data } = imagesApi.endpoints.listImages.select({
            categories,
            board_id,
          })(getState());

          const isInDateRange = getIsImageInDateRange(data, oldImageDTO);
          const isCacheFullyPopulated = data && data.ids.length === data.total;

          if (isCacheFullyPopulated || isInDateRange) {
            // Do the "Add to" cache updates
            patches.push(
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  { board_id, categories },
                  (draft) => {
                    imagesAdapter.addOne(draft, newImageDTO);
                    draft.total += 1;
                  }
                )
              )
            );
          }
        }

        try {
          await queryFulfilled;
        } catch {
          patches.forEach((patchResult) => patchResult.undo());
        }
      },
    }),
    removeImageFromBoard: build.mutation<void, { imageDTO: ImageDTO }>({
      query: ({ imageDTO }) => {
        const { board_id, image_name } = imageDTO;
        return {
          url: `board_images/`,
          method: 'DELETE',
          body: { board_id, image_name },
        };
      },
      invalidatesTags: (result, error, arg) => [
        { type: 'BoardImage' },
        { type: 'Board', id: arg.imageDTO.board_id },
      ],
      async onQueryStarted(
        { imageDTO },
        { dispatch, queryFulfilled, getState }
      ) {
        /**
         * Cache changes for removeImageFromBoard:
         * - Add to "No Board"
         *    - IF the image's `created_at` is within the range of the board's cached images
         * - Add to "All Images"
         *    - IF the image's `created_at` is within the range of the board's cached images
         * - Remove from `old_board_id`
         * - Update the ImageDTO
         */

        const { image_name, image_category, board_id: old_board_id } = imageDTO;

        const categories = IMAGE_CATEGORIES.includes(image_category)
          ? IMAGE_CATEGORIES
          : ASSETS_CATEGORIES;

        // TODO: Batch - do we want to artificially create an RTK query cache for batch?
        // cacheKeys.push({ board_id: 'batch', categories });

        const patches: PatchCollection[] = [];

        // Updated imageDTO with new board_id
        const newImageDTO = { ...imageDTO, board_id: undefined };

        // Update getImageDTO cache
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'getImageDTO',
              image_name,
              (draft) => {
                Object.assign(draft, newImageDTO);
              }
            )
          )
        );

        // Remove from old board
        if (old_board_id) {
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                { board_id: old_board_id, categories },
                (draft) => {
                  imagesAdapter.removeOne(draft, image_name);
                  draft.total -= 1;
                }
              )
            )
          );
        }

        // All Images
        const { data: allImagesData } = imagesApi.endpoints.listImages.select({
          categories,
        })(getState());

        const shouldAddToAllImagesCache = getIsImageInDateRange(
          allImagesData,
          imageDTO
        );

        if (shouldAddToAllImagesCache) {
          // Do the "Add to" cache updates
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                { categories },
                (draft) => {
                  imagesAdapter.addOne(draft, imageDTO);
                  draft.total += 1;
                }
              )
            )
          );
        }

        // TODO: No Board
        // same same but diffrent

        try {
          await queryFulfilled;
        } catch {
          patches.forEach((patchResult) => patchResult.undo());
        }
      },
    }),
  }),
});

export const {
  useListImagesQuery,
  useLazyListImagesQuery,
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
  useDeleteImageMutation,
  useUpdateImageMutation,
  useUploadImageMutation,
  useAddImageToBoardMutation,
  useRemoveImageFromBoardMutation,
  useClearIntermediatesMutation,
} = imagesApi;
