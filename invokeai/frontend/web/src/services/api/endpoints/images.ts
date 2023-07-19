import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { PatchCollection } from '@reduxjs/toolkit/dist/query/core/buildThunks';
import { dateComparator } from 'common/util/dateComparator';
import {
  ASSETS_CATEGORIES,
  BoardId,
  IMAGE_CATEGORIES,
} from 'features/gallery/store/gallerySlice';
import { omit } from 'lodash-es';
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
import { getCacheAction } from './util';

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

export type ImageCache = EntityState<ImageDTO> & { total: number };

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

export const SYSTEM_BOARDS = ['images', 'assets', 'no_board', 'batch'];

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
      providesTags: (result, error, { board_id, categories }) => [
        // Make the tags the same as the cache key
        { type: 'ImageList', id: getListImagesUrl({ board_id, categories }) },
      ],
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

          // update the `getImageDTO` cache for each image
          imagesSelectors.selectAll(data).forEach((imageDTO) => {
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
    getIntermediatesCount: build.query<number, void>({
      query: () => ({ url: getListImagesUrl({ is_intermediate: true }) }),
      providesTags: ['IntermediatesCount'],
      transformResponse: (response: OffsetPaginatedResults_ImageDTO_) => {
        return response.total;
      },
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
    clearIntermediates: build.mutation<number, void>({
      query: () => ({ url: `images/clear-intermediates`, method: 'POST' }),
      invalidatesTags: ['IntermediatesCount'],
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
          removeFromCacheKeys.push({ board_id });
        } else {
          // TODO: No Board
        }

        // TODO: Batch

        const patches: PatchCollection[] = [];
        removeFromCacheKeys.forEach((cacheKey) => {
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                cacheKey,
                (draft) => {
                  imagesAdapter.removeOne(draft, image_name);
                  draft.total = Math.max(draft.total - 1, 0);
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
      ImageDTO,
      {
        imageDTO: ImageDTO;
        // For now, we will not allow image categories to change
        changes: Omit<ImageChanges, 'image_category'>;
      }
    >({
      query: ({ imageDTO, changes }) => ({
        url: `images/${imageDTO.image_name}`,
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (result, error, { imageDTO }) => [
        { type: 'Image', id: imageDTO.image_name },
      ],
      async onQueryStarted(
        { imageDTO: oldImageDTO, changes: _changes },
        { dispatch, queryFulfilled, getState }
      ) {
        // TODO: Should we handle changes to boards via this mutation? Seems reasonable...

        // let's be extra-sure we do not accidentally change categories
        const changes = omit(_changes, 'image_category');

        /**
         * Cache changes for `updateImage`:
         * - Update the ImageDTO
         * - Update the image in "All Images" board:
         *   - IF it is in the date range represented by the cache:
         *     - add the image IF it is not already in the cache & update the total
         *     - ELSE update the image IF it is already in the cache
         * - IF the image has a board:
         *   - Update the image in it's own board
         *   - ELSE Update the image in the "No Board" board (TODO)
         */

        const patches: PatchCollection[] = [];
        const { image_name, board_id, image_category } = oldImageDTO;
        const categories = IMAGE_CATEGORIES.includes(image_category)
          ? IMAGE_CATEGORIES
          : ASSETS_CATEGORIES;

        // TODO: No Board

        // Update `getImageDTO` cache
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'getImageDTO',
              image_name,
              (draft) => {
                Object.assign(draft, changes);
              }
            )
          )
        );

        // Update the "All Image" or "All Assets" board
        const queryArgsToUpdate: ListImagesArgs[] = [{ categories }];

        if (board_id) {
          // We also need to update the user board
          queryArgsToUpdate.push({ board_id });
        }

        queryArgsToUpdate.forEach((queryArg) => {
          const { data } = imagesApi.endpoints.listImages.select(queryArg)(
            getState()
          );

          const cacheAction = getCacheAction(data, oldImageDTO);

          if (['update', 'add'].includes(cacheAction)) {
            patches.push(
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArg,
                  (draft) => {
                    // One of the common changes is to make a canvas intermediate a non-intermediate,
                    // i.e. save a canvas image to the gallery.
                    // If that was the change, need to add the image to the cache instead of updating
                    // the existing cache entry.
                    if (
                      changes.is_intermediate === false ||
                      cacheAction === 'add'
                    ) {
                      // add it to the cache
                      imagesAdapter.addOne(draft, {
                        ...oldImageDTO,
                        ...changes,
                      });
                      draft.total += 1;
                    } else if (cacheAction === 'update') {
                      // just update it
                      imagesAdapter.updateOne(draft, {
                        id: image_name,
                        changes,
                      });
                    }
                  }
                )
              )
            );
          }
        });

        try {
          await queryFulfilled;
        } catch {
          patches.forEach((patchResult) => patchResult.undo());
        }
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
      async onQueryStarted(
        { file, image_category, is_intermediate, postUploadAction },
        { dispatch, queryFulfilled }
      ) {
        try {
          const { data: imageDTO } = await queryFulfilled;

          if (imageDTO.is_intermediate) {
            // Don't add it to anything
            return;
          }

          // Add the image to the "All Images" / "All Assets" board
          const queryArg = {
            categories: IMAGE_CATEGORIES.includes(image_category)
              ? IMAGE_CATEGORIES
              : ASSETS_CATEGORIES,
          };

          dispatch(
            imagesApi.util.updateQueryData('listImages', queryArg, (draft) => {
              imagesAdapter.addOne(draft, imageDTO);
              draft.total = draft.total + 1;
            })
          );
        } catch {
          // no-op
        }
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

        const { image_name, board_id: old_board_id } = oldImageDTO;

        // Figure out the `listImages` caches that we need to update
        const removeFromQueryArgs: ListImagesArgs[] = [];

        // TODO: No Board
        // TODO: Batch

        // Remove from No Board
        removeFromQueryArgs.push({ board_id: 'none' });

        // Remove from old board
        if (old_board_id) {
          removeFromQueryArgs.push({ board_id: old_board_id });
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
                  // sanity check
                  if (draft.ids.includes(image_name)) {
                    imagesAdapter.removeOne(draft, image_name);
                    draft.total = Math.max(draft.total - 1, 0);
                  }
                }
              )
            )
          );
        });

        // We only need to add to the cache if the board is not a system board
        if (!SYSTEM_BOARDS.includes(board_id)) {
          const queryArgs = { board_id };
          const { data } = imagesApi.endpoints.listImages.select(queryArgs)(
            getState()
          );

          const cacheAction = getCacheAction(data, oldImageDTO);

          if (['add', 'update'].includes(cacheAction)) {
            // Do the "Add to" cache updates
            patches.push(
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArgs,
                  (draft) => {
                    if (cacheAction === 'add') {
                      imagesAdapter.addOne(draft, newImageDTO);
                      draft.total += 1;
                    } else {
                      imagesAdapter.updateOne(draft, {
                        id: image_name,
                        changes: { board_id },
                      });
                    }
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
         * - Remove from `old_board_id`
         * - Update the ImageDTO
         */

        const { image_name, board_id: old_board_id } = imageDTO;

        // TODO: Batch

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
          const oldBoardQueryArgs = { board_id: old_board_id };
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                oldBoardQueryArgs,
                (draft) => {
                  // sanity check
                  if (draft.ids.includes(image_name)) {
                    imagesAdapter.removeOne(draft, image_name);
                    draft.total = Math.max(draft.total - 1, 0);
                  }
                }
              )
            )
          );
        }

        // Add to "No Board"
        const noBoardQueryArgs = { board_id: 'none' };
        const { data } = imagesApi.endpoints.listImages.select(
          noBoardQueryArgs
        )(getState());

        // Check if we need to make any cache changes
        const cacheAction = getCacheAction(data, imageDTO);

        if (['add', 'update'].includes(cacheAction)) {
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                noBoardQueryArgs,
                (draft) => {
                  if (cacheAction === 'add') {
                    imagesAdapter.addOne(draft, imageDTO);
                    draft.total += 1;
                  } else {
                    imagesAdapter.updateOne(draft, {
                      id: image_name,
                      changes: { board_id: undefined },
                    });
                  }
                }
              )
            )
          );
        }

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
  useGetIntermediatesCountQuery,
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
