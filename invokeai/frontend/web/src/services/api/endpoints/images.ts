import { EntityState, Update } from '@reduxjs/toolkit';
import { PatchCollection } from '@reduxjs/toolkit/dist/query/core/buildThunks';
import {
  ASSETS_CATEGORIES,
  BoardId,
  IMAGE_CATEGORIES,
  IMAGE_LIMIT,
} from 'features/gallery/store/types';
import { getMetadataAndWorkflowFromImageBlob } from 'features/nodes/util/getMetadataAndWorkflowFromImageBlob';
import { keyBy } from 'lodash-es';
import { ApiFullTagDescription, LIST_TAG, api } from '..';
import { components, paths } from '../schema';
import {
  DeleteBoardResult,
  ImageCategory,
  ImageDTO,
  ListImagesArgs,
  OffsetPaginatedResults_ImageDTO_,
  PostUploadAction,
  UnsafeImageMetadata,
} from '../types';
import {
  getCategories,
  getIsImageInDateRange,
  getListImagesUrl,
  imagesAdapter,
  imagesSelectors,
} from '../util';
import { boardsApi } from './boards';
import { ImageMetadataAndWorkflow } from 'features/nodes/types/types';
import { fetchBaseQuery } from '@reduxjs/toolkit/dist/query';
import { $authToken, $projectId } from '../client';

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    listImages: build.query<EntityState<ImageDTO>, ListImagesArgs>({
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
        const { items: images } = response;
        // Use the adapter to convert the response to the right shape.
        // The trick is to just provide an empty state and add the images array to it. This returns
        // a properly shaped EntityState.
        return imagesAdapter.addMany(imagesAdapter.getInitialState(), images);
      },
      merge: (cache, response) => {
        // Here we actually update the cache. `response` here is the output of `transformResponse`
        // above. In a similar vein to `transformResponse`, we can use the imagesAdapter to get
        // things in the right shape.
        imagesAdapter.addMany(cache, imagesSelectors.selectAll(response));
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
      query: (image_name) => ({ url: `images/i/${image_name}` }),
      providesTags: (result, error, image_name) => [
        { type: 'Image', id: image_name },
      ],
      keepUnusedDataFor: 86400, // 24 hours
    }),
    getImageMetadata: build.query<UnsafeImageMetadata, string>({
      query: (image_name) => ({ url: `images/i/${image_name}/metadata` }),
      providesTags: (result, error, image_name) => [
        { type: 'ImageMetadata', id: image_name },
      ],
      keepUnusedDataFor: 86400, // 24 hours
    }),
    getImageMetadataFromFile: build.query<ImageMetadataAndWorkflow, ImageDTO>({
      queryFn: async (args: ImageDTO, api, extraOptions) => {
        const authToken = $authToken.get();
        const projectId = $projectId.get();
        const customBaseQuery = fetchBaseQuery({
          baseUrl: '',
          prepareHeaders: (headers) => {
            if (authToken) {
              headers.set('Authorization', `Bearer ${authToken}`);
            }
            if (projectId) {
              headers.set('project-id', projectId);
            }

            return headers;
          },
          responseHandler: async (res) => {
            return await res.blob();
          },
        });

        const response = await customBaseQuery(
          args.image_url,
          api,
          extraOptions
        );
        const data = await getMetadataAndWorkflowFromImageBlob(
          response.data as Blob
        );
        return { data };
      },
      providesTags: (result, error, image_dto) => [
        { type: 'ImageMetadataFromFile', id: image_dto.image_name },
      ],
      keepUnusedDataFor: 86400, // 24 hours
    }),
    clearIntermediates: build.mutation<number, void>({
      query: () => ({ url: `images/clear-intermediates`, method: 'POST' }),
      invalidatesTags: ['IntermediatesCount'],
    }),
    deleteImage: build.mutation<void, ImageDTO>({
      query: ({ image_name }) => ({
        url: `images/i/${image_name}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, { board_id }) => [
        { type: 'BoardImagesTotal', id: board_id ?? 'none' },
        { type: 'BoardAssetsTotal', id: board_id ?? 'none' },
      ],
      async onQueryStarted(imageDTO, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for `deleteImage`:
         * - NOT POSSIBLE: *remove* from getImageDTO
         * - $cache = [board_id|no_board]/[images|assets]
         * - *remove* from $cache
         */

        const { image_name, board_id } = imageDTO;

        const queryArg = {
          board_id: board_id ?? 'none',
          categories: getCategories(imageDTO),
        };

        const patch = dispatch(
          imagesApi.util.updateQueryData('listImages', queryArg, (draft) => {
            imagesAdapter.removeOne(draft, image_name);
          })
        );

        try {
          await queryFulfilled;
        } catch {
          patch.undo();
        }
      },
    }),
    deleteImages: build.mutation<
      components['schemas']['DeleteImagesFromListResult'],
      { imageDTOs: ImageDTO[] }
    >({
      query: ({ imageDTOs }) => {
        const image_names = imageDTOs.map((imageDTO) => imageDTO.image_name);
        return {
          url: `images/delete`,
          method: 'POST',
          body: {
            image_names,
          },
        };
      },
      invalidatesTags: (result, error, { imageDTOs }) => {
        // for now, assume bulk delete is all on one board
        const boardId = imageDTOs[0]?.board_id;
        return [
          { type: 'BoardImagesTotal', id: boardId ?? 'none' },
          { type: 'BoardAssetsTotal', id: boardId ?? 'none' },
        ];
      },
      async onQueryStarted({ imageDTOs }, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for `deleteImages`:
         * - *remove* the deleted images from their boards
         *
         * Unfortunately we cannot do an optimistic update here due to how immer handles patching
         * arrays. You have to undo *all* patches, else the entity adapter's `ids` array is borked.
         * So we have to wait for the query to complete before updating the cache.
         */
        try {
          const { data } = await queryFulfilled;

          // convert to an object so we can access the successfully delete image DTOs by name
          const groupedImageDTOs = keyBy(imageDTOs, 'image_name');

          data.deleted_images.forEach((image_name) => {
            const imageDTO = groupedImageDTOs[image_name];

            // should never be undefined
            if (imageDTO) {
              const queryArg = {
                board_id: imageDTO.board_id ?? 'none',
                categories: getCategories(imageDTO),
              };
              // remove all deleted images from their boards
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArg,
                  (draft) => {
                    imagesAdapter.removeOne(draft, image_name);
                  }
                )
              );
            }
          });
        } catch {
          //
        }
      },
    }),
    /**
     * Change an image's `is_intermediate` property.
     */
    changeImageIsIntermediate: build.mutation<
      ImageDTO,
      { imageDTO: ImageDTO; is_intermediate: boolean }
    >({
      query: ({ imageDTO, is_intermediate }) => ({
        url: `images/i/${imageDTO.image_name}`,
        method: 'PATCH',
        body: { is_intermediate },
      }),
      invalidatesTags: (result, error, { imageDTO }) => [
        { type: 'BoardImagesTotal', id: imageDTO.board_id ?? 'none' },
        { type: 'BoardAssetsTotal', id: imageDTO.board_id ?? 'none' },
      ],
      async onQueryStarted(
        { imageDTO, is_intermediate },
        { dispatch, queryFulfilled, getState }
      ) {
        /**
         * Cache changes for `changeImageIsIntermediate`:
         * - *update* getImageDTO
         * - $cache = [board_id|no_board]/[images|assets]
         * - IF it is being changed to an intermediate:
         *    - remove from $cache
         * - ELSE (it is being changed to a non-intermediate):
         *    - IF it eligible for insertion into existing $cache:
         *      - *upsert* to $cache
         */

        // Store patches so we can undo if the query fails
        const patches: PatchCollection[] = [];

        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'getImageDTO',
              imageDTO.image_name,
              (draft) => {
                Object.assign(draft, { is_intermediate });
              }
            )
          )
        );

        // $cache = [board_id|no_board]/[images|assets]
        const categories = getCategories(imageDTO);

        if (is_intermediate) {
          // IF it is being changed to an intermediate:
          // remove from $cache
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                { board_id: imageDTO.board_id ?? 'none', categories },
                (draft) => {
                  imagesAdapter.removeOne(draft, imageDTO.image_name);
                }
              )
            )
          );
        } else {
          // ELSE (it is being changed to a non-intermediate):
          const queryArgs = {
            board_id: imageDTO.board_id ?? 'none',
            categories,
          };

          const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(
            getState()
          );

          const { data: total } = IMAGE_CATEGORIES.includes(
            imageDTO.image_category
          )
            ? boardsApi.endpoints.getBoardImagesTotal.select(
                imageDTO.board_id ?? 'none'
              )(getState())
            : boardsApi.endpoints.getBoardAssetsTotal.select(
                imageDTO.board_id ?? 'none'
              )(getState());

          // IF it eligible for insertion into existing $cache
          // "eligible" means either:
          // - The cache is fully populated, with all images in the db cached
          //    OR
          // - The image's `created_at` is within the range of the cached images

          const isCacheFullyPopulated =
            currentCache.data && currentCache.data.ids.length >= (total ?? 0);

          const isInDateRange = getIsImageInDateRange(
            currentCache.data,
            imageDTO
          );

          if (isCacheFullyPopulated || isInDateRange) {
            // *upsert* to $cache
            patches.push(
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArgs,
                  (draft) => {
                    imagesAdapter.upsertOne(draft, imageDTO);
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
    /**
     * Change an image's `session_id` association.
     */
    changeImageSessionId: build.mutation<
      ImageDTO,
      { imageDTO: ImageDTO; session_id: string }
    >({
      query: ({ imageDTO, session_id }) => ({
        url: `images/i/${imageDTO.image_name}`,
        method: 'PATCH',
        body: { session_id },
      }),
      invalidatesTags: (result, error, { imageDTO }) => [
        { type: 'BoardImagesTotal', id: imageDTO.board_id ?? 'none' },
        { type: 'BoardAssetsTotal', id: imageDTO.board_id ?? 'none' },
      ],
      async onQueryStarted(
        { imageDTO, session_id },
        { dispatch, queryFulfilled }
      ) {
        /**
         * Cache changes for `changeImageSessionId`:
         * - *update* getImageDTO
         */

        // Store patches so we can undo if the query fails
        const patches: PatchCollection[] = [];

        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'getImageDTO',
              imageDTO.image_name,
              (draft) => {
                Object.assign(draft, { session_id });
              }
            )
          )
        );

        try {
          await queryFulfilled;
        } catch {
          patches.forEach((patchResult) => patchResult.undo());
        }
      },
    }),
    /**
     * Star a list of images.
     */
    starImages: build.mutation<
      paths['/api/v1/images/unstar']['post']['responses']['200']['content']['application/json'],
      { imageDTOs: ImageDTO[] }
    >({
      query: ({ imageDTOs: images }) => ({
        url: `images/star`,
        method: 'POST',
        body: { image_names: images.map((img) => img.image_name) },
      }),
      invalidatesTags: (result, error, { imageDTOs: images }) => {
        // assume all images are on the same board/category
        if (images[0]) {
          const categories = getCategories(images[0]);
          const boardId = images[0].board_id;
          return [
            {
              type: 'ImageList',
              id: getListImagesUrl({
                board_id: boardId,
                categories,
              }),
            },
          ];
        }
        return [];
      },
      async onQueryStarted(
        { imageDTOs },
        { dispatch, queryFulfilled, getState }
      ) {
        try {
          /**
           * Cache changes for pinImages:
           * - *update* getImageDTO for each image
           * - *upsert* into list for each image
           */

          const { data } = await queryFulfilled;
          const updatedImages = imageDTOs.filter((i) =>
            data.updated_image_names.includes(i.image_name)
          );

          if (!updatedImages[0]) {
            return;
          }

          // assume all images are on the same board/category
          const categories = getCategories(updatedImages[0]);
          const boardId = updatedImages[0].board_id;

          updatedImages.forEach((imageDTO) => {
            const { image_name } = imageDTO;
            dispatch(
              imagesApi.util.updateQueryData(
                'getImageDTO',
                image_name,
                (draft) => {
                  draft.starred = true;
                }
              )
            );

            const queryArgs = {
              board_id: boardId ?? 'none',
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(
              queryArgs
            )(getState());

            const { data: previousTotal } = IMAGE_CATEGORIES.includes(
              imageDTO.image_category
            )
              ? boardsApi.endpoints.getBoardImagesTotal.select(
                  boardId ?? 'none'
                )(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(
                  boardId ?? 'none'
                )(getState());

            const isCacheFullyPopulated =
              currentCache.data &&
              currentCache.data.ids.length >= (previousTotal ?? 0);

            const isInDateRange =
              (previousTotal || 0) >= IMAGE_LIMIT
                ? getIsImageInDateRange(currentCache.data, imageDTO)
                : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArgs,
                  (draft) => {
                    imagesAdapter.upsertOne(draft, {
                      ...imageDTO,
                      starred: true,
                    });
                  }
                )
              );
            }
          });
        } catch {
          // no-op
        }
      },
    }),
    /**
     * Unstar a list of images.
     */
    unstarImages: build.mutation<
      paths['/api/v1/images/unstar']['post']['responses']['200']['content']['application/json'],
      { imageDTOs: ImageDTO[] }
    >({
      query: ({ imageDTOs: images }) => ({
        url: `images/unstar`,
        method: 'POST',
        body: { image_names: images.map((img) => img.image_name) },
      }),
      invalidatesTags: (result, error, { imageDTOs: images }) => {
        // assume all images are on the same board/category
        if (images[0]) {
          const categories = getCategories(images[0]);
          const boardId = images[0].board_id;
          return [
            {
              type: 'ImageList',
              id: getListImagesUrl({
                board_id: boardId,
                categories,
              }),
            },
          ];
        }
        return [];
      },
      async onQueryStarted(
        { imageDTOs },
        { dispatch, queryFulfilled, getState }
      ) {
        try {
          /**
           * Cache changes for unstarImages:
           * - *update* getImageDTO for each image
           * - *upsert* into list for each image
           */

          const { data } = await queryFulfilled;
          const updatedImages = imageDTOs.filter((i) =>
            data.updated_image_names.includes(i.image_name)
          );

          if (!updatedImages[0]) {
            return;
          }
          // assume all images are on the same board/category
          const categories = getCategories(updatedImages[0]);
          const boardId = updatedImages[0].board_id;

          updatedImages.forEach((imageDTO) => {
            const { image_name } = imageDTO;
            dispatch(
              imagesApi.util.updateQueryData(
                'getImageDTO',
                image_name,
                (draft) => {
                  draft.starred = false;
                }
              )
            );

            const queryArgs = {
              board_id: boardId ?? 'none',
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(
              queryArgs
            )(getState());

            const { data: previousTotal } = IMAGE_CATEGORIES.includes(
              imageDTO.image_category
            )
              ? boardsApi.endpoints.getBoardImagesTotal.select(
                  boardId ?? 'none'
                )(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(
                  boardId ?? 'none'
                )(getState());

            const isCacheFullyPopulated =
              currentCache.data &&
              currentCache.data.ids.length >= (previousTotal ?? 0);

            const isInDateRange =
              (previousTotal || 0) >= IMAGE_LIMIT
                ? getIsImageInDateRange(currentCache.data, imageDTO)
                : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArgs,
                  (draft) => {
                    imagesAdapter.upsertOne(draft, {
                      ...imageDTO,
                      starred: false,
                    });
                  }
                )
              );
            }
          });
        } catch {
          // no-op
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
        board_id?: string;
        crop_visible?: boolean;
      }
    >({
      query: ({
        file,
        image_category,
        is_intermediate,
        session_id,
        board_id,
        crop_visible,
      }) => {
        const formData = new FormData();
        formData.append('file', file);
        return {
          url: `images/upload`,
          method: 'POST',
          body: formData,
          params: {
            image_category,
            is_intermediate,
            session_id,
            board_id: board_id === 'none' ? undefined : board_id,
            crop_visible,
          },
        };
      },
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        try {
          /**
           * NOTE: PESSIMISTIC UPDATE
           * Cache changes for `uploadImage`:
           * - IF the image is an intermediate:
           *    - BAIL OUT
           * - *add* to `getImageDTO`
           * - *add* to no_board/assets
           */

          const { data: imageDTO } = await queryFulfilled;

          if (imageDTO.is_intermediate) {
            // Don't add it to anything
            return;
          }

          // *add* to `getImageDTO`
          dispatch(
            imagesApi.util.upsertQueryData(
              'getImageDTO',
              imageDTO.image_name,
              imageDTO
            )
          );

          const categories = getCategories(imageDTO);

          // *add* to no_board/assets
          dispatch(
            imagesApi.util.updateQueryData(
              'listImages',
              {
                board_id: imageDTO.board_id ?? 'none',
                categories,
              },
              (draft) => {
                imagesAdapter.addOne(draft, imageDTO);
              }
            )
          );

          dispatch(
            imagesApi.util.invalidateTags([
              { type: 'BoardImagesTotal', id: imageDTO.board_id ?? 'none' },
              { type: 'BoardAssetsTotal', id: imageDTO.board_id ?? 'none' },
            ])
          );
        } catch {
          // query failed, no action needed
        }
      },
    }),

    deleteBoard: build.mutation<DeleteBoardResult, string>({
      query: (board_id) => ({ url: `boards/${board_id}`, method: 'DELETE' }),
      invalidatesTags: () => [
        { type: 'Board', id: LIST_TAG },
        // invalidate the 'No Board' cache
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: IMAGE_CATEGORIES,
          }),
        },
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: ASSETS_CATEGORIES,
          }),
        },
        { type: 'BoardImagesTotal', id: 'none' },
        { type: 'BoardAssetsTotal', id: 'none' },
      ],
      async onQueryStarted(board_id, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for deleteBoard:
         * - Update every image in the 'getImageDTO' cache that has the board_id
         * - Update every image in the 'All Images' cache that has the board_id
         * - Update every image in the 'All Assets' cache that has the board_id
         * - Invalidate the 'No Board' cache:
         *   Ideally we'd be able to insert all deleted images into the cache, but we don't
         *   have access to the deleted images DTOs - only the names, and a network request
         *   for all of a board's DTOs could be very large. Instead, we invalidate the 'No Board'
         *   cache.
         */

        try {
          const { data } = await queryFulfilled;
          const { deleted_board_images } = data;

          // update getImageDTO caches
          deleted_board_images.forEach((image_id) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'getImageDTO',
                image_id,
                (draft) => {
                  draft.board_id = undefined;
                }
              )
            );
          });

          // update 'All Images' & 'All Assets' caches
          const queryArgsToUpdate = [
            {
              categories: IMAGE_CATEGORIES,
            },
            {
              categories: ASSETS_CATEGORIES,
            },
          ];

          const updates: Update<ImageDTO>[] = deleted_board_images.map(
            (image_name) => ({
              id: image_name,
              changes: { board_id: undefined },
            })
          );

          queryArgsToUpdate.forEach((queryArgs) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                queryArgs,
                (draft) => {
                  imagesAdapter.updateMany(draft, updates);
                }
              )
            );
          });
        } catch {
          //no-op
        }
      },
    }),

    deleteBoardAndImages: build.mutation<DeleteBoardResult, string>({
      query: (board_id) => ({
        url: `boards/${board_id}`,
        method: 'DELETE',
        params: { include_images: true },
      }),
      invalidatesTags: () => [
        { type: 'Board', id: LIST_TAG },
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: IMAGE_CATEGORIES,
          }),
        },
        {
          type: 'ImageList',
          id: getListImagesUrl({
            board_id: 'none',
            categories: ASSETS_CATEGORIES,
          }),
        },
        { type: 'BoardImagesTotal', id: 'none' },
        { type: 'BoardAssetsTotal', id: 'none' },
      ],
      async onQueryStarted(board_id, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for deleteBoardAndImages:
         * - ~~Remove every image in the 'getImageDTO' cache that has the board_id~~
         *   This isn't actually possible, you cannot remove cache entries with RTK Query.
         *   Instead, we rely on the UI to remove all components that use the deleted images.
         * - Remove every image in the 'All Images' cache that has the board_id
         * - Remove every image in the 'All Assets' cache that has the board_id
         */

        try {
          const { data } = await queryFulfilled;
          const { deleted_images } = data;

          // update 'All Images' & 'All Assets' caches
          const queryArgsToUpdate = [
            {
              categories: IMAGE_CATEGORIES,
            },
            {
              categories: ASSETS_CATEGORIES,
            },
          ];

          queryArgsToUpdate.forEach((queryArgs) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                queryArgs,
                (draft) => {
                  imagesAdapter.removeMany(draft, deleted_images);
                }
              )
            );
          });
        } catch {
          //no-op
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
      invalidatesTags: (result, error, { board_id, imageDTO }) => [
        // refresh the board itself
        { type: 'Board', id: board_id },
        // update old board totals
        { type: 'BoardImagesTotal', id: board_id },
        { type: 'BoardAssetsTotal', id: board_id },
        // update new board totals
        { type: 'BoardImagesTotal', id: imageDTO.board_id ?? 'none' },
        { type: 'BoardAssetsTotal', id: imageDTO.board_id ?? 'none' },
      ],
      async onQueryStarted(
        { board_id, imageDTO },
        { dispatch, queryFulfilled, getState }
      ) {
        /**
         * Cache changes for `addImageToBoard`:
         * - *update* getImageDTO
         * - IF it is intermediate:
         *    - BAIL OUT ON FURTHER CHANGES
         * - IF it has an old board_id:
         *    - THEN *remove* from old board_id/[images|assets]
         *    - ELSE *remove* from no_board/[images|assets]
         * - $cache = board_id/[images|assets]
         * - IF it eligible for insertion into existing $cache:
         *    - THEN *add* to $cache
         */

        const patches: PatchCollection[] = [];
        const categories = getCategories(imageDTO);

        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'getImageDTO',
              imageDTO.image_name,
              (draft) => {
                draft.board_id = board_id;
              }
            )
          )
        );

        if (!imageDTO.is_intermediate) {
          // *remove* from [no_board|board_id]/[images|assets]
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                {
                  board_id: imageDTO.board_id ?? 'none',
                  categories,
                },
                (draft) => {
                  imagesAdapter.removeOne(draft, imageDTO.image_name);
                }
              )
            )
          );

          // $cache = board_id/[images|assets]
          const queryArgs = { board_id: board_id ?? 'none', categories };
          const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(
            getState()
          );

          // IF it eligible for insertion into existing $cache
          // "eligible" means either:
          // - The cache is fully populated, with all images in the db cached
          //    OR
          // - The image's `created_at` is within the range of the cached images

          const { data: total } = IMAGE_CATEGORIES.includes(
            imageDTO.image_category
          )
            ? boardsApi.endpoints.getBoardImagesTotal.select(
                imageDTO.board_id ?? 'none'
              )(getState())
            : boardsApi.endpoints.getBoardAssetsTotal.select(
                imageDTO.board_id ?? 'none'
              )(getState());

          const isCacheFullyPopulated =
            currentCache.data && currentCache.data.ids.length >= (total ?? 0);

          const isInDateRange = getIsImageInDateRange(
            currentCache.data,
            imageDTO
          );

          if (isCacheFullyPopulated || isInDateRange) {
            // THEN *add* to $cache
            patches.push(
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArgs,
                  (draft) => {
                    imagesAdapter.addOne(draft, imageDTO);
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
        const { image_name } = imageDTO;
        return {
          url: `board_images/`,
          method: 'DELETE',
          body: { image_name },
        };
      },
      invalidatesTags: (result, error, { imageDTO }) => {
        const { board_id } = imageDTO;
        return [
          // invalidate the image's old board
          { type: 'Board', id: board_id ?? 'none' },
          // update old board totals
          { type: 'BoardImagesTotal', id: board_id ?? 'none' },
          { type: 'BoardAssetsTotal', id: board_id ?? 'none' },
          // update the no_board totals
          { type: 'BoardImagesTotal', id: 'none' },
          { type: 'BoardAssetsTotal', id: 'none' },
        ];
      },
      async onQueryStarted(
        { imageDTO },
        { dispatch, queryFulfilled, getState }
      ) {
        /**
         * Cache changes for removeImageFromBoard:
         * - *update* getImageDTO
         * - *remove* from board_id/[images|assets]
         * - $cache = no_board/[images|assets]
         * - IF it eligible for insertion into existing $cache:
         *    - THEN *upsert* to $cache
         */

        const categories = getCategories(imageDTO);
        const patches: PatchCollection[] = [];

        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'getImageDTO',
              imageDTO.image_name,
              (draft) => {
                draft.board_id = undefined;
              }
            )
          )
        );

        // *remove* from board_id/[images|assets]
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'listImages',
              {
                board_id: imageDTO.board_id ?? 'none',
                categories,
              },
              (draft) => {
                imagesAdapter.removeOne(draft, imageDTO.image_name);
              }
            )
          )
        );

        // $cache = no_board/[images|assets]
        const queryArgs = { board_id: 'none', categories };
        const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(
          getState()
        );

        // IF it eligible for insertion into existing $cache
        // "eligible" means either:
        // - The cache is fully populated, with all images in the db cached
        //    OR
        // - The image's `created_at` is within the range of the cached images

        const { data: total } = IMAGE_CATEGORIES.includes(
          imageDTO.image_category
        )
          ? boardsApi.endpoints.getBoardImagesTotal.select(
              imageDTO.board_id ?? 'none'
            )(getState())
          : boardsApi.endpoints.getBoardAssetsTotal.select(
              imageDTO.board_id ?? 'none'
            )(getState());

        const isCacheFullyPopulated =
          currentCache.data && currentCache.data.ids.length >= (total ?? 0);

        const isInDateRange = getIsImageInDateRange(
          currentCache.data,
          imageDTO
        );

        if (isCacheFullyPopulated || isInDateRange) {
          // THEN *upsert* to $cache
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                queryArgs,
                (draft) => {
                  imagesAdapter.upsertOne(draft, imageDTO);
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
    addImagesToBoard: build.mutation<
      components['schemas']['AddImagesToBoardResult'],
      {
        board_id: string;
        imageDTOs: ImageDTO[];
      }
    >({
      query: ({ board_id, imageDTOs }) => ({
        url: `board_images/batch`,
        method: 'POST',
        body: {
          image_names: imageDTOs.map((i) => i.image_name),
          board_id,
        },
      }),
      invalidatesTags: (result, error, { imageDTOs, board_id }) => {
        //assume all images are being moved from one board for now
        const oldBoardId = imageDTOs[0]?.board_id;
        return [
          // update the destination board
          { type: 'Board', id: board_id ?? 'none' },
          // update new board totals
          { type: 'BoardImagesTotal', id: board_id ?? 'none' },
          { type: 'BoardAssetsTotal', id: board_id ?? 'none' },
          // update old board totals
          { type: 'BoardImagesTotal', id: oldBoardId ?? 'none' },
          { type: 'BoardAssetsTotal', id: oldBoardId ?? 'none' },
          // update the no_board totals
          { type: 'BoardImagesTotal', id: 'none' },
          { type: 'BoardAssetsTotal', id: 'none' },
        ];
      },
      async onQueryStarted(
        { board_id: new_board_id, imageDTOs },
        { dispatch, queryFulfilled, getState }
      ) {
        try {
          const { data } = await queryFulfilled;
          const { added_image_names } = data;

          /**
           * Cache changes for addImagesToBoard:
           * - *update* getImageDTO for each image
           * - *add* to board_id/[images|assets]
           * - *remove* from [old_board_id|no_board]/[images|assets]
           */

          added_image_names.forEach((image_name) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'getImageDTO',
                image_name,
                (draft) => {
                  draft.board_id = new_board_id;
                }
              )
            );

            const imageDTO = imageDTOs.find((i) => i.image_name === image_name);

            if (!imageDTO) {
              return;
            }

            const categories = getCategories(imageDTO);
            const old_board_id = imageDTO.board_id;

            // remove from the old board
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                { board_id: old_board_id ?? 'none', categories },
                (draft) => {
                  imagesAdapter.removeOne(draft, imageDTO.image_name);
                }
              )
            );

            const queryArgs = {
              board_id: new_board_id,
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(
              queryArgs
            )(getState());

            const { data: previousTotal } = IMAGE_CATEGORIES.includes(
              imageDTO.image_category
            )
              ? boardsApi.endpoints.getBoardImagesTotal.select(
                  new_board_id ?? 'none'
                )(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(
                  new_board_id ?? 'none'
                )(getState());

            const isCacheFullyPopulated =
              currentCache.data &&
              currentCache.data.ids.length >= (previousTotal ?? 0);

            const isInDateRange =
              (previousTotal || 0) >= IMAGE_LIMIT
                ? getIsImageInDateRange(currentCache.data, imageDTO)
                : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArgs,
                  (draft) => {
                    imagesAdapter.upsertOne(draft, {
                      ...imageDTO,
                      board_id: new_board_id,
                    });
                  }
                )
              );
            }
          });
        } catch {
          // no-op
        }
      },
    }),
    removeImagesFromBoard: build.mutation<
      components['schemas']['RemoveImagesFromBoardResult'],
      {
        imageDTOs: ImageDTO[];
      }
    >({
      query: ({ imageDTOs }) => ({
        url: `board_images/batch/delete`,
        method: 'POST',
        body: {
          image_names: imageDTOs.map((i) => i.image_name),
        },
      }),
      invalidatesTags: (result, error, { imageDTOs }) => {
        const touchedBoardIds: string[] = [];
        const tags: ApiFullTagDescription[] = [
          { type: 'BoardImagesTotal', id: 'none' },
          { type: 'BoardAssetsTotal', id: 'none' },
        ];

        result?.removed_image_names.forEach((image_name) => {
          const board_id = imageDTOs.find((i) => i.image_name === image_name)
            ?.board_id;

          if (!board_id || touchedBoardIds.includes(board_id)) {
            return;
          }

          tags.push({ type: 'Board', id: board_id });
          tags.push({ type: 'BoardImagesTotal', id: board_id });
          tags.push({ type: 'BoardAssetsTotal', id: board_id });
        });

        return tags;
      },
      async onQueryStarted(
        { imageDTOs },
        { dispatch, queryFulfilled, getState }
      ) {
        try {
          const { data } = await queryFulfilled;
          const { removed_image_names } = data;

          /**
           * Cache changes for removeImagesFromBoard:
           * - *update* getImageDTO for each image
           * - *remove* from old_board_id/[images|assets]
           * - *add* to no_board/[images|assets]
           */

          removed_image_names.forEach((image_name) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'getImageDTO',
                image_name,
                (draft) => {
                  draft.board_id = undefined;
                }
              )
            );

            const imageDTO = imageDTOs.find((i) => i.image_name === image_name);

            if (!imageDTO) {
              return;
            }

            const categories = getCategories(imageDTO);

            // remove from the old board
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                { board_id: imageDTO.board_id ?? 'none', categories },
                (draft) => {
                  imagesAdapter.removeOne(draft, imageDTO.image_name);
                }
              )
            );

            // add to `no_board`
            const queryArgs = {
              board_id: 'none',
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(
              queryArgs
            )(getState());

            const { data: total } = IMAGE_CATEGORIES.includes(
              imageDTO.image_category
            )
              ? boardsApi.endpoints.getBoardImagesTotal.select(
                  imageDTO.board_id ?? 'none'
                )(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(
                  imageDTO.board_id ?? 'none'
                )(getState());

            const isCacheFullyPopulated =
              currentCache.data && currentCache.data.ids.length >= (total ?? 0);

            const isInDateRange =
              (total || 0) >= IMAGE_LIMIT
                ? getIsImageInDateRange(currentCache.data, imageDTO)
                : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData(
                  'listImages',
                  queryArgs,
                  (draft) => {
                    imagesAdapter.upsertOne(draft, {
                      ...imageDTO,
                      board_id: 'none',
                    });
                  }
                )
              );
            }
          });
        } catch {
          // no-op
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
  useDeleteImagesMutation,
  useUploadImageMutation,
  useClearIntermediatesMutation,
  useAddImagesToBoardMutation,
  useRemoveImagesFromBoardMutation,
  useAddImageToBoardMutation,
  useRemoveImageFromBoardMutation,
  useChangeImageIsIntermediateMutation,
  useChangeImageSessionIdMutation,
  useDeleteBoardAndImagesMutation,
  useDeleteBoardMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
  useGetImageMetadataFromFileQuery,
} = imagesApi;
