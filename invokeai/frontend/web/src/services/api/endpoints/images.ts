import type { EntityState, Update } from '@reduxjs/toolkit';
import type { PatchCollection } from '@reduxjs/toolkit/dist/query/core/buildThunks';
import { logger } from 'app/logging/logger';
import type { BoardId } from 'features/gallery/store/types';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES, IMAGE_LIMIT } from 'features/gallery/store/types';
import type { CoreMetadata } from 'features/nodes/types/metadata';
import { zCoreMetadata } from 'features/nodes/types/metadata';
import { addToast } from 'features/system/store/systemSlice';
import { t } from 'i18next';
import { keyBy } from 'lodash-es';
import type { components, paths } from 'services/api/schema';
import type {
  DeleteBoardResult,
  ImageCategory,
  ImageDTO,
  ListImagesArgs,
  OffsetPaginatedResults_ImageDTO_,
  PostUploadAction,
} from 'services/api/types';
import {
  getCategories,
  getIsImageInDateRange,
  getListImagesUrl,
  imagesAdapter,
  imagesSelectors,
} from 'services/api/util';

import type { ApiTagDescription } from '..';
import { api, LIST_TAG } from '..';
import { boardsApi } from './boards';

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    listImages: build.query<EntityState<ImageDTO, string>, ListImagesArgs>({
      query: (queryArgs) => ({
        // Use the helper to create the URL.
        url: getListImagesUrl(queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, { board_id, categories }) => [
        // Make the tags the same as the cache key
        { type: 'ImageList', id: getListImagesUrl({ board_id, categories }) },
        'FetchOnReconnect',
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
            dispatch(imagesApi.util.upsertQueryData('getImageDTO', imageDTO.image_name, imageDTO));
          });
        } catch {
          // no-op
        }
      },
      // 24 hours - reducing this to a few minutes would reduce memory usage.
      keepUnusedDataFor: 86400,
    }),
    getIntermediatesCount: build.query<number, void>({
      query: () => ({ url: 'images/intermediates' }),
      providesTags: ['IntermediatesCount', 'FetchOnReconnect'],
    }),
    clearIntermediates: build.mutation<number, void>({
      query: () => ({ url: `images/intermediates`, method: 'DELETE' }),
      invalidatesTags: ['IntermediatesCount'],
    }),
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: `images/i/${image_name}` }),
      providesTags: (result, error, image_name) => [{ type: 'Image', id: image_name }],
      keepUnusedDataFor: 86400, // 24 hours
    }),
    getImageMetadata: build.query<CoreMetadata | undefined, string>({
      query: (image_name) => ({ url: `images/i/${image_name}/metadata` }),
      providesTags: (result, error, image_name) => [{ type: 'ImageMetadata', id: image_name }],
      transformResponse: (
        response: paths['/api/v1/images/i/{image_name}/metadata']['get']['responses']['200']['content']['application/json']
      ) => {
        if (response) {
          const result = zCoreMetadata.safeParse(response);
          if (result.success) {
            return result.data;
          } else {
            logger('images').warn('Problem parsing metadata');
          }
        }
        return;
      },
      keepUnusedDataFor: 86400, // 24 hours
    }),
    getImageWorkflow: build.query<
      paths['/api/v1/images/i/{image_name}/workflow']['get']['responses']['200']['content']['application/json'],
      string
    >({
      query: (image_name) => ({ url: `images/i/${image_name}/workflow` }),
      providesTags: (result, error, image_name) => [{ type: 'ImageWorkflow', id: image_name }],
      keepUnusedDataFor: 86400, // 24 hours
    }),
    deleteImage: build.mutation<void, ImageDTO>({
      query: ({ image_name }) => ({
        url: `images/i/${image_name}`,
        method: 'DELETE',
      }),
      async onQueryStarted(imageDTO, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for `deleteImage`:
         * - NOT POSSIBLE: *remove* from getImageDTO
         * - $cache = [board_id|no_board]/[images|assets]
         * - *remove* from $cache
         * - decrement the image's board's total
         */

        const { image_name, board_id } = imageDTO;
        const isAsset = ASSETS_CATEGORIES.includes(imageDTO.image_category);

        const queryArg = {
          board_id: board_id ?? 'none',
          categories: getCategories(imageDTO),
        };

        const patches: PatchCollection[] = [];

        patches.push(
          dispatch(
            imagesApi.util.updateQueryData('listImages', queryArg, (draft) => {
              imagesAdapter.removeOne(draft, image_name);
            })
          )
        );

        patches.push(
          dispatch(
            boardsApi.util.updateQueryData(
              isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
              imageDTO.board_id ?? 'none',
              (draft) => {
                draft.total = Math.max(draft.total - 1, 0);
              }
            )
          )
        ); // decrement the image board's total

        try {
          await queryFulfilled;
        } catch {
          patches.forEach((patch) => {
            patch.undo();
          });
        }
      },
    }),
    deleteImages: build.mutation<components['schemas']['DeleteImagesFromListResult'], { imageDTOs: ImageDTO[] }>({
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
      async onQueryStarted({ imageDTOs }, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for `deleteImages`:
         * - *remove* the deleted images from their boards
         * - decrement the images' board's totals
         *
         * Unfortunately we cannot do an optimistic update here due to how immer handles patching
         * arrays. You have to undo *all* patches, else the entity adapter's `ids` array is borked.
         * So we have to wait for the query to complete before updating the cache.
         */
        try {
          const { data } = await queryFulfilled;

          if (data.deleted_images.length < imageDTOs.length) {
            dispatch(
              addToast({
                title: t('gallery.problemDeletingImages'),
                description: t('gallery.problemDeletingImagesDesc'),
                status: 'warning',
              })
            );
          }

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
                imagesApi.util.updateQueryData('listImages', queryArg, (draft) => {
                  imagesAdapter.removeOne(draft, image_name);
                })
              );

              const isAsset = ASSETS_CATEGORIES.includes(imageDTO.image_category);

              // decrement the image board's total
              dispatch(
                boardsApi.util.updateQueryData(
                  isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                  imageDTO.board_id ?? 'none',
                  (draft) => {
                    draft.total = Math.max(draft.total - 1, 0);
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
    changeImageIsIntermediate: build.mutation<ImageDTO, { imageDTO: ImageDTO; is_intermediate: boolean }>({
      query: ({ imageDTO, is_intermediate }) => ({
        url: `images/i/${imageDTO.image_name}`,
        method: 'PATCH',
        body: { is_intermediate },
      }),
      async onQueryStarted({ imageDTO, is_intermediate }, { dispatch, queryFulfilled, getState }) {
        /**
         * Cache changes for `changeImageIsIntermediate`:
         * - *update* getImageDTO
         * - $cache = [board_id|no_board]/[images|assets]
         * - IF it is being changed to an intermediate:
         *    - remove from $cache
         *    - decrement the image's board's total
         * - ELSE (it is being changed to a non-intermediate):
         *    - IF it eligible for insertion into existing $cache:
         *      - *upsert* to $cache
         *    - increment the image's board's total
         */

        // Store patches so we can undo if the query fails
        const patches: PatchCollection[] = [];

        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData('getImageDTO', imageDTO.image_name, (draft) => {
              Object.assign(draft, { is_intermediate });
            })
          )
        );

        // $cache = [board_id|no_board]/[images|assets]
        const categories = getCategories(imageDTO);
        const isAsset = ASSETS_CATEGORIES.includes(imageDTO.image_category);

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

          // decrement the image board's total
          patches.push(
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                imageDTO.board_id ?? 'none',
                (draft) => {
                  draft.total = Math.max(draft.total - 1, 0);
                }
              )
            )
          );
        } else {
          // ELSE (it is being changed to a non-intermediate):

          // increment the image board's total
          patches.push(
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                imageDTO.board_id ?? 'none',
                (draft) => {
                  draft.total += 1;
                }
              )
            )
          );

          const queryArgs = {
            board_id: imageDTO.board_id ?? 'none',
            categories,
          };

          const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(getState());

          const { data } = IMAGE_CATEGORIES.includes(imageDTO.image_category)
            ? boardsApi.endpoints.getBoardImagesTotal.select(imageDTO.board_id ?? 'none')(getState())
            : boardsApi.endpoints.getBoardAssetsTotal.select(imageDTO.board_id ?? 'none')(getState());

          // IF it eligible for insertion into existing $cache
          // "eligible" means either:
          // - The cache is fully populated, with all images in the db cached
          //    OR
          // - The image's `created_at` is within the range of the cached images

          const isCacheFullyPopulated = currentCache.data && currentCache.data.ids.length >= (data?.total ?? 0);

          const isInDateRange = getIsImageInDateRange(currentCache.data, imageDTO);

          if (isCacheFullyPopulated || isInDateRange) {
            // *upsert* to $cache
            patches.push(
              dispatch(
                imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                  imagesAdapter.upsertOne(draft, imageDTO);
                })
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
    changeImageSessionId: build.mutation<ImageDTO, { imageDTO: ImageDTO; session_id: string }>({
      query: ({ imageDTO, session_id }) => ({
        url: `images/i/${imageDTO.image_name}`,
        method: 'PATCH',
        body: { session_id },
      }),
      async onQueryStarted({ imageDTO, session_id }, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for `changeImageSessionId`:
         * - *update* getImageDTO
         */

        // Store patches so we can undo if the query fails
        const patches: PatchCollection[] = [];

        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData('getImageDTO', imageDTO.image_name, (draft) => {
              Object.assign(draft, { session_id });
            })
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
          const boardId = images[0].board_id ?? undefined;

          return [
            {
              type: 'ImageList',
              id: getListImagesUrl({
                board_id: boardId,
                categories,
              }),
            },
            {
              type: 'Board',
              id: boardId,
            },
          ];
        }
        return [];
      },
      async onQueryStarted({ imageDTOs }, { dispatch, queryFulfilled, getState }) {
        try {
          /**
           * Cache changes for pinImages:
           * - *update* getImageDTO for each image
           * - *upsert* into list for each image
           */

          const { data } = await queryFulfilled;
          const updatedImages = imageDTOs.filter((i) => data.updated_image_names.includes(i.image_name));

          if (!updatedImages[0]) {
            return;
          }

          // assume all images are on the same board/category
          const categories = getCategories(updatedImages[0]);
          const boardId = updatedImages[0].board_id;

          updatedImages.forEach((imageDTO) => {
            const { image_name } = imageDTO;
            dispatch(
              imagesApi.util.updateQueryData('getImageDTO', image_name, (draft) => {
                draft.starred = true;
              })
            );

            const queryArgs = {
              board_id: boardId ?? 'none',
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(getState());

            const { data } = IMAGE_CATEGORIES.includes(imageDTO.image_category)
              ? boardsApi.endpoints.getBoardImagesTotal.select(boardId ?? 'none')(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(boardId ?? 'none')(getState());

            const isCacheFullyPopulated = currentCache.data && currentCache.data.ids.length >= (data?.total ?? 0);

            const isInDateRange =
              (data?.total ?? 0) >= IMAGE_LIMIT ? getIsImageInDateRange(currentCache.data, imageDTO) : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                  imagesAdapter.upsertOne(draft, {
                    ...imageDTO,
                    starred: true,
                  });
                })
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
          const boardId = images[0].board_id ?? undefined;
          return [
            {
              type: 'ImageList',
              id: getListImagesUrl({
                board_id: boardId,
                categories,
              }),
            },
            {
              type: 'Board',
              id: boardId,
            },
          ];
        }
        return [];
      },
      async onQueryStarted({ imageDTOs }, { dispatch, queryFulfilled, getState }) {
        try {
          /**
           * Cache changes for unstarImages:
           * - *update* getImageDTO for each image
           * - *upsert* into list for each image
           */

          const { data } = await queryFulfilled;
          const updatedImages = imageDTOs.filter((i) => data.updated_image_names.includes(i.image_name));

          if (!updatedImages[0]) {
            return;
          }
          // assume all images are on the same board/category
          const categories = getCategories(updatedImages[0]);
          const boardId = updatedImages[0].board_id;

          updatedImages.forEach((imageDTO) => {
            const { image_name } = imageDTO;
            dispatch(
              imagesApi.util.updateQueryData('getImageDTO', image_name, (draft) => {
                draft.starred = false;
              })
            );

            const queryArgs = {
              board_id: boardId ?? 'none',
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(getState());

            const { data } = IMAGE_CATEGORIES.includes(imageDTO.image_category)
              ? boardsApi.endpoints.getBoardImagesTotal.select(boardId ?? 'none')(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(boardId ?? 'none')(getState());

            const isCacheFullyPopulated = currentCache.data && currentCache.data.ids.length >= (data?.total ?? 0);

            const isInDateRange =
              (data?.total ?? 0) >= IMAGE_LIMIT ? getIsImageInDateRange(currentCache.data, imageDTO) : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                  imagesAdapter.upsertOne(draft, {
                    ...imageDTO,
                    starred: false,
                  });
                })
              );
            }
          });
        } catch {
          // no-op
        }
      },
    }),
    /**
     * Upload a multiple images.
     */
    uploadMultipleImages: build.mutation<
      ImageDTO[],
      {
        formData: FormData;
        image_category: ImageCategory;
        is_intermediate: boolean;
        postUploadAction?: PostUploadAction;
        session_id?: string;
        board_id?: string;
        crop_visible?: boolean;
      }
    >({
      query: ({
        formData,
        image_category,
        is_intermediate,
        session_id,
        board_id,
        crop_visible,
      }) => {
        return {
          url: `images/upload_multiple`,
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
           * - update the image's board's assets total
           */
          //   const { data: imageDTOs } = await queryFulfilled;
          //   imageDTOs.forEach((imageDTO) => {
          //     if (imageDTO.is_intermediate) {
          //       // Don't add it to anything
          //       return;
          //     }
          //     // *add* to `getImageDTO`
          //     dispatch(
          //       imagesApi.util.upsertQueryData(
          //         'getImageDTO',
          //         imageDTO.image_name,
          //         imageDTO
          //       )
          //     );
          //     const categories = getCategories(imageDTO);
          //     // *add* to no_board/assets
          //     dispatch(
          //       imagesApi.util.updateQueryData(
          //         'listImages',
          //         {
          //           board_id: imageDTO.board_id ?? 'none',
          //           categories,
          //         },
          //         (draft) => {
          //           imagesAdapter.addOne(draft, imageDTO);
          //         }
          //       )
          //     );
          //     // increment new board's total
          //     dispatch(
          //       boardsApi.util.updateQueryData(
          //         'getBoardAssetsTotal',
          //         imageDTO.board_id ?? 'none',
          //         (draft) => {
          //           draft.total += 1;
          //         }
          //       )
          //     );
          //   });
        } catch (error) {
          console.error('Error in onQueryStarted:', error);
          // query failed, no action needed
        }
      },
    }),

    uploadImage: build.mutation<
      ImageDTO,
      {
        file: File; // Expecting a single File object
        image_category: ImageCategory;
        is_intermediate: boolean;
        postUploadAction?: PostUploadAction;
        session_id?: string;
        board_id?: string;
        crop_visible?: boolean;
      }
    >({
      query: ({ file, image_category, is_intermediate, session_id, board_id, crop_visible }) => {
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
           * - update the image's board's assets total
           */

          const { data: imageDTO } = await queryFulfilled;

          if (imageDTO.is_intermediate) {
            // Don't add it to anything
            return;
          }

          // *add* to `getImageDTO`
          dispatch(imagesApi.util.upsertQueryData('getImageDTO', imageDTO.image_name, imageDTO));

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

          // increment new board's total
          dispatch(
            boardsApi.util.updateQueryData('getBoardAssetsTotal', imageDTO.board_id ?? 'none', (draft) => {
              draft.total += 1;
            })
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
         * - set the board's totals to zero
         */

        try {
          const { data } = await queryFulfilled;
          const { deleted_board_images } = data;

          // update getImageDTO caches
          deleted_board_images.forEach((image_id) => {
            dispatch(
              imagesApi.util.updateQueryData('getImageDTO', image_id, (draft) => {
                draft.board_id = undefined;
              })
            );
          });

          // set the board's asset total to 0 (feels unnecessary since we are deleting it?)
          dispatch(
            boardsApi.util.updateQueryData('getBoardAssetsTotal', board_id, (draft) => {
              draft.total = 0;
            })
          );

          // set the board's images total to 0 (feels unnecessary since we are deleting it?)
          dispatch(
            boardsApi.util.updateQueryData('getBoardImagesTotal', board_id, (draft) => {
              draft.total = 0;
            })
          );

          // update 'All Images' & 'All Assets' caches
          const queryArgsToUpdate = [
            {
              categories: IMAGE_CATEGORIES,
            },
            {
              categories: ASSETS_CATEGORIES,
            },
          ];

          const updates: Update<ImageDTO, string>[] = deleted_board_images.map((image_name) => ({
            id: image_name,
            changes: { board_id: undefined },
          }));

          queryArgsToUpdate.forEach((queryArgs) => {
            dispatch(
              imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                imagesAdapter.updateMany(draft, updates);
              })
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
      ],
      async onQueryStarted(board_id, { dispatch, queryFulfilled }) {
        /**
         * Cache changes for deleteBoardAndImages:
         * - ~~Remove every image in the 'getImageDTO' cache that has the board_id~~
         *   This isn't actually possible, you cannot remove cache entries with RTK Query.
         *   Instead, we rely on the UI to remove all components that use the deleted images.
         * - Remove every image in the 'All Images' cache that has the board_id
         * - Remove every image in the 'All Assets' cache that has the board_id
         * - set the board's totals to zero
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
              imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                imagesAdapter.removeMany(draft, deleted_images);
              })
            );
          });

          // set the board's asset total to 0 (feels unnecessary since we are deleting it?)
          dispatch(
            boardsApi.util.updateQueryData('getBoardAssetsTotal', board_id, (draft) => {
              draft.total = 0;
            })
          );

          // set the board's images total to 0 (feels unnecessary since we are deleting it?)
          dispatch(
            boardsApi.util.updateQueryData('getBoardImagesTotal', board_id, (draft) => {
              draft.total = 0;
            })
          );
        } catch {
          //no-op
        }
      },
    }),
    addImageToBoard: build.mutation<void, { board_id: BoardId; imageDTO: ImageDTO }>({
      query: ({ board_id, imageDTO }) => {
        const { image_name } = imageDTO;
        return {
          url: `board_images/`,
          method: 'POST',
          body: { board_id, image_name },
        };
      },
      invalidatesTags: (result, error, { board_id }) => [
        // refresh the board itself
        { type: 'Board', id: board_id },
      ],
      async onQueryStarted({ board_id, imageDTO }, { dispatch, queryFulfilled, getState }) {
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
         * - decrement both old board's total
         * - increment the new board's total
         */

        const patches: PatchCollection[] = [];
        const categories = getCategories(imageDTO);
        const isAsset = ASSETS_CATEGORIES.includes(imageDTO.image_category);
        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData('getImageDTO', imageDTO.image_name, (draft) => {
              draft.board_id = board_id;
            })
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

          // decrement old board's total
          patches.push(
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                imageDTO.board_id ?? 'none',
                (draft) => {
                  draft.total = Math.max(draft.total - 1, 0);
                }
              )
            )
          );

          // increment new board's total
          patches.push(
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                board_id ?? 'none',
                (draft) => {
                  draft.total += 1;
                }
              )
            )
          );

          // $cache = board_id/[images|assets]
          const queryArgs = { board_id: board_id ?? 'none', categories };
          const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(getState());

          // IF it eligible for insertion into existing $cache
          // "eligible" means either:
          // - The cache is fully populated, with all images in the db cached
          //    OR
          // - The image's `created_at` is within the range of the cached images

          const { data } = IMAGE_CATEGORIES.includes(imageDTO.image_category)
            ? boardsApi.endpoints.getBoardImagesTotal.select(imageDTO.board_id ?? 'none')(getState())
            : boardsApi.endpoints.getBoardAssetsTotal.select(imageDTO.board_id ?? 'none')(getState());

          const isCacheFullyPopulated = currentCache.data && currentCache.data.ids.length >= (data?.total ?? 0);

          const isInDateRange = getIsImageInDateRange(currentCache.data, imageDTO);

          if (isCacheFullyPopulated || isInDateRange) {
            // THEN *add* to $cache
            patches.push(
              dispatch(
                imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                  imagesAdapter.addOne(draft, imageDTO);
                })
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
        ];
      },
      async onQueryStarted({ imageDTO }, { dispatch, queryFulfilled, getState }) {
        /**
         * Cache changes for removeImageFromBoard:
         * - *update* getImageDTO
         * - *remove* from board_id/[images|assets]
         * - $cache = no_board/[images|assets]
         * - IF it eligible for insertion into existing $cache:
         *    - THEN *upsert* to $cache
         * - decrement old board's total
         * - increment the new board's total (no board)
         */

        const categories = getCategories(imageDTO);
        const patches: PatchCollection[] = [];
        const isAsset = ASSETS_CATEGORIES.includes(imageDTO.image_category);

        // *update* getImageDTO
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData('getImageDTO', imageDTO.image_name, (draft) => {
              draft.board_id = undefined;
            })
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

        // decrement old board's total
        patches.push(
          dispatch(
            boardsApi.util.updateQueryData(
              isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
              imageDTO.board_id ?? 'none',
              (draft) => {
                draft.total = Math.max(draft.total - 1, 0);
              }
            )
          )
        );

        // increment new board's total (no board)
        patches.push(
          dispatch(
            boardsApi.util.updateQueryData(isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal', 'none', (draft) => {
              draft.total += 1;
            })
          )
        );

        // $cache = no_board/[images|assets]
        const queryArgs = { board_id: 'none', categories };
        const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(getState());

        // IF it eligible for insertion into existing $cache
        // "eligible" means either:
        // - The cache is fully populated, with all images in the db cached
        //    OR
        // - The image's `created_at` is within the range of the cached images

        const { data } = IMAGE_CATEGORIES.includes(imageDTO.image_category)
          ? boardsApi.endpoints.getBoardImagesTotal.select(imageDTO.board_id ?? 'none')(getState())
          : boardsApi.endpoints.getBoardAssetsTotal.select(imageDTO.board_id ?? 'none')(getState());

        const isCacheFullyPopulated = currentCache.data && currentCache.data.ids.length >= (data?.total ?? 0);

        const isInDateRange = getIsImageInDateRange(currentCache.data, imageDTO);

        if (isCacheFullyPopulated || isInDateRange) {
          // THEN *upsert* to $cache
          patches.push(
            dispatch(
              imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                imagesAdapter.upsertOne(draft, imageDTO);
              })
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
      invalidatesTags: (result, error, { board_id }) => {
        return [
          // update the destination board
          { type: 'Board', id: board_id ?? 'none' },
        ];
      },
      async onQueryStarted({ board_id: new_board_id, imageDTOs }, { dispatch, queryFulfilled, getState }) {
        try {
          const { data } = await queryFulfilled;
          const { added_image_names } = data;

          /**
           * Cache changes for addImagesToBoard:
           * - *update* getImageDTO for each image
           * - *add* to board_id/[images|assets]
           * - *remove* from [old_board_id|no_board]/[images|assets]
           * - decrement old board's totals for each image
           * - increment new board's totals for each image
           */

          added_image_names.forEach((image_name) => {
            dispatch(
              imagesApi.util.updateQueryData('getImageDTO', image_name, (draft) => {
                draft.board_id = new_board_id === 'none' ? undefined : new_board_id;
              })
            );

            const imageDTO = imageDTOs.find((i) => i.image_name === image_name);

            if (!imageDTO) {
              return;
            }

            const categories = getCategories(imageDTO);
            const old_board_id = imageDTO.board_id;
            const isAsset = ASSETS_CATEGORIES.includes(imageDTO.image_category);

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

            // decrement old board's total
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                old_board_id ?? 'none',
                (draft) => {
                  draft.total = Math.max(draft.total - 1, 0);
                }
              )
            );

            // increment new board's total
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                new_board_id ?? 'none',
                (draft) => {
                  draft.total += 1;
                }
              )
            );

            const queryArgs = {
              board_id: new_board_id,
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(getState());

            const { data } = IMAGE_CATEGORIES.includes(imageDTO.image_category)
              ? boardsApi.endpoints.getBoardImagesTotal.select(new_board_id ?? 'none')(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(new_board_id ?? 'none')(getState());

            const isCacheFullyPopulated = currentCache.data && currentCache.data.ids.length >= (data?.total ?? 0);

            const isInDateRange =
              (data?.total ?? 0) >= IMAGE_LIMIT ? getIsImageInDateRange(currentCache.data, imageDTO) : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                  imagesAdapter.upsertOne(draft, {
                    ...imageDTO,
                    board_id: new_board_id,
                  });
                })
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
        const tags: ApiTagDescription[] = [];

        result?.removed_image_names.forEach((image_name) => {
          const board_id = imageDTOs.find((i) => i.image_name === image_name)?.board_id;

          if (!board_id || touchedBoardIds.includes(board_id)) {
            return;
          }

          tags.push({ type: 'Board', id: board_id });
        });

        return tags;
      },
      async onQueryStarted({ imageDTOs }, { dispatch, queryFulfilled, getState }) {
        try {
          const { data } = await queryFulfilled;
          const { removed_image_names } = data;

          /**
           * Cache changes for removeImagesFromBoard:
           * - *update* getImageDTO for each image
           * - *remove* from old_board_id/[images|assets]
           * - *add* to no_board/[images|assets]
           * - decrement old board's totals for each image
           * - increment new board's (no board) totals for each image
           */

          removed_image_names.forEach((image_name) => {
            dispatch(
              imagesApi.util.updateQueryData('getImageDTO', image_name, (draft) => {
                draft.board_id = undefined;
              })
            );

            const imageDTO = imageDTOs.find((i) => i.image_name === image_name);

            if (!imageDTO) {
              return;
            }

            const categories = getCategories(imageDTO);
            const isAsset = ASSETS_CATEGORIES.includes(imageDTO.image_category);

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

            // decrement old board's total
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                imageDTO.board_id ?? 'none',
                (draft) => {
                  draft.total = Math.max(draft.total - 1, 0);
                }
              )
            );

            // increment new board's total (no board)
            dispatch(
              boardsApi.util.updateQueryData(
                isAsset ? 'getBoardAssetsTotal' : 'getBoardImagesTotal',
                'none',
                (draft) => {
                  draft.total += 1;
                }
              )
            );

            // add to `no_board`
            const queryArgs = {
              board_id: 'none',
              categories,
            };

            const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(getState());

            const { data } = IMAGE_CATEGORIES.includes(imageDTO.image_category)
              ? boardsApi.endpoints.getBoardImagesTotal.select(imageDTO.board_id ?? 'none')(getState())
              : boardsApi.endpoints.getBoardAssetsTotal.select(imageDTO.board_id ?? 'none')(getState());

            const isCacheFullyPopulated = currentCache.data && currentCache.data.ids.length >= (data?.total ?? 0);

            const isInDateRange =
              (data?.total ?? 0) >= IMAGE_LIMIT ? getIsImageInDateRange(currentCache.data, imageDTO) : true;

            if (isCacheFullyPopulated || isInDateRange) {
              // *upsert* to $cache
              dispatch(
                imagesApi.util.updateQueryData('listImages', queryArgs, (draft) => {
                  imagesAdapter.upsertOne(draft, {
                    ...imageDTO,
                    board_id: 'none',
                  });
                })
              );
            }
          });
        } catch {
          // no-op
        }
      },
    }),
    bulkDownloadImages: build.mutation<
      components['schemas']['ImagesDownloaded'],
      components['schemas']['Body_download_images_from_list']
    >({
      query: ({ image_names, board_id }) => ({
        url: `images/download`,
        method: 'POST',
        body: {
          image_names,
          board_id,
        },
      }),
    }),
  }),
});

export const {
  useGetIntermediatesCountQuery,
  useListImagesQuery,
  useLazyListImagesQuery,
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
  useGetImageWorkflowQuery,
  useLazyGetImageWorkflowQuery,
  useDeleteImageMutation,
  useDeleteImagesMutation,

  useUploadMultipleImagesMutation,

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
  useBulkDownloadImagesMutation,
} = imagesApi;
