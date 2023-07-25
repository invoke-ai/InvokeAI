import { EntityState, createEntityAdapter } from '@reduxjs/toolkit';
import { PatchCollection } from '@reduxjs/toolkit/dist/query/core/buildThunks';
import { dateComparator } from 'common/util/dateComparator';
import {
  ASSETS_CATEGORIES,
  BoardId,
  IMAGE_CATEGORIES,
} from 'features/gallery/store/types';
import queryString from 'query-string';
import { ApiFullTagDescription, api } from '..';
import { components, paths } from '../schema';
import {
  ImageCategory,
  ImageDTO,
  OffsetPaginatedResults_ImageDTO_,
  PostUploadAction,
} from '../types';

const getIsImageInDateRange = (
  data: ImageCache | undefined,
  imageDTO: ImageDTO
) => {
  if (!data) {
    return false;
  }
  const cacheImageDTOS = imagesSelectors.selectAll(data);

  if (cacheImageDTOS.length > 1) {
    // Images are sorted by `created_at` DESC
    // check if the image is newer than the oldest image in the cache
    const createdDate = new Date(imageDTO.created_at);
    const oldestDate = new Date(
      cacheImageDTOS[cacheImageDTOS.length - 1].created_at
    );
    return createdDate >= oldestDate;
  } else if ([0, 1].includes(cacheImageDTOS.length)) {
    // if there are only 1 or 0 images in the cache, we consider the image to be in the date range
    return true;
  }
  return false;
};

const getCategories = (imageDTO: ImageDTO) => {
  if (IMAGE_CATEGORIES.includes(imageDTO.image_category)) {
    return IMAGE_CATEGORIES;
  }
  return ASSETS_CATEGORIES;
};

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
    getBoardImagesTotal: build.query<number, string | undefined>({
      query: (board_id) => ({
        url: getListImagesUrl({
          board_id: board_id ?? 'none',
          categories: IMAGE_CATEGORIES,
          is_intermediate: false,
          limit: 0,
          offset: 0,
        }),
        method: 'GET',
      }),
      providesTags: (result, error, arg) => [
        { type: 'BoardImagesTotal', id: arg ?? 'none' },
      ],
      transformResponse: (response: OffsetPaginatedResults_ImageDTO_) => {
        return response.total;
      },
    }),
    getBoardAssetsTotal: build.query<number, string | undefined>({
      query: (board_id) => ({
        url: getListImagesUrl({
          board_id: board_id ?? 'none',
          categories: ASSETS_CATEGORIES,
          is_intermediate: false,
          limit: 0,
          offset: 0,
        }),
        method: 'GET',
      }),
      providesTags: (result, error, arg) => [
        { type: 'BoardAssetsTotal', id: arg ?? 'none' },
      ],
      transformResponse: (response: OffsetPaginatedResults_ImageDTO_) => {
        return response.total;
      },
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

        // Store patches so we can undo if the query fails
        const patches: PatchCollection[] = [];

        // determine `categories`, i.e. do we update "All Images" or "All Assets"
        // $cache = [board_id|no_board]/[images|assets]
        const categories = getCategories(imageDTO);

        // *remove* from $cache
        patches.push(
          dispatch(
            imagesApi.util.updateQueryData(
              'listImages',
              { board_id: board_id ?? 'none', categories },
              (draft) => {
                const oldTotal = draft.total;
                const newState = imagesAdapter.removeOne(draft, image_name);
                const delta = newState.total - oldTotal;
                draft.total = draft.total + delta;
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
     * Change an image's `is_intermediate` property.
     */
    changeImageIsIntermediate: build.mutation<
      ImageDTO,
      { imageDTO: ImageDTO; is_intermediate: boolean }
    >({
      query: ({ imageDTO, is_intermediate }) => ({
        url: `images/${imageDTO.image_name}`,
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
                  const oldTotal = draft.total;
                  const newState = imagesAdapter.removeOne(
                    draft,
                    imageDTO.image_name
                  );
                  const delta = newState.total - oldTotal;
                  draft.total = draft.total + delta;
                }
              )
            )
          );
        } else {
          // ELSE (it is being changed to a non-intermediate):
          console.log(imageDTO);
          const queryArgs = {
            board_id: imageDTO.board_id ?? 'none',
            categories,
          };

          const currentCache = imagesApi.endpoints.listImages.select(queryArgs)(
            getState()
          );

          // IF it eligible for insertion into existing $cache
          // "eligible" means either:
          // - The cache is fully populated, with all images in the db cached
          //    OR
          // - The image's `created_at` is within the range of the cached images

          const isCacheFullyPopulated =
            currentCache.data &&
            currentCache.data.ids.length >= currentCache.data.total;

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
                    const oldTotal = draft.total;
                    const newState = imagesAdapter.upsertOne(draft, imageDTO);
                    const delta = newState.total - oldTotal;
                    draft.total = draft.total + delta;
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
        url: `images/${imageDTO.image_name}`,
        method: 'PATCH',
        body: { session_id },
      }),
      invalidatesTags: (result, error, { imageDTO }) => [
        { type: 'BoardImagesTotal', id: imageDTO.board_id ?? 'none' },
        { type: 'BoardAssetsTotal', id: imageDTO.board_id ?? 'none' },
      ],
      async onQueryStarted(
        { imageDTO, session_id },
        { dispatch, queryFulfilled, getState }
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
          url: `images/`,
          method: 'POST',
          body: formData,
          params: {
            image_category,
            is_intermediate,
            session_id,
            board_id,
            crop_visible,
          },
        };
      },
      async onQueryStarted(
        {
          file,
          image_category,
          is_intermediate,
          postUploadAction,
          session_id,
          board_id,
        },
        { dispatch, queryFulfilled }
      ) {
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
                const oldTotal = draft.total;
                const newState = imagesAdapter.addOne(draft, imageDTO);
                const delta = newState.total - oldTotal;
                draft.total = draft.total + delta;
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
        { type: 'Board', id: board_id },
        { type: 'BoardImagesTotal', id: board_id },
        { type: 'BoardImagesTotal', id: imageDTO.board_id ?? 'none' },
        { type: 'BoardAssetsTotal', id: board_id },
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
                Object.assign(draft, { board_id });
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
                  const oldTotal = draft.total;
                  const newState = imagesAdapter.removeOne(
                    draft,
                    imageDTO.image_name
                  );
                  const delta = newState.total - oldTotal;
                  draft.total = draft.total + delta;
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

          const isCacheFullyPopulated =
            currentCache.data &&
            currentCache.data.ids.length >= currentCache.data.total;

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
                    const oldTotal = draft.total;
                    const newState = imagesAdapter.addOne(draft, imageDTO);
                    const delta = newState.total - oldTotal;
                    draft.total = draft.total + delta;
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
      invalidatesTags: (result, error, { imageDTO }) => [
        { type: 'Board', id: imageDTO.board_id },
        { type: 'BoardImagesTotal', id: imageDTO.board_id },
        { type: 'BoardImagesTotal', id: 'none' },
        { type: 'BoardAssetsTotal', id: imageDTO.board_id },
        { type: 'BoardAssetsTotal', id: 'none' },
      ],
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
                Object.assign(draft, { board_id: undefined });
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
                const oldTotal = draft.total;
                const newState = imagesAdapter.removeOne(
                  draft,
                  imageDTO.image_name
                );
                const delta = newState.total - oldTotal;
                draft.total = draft.total + delta;
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

        const isCacheFullyPopulated =
          currentCache.data &&
          currentCache.data.ids.length >= currentCache.data.total;

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
                  const oldTotal = draft.total;
                  const newState = imagesAdapter.upsertOne(draft, imageDTO);
                  const delta = newState.total - oldTotal;
                  draft.total = draft.total + delta;
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
  useGetBoardImagesTotalQuery,
  useGetBoardAssetsTotalQuery,
  useUploadImageMutation,
  useAddImageToBoardMutation,
  useRemoveImageFromBoardMutation,
  useClearIntermediatesMutation,
} = imagesApi;
