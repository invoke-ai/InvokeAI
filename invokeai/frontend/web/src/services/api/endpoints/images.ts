import { skipToken } from '@reduxjs/toolkit/query';
import { $authToken } from 'app/store/nanostores/authToken';
import { getStore } from 'app/store/nanostores/store';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { components, paths } from 'services/api/schema';
import type {
  GraphAndWorkflowResponse,
  ImageCategory,
  ImageDTO,
  ImageUploadEntryRequest,
  ImageUploadEntryResponse,
  ListImagesArgs,
  ListImagesResponse,
  SQLiteDirection,
  UploadImageArg,
} from 'services/api/types';
import { getCategories, getListImagesUrl } from 'services/api/util';
import type { Param0 } from 'tsafe';
import type { JsonObject } from 'type-fest';

import type { ApiTagDescription } from '..';
import { api, buildV1Url, LIST_TAG } from '..';
import { buildBoardsUrl } from './boards';

/**
 * Builds an endpoint URL for the images router
 * @example
 * buildImagesUrl('some-path')
 * // '/api/v1/images/some-path'
 */
const buildImagesUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`images/${path}`, query);

/**
 * Builds an endpoint URL for the board_images router
 * @example
 * buildBoardImagesUrl('some-path')
 * // '/api/v1/board_images/some-path'
 */
const buildBoardImagesUrl = (path: string = '') => buildV1Url(`board_images/${path}`);

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    listImages: build.query<ListImagesResponse, ListImagesArgs>({
      query: (queryArgs) => ({
        // Use the helper to create the URL.
        url: getListImagesUrl(queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => {
        return [
          // Make the tags the same as the cache key
          { type: 'ImageList', id: JSON.stringify(queryArgs) },
          'FetchOnReconnect',
        ];
      },
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        // Populate the getImageDTO cache with these images. This makes image selection smoother, because it doesn't
        // need to re-fetch image data when the user selects an image. The getImageDTO cache keeps data for the default
        // of 60s, so this data won't stick around too long.
        const res = await queryFulfilled;
        const imageDTOs = res.data.items;
        const updates: Param0<typeof imagesApi.util.upsertQueryEntries> = [];
        for (const imageDTO of imageDTOs) {
          updates.push({
            endpointName: 'getImageDTO',
            arg: imageDTO.image_name,
            value: imageDTO,
          });
        }
        dispatch(imagesApi.util.upsertQueryEntries(updates));
      },
    }),
    getIntermediatesCount: build.query<number, void>({
      query: () => ({ url: buildImagesUrl('intermediates') }),
      providesTags: ['IntermediatesCount', 'FetchOnReconnect'],
    }),
    clearIntermediates: build.mutation<number, void>({
      query: () => ({ url: buildImagesUrl('intermediates'), method: 'DELETE' }),
      invalidatesTags: [
        'IntermediatesCount',
        'InvocationCacheStatus',
        'ImageCollectionCounts',
        { type: 'ImageCollection', id: LIST_TAG },
      ],
    }),
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}`) }),
      providesTags: (result, error, image_name) => [{ type: 'Image', id: image_name }],
    }),
    getImageMetadata: build.query<JsonObject | undefined, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}/metadata`) }),
      providesTags: (result, error, image_name) => [{ type: 'ImageMetadata', id: image_name }],
    }),
    getImageWorkflow: build.query<GraphAndWorkflowResponse, string>({
      query: (image_name) => ({ url: buildImagesUrl(`i/${image_name}/workflow`) }),
      providesTags: (result, error, image_name) => [{ type: 'ImageWorkflow', id: image_name }],
    }),
    deleteImage: build.mutation<
      paths['/api/v1/images/i/{image_name}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/i/{image_name}']['delete']['parameters']['path']
    >({
      query: ({ image_name }) => ({
        url: buildImagesUrl(`i/${image_name}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
        // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
        // will force those queries to re-fetch, and the requests will of course 404.
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
        ];
      },
    }),
    deleteImages: build.mutation<
      paths['/api/v1/images/delete']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/delete']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildImagesUrl('delete'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
        // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
        // will force those queries to re-fetch, and the requests will of course 404.
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
        ];
      },
    }),
    deleteUncategorizedImages: build.mutation<
      paths['/api/v1/images/uncategorized']['delete']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({ url: buildImagesUrl('uncategorized'), method: 'DELETE' }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // We ignore the deleted images when getting tags to invalidate. If we did not, we will invalidate the queries
        // that fetch image DTOs, metadata, and workflows. But we have just deleted those images! Invalidating the tags
        // will force those queries to re-fetch, and the requests will of course 404.
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: LIST_TAG },
        ];
      },
    }),
    /**
     * Change an image's `is_intermediate` property.
     */
    changeImageIsIntermediate: build.mutation<
      paths['/api/v1/images/i/{image_name}']['patch']['responses']['200']['content']['application/json'],
      { image_name: string; is_intermediate: boolean }
    >({
      query: ({ image_name, is_intermediate }) => ({
        url: buildImagesUrl(`i/${image_name}`),
        method: 'PATCH',
        body: { is_intermediate },
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation([result.image_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([result.board_id ?? 'none']),
        ];
      },
    }),
    /**
     * Star a list of images.
     */
    starImages: build.mutation<
      paths['/api/v1/images/star']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/star']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildImagesUrl('star'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.starred_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: 'starred' },
          { type: 'ImageCollection', id: 'unstarred' },
        ];
      },
    }),
    /**
     * Unstar a list of images.
     */
    unstarImages: build.mutation<
      paths['/api/v1/images/unstar']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/unstar']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildImagesUrl('unstar'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.unstarred_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'ImageCollectionCounts',
          { type: 'ImageCollection', id: 'starred' },
          { type: 'ImageCollection', id: 'unstarred' },
        ];
      },
    }),
    uploadImage: build.mutation<
      paths['/api/v1/images/upload']['post']['responses']['201']['content']['application/json'],
      UploadImageArg
    >({
      query: ({ file, image_category, is_intermediate, session_id, board_id, crop_visible, metadata, resize_to }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }
        if (resize_to) {
          formData.append('resize_to', JSON.stringify(resize_to));
        }
        return {
          url: buildImagesUrl('upload'),
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

      invalidatesTags: (result) => {
        if (!result || result.is_intermediate) {
          // Don't add it to anything
          return [];
        }
        const categories = getCategories(result);
        const boardId = result.board_id ?? 'none';

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
          {
            type: 'BoardImagesTotal',
            id: boardId,
          },
        ];
      },
    }),
    createImageUploadEntry: build.mutation<ImageUploadEntryResponse, ImageUploadEntryRequest>({
      query: ({ width, height, board_id }) => ({
        url: buildImagesUrl(),
        method: 'POST',
        body: { width, height, board_id },
      }),
    }),
    deleteBoard: build.mutation<
      paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/boards/{board_id}']['delete']['parameters']['path']
    >({
      query: ({ board_id }) => ({ url: buildBoardsUrl(board_id), method: 'DELETE' }),
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
    }),

    deleteBoardAndImages: build.mutation<
      paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/boards/{board_id}']['delete']['parameters']['path']
    >({
      query: ({ board_id }) => ({
        url: buildBoardsUrl(board_id),
        method: 'DELETE',
        params: { include_images: true },
      }),
      invalidatesTags: () => [{ type: 'Board', id: LIST_TAG }],
    }),
    addImageToBoard: build.mutation<
      paths['/api/v1/board_images/']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => {
        return {
          url: buildBoardImagesUrl(),
          method: 'POST',
          body,
        };
      },
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.added_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    removeImageFromBoard: build.mutation<
      paths['/api/v1/board_images/']['delete']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/']['delete']['requestBody']['content']['application/json']
    >({
      query: (body) => {
        return {
          url: buildBoardImagesUrl(),
          method: 'DELETE',
          body,
        };
      },
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.removed_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    addImagesToBoard: build.mutation<
      paths['/api/v1/board_images/batch']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/batch']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildBoardImagesUrl('batch'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.added_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    removeImagesFromBoard: build.mutation<
      paths['/api/v1/board_images/batch/delete']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_images/batch/delete']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildBoardImagesUrl('batch/delete'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForImageMutation(result.removed_images),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    bulkDownloadImages: build.mutation<
      components['schemas']['ImagesDownloaded'],
      components['schemas']['Body_download_images_from_list']
    >({
      query: ({ image_names, board_id }) => ({
        url: buildImagesUrl('download'),
        method: 'POST',
        body: {
          image_names,
          board_id,
        },
      }),
    }),
    /**
     * Get counts for starred and unstarred image collections
     */
    getImageCollectionCounts: build.query<
      paths['/api/v1/images/collections/counts']['get']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/collections/counts']['get']['parameters']['query']
    >({
      query: (queryArgs) => ({
        url: buildImagesUrl('collections/counts', queryArgs),
        method: 'GET',
      }),
      providesTags: ['ImageCollectionCounts', 'FetchOnReconnect'],
    }),
    /**
     * Get images from a specific collection (starred or unstarred)
     */
    getImageCollection: build.query<
      paths['/api/v1/images/collections/{collection}']['get']['responses']['200']['content']['application/json'],
      paths['/api/v1/images/collections/{collection}']['get']['parameters']['path'] &
        paths['/api/v1/images/collections/{collection}']['get']['parameters']['query']
    >({
      query: ({ collection, ...queryArgs }) => ({
        url: buildImagesUrl(`collections/${collection}`, queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, { collection, board_id, categories }) => {
        const cacheKey = `${collection}-${board_id || 'all'}-${categories?.join(',') || 'all'}`;
        return [
          { type: 'ImageCollection', id: collection },
          { type: 'ImageCollection', id: cacheKey },
          'FetchOnReconnect',
        ];
      },
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        // Populate the getImageDTO cache with these images, similar to listImages
        const res = await queryFulfilled;
        const imageDTOs = res.data.items;
        const updates: Param0<typeof imagesApi.util.upsertQueryEntries> = [];
        for (const imageDTO of imageDTOs) {
          updates.push({
            endpointName: 'getImageDTO',
            arg: imageDTO.image_name,
            value: imageDTO,
          });
        }
        dispatch(imagesApi.util.upsertQueryEntries(updates));
      },
    }),
    /**
     * Get ordered list of image names for selection operations
     */
    getImageNames: build.query<
      string[],
      {
        image_origin?: 'internal' | 'external' | null;
        categories?: ImageCategory[] | null;
        is_intermediate?: boolean | null;
        board_id?: string | null;
        search_term?: string | null;
        order_dir?: SQLiteDirection;
      }
    >({
      query: (queryArgs) => ({
        url: buildImagesUrl('names', queryArgs),
        method: 'GET',
      }),
      providesTags: ['ImageNameList', 'FetchOnReconnect'],
    }),
    /**
     * Get paginated images with starred first (unified list)
     */
    getUnifiedImageList: build.query<
      ListImagesResponse,
      {
        offset?: number;
        limit?: number;
        image_origin?: 'internal' | 'external' | null;
        categories?: ImageCategory[] | null;
        is_intermediate?: boolean | null;
        board_id?: string | null;
        search_term?: string | null;
        order_dir?: SQLiteDirection;
      }
    >({
      query: (queryArgs) => ({
        url: getListImagesUrl({ ...queryArgs, starred_first: true }),
        method: 'GET',
      }),
      providesTags: (result, error, { board_id, categories }) => [
        { type: 'ImageList', id: getListImagesUrl({ board_id, categories }) },
        'FetchOnReconnect',
      ],
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        // Populate the getImageDTO cache with these images
        const res = await queryFulfilled;
        const imageDTOs = res.data.items;
        const updates: Param0<typeof imagesApi.util.upsertQueryEntries> = [];
        for (const imageDTO of imageDTOs) {
          updates.push({
            endpointName: 'getImageDTO',
            arg: imageDTO.image_name,
            value: imageDTO,
          });
        }
        dispatch(imagesApi.util.upsertQueryEntries(updates));
      },
    }),
  }),
});

export const {
  useGetIntermediatesCountQuery,
  useListImagesQuery,
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
  useGetImageWorkflowQuery,
  useLazyGetImageWorkflowQuery,
  useUploadImageMutation,
  useCreateImageUploadEntryMutation,
  useClearIntermediatesMutation,
  useAddImagesToBoardMutation,
  useRemoveImagesFromBoardMutation,
  useDeleteBoardAndImagesMutation,
  useDeleteUncategorizedImagesMutation,
  useDeleteBoardMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
  useBulkDownloadImagesMutation,
  useGetImageCollectionCountsQuery,
  useGetImageCollectionQuery,
  useLazyGetImageCollectionQuery,
  useGetImageNamesQuery,
  useGetUnifiedImageListQuery,
} = imagesApi;

/**
 * Imperative RTKQ helper to fetch an ImageDTO.
 * @param image_name The name of the image to fetch
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @returns The ImageDTO if found, otherwise null
 */
export const getImageDTOSafe = async (
  image_name: string,
  options?: Parameters<typeof imagesApi.endpoints.getImageDTO.initiate>[1]
): Promise<ImageDTO | null> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(imagesApi.endpoints.getImageDTO.initiate(image_name, _options));
  try {
    return await req.unwrap();
  } catch {
    return null;
  }
};

/**
 * Imperative RTKQ helper to fetch an ImageDTO.
 * @param image_name The name of the image to fetch
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @raises Error if the image is not found or there is an error fetching the image
 */
export const getImageDTO = (
  image_name: string,
  options?: Parameters<typeof imagesApi.endpoints.getImageDTO.initiate>[1]
): Promise<ImageDTO> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(imagesApi.endpoints.getImageDTO.initiate(image_name, _options));
  return req.unwrap();
};

/**
 * Imperative RTKQ helper to fetch an image's metadata.
 * @param image_name The name of the image
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @raises Error if the image metadata is not found or there is an error fetching the image metadata. Images without
 * metadata will return undefined.
 */
export const getImageMetadata = (
  image_name: string,
  options?: Parameters<typeof imagesApi.endpoints.getImageMetadata.initiate>[1]
): Promise<JsonObject | undefined> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(imagesApi.endpoints.getImageMetadata.initiate(image_name, _options));
  return req.unwrap();
};

export const uploadImage = (arg: UploadImageArg): Promise<ImageDTO> => {
  const { dispatch } = getStore();
  const req = dispatch(imagesApi.endpoints.uploadImage.initiate(arg, { track: false }));
  return req.unwrap();
};

export const copyImage = async (imageName: string, uploadImageArg: Omit<UploadImageArg, 'file'>): Promise<ImageDTO> => {
  const originalImageDTO = await getImageDTO(imageName);
  const file = await imageDTOToFile(originalImageDTO);
  const imageDTO = await uploadImage({ file, ...uploadImageArg });
  return imageDTO;
};

export const uploadImages = async (args: UploadImageArg[]): Promise<ImageDTO[]> => {
  const { dispatch } = getStore();
  const results = await Promise.allSettled(
    args.map((arg) => {
      const req = dispatch(imagesApi.endpoints.uploadImage.initiate(arg, { track: false }));
      return req.unwrap();
    })
  );
  return results.filter((r): r is PromiseFulfilledResult<ImageDTO> => r.status === 'fulfilled').map((r) => r.value);
};

/**
 * Convert an ImageDTO to a File by downloading the image from the server.
 * @param imageDTO The image to download and convert to a File
 */
export const imageDTOToFile = async (imageDTO: ImageDTO): Promise<File> => {
  const init: RequestInit = {};
  const authToken = $authToken.get();
  if (authToken) {
    init.headers = { Authorization: `Bearer ${authToken}` };
  }
  const res = await fetch(imageDTO.image_url, init);
  const blob = await res.blob();
  // Create a new file with the same name, which we will upload
  const file = new File([blob], `copy_of_${imageDTO.image_name}`, { type: 'image/png' });
  return file;
};

export const useImageDTO = (imageName: string | null | undefined) => {
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);
  return imageDTO ?? null;
};

export const getTagsToInvalidateForImageMutation = (image_names: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = [];

  for (const image_name of image_names) {
    tags.push({
      type: 'Image',
      id: image_name,
    });
    tags.push({
      type: 'ImageMetadata',
      id: image_name,
    });
    tags.push({
      type: 'ImageWorkflow',
      id: image_name,
    });
  }

  return tags;
};

export const getTagsToInvalidateForBoardAffectingMutation = (affected_boards: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = [];

  for (const board_id of affected_boards) {
    tags.push({
      type: 'ImageList',
      id: getListImagesUrl({
        board_id,
        categories: IMAGE_CATEGORIES,
      }),
    });

    tags.push({
      type: 'ImageList',
      id: getListImagesUrl({
        board_id,
        categories: ASSETS_CATEGORIES,
      }),
    });

    tags.push({
      type: 'Board',
      id: board_id,
    });

    tags.push({
      type: 'BoardImagesTotal',
      id: board_id,
    });
  }

  return tags;
};
