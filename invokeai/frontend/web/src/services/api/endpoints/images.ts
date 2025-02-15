import { $authToken } from 'app/store/nanostores/authToken';
import { getStore } from 'app/store/nanostores/store';
import type { BoardId } from 'features/gallery/store/types';
import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { components, paths } from 'services/api/schema';
import type {
  DeleteBoardResult,
  GraphAndWorkflowResponse,
  ImageDTO,
  ListImagesArgs,
  ListImagesResponse,
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
const buildImagesUrl = (path: string = '') => buildV1Url(`images/${path}`);

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
      providesTags: (result, error, { board_id, categories }) => {
        return [
          // Make the tags the same as the cache key
          { type: 'ImageList', id: getListImagesUrl({ board_id, categories }) },
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
      invalidatesTags: ['IntermediatesCount', 'InvocationCacheStatus'],
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
    deleteImage: build.mutation<void, ImageDTO>({
      query: ({ image_name }) => ({
        url: buildImagesUrl(`i/${image_name}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, imageDTO) => {
        const categories = getCategories(imageDTO);
        const boardId = imageDTO.board_id ?? 'none';

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

    deleteImages: build.mutation<components['schemas']['DeleteImagesFromListResult'], { imageDTOs: ImageDTO[] }>({
      query: ({ imageDTOs }) => {
        const image_names = imageDTOs.map((imageDTO) => imageDTO.image_name);
        return {
          url: buildImagesUrl('delete'),
          method: 'POST',
          body: {
            image_names,
          },
        };
      },
      invalidatesTags: (result, error, { imageDTOs }) => {
        if (imageDTOs[0]) {
          const categories = getCategories(imageDTOs[0]);
          const boardId = imageDTOs[0].board_id ?? 'none';

          const tags: ApiTagDescription[] = [
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

          return tags;
        }
        return [];
      },
    }),
    /**
     * Change an image's `is_intermediate` property.
     */
    changeImageIsIntermediate: build.mutation<ImageDTO, { imageDTO: ImageDTO; is_intermediate: boolean }>({
      query: ({ imageDTO, is_intermediate }) => ({
        url: buildImagesUrl(`i/${imageDTO.image_name}`),
        method: 'PATCH',
        body: { is_intermediate },
      }),
      invalidatesTags: (result, error, { imageDTO }) => {
        const categories = getCategories(imageDTO);
        const boardId = imageDTO.board_id ?? 'none';

        return [
          { type: 'Image', id: imageDTO.image_name },
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
    /**
     * Star a list of images.
     */
    starImages: build.mutation<
      paths['/api/v1/images/unstar']['post']['responses']['200']['content']['application/json'],
      { imageDTOs: ImageDTO[] }
    >({
      query: ({ imageDTOs: images }) => ({
        url: buildImagesUrl('star'),
        method: 'POST',
        body: { image_names: images.map((img) => img.image_name) },
      }),
      invalidatesTags: (result, error, { imageDTOs }) => {
        // assume all images are on the same board/category
        if (imageDTOs[0]) {
          const categories = getCategories(imageDTOs[0]);
          const boardId = imageDTOs[0].board_id ?? 'none';
          const tags: ApiTagDescription[] = [
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
          for (const imageDTO of imageDTOs) {
            tags.push({ type: 'Image', id: imageDTO.image_name });
          }
          return tags;
        }
        return [];
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
        url: buildImagesUrl('unstar'),
        method: 'POST',
        body: { image_names: images.map((img) => img.image_name) },
      }),
      invalidatesTags: (result, error, { imageDTOs }) => {
        // assume all images are on the same board/category
        if (imageDTOs[0]) {
          const categories = getCategories(imageDTOs[0]);
          const boardId = imageDTOs[0].board_id ?? 'none';
          const tags: ApiTagDescription[] = [
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
          for (const imageDTO of imageDTOs) {
            tags.push({ type: 'Image', id: imageDTO.image_name });
          }
          return tags;
        }
        return [];
      },
    }),
    uploadImage: build.mutation<ImageDTO, UploadImageArg>({
      query: ({ file, image_category, is_intermediate, session_id, board_id, crop_visible, metadata }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
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

    deleteBoard: build.mutation<DeleteBoardResult, string>({
      query: (board_id) => ({ url: buildBoardsUrl(board_id), method: 'DELETE' }),
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

    deleteBoardAndImages: build.mutation<DeleteBoardResult, string>({
      query: (board_id) => ({
        url: buildBoardsUrl(board_id),
        method: 'DELETE',
        params: { include_images: true },
      }),
      invalidatesTags: () => [{ type: 'Board', id: LIST_TAG }],
    }),
    addImageToBoard: build.mutation<void, { board_id: BoardId; imageDTO: ImageDTO }>({
      query: ({ board_id, imageDTO }) => {
        const { image_name } = imageDTO;
        return {
          url: buildBoardImagesUrl(),
          method: 'POST',
          body: { board_id, image_name },
        };
      },
      invalidatesTags: (result, error, { board_id, imageDTO }) => {
        return [
          { type: 'Image', id: imageDTO.image_name },
          {
            type: 'ImageList',
            id: getListImagesUrl({
              board_id,
              categories: getCategories(imageDTO),
            }),
          },
          {
            type: 'ImageList',
            id: getListImagesUrl({
              board_id: imageDTO.board_id ?? 'none',
              categories: getCategories(imageDTO),
            }),
          },
          { type: 'Board', id: board_id },
          { type: 'Board', id: imageDTO.board_id ?? 'none' },
          {
            type: 'BoardImagesTotal',
            id: imageDTO.board_id ?? 'none',
          },
          {
            type: 'BoardImagesTotal',
            id: board_id,
          },
        ];
      },
    }),
    removeImageFromBoard: build.mutation<void, { imageDTO: ImageDTO }>({
      query: ({ imageDTO }) => {
        const { image_name } = imageDTO;
        return {
          url: buildBoardImagesUrl(),
          method: 'DELETE',
          body: { image_name },
        };
      },
      invalidatesTags: (result, error, { imageDTO }) => {
        return [
          { type: 'Image', id: imageDTO.image_name },
          {
            type: 'ImageList',
            id: getListImagesUrl({
              board_id: imageDTO.board_id,
              categories: getCategories(imageDTO),
            }),
          },
          {
            type: 'ImageList',
            id: getListImagesUrl({
              board_id: 'none',
              categories: getCategories(imageDTO),
            }),
          },
          { type: 'Board', id: imageDTO.board_id ?? 'none' },
          { type: 'Board', id: 'none' },
          {
            type: 'BoardImagesTotal',
            id: imageDTO.board_id ?? 'none',
          },
          { type: 'BoardImagesTotal', id: 'none' },
        ];
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
        url: buildBoardImagesUrl('batch'),
        method: 'POST',
        body: {
          image_names: imageDTOs.map((i) => i.image_name),
          board_id,
        },
      }),
      invalidatesTags: (result, error, { board_id, imageDTOs }) => {
        const tags: ApiTagDescription[] = [];
        if (imageDTOs[0]) {
          tags.push({
            type: 'ImageList',
            id: getListImagesUrl({
              board_id: imageDTOs[0].board_id ?? 'none',
              categories: getCategories(imageDTOs[0]),
            }),
          });
          tags.push({
            type: 'ImageList',
            id: getListImagesUrl({
              board_id: board_id,
              categories: getCategories(imageDTOs[0]),
            }),
          });
          tags.push({ type: 'Board', id: imageDTOs[0].board_id ?? 'none' });
          tags.push({
            type: 'BoardImagesTotal',
            id: imageDTOs[0].board_id ?? 'none',
          });
        }
        for (const imageDTO of imageDTOs) {
          tags.push({ type: 'Image', id: imageDTO.image_name });
        }
        tags.push({ type: 'Board', id: board_id });
        tags.push({
          type: 'BoardImagesTotal',
          id: board_id ?? 'none',
        });
        return tags;
      },
    }),
    removeImagesFromBoard: build.mutation<
      components['schemas']['RemoveImagesFromBoardResult'],
      {
        imageDTOs: ImageDTO[];
      }
    >({
      query: ({ imageDTOs }) => ({
        url: buildBoardImagesUrl('batch/delete'),
        method: 'POST',
        body: {
          image_names: imageDTOs.map((i) => i.image_name),
        },
      }),
      invalidatesTags: (result, error, { imageDTOs }) => {
        const touchedBoardIds: string[] = [];
        const tags: ApiTagDescription[] = [];

        if (imageDTOs[0]) {
          tags.push({
            type: 'ImageList',
            id: getListImagesUrl({
              board_id: imageDTOs[0].board_id,
              categories: getCategories(imageDTOs[0]),
            }),
          });
          tags.push({
            type: 'ImageList',
            id: getListImagesUrl({
              board_id: 'none',
              categories: getCategories(imageDTOs[0]),
            }),
          });
          tags.push({
            type: 'BoardImagesTotal',
            id: 'none',
          });
        }

        result?.removed_image_names.forEach((image_name) => {
          const board_id = imageDTOs.find((i) => i.image_name === image_name)?.board_id;

          if (!board_id || touchedBoardIds.includes(board_id)) {
            tags.push({ type: 'Board', id: 'none' });
            return;
          }
          tags.push({ type: 'Image', id: image_name });
          tags.push({ type: 'Board', id: board_id });
          tags.push({
            type: 'BoardImagesTotal',
            id: board_id ?? 'none',
          });
        });

        return tags;
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
  useClearIntermediatesMutation,
  useAddImagesToBoardMutation,
  useRemoveImagesFromBoardMutation,
  useDeleteBoardAndImagesMutation,
  useDeleteBoardMutation,
  useStarImagesMutation,
  useUnstarImagesMutation,
  useBulkDownloadImagesMutation,
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
