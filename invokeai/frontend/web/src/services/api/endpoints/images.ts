import { ApiFullTagDescription, LIST_TAG, api } from '..';
import { components, paths } from '../schema';
import { ImageDTO, OffsetPaginatedResults_ImageDTO_ } from '../types';

type ListImagesArg = NonNullable<
  paths['/api/v1/images/']['get']['parameters']['query']
>;

/**
 * This is an unsafe type; the object inside is not guaranteed to be valid.
 */
export type UnsafeImageMetadata = {
  metadata: components['schemas']['CoreMetadata'];
  graph: NonNullable<components['schemas']['Graph']>;
};

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    listImages: build.query<OffsetPaginatedResults_ImageDTO_, ListImagesArg>({
      query: (arg) => ({ url: 'images/', params: arg }),
      providesTags: (result, error, arg) => {
        // any list of images
        const tags: ApiFullTagDescription[] = [{ id: 'Image', type: LIST_TAG }];

        if (result) {
          // and individual tags for each image
          tags.push(
            ...result.items.map(({ image_name, board_id }) => ({
              type: 'Image' as const,
              id: image_name,
            }))
          );
        }

        if (result && arg.board_id) {
          tags.push(
            ...result.items.map(({ image_name, board_id }) => ({
              type: 'BoardImage' as const,
              id: `${image_name}_${board_id}`,
            }))
          );
        }

        return tags;
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
  }),
});

export const { useGetImageDTOQuery, useGetImageMetadataQuery, useListImagesQuery } = imagesApi;
