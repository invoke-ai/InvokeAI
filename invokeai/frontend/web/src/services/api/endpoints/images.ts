import { ApiFullTagDescription, api } from '..';
import { components } from '../schema';
import { ImageDTO } from '../types';


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

export const { useGetImageDTOQuery, useGetImageMetadataQuery, useClearIntermediatesMutation } = imagesApi;
