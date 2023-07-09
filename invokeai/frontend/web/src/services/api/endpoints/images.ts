import { api } from '..';
import { ImageDTO } from '../types';

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: `images/${image_name}/metadata` }),
      providesTags: (result, error, arg) => [{ type: 'Image', id: arg }],
      keepUnusedDataFor: 86400, // 24 hours
    }),
  }),
});

export const { useGetImageDTOQuery } = imagesApi;
