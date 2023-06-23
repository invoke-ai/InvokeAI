import { ApiFullTagDescription, api } from '..';
import { ImageDTO } from '../types';

export const imagesApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Image Queries
     */
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: `images/${image_name}/metadata` }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [{ type: 'Image', id: arg }];
        if (result?.board_id) {
          tags.push({ type: 'Board', id: result.board_id });
        }
        return tags;
      },
    }),
  }),
});

export const { useGetImageDTOQuery } = imagesApi;
