import type { paths } from 'services/api/schema';
import type {
  GetVideoIdsArgs,
  GetVideoIdsResult,
  VideoDTO,
} from 'services/api/types';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';

import { api, buildV1Url, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the videos router
 * @example
 * buildVideosUrl('some-path')
 * // '/api/v1/videos/some-path'
 */
const buildVideosUrl  = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`videos/${path}`, query);

export const videosApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Video Queries
     */

    getVideoDTO: build.query<VideoDTO, string>({
      query: (video_id) => ({ url: buildVideosUrl(`i/${video_id}`) }),
      providesTags: (result, error, video_id) => [{ type: 'Video', id: video_id }],
    }),
    

    /**
     * Get ordered list of image names for selection operations
     */
    getVideoIds: build.query<GetVideoIdsResult, GetVideoIdsArgs>({
      query: (queryArgs) => ({
        url: buildVideosUrl('ids', queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        'VideoIdList',
        'FetchOnReconnect',
        { type: 'VideoIdList', id: stableHash(queryArgs) },
      ],
    }),
    /**
     * Get image DTOs for the specified image names. Maintains order of input names.
     */
    getVideoDTOsByNames: build.mutation<
      paths['/api/v1/videos/videos_by_ids']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/videos_by_ids']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildVideosUrl('videos_by_ids'),
        method: 'POST',
        body,
      }),
      // Don't provide cache tags - we'll manually upsert into individual getImageDTO caches
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        try {
          const { data: videoDTOs } = await queryFulfilled;

          // Upsert each DTO into the individual image cache
          const updates: Param0<typeof videosApi.util.upsertQueryEntries> = [];
          for (const videoDTO of videoDTOs) {
            updates.push({
              endpointName: 'getVideoDTO',
              arg: videoDTO.video_id,
              value: videoDTO,
            });
          }
          dispatch(videosApi.util.upsertQueryEntries(updates));
        } catch {
          // Handle error if needed
        }
      },
    }),
  }),
});

export const {
  useGetVideoDTOQuery,
  useGetVideoIdsQuery,
  useGetVideoDTOsByNamesMutation,
} = videosApi;


