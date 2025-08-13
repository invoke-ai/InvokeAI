import type { paths } from 'services/api/schema';

import type { ApiTagDescription } from '..';
import { buildV1Url } from '..';
import { api, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the videos router
 * @example
 * buildVideosUrl('some-path')
 * // '/api/v1/videos/some-path'
 */
const buildVideosUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`videos/${path}`, query);

export const videosApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Video Queries
     */
    listVideos: build.query<
      paths['/api/v1/videos/']['get']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/']['get']['parameters']['query']
    >({
      query: (queryArgs) => ({
        url: buildVideosUrl(),
        method: 'GET',
        params: queryArgs,
      }),
      providesTags: (result, error, queryArgs) => {
        const tags: ApiTagDescription[] = [
          { type: 'VideoList', id: JSON.stringify(queryArgs) },
          { type: 'Board', id: queryArgs?.board_id ?? 'none' },
          'FetchOnReconnect',
        ];
        return tags;
      },
    }),

    getVideoDTO: build.query<
      paths['/api/v1/videos/i/{video_id}']['get']['responses']['200']['content']['application/json'],
      string
    >({
      query: (video_id) => ({ url: buildVideosUrl(`i/${video_id}`) }),
      providesTags: (result, error, video_id) => [
        { type: 'Video', id: video_id },
        'FetchOnReconnect',
      ],
    }),

    getVideoIds: build.query<
      paths['/api/v1/videos/ids']['get']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/ids']['get']['parameters']['query']
    >({
      query: (queryArgs) => ({
        url: buildVideosUrl('ids'),
        method: 'GET',
        params: queryArgs,
      }),
      providesTags: (result, error, queryArgs) => [
        { type: 'VideoNameList', id: JSON.stringify(queryArgs) },
        'FetchOnReconnect',
      ],
    }),

    getVideosByIds: build.query<
      paths['/api/v1/videos/videos_by_ids']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/videos_by_ids']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildVideosUrl('videos_by_ids'),
        method: 'POST',
        body,
      }),
      
    }),

    /**
     * Video Mutations
     */

    updateVideo: build.mutation<
      paths['/api/v1/videos/i/{video_id}']['patch']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/i/{video_id}']['patch']['parameters']['path'] &
        paths['/api/v1/videos/i/{video_id}']['patch']['requestBody']['content']['application/json']
    >({
      query: ({ video_id, ...body }) => ({
        url: buildVideosUrl(`i/${video_id}`),
        method: 'PATCH',
        body,
      }),
      
    }),



    

  }),
});

export const {
  useListVideosQuery,
  useGetVideoDTOQuery,
  useGetVideoIdsQuery,
  useGetVideosByIdsQuery,
  useUpdateVideoMutation,
} = videosApi;

export const getTagsToInvalidateForVideoMutation = (video_ids: string[]): ApiTagDescription[] => {
  const tags: ApiTagDescription[] = [];

  for (const video_id of video_ids) {
    tags.push({
      type: 'Video',
      id: video_id,
    });

  }

  return tags;
};