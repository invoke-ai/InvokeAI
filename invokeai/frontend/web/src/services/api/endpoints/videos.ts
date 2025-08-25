import { skipToken } from '@reduxjs/toolkit/query';
import { getStore } from 'app/store/nanostores/store';
import type { paths } from 'services/api/schema';
import type { GetVideoIdsArgs, GetVideoIdsResult, VideoDTO } from 'services/api/types';
import {
  getTagsToInvalidateForBoardAffectingMutation,
  getTagsToInvalidateForVideoMutation,
} from 'services/api/util/tagInvalidation';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';
import type { JsonObject } from 'type-fest';

import { api, buildV1Url, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the videos router
 * @example
 * buildVideosUrl('some-path')
 * // '/api/v1/videos/some-path'
 */
export const buildVideosUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`videos/${path}`, query);

const buildBoardVideosUrl = (path: string = '') => buildV1Url(`board_videos/${path}`);

export const videosApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Video Queries
     */

    getVideoDTO: build.query<VideoDTO, string>({
      query: (video_id) => ({ url: buildVideosUrl(`i/${video_id}`) }),
      providesTags: (result, error, video_id) => [{ type: 'Video', id: video_id }],
    }),

    getVideoMetadata: build.query<JsonObject | undefined, string>({
      query: (video_id) => ({ url: buildVideosUrl(`i/${video_id}/metadata`) }),
      providesTags: (result, error, video_id) => [{ type: 'VideoMetadata', id: video_id }],
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
    /**
     * Star a list of videos.
     */
    starVideos: build.mutation<
      paths['/api/v1/videos/star']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/star']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildVideosUrl('star'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForVideoMutation(result.starred_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'VideoCollectionCounts',
          { type: 'VideoCollection', id: 'starred' },
          { type: 'VideoCollection', id: 'unstarred' },
        ];
      },
    }),
    /**
     * Unstar a list of videos.
     */
    unstarVideos: build.mutation<
      paths['/api/v1/videos/unstar']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/unstar']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildVideosUrl('unstar'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForVideoMutation(result.unstarred_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          'VideoCollectionCounts',
          { type: 'VideoCollection', id: 'starred' },
          { type: 'VideoCollection', id: 'unstarred' },
        ];
      },
    }),
    deleteVideos: build.mutation<
      paths['/api/v1/videos/delete']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/delete']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildVideosUrl('delete'),
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
          'VideoCollectionCounts',
          { type: 'VideoCollection', id: LIST_TAG },
        ];
      },
    }),
    addVideosToBoard: build.mutation<
      paths['/api/v1/board_videos/batch']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_videos/batch']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildBoardVideosUrl('batch'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForVideoMutation(result.added_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
    removeVideosFromBoard: build.mutation<
      paths['/api/v1/board_videos/batch/delete']['post']['responses']['201']['content']['application/json'],
      paths['/api/v1/board_videos/batch/delete']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildBoardVideosUrl('batch/delete'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForVideoMutation(result.removed_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
  }),
});

export const {
  useGetVideoDTOQuery,
  useGetVideoIdsQuery,
  useGetVideoDTOsByNamesMutation,
  useStarVideosMutation,
  useUnstarVideosMutation,
  useDeleteVideosMutation,
  useAddVideosToBoardMutation,
  useRemoveVideosFromBoardMutation,
  useGetVideoMetadataQuery,
} = videosApi;

/**
 * Imperative RTKQ helper to fetch an VideoDTO.
 * @param id The id of the video to fetch
 * @param options The options for the query. By default, the query will not subscribe to the store.
 * @returns The ImageDTO if found, otherwise null
 */
export const getVideoDTOSafe = async (
  id: string,
  options?: Parameters<typeof videosApi.endpoints.getVideoDTOsByNames.initiate>[1]
): Promise<VideoDTO | null> => {
  const _options = {
    subscribe: false,
    ...options,
  };
  const req = getStore().dispatch(videosApi.endpoints.getVideoDTOsByNames.initiate({ video_ids: [id] }, _options));
  try {
    return (await req.unwrap())[0] ?? null;
  } catch {
    return null;
  }
};

export const useVideoDTO = (video_id: string | null | undefined) => {
  const { currentData: videoDTO } = useGetVideoDTOQuery(video_id ?? skipToken);
  return videoDTO ?? null;
};
