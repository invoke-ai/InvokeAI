import { skipToken } from '@reduxjs/toolkit/query';
import { getStore } from 'app/store/nanostores/store';
import type { paths } from 'services/api/schema';
import type {
  GetVideoNamesArgs,
  GetVideoNamesResult,
  ListVideosArgs,
  ListVideosResponse,
  UploadVideoArg,
  VideoDTO,
} from 'services/api/types';
import { getListVideosUrl } from 'services/api/util';
import {
  getTagsToInvalidateForBoardAffectingMutation,
  getTagsToInvalidateForVideoMutation,
} from 'services/api/util/tagInvalidation';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';
import type { JsonObject } from 'type-fest';

import { api, buildV1Url, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the videos router.
 * @example
 * buildVideosUrl('some-path') // 'api/v1/videos/some-path'
 */
const buildVideosUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`videos/${path}`, query);

/**
 * Video RTK Query endpoints — parallel to imagesApi. Used by the gallery (Phase 4) and the
 * viewer / linear flows that land in later phases.
 */
export const videosApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * List videos (paginated). Used directly when a video-only view is needed; the gallery
     * itself uses the polymorphic /gallery/items/ endpoint.
     */
    listVideos: build.query<ListVideosResponse, ListVideosArgs>({
      query: (queryArgs) => ({
        url: getListVideosUrl(queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        { type: 'VideoList', id: stableHash(queryArgs) },
        // The LIST_TAG-scoped tag is what mutations invalidate when they can't know which
        // query-arg-specific lists changed (star/unstar, board cascades). Without
        // providing it here those invalidations match nothing and the lists go stale.
        { type: 'VideoList', id: LIST_TAG },
        { type: 'Board', id: queryArgs.board_id ?? 'none' },
        'FetchOnReconnect',
      ],
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        // Pre-populate the per-video getVideoDTO cache so selection feels snappy.
        const res = await queryFulfilled;
        const videoDTOs = res.data.items;
        const updates: Param0<typeof videosApi.util.upsertQueryEntries> = [];
        for (const videoDTO of videoDTOs) {
          updates.push({
            endpointName: 'getVideoDTO',
            arg: videoDTO.video_name,
            value: videoDTO,
          });
        }
        dispatch(videosApi.util.upsertQueryEntries(updates));
      },
    }),

    getVideoDTO: build.query<VideoDTO, string>({
      query: (video_name) => ({ url: buildVideosUrl(`i/${video_name}`) }),
      providesTags: (result, error, video_name) => [{ type: 'Video', id: video_name }],
    }),

    getVideoMetadata: build.query<JsonObject | undefined, string>({
      query: (video_name) => ({ url: buildVideosUrl(`i/${video_name}/metadata`) }),
      providesTags: (result, error, video_name) => [{ type: 'VideoMetadata', id: video_name }],
    }),

    getVideoWorkflow: build.query<
      paths['/api/v1/videos/i/{video_name}/workflow']['get']['responses']['200']['content']['application/json'],
      string
    >({
      query: (video_name) => ({ url: buildVideosUrl(`i/${video_name}/workflow`) }),
      providesTags: (result, error, video_name) => [{ type: 'VideoWorkflow', id: video_name }],
    }),

    getVideoNames: build.query<GetVideoNamesResult, GetVideoNamesArgs>({
      query: (queryArgs) => ({
        url: buildVideosUrl('names', queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        'VideoNameList',
        'FetchOnReconnect',
        { type: 'VideoNameList', id: stableHash(queryArgs) },
      ],
    }),

    deleteVideo: build.mutation<
      paths['/api/v1/videos/i/{video_name}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/i/{video_name}']['delete']['parameters']['path']
    >({
      query: ({ video_name }) => ({
        url: buildVideosUrl(`i/${video_name}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        // Per-video tags too, so the deleted video's DTO/metadata caches refetch and 404
        // instead of serving a stale entry (e.g. to node inputs still referencing it).
        return [
          ...getTagsToInvalidateForVideoMutation(result.deleted_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          { type: 'VideoList', id: LIST_TAG },
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
        // Only the server-confirmed deletions — videos that failed to delete keep their
        // live cache entries.
        return [
          ...getTagsToInvalidateForVideoMutation(result.deleted_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          { type: 'VideoList', id: LIST_TAG },
        ];
      },
    }),

    /** Companion to deleteUncategorizedImages: the "Delete All Uncategorized Images/Videos"
     * board action fires both so the polymorphic uncategorized bucket is fully cleared. */
    deleteUncategorizedVideos: build.mutation<
      paths['/api/v1/videos/uncategorized']['delete']['responses']['200']['content']['application/json'],
      void
    >({
      query: () => ({
        url: buildVideosUrl('uncategorized'),
        method: 'DELETE',
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForVideoMutation(result.deleted_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          { type: 'VideoList', id: LIST_TAG },
        ];
      },
    }),

    /** Toggle a video's is_intermediate flag. */
    changeVideoIsIntermediate: build.mutation<
      paths['/api/v1/videos/i/{video_name}']['patch']['responses']['200']['content']['application/json'],
      { video_name: string; is_intermediate: boolean }
    >({
      query: ({ video_name, is_intermediate }) => ({
        url: buildVideosUrl(`i/${video_name}`),
        method: 'PATCH',
        body: { is_intermediate },
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForVideoMutation([result.video_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([result.board_id ?? 'none']),
        ];
      },
    }),

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
        // Starring reorders every video list (starred_first), but this mutation can't
        // know which query-arg-specific lists are cached — invalidate the LIST_TAG-scoped
        // tag that all listVideos queries provide.
        return [
          ...getTagsToInvalidateForVideoMutation(result.starred_videos),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          { type: 'VideoList', id: LIST_TAG },
        ];
      },
    }),

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
          { type: 'VideoList', id: LIST_TAG },
        ];
      },
    }),

    uploadVideo: build.mutation<
      paths['/api/v1/videos/upload']['post']['responses']['201']['content']['application/json'],
      UploadVideoArg
    >({
      query: ({ file, video_category, is_intermediate, session_id, board_id, metadata }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }
        return {
          url: buildVideosUrl('upload'),
          method: 'POST',
          body: formData,
          params: {
            video_category,
            is_intermediate,
            session_id,
            board_id: board_id === 'none' ? undefined : board_id,
          },
        };
      },
      invalidatesTags: (result) => {
        if (!result || result.is_intermediate) {
          return [];
        }
        const boardId = result.board_id ?? 'none';
        return [
          ...getTagsToInvalidateForVideoMutation([result.video_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([boardId]),
          { type: 'VideoList', id: LIST_TAG },
        ];
      },
    }),

    addVideoToBoard: build.mutation<
      paths['/api/v1/videos/board']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/board']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildVideosUrl('board'),
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

    removeVideoFromBoard: build.mutation<
      paths['/api/v1/videos/board']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/videos/board']['delete']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildVideosUrl('board'),
        method: 'DELETE',
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
  useUploadVideoMutation,
  useGetVideoDTOQuery,
  useStarVideosMutation,
  useUnstarVideosMutation,
  useAddVideoToBoardMutation,
  useRemoveVideoFromBoardMutation,
  useDeleteUncategorizedVideosMutation,
} = videosApi;

/** @knipignore Reserved for follow-up phases (bulk delete / intermediate toggle / video-only views).
 * useDeleteVideoMutation is here because the only call site uses videosApi.endpoints.deleteVideo.initiate
 * via the delete-video modal, but a future bulk/multi-select flow may want the React hook form. */
export const {
  useListVideosQuery,
  useGetVideoMetadataQuery,
  useGetVideoWorkflowQuery,
  useLazyGetVideoWorkflowQuery,
  useGetVideoNamesQuery,
  useDeleteVideoMutation,
  useDeleteVideosMutation,
  useChangeVideoIsIntermediateMutation,
} = videosApi;

/**
 * Imperative helper to fetch a VideoDTO. Mirrors `getImageDTOSafe`.
 */
export const getVideoDTOSafe = async (
  video_name: string,
  options?: Parameters<typeof videosApi.endpoints.getVideoDTO.initiate>[1]
): Promise<VideoDTO | null> => {
  const _options = { subscribe: false, ...options };
  const req = getStore().dispatch(videosApi.endpoints.getVideoDTO.initiate(video_name, _options));
  try {
    return await req.unwrap();
  } catch {
    return null;
  }
};

/** @knipignore Multi-phase rollout; consumed by Phase 5 viewer code. */
export const getVideoDTO = (
  video_name: string,
  options?: Parameters<typeof videosApi.endpoints.getVideoDTO.initiate>[1]
): Promise<VideoDTO> => {
  const _options = { subscribe: false, ...options };
  const req = getStore().dispatch(videosApi.endpoints.getVideoDTO.initiate(video_name, _options));
  return req.unwrap();
};

/** @knipignore Multi-phase rollout; imperative form consumed by Phase 6 invocations. */
export const uploadVideo = (arg: UploadVideoArg): Promise<VideoDTO> => {
  const { dispatch } = getStore();
  const req = dispatch(videosApi.endpoints.uploadVideo.initiate(arg, { track: false }));
  return req.unwrap();
};

/**
 * Uploads a batch of videos and resolves with the DTOs that succeeded.
 *
 * Rejections are NOT re-thrown, mirroring `uploadImages`: per-file failure feedback
 * (an error toast naming the failed file) is handled by the `uploadVideo.matchRejected`
 * listener in `videoUploaded.ts`, which fires for every rejected mutation regardless of
 * how the caller aggregates the promises. Callers should treat the resolved array as
 * "what actually made it" and must not assume it matches the request 1:1.
 */
export const uploadVideos = async (args: UploadVideoArg[]): Promise<VideoDTO[]> => {
  const { dispatch } = getStore();
  const results = await Promise.allSettled(
    args.map((arg) => {
      const req = dispatch(videosApi.endpoints.uploadVideo.initiate(arg, { track: false }));
      return req.unwrap();
    })
  );
  return results.filter((r): r is PromiseFulfilledResult<VideoDTO> => r.status === 'fulfilled').map((r) => r.value);
};

export const useVideoDTO = (videoName: string | null | undefined) => {
  const { currentData: videoDTO } = useGetVideoDTOQuery(videoName ?? skipToken);
  return videoDTO ?? null;
};
