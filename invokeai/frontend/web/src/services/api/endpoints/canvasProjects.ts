import { getStore } from 'app/store/nanostores/store';
import type { paths } from 'services/api/schema';
import type {
  CanvasProjectDTO,
  ListCanvasProjectsArgs,
  ListCanvasProjectsResponse,
  ReplaceCanvasProjectFileArg,
  UploadCanvasProjectArg,
} from 'services/api/types';
import {
  getTagsToInvalidateForBoardAffectingMutation,
  getTagsToInvalidateForCanvasProjectMutation,
} from 'services/api/util/tagInvalidation';
import stableHash from 'stable-hash';
import type { Param0 } from 'tsafe';

import { api, buildV1Url, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the canvas projects router.
 * @example
 * buildCanvasProjectsUrl('some-path') // 'api/v1/canvas_projects/some-path'
 */
const buildCanvasProjectsUrl = (path: string = '', query?: Parameters<typeof buildV1Url>[1]) =>
  buildV1Url(`canvas_projects/${path}`, query);

/**
 * Canvas-project RTK Query endpoints — parallel to imagesApi / videosApi. Used by the canvas
 * save/load flows and the gallery integration.
 */
export const canvasProjectsApi = api.injectEndpoints({
  endpoints: (build) => ({
    listCanvasProjects: build.query<ListCanvasProjectsResponse, ListCanvasProjectsArgs>({
      query: (queryArgs) => ({
        url: buildCanvasProjectsUrl('', queryArgs),
        method: 'GET',
      }),
      providesTags: (result, error, queryArgs) => [
        { type: 'CanvasProjectList', id: stableHash(queryArgs) },
        { type: 'Board', id: queryArgs.board_id ?? 'none' },
        'FetchOnReconnect',
      ],
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        // Pre-populate the per-project getCanvasProjectDTO cache so click-to-load feels snappy.
        const res = await queryFulfilled;
        const projectDTOs = res.data.items;
        const updates: Param0<typeof canvasProjectsApi.util.upsertQueryEntries> = [];
        for (const projectDTO of projectDTOs) {
          updates.push({
            endpointName: 'getCanvasProjectDTO',
            arg: projectDTO.project_name,
            value: projectDTO,
          });
        }
        dispatch(canvasProjectsApi.util.upsertQueryEntries(updates));
      },
    }),

    getCanvasProjectDTO: build.query<CanvasProjectDTO, string>({
      query: (project_name) => ({ url: buildCanvasProjectsUrl(`i/${project_name}`) }),
      providesTags: (result, error, project_name) => [{ type: 'CanvasProject', id: project_name }],
    }),

    deleteCanvasProject: build.mutation<
      paths['/api/v1/canvas_projects/i/{project_name}']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/canvas_projects/i/{project_name}']['delete']['parameters']['path']
    >({
      query: ({ project_name }) => ({
        url: buildCanvasProjectsUrl(`i/${project_name}`),
        method: 'DELETE',
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          { type: 'CanvasProjectList', id: LIST_TAG },
        ];
      },
    }),

    deleteCanvasProjects: build.mutation<
      paths['/api/v1/canvas_projects/delete']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/canvas_projects/delete']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildCanvasProjectsUrl('delete'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
          { type: 'CanvasProjectList', id: LIST_TAG },
        ];
      },
    }),

    updateCanvasProject: build.mutation<
      paths['/api/v1/canvas_projects/i/{project_name}']['patch']['responses']['200']['content']['application/json'],
      {
        project_name: string;
        changes: paths['/api/v1/canvas_projects/i/{project_name}']['patch']['requestBody']['content']['application/json'];
      }
    >({
      query: ({ project_name, changes }) => ({
        url: buildCanvasProjectsUrl(`i/${project_name}`),
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForCanvasProjectMutation([result.project_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([result.board_id ?? 'none']),
        ];
      },
    }),

    starCanvasProjects: build.mutation<
      paths['/api/v1/canvas_projects/star']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/canvas_projects/star']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildCanvasProjectsUrl('star'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForCanvasProjectMutation(result.starred_projects),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),

    unstarCanvasProjects: build.mutation<
      paths['/api/v1/canvas_projects/unstar']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/canvas_projects/unstar']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildCanvasProjectsUrl('unstar'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForCanvasProjectMutation(result.unstarred_projects),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),

    uploadCanvasProject: build.mutation<
      paths['/api/v1/canvas_projects/upload']['post']['responses']['201']['content']['application/json'],
      UploadCanvasProjectArg
    >({
      query: ({ file, name, app_version, width, height, image_count, thumbnail, board_id, is_intermediate }) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('name', name);
        formData.append('app_version', app_version);
        formData.append('width', String(width));
        formData.append('height', String(height));
        formData.append('image_count', String(image_count));
        formData.append('is_intermediate', String(is_intermediate ?? false));
        if (thumbnail) {
          formData.append('thumbnail', thumbnail, 'preview.webp');
        }
        if (board_id && board_id !== 'none') {
          formData.append('board_id', board_id);
        }
        return {
          url: buildCanvasProjectsUrl('upload'),
          method: 'POST',
          body: formData,
        };
      },
      invalidatesTags: (result) => {
        if (!result || result.is_intermediate) {
          return [];
        }
        const boardId = result.board_id ?? 'none';
        return [
          ...getTagsToInvalidateForCanvasProjectMutation([result.project_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([boardId]),
          { type: 'CanvasProjectList', id: LIST_TAG },
        ];
      },
    }),

    replaceCanvasProjectFile: build.mutation<CanvasProjectDTO, ReplaceCanvasProjectFileArg>({
      query: ({ project_name, file, name, app_version, width, height, image_count, thumbnail }) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('app_version', app_version);
        formData.append('width', String(width));
        formData.append('height', String(height));
        formData.append('image_count', String(image_count));
        if (name !== undefined) {
          formData.append('name', name);
        }
        if (thumbnail) {
          formData.append('thumbnail', thumbnail, 'preview.webp');
        }
        return {
          url: buildCanvasProjectsUrl(`i/${project_name}/file`),
          method: 'PUT',
          body: formData,
        };
      },
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForCanvasProjectMutation([result.project_name]),
          ...getTagsToInvalidateForBoardAffectingMutation([result.board_id ?? 'none']),
          { type: 'CanvasProjectList', id: LIST_TAG },
        ];
      },
    }),

    addCanvasProjectToBoard: build.mutation<
      paths['/api/v1/board_canvas_projects/']['post']['responses']['200']['content']['application/json'],
      paths['/api/v1/board_canvas_projects/']['post']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildV1Url('board_canvas_projects/'),
        method: 'POST',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForCanvasProjectMutation(result.added_projects),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),

    removeCanvasProjectFromBoard: build.mutation<
      paths['/api/v1/board_canvas_projects/']['delete']['responses']['200']['content']['application/json'],
      paths['/api/v1/board_canvas_projects/']['delete']['requestBody']['content']['application/json']
    >({
      query: (body) => ({
        url: buildV1Url('board_canvas_projects/'),
        method: 'DELETE',
        body,
      }),
      invalidatesTags: (result) => {
        if (!result) {
          return [];
        }
        return [
          ...getTagsToInvalidateForCanvasProjectMutation(result.removed_projects),
          ...getTagsToInvalidateForBoardAffectingMutation(result.affected_boards),
        ];
      },
    }),
  }),
});

export const {
  useListCanvasProjectsQuery,
  useGetCanvasProjectDTOQuery,
  useUploadCanvasProjectMutation,
  useReplaceCanvasProjectFileMutation,
  useDeleteCanvasProjectMutation,
  useDeleteCanvasProjectsMutation,
  useUpdateCanvasProjectMutation,
  useStarCanvasProjectsMutation,
  useUnstarCanvasProjectsMutation,
  useAddCanvasProjectToBoardMutation,
  useRemoveCanvasProjectFromBoardMutation,
} = canvasProjectsApi;

/**
 * Imperative helper to fetch a CanvasProjectDTO. Mirrors `getVideoDTOSafe` / `getImageDTOSafe`.
 */
export const getCanvasProjectDTOSafe = async (
  project_name: string,
  options?: Parameters<typeof canvasProjectsApi.endpoints.getCanvasProjectDTO.initiate>[1]
): Promise<CanvasProjectDTO | null> => {
  const _options = { subscribe: false, ...options };
  const req = getStore().dispatch(canvasProjectsApi.endpoints.getCanvasProjectDTO.initiate(project_name, _options));
  try {
    return await req.unwrap();
  } catch {
    return null;
  }
};

/**
 * Imperative ZIP fetch for a server-stored canvas project. Returns the raw `.invk` Blob so the
 * caller can hand it to the existing `parseCanvasProjectZip` parser.
 */
export const fetchCanvasProjectZip = async (project_name: string): Promise<Blob> => {
  const response = await fetch(buildCanvasProjectsUrl(`i/${project_name}/full`));
  if (!response.ok) {
    throw new Error(`Failed to fetch canvas project ${project_name}: HTTP ${response.status}`);
  }
  return await response.blob();
};
