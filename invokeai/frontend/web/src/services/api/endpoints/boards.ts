import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import queryString from 'query-string';
import type {
  BoardDTO,
  CreateBoardArg,
  GetImageNamesResult,
  ImageCategory,
  ListBoardsArgs,
  ResourceType,
  UpdateBoardArg,
} from 'services/api/types';
import { getListImagesUrl } from 'services/api/util';

import type { ApiTagDescription } from '..';
import { api, buildV1Url, LIST_TAG } from '..';

/**
 * Builds an endpoint URL for the boards router
 * @example
 * buildBoardsUrl('some-path')
 * // '/api/v1/boards/some-path'
 */
export const buildBoardsUrl = (path: string = '') => buildV1Url(`boards/${path}`);

export const boardsApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Boards Queries
     */
    listAllBoards: build.query<Array<BoardDTO>, ListBoardsArgs>({
      query: (args) => ({
        url: buildBoardsUrl(),
        params: { all: true, ...args },
      }),
      providesTags: (result) => {
        // any list of boards
        const tags: ApiTagDescription[] = [{ type: 'Board', id: LIST_TAG }, 'FetchOnReconnect'];

        if (result) {
          // and individual tags for each board
          tags.push(
            ...result.map(({ board_id }) => ({
              type: 'Board' as const,
              id: board_id,
            }))
          );
        }

        return tags;
      },
    }),

    listAllResourceIdsForBoard: build.query<
      Array<string>,
      { board_id: string | 'none'; categories: ImageCategory[] | undefined; is_intermediate: boolean | undefined; resource_type: ResourceType }
    >({
      query: ({ board_id, categories, is_intermediate, resource_type }) => ({
        url: buildBoardsUrl(
          `${board_id}/resource_ids?${queryString.stringify({ categories, is_intermediate, resource_type }, { arrayFormat: 'none' })}`
        ),
      }),
      providesTags: (result, error, arg) => [{ type: 'ImageNameList', id: JSON.stringify(arg) }, 'FetchOnReconnect'],
    }),

    getBoardImagesTotal: build.query<{ total: number }, string | undefined>({
      query: (board_id) => ({
        url: getListImagesUrl({
          board_id: board_id ?? 'none',
          categories: IMAGE_CATEGORIES,
          is_intermediate: false,
        }),
        method: 'GET',
      }),
      providesTags: (result, error, arg) => [{ type: 'BoardImagesTotal', id: arg ?? 'none' }, 'FetchOnReconnect'],
      transformResponse: (response: GetImageNamesResult) => {
        return { total: response.total_count };
      },
    }),

    getBoardAssetsTotal: build.query<{ total: number }, string | undefined>({
      query: (board_id) => ({
        url: getListImagesUrl({
          board_id: board_id ?? 'none',
          categories: ASSETS_CATEGORIES,
          is_intermediate: false,
        }),
        method: 'GET',
      }),
      providesTags: (result, error, arg) => [{ type: 'BoardAssetsTotal', id: arg ?? 'none' }, 'FetchOnReconnect'],
      transformResponse: (response: GetImageNamesResult) => {
        return { total: response.total_count };
      },
    }),

    /**
     * Boards Mutations
     */

    createBoard: build.mutation<BoardDTO, CreateBoardArg>({
      query: ({ board_name, is_private }) => ({
        url: buildBoardsUrl(),
        method: 'POST',
        params: { board_name, is_private },
      }),
      invalidatesTags: [{ type: 'Board', id: LIST_TAG }],
    }),

    updateBoard: build.mutation<BoardDTO, UpdateBoardArg>({
      query: ({ board_id, changes }) => ({
        url: buildBoardsUrl(board_id),
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (result, error, arg) => {
        const tags: ApiTagDescription[] = [];
        if (Object.keys(arg.changes).includes('archived')) {
          tags.push({ type: 'Board', id: LIST_TAG });
        }

        tags.push({ type: 'Board', id: arg.board_id });

        return tags;
      },
    }),
  }),
});

export const {
  useListAllBoardsQuery,
  useGetBoardImagesTotalQuery,
  useGetBoardAssetsTotalQuery,
  useCreateBoardMutation,
  useUpdateBoardMutation,
  useListAllResourceIdsForBoardQuery,
} = boardsApi;
