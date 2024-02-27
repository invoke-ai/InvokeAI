import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type { BoardDTO, OffsetPaginatedResults_ImageDTO_, UpdateBoardArg } from 'services/api/types';
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
    listAllBoards: build.query<Array<BoardDTO>, void>({
      query: () => ({
        url: buildBoardsUrl(),
        params: { all: true },
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

    listAllImageNamesForBoard: build.query<Array<string>, string>({
      query: (board_id) => ({
        url: buildBoardsUrl(`${board_id}/image_names`),
      }),
      providesTags: (result, error, arg) => [{ type: 'ImageNameList', id: arg }, 'FetchOnReconnect'],
      keepUnusedDataFor: 0,
    }),

    getBoardImagesTotal: build.query<{ total: number }, string | undefined>({
      query: (board_id) => ({
        url: getListImagesUrl({
          board_id: board_id ?? 'none',
          categories: IMAGE_CATEGORIES,
          is_intermediate: false,
          limit: 0,
          offset: 0,
        }),
        method: 'GET',
      }),
      providesTags: (result, error, arg) => [{ type: 'BoardImagesTotal', id: arg ?? 'none' }, 'FetchOnReconnect'],
      transformResponse: (response: OffsetPaginatedResults_ImageDTO_) => {
        return { total: response.total };
      },
    }),

    getBoardAssetsTotal: build.query<{ total: number }, string | undefined>({
      query: (board_id) => ({
        url: getListImagesUrl({
          board_id: board_id ?? 'none',
          categories: ASSETS_CATEGORIES,
          is_intermediate: false,
          limit: 0,
          offset: 0,
        }),
        method: 'GET',
      }),
      providesTags: (result, error, arg) => [{ type: 'BoardAssetsTotal', id: arg ?? 'none' }, 'FetchOnReconnect'],
      transformResponse: (response: OffsetPaginatedResults_ImageDTO_) => {
        return { total: response.total };
      },
    }),

    /**
     * Boards Mutations
     */

    createBoard: build.mutation<BoardDTO, string>({
      query: (board_name) => ({
        url: buildBoardsUrl(),
        method: 'POST',
        params: { board_name },
      }),
      invalidatesTags: [{ type: 'Board', id: LIST_TAG }],
    }),

    updateBoard: build.mutation<BoardDTO, UpdateBoardArg>({
      query: ({ board_id, changes }) => ({
        url: buildBoardsUrl(board_id),
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (result, error, arg) => [{ type: 'Board', id: arg.board_id }],
    }),
  }),
});

export const {
  useListAllBoardsQuery,
  useGetBoardImagesTotalQuery,
  useGetBoardAssetsTotalQuery,
  useCreateBoardMutation,
  useUpdateBoardMutation,
  useListAllImageNamesForBoardQuery,
} = boardsApi;
