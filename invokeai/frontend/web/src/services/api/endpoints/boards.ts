import { ASSETS_CATEGORIES, IMAGE_CATEGORIES } from 'features/gallery/store/types';
import type {
  BoardDTO,
  ListBoardsArg,
  OffsetPaginatedResults_BoardDTO_,
  OffsetPaginatedResults_ImageDTO_,
  UpdateBoardArg,
} from 'services/api/types';
import { getListImagesUrl } from 'services/api/util';

import type { ApiTagDescription } from '..';
import { api, LIST_TAG } from '..';

export const boardsApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Boards Queries
     */
    listBoards: build.query<OffsetPaginatedResults_BoardDTO_, ListBoardsArg>({
      query: (arg) => ({ url: 'boards/', params: arg }),
      providesTags: (result) => {
        // any list of boards
        const tags: ApiTagDescription[] = [{ type: 'Board', id: LIST_TAG }, 'FetchOnReconnect'];

        if (result) {
          // and individual tags for each board
          tags.push(
            ...result.items.map(({ board_id }) => ({
              type: 'Board' as const,
              id: board_id,
            }))
          );
        }

        return tags;
      },
    }),

    listAllBoards: build.query<Array<BoardDTO>, void>({
      query: () => ({
        url: 'boards/',
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
        url: `boards/${board_id}/image_names`,
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
        url: `boards/`,
        method: 'POST',
        params: { board_name },
      }),
      invalidatesTags: [{ type: 'Board', id: LIST_TAG }],
    }),

    updateBoard: build.mutation<BoardDTO, UpdateBoardArg>({
      query: ({ board_id, changes }) => ({
        url: `boards/${board_id}`,
        method: 'PATCH',
        body: changes,
      }),
      invalidatesTags: (result, error, arg) => [{ type: 'Board', id: arg.board_id }],
    }),
  }),
});

export const {
  useListBoardsQuery,
  useListAllBoardsQuery,
  useGetBoardImagesTotalQuery,
  useGetBoardAssetsTotalQuery,
  useCreateBoardMutation,
  useUpdateBoardMutation,
  useListAllImageNamesForBoardQuery,
} = boardsApi;
