import { BoardDTO, OffsetPaginatedResults_BoardDTO_ } from 'services/api/types';
import { ApiFullTagDescription, LIST_TAG, api } from '..';
import { paths } from '../schema';

type ListBoardsArg = NonNullable<
  paths['/api/v1/boards/']['get']['parameters']['query']
>;

type UpdateBoardArg =
  paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
    changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
  };

export const boardsApi = api.injectEndpoints({
  endpoints: (build) => ({
    /**
     * Boards Queries
     */
    listBoards: build.query<OffsetPaginatedResults_BoardDTO_, ListBoardsArg>({
      query: (arg) => ({ url: 'boards/', params: arg }),
      providesTags: (result, error, arg) => {
        // any list of boards
        const tags: ApiFullTagDescription[] = [{ type: 'Board', id: LIST_TAG }];

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
      providesTags: (result, error, arg) => {
        // any list of boards
        const tags: ApiFullTagDescription[] = [{ type: 'Board', id: LIST_TAG }];

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
      invalidatesTags: (result, error, arg) => [
        { type: 'Board', id: arg.board_id },
      ],
    }),
    deleteBoard: build.mutation<void, string>({
      query: (board_id) => ({ url: `boards/${board_id}`, method: 'DELETE' }),
      invalidatesTags: (result, error, arg) => [{ type: 'Board', id: arg }],
    }),
    deleteBoardAndImages: build.mutation<void, string>({
      query: (board_id) => ({
        url: `boards/${board_id}`,
        method: 'DELETE',
        params: { include_images: true },
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Board', id: arg },
        { type: 'Image', id: LIST_TAG },
      ],
    }),
  }),
});

export const {
  useListBoardsQuery,
  useListAllBoardsQuery,
  useCreateBoardMutation,
  useUpdateBoardMutation,
  useDeleteBoardMutation,
  useDeleteBoardAndImagesMutation,
} = boardsApi;
