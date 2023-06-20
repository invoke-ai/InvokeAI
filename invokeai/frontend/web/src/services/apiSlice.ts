import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { BoardDTO } from './api/models/BoardDTO';
import { OffsetPaginatedResults_BoardDTO_ } from './api/models/OffsetPaginatedResults_BoardDTO_';
import { BoardChanges } from './api/models/BoardChanges';
import { OffsetPaginatedResults_ImageDTO_ } from './api/models/OffsetPaginatedResults_ImageDTO_';

type ListBoardsArg = { offset: number; limit: number };
type UpdateBoardArg = { board_id: string; changes: BoardChanges };
type AddImageToBoardArg = { board_id: string; image_name: string };
type RemoveImageFromBoardArg = { board_id: string; image_name: string };
type ListBoardImagesArg = { board_id: string; offset: number; limit: number };

export const api = createApi({
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:5173/api/v1/' }),
  reducerPath: 'api',
  tagTypes: ['Board'],
  endpoints: (build) => ({
    /**
     * Boards Queries
     */
    listBoards: build.query<OffsetPaginatedResults_BoardDTO_, ListBoardsArg>({
      query: (arg) => ({ url: 'boards/', params: arg }),
      providesTags: (result, error, arg) => {
        if (!result) {
          // Provide the broad 'Board' tag until there is a response
          return ['Board'];
        }

        // Provide the broad 'Board' tab, and individual tags for each board
        return [
          ...result.items.map(({ board_id }) => ({
            type: 'Board' as const,
            id: board_id,
          })),
          'Board',
        ];
      },
    }),

    listAllBoards: build.query<Array<BoardDTO>, void>({
      query: () => ({
        url: 'boards/',
        params: { all: true },
      }),
      providesTags: (result, error, arg) => {
        if (!result) {
          // Provide the broad 'Board' tag until there is a response
          return ['Board'];
        }

        // Provide the broad 'Board' tab, and individual tags for each board
        return [
          ...result.map(({ board_id }) => ({
            type: 'Board' as const,
            id: board_id,
          })),
          'Board',
        ];
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
      invalidatesTags: ['Board'],
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

    /**
     * Board Images Queries
     */

    listBoardImages: build.query<
      OffsetPaginatedResults_ImageDTO_,
      ListBoardImagesArg
    >({
      query: ({ board_id, offset, limit }) => ({
        url: `board_images/${board_id}`,
        method: 'DELETE',
        body: { offset, limit },
      }),
    }),

    /**
     * Board Images Mutations
     */

    addImageToBoard: build.mutation<void, AddImageToBoardArg>({
      query: ({ board_id, image_name }) => ({
        url: `board_images/`,
        method: 'POST',
        body: { board_id, image_name },
      }),
      invalidatesTags: ['Board'],
      // invalidatesTags: (result, error, arg) => [
      //   { type: 'Board', id: arg.board_id },
      // ],
    }),

    removeImageFromBoard: build.mutation<void, RemoveImageFromBoardArg>({
      query: ({ board_id, image_name }) => ({
        url: `board_images/`,
        method: 'DELETE',
        body: { board_id, image_name },
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Board', id: arg.board_id },
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
  useAddImageToBoardMutation,
  useRemoveImageFromBoardMutation,
  useListBoardImagesQuery,
} = api;
