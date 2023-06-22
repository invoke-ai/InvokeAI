import {
  TagDescription,
  createApi,
  fetchBaseQuery,
} from '@reduxjs/toolkit/query/react';
import { BoardDTO } from './api/models/BoardDTO';
import { OffsetPaginatedResults_BoardDTO_ } from './api/models/OffsetPaginatedResults_BoardDTO_';
import { BoardChanges } from './api/models/BoardChanges';
import { OffsetPaginatedResults_ImageDTO_ } from './api/models/OffsetPaginatedResults_ImageDTO_';
import { ImageDTO } from './api/models/ImageDTO';
import {
  FullTagDescription,
  TagTypesFrom,
  TagTypesFromApi,
} from '@reduxjs/toolkit/dist/query/endpointDefinitions';

type ListBoardsArg = { offset: number; limit: number };
type UpdateBoardArg = { board_id: string; changes: BoardChanges };
type AddImageToBoardArg = { board_id: string; image_name: string };
type RemoveImageFromBoardArg = { board_id: string; image_name: string };
type ListBoardImagesArg = { board_id: string; offset: number; limit: number };

const tagTypes = ['Board', 'Image'];
type ApiFullTagDescription = FullTagDescription<(typeof tagTypes)[number]>;

const LIST = 'LIST';

export const api = createApi({
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:5173/api/v1/' }),
  reducerPath: 'api',
  tagTypes,
  endpoints: (build) => ({
    /**
     * Boards Queries
     */
    listBoards: build.query<OffsetPaginatedResults_BoardDTO_, ListBoardsArg>({
      query: (arg) => ({ url: 'boards/', params: arg }),
      providesTags: (result, error, arg) => {
        // any list of boards
        const tags: ApiFullTagDescription[] = [{ id: 'Board', type: LIST }];

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
        const tags: ApiFullTagDescription[] = [{ id: 'Board', type: LIST }];

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
      invalidatesTags: [{ id: 'Board', type: LIST }],
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
      invalidatesTags: (result, error, arg) => [
        { type: 'Board', id: arg.board_id },
        { type: 'Image', id: arg.image_name },
      ],
    }),

    removeImageFromBoard: build.mutation<void, RemoveImageFromBoardArg>({
      query: ({ board_id, image_name }) => ({
        url: `board_images/`,
        method: 'DELETE',
        body: { board_id, image_name },
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Board', id: arg.board_id },
        { type: 'Image', id: arg.image_name },
      ],
    }),

    /**
     * Image Queries
     */
    getImageDTO: build.query<ImageDTO, string>({
      query: (image_name) => ({ url: `images/${image_name}/metadata` }),
      providesTags: (result, error, arg) => {
        const tags: ApiFullTagDescription[] = [{ type: 'Image', id: arg }];
        if (result?.board_id) {
          tags.push({ type: 'Board', id: result.board_id });
        }
        return tags;
      },
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
  useGetImageDTOQuery,
} = api;
