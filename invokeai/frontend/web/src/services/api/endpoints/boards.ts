import { Update } from '@reduxjs/toolkit';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  boardIdSelected,
} from 'features/gallery/store/gallerySlice';
import {
  BoardDTO,
  ImageDTO,
  OffsetPaginatedResults_BoardDTO_,
} from 'services/api/types';
import { ApiFullTagDescription, LIST_TAG, api } from '..';
import { paths } from '../schema';
import { getListImagesUrl, imagesAdapter, imagesApi } from './images';

type ListBoardsArg = NonNullable<
  paths['/api/v1/boards/']['get']['parameters']['query']
>;

type UpdateBoardArg =
  paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
    changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
  };

type DeleteBoardResult =
  paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'];

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

    listAllImageNamesForBoard: build.query<Array<string>, string>({
      query: (board_id) => ({
        url: `boards/${board_id}/image_names`,
      }),
      providesTags: (result, error, arg) => [
        { type: 'ImageNameList', id: arg },
      ],
      keepUnusedDataFor: 0,
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

    deleteBoard: build.mutation<DeleteBoardResult, string>({
      query: (board_id) => ({ url: `boards/${board_id}`, method: 'DELETE' }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Board', id: arg },
        // invalidate the 'No Board' cache
        { type: 'ImageList', id: getListImagesUrl({ board_id: 'none' }) },
      ],
      async onQueryStarted(board_id, { dispatch, queryFulfilled, getState }) {
        /**
         * Cache changes for deleteBoard:
         * - Update every image in the 'getImageDTO' cache that has the board_id
         * - Update every image in the 'All Images' cache that has the board_id
         * - Update every image in the 'All Assets' cache that has the board_id
         * - Invalidate the 'No Board' cache:
         *   Ideally we'd be able to insert all deleted images into the cache, but we don't
         *   have access to the deleted images DTOs - only the names, and a network request
         *   for all of a board's DTOs could be very large. Instead, we invalidate the 'No Board'
         *   cache.
         */

        try {
          const { data } = await queryFulfilled;
          const { deleted_board_images } = data;

          // update getImageDTO caches
          deleted_board_images.forEach((image_id) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'getImageDTO',
                image_id,
                (draft) => {
                  draft.board_id = undefined;
                }
              )
            );
          });

          // update 'All Images' & 'All Assets' caches
          const queryArgsToUpdate = [
            {
              categories: IMAGE_CATEGORIES,
            },
            {
              categories: ASSETS_CATEGORIES,
            },
          ];

          const updates: Update<ImageDTO>[] = deleted_board_images.map(
            (image_name) => ({
              id: image_name,
              changes: { board_id: undefined },
            })
          );

          queryArgsToUpdate.forEach((queryArgs) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                queryArgs,
                (draft) => {
                  const oldCount = imagesAdapter
                    .getSelectors()
                    .selectTotal(draft);
                  const newState = imagesAdapter.updateMany(draft, updates);
                  const newCount = imagesAdapter
                    .getSelectors()
                    .selectTotal(newState);
                  draft.total = Math.max(
                    draft.total - (oldCount - newCount),
                    0
                  );
                }
              )
            );
          });

          // after deleting a board, select the 'All Images' board
          dispatch(boardIdSelected('images'));
        } catch {
          //no-op
        }
      },
    }),

    deleteBoardAndImages: build.mutation<DeleteBoardResult, string>({
      query: (board_id) => ({
        url: `boards/${board_id}`,
        method: 'DELETE',
        params: { include_images: true },
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'Board', id: arg },
        { type: 'ImageList', id: getListImagesUrl({ board_id: 'none' }) },
      ],
      async onQueryStarted(board_id, { dispatch, queryFulfilled, getState }) {
        /**
         * Cache changes for deleteBoardAndImages:
         * - ~~Remove every image in the 'getImageDTO' cache that has the board_id~~
         *   This isn't actually possible, you cannot remove cache entries with RTK Query.
         *   Instead, we rely on the UI to remove all components that use the deleted images.
         * - Remove every image in the 'All Images' cache that has the board_id
         * - Remove every image in the 'All Assets' cache that has the board_id
         */

        try {
          const { data } = await queryFulfilled;
          const { deleted_images } = data;

          // update 'All Images' & 'All Assets' caches
          const queryArgsToUpdate = [
            {
              categories: IMAGE_CATEGORIES,
            },
            {
              categories: ASSETS_CATEGORIES,
            },
          ];

          queryArgsToUpdate.forEach((queryArgs) => {
            dispatch(
              imagesApi.util.updateQueryData(
                'listImages',
                queryArgs,
                (draft) => {
                  const oldCount = imagesAdapter
                    .getSelectors()
                    .selectTotal(draft);
                  const newState = imagesAdapter.removeMany(
                    draft,
                    deleted_images
                  );
                  const newCount = imagesAdapter
                    .getSelectors()
                    .selectTotal(newState);
                  draft.total = Math.max(
                    draft.total - (oldCount - newCount),
                    0
                  );
                }
              )
            );
          });

          // after deleting a board, select the 'All Images' board
          dispatch(boardIdSelected('images'));
        } catch {
          //no-op
        }
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
  useDeleteBoardAndImagesMutation,
  useListAllImageNamesForBoardQuery,
} = boardsApi;
