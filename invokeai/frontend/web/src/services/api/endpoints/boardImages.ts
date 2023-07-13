import { OffsetPaginatedResults_ImageDTO_ } from 'services/api/types';
import { api } from '..';
import { paths } from '../schema';
import { imagesApi } from './images';

type ListBoardImagesArg =
  paths['/api/v1/board_images/{board_id}']['get']['parameters']['path'] &
    paths['/api/v1/board_images/{board_id}']['get']['parameters']['query'];

type AddImageToBoardArg =
  paths['/api/v1/board_images/']['post']['requestBody']['content']['application/json'];

type RemoveImageFromBoardArg =
  paths['/api/v1/board_images/']['delete']['requestBody']['content']['application/json'];

export const boardImagesApi = api.injectEndpoints({
  endpoints: (build) => ({
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
      ],
      async onQueryStarted(
        { image_name, ...patch },
        { dispatch, queryFulfilled }
      ) {
        const patchResult = dispatch(
          imagesApi.util.updateQueryData('getImageDTO', image_name, (draft) => {
            Object.assign(draft, patch);
          })
        );
        try {
          await queryFulfilled;
        } catch {
          patchResult.undo();
        }
      },
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
      async onQueryStarted(
        { image_name, ...patch },
        { dispatch, queryFulfilled }
      ) {
        const patchResult = dispatch(
          imagesApi.util.updateQueryData('getImageDTO', image_name, (draft) => {
            Object.assign(draft, { board_id: null });
          })
        );
        try {
          await queryFulfilled;
        } catch {
          patchResult.undo();
        }
      },
    }),
  }),
});

export const {
  useAddImageToBoardMutation,
  useRemoveImageFromBoardMutation,
  useListBoardImagesQuery,
} = boardImagesApi;
