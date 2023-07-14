import { api } from '..';
import { paths } from '../schema';

type AddImageToBoardArg =
  paths['/api/v1/board_images/']['post']['requestBody']['content']['application/json'];

type RemoveImageFromBoardArg =
  paths['/api/v1/board_images/']['delete']['requestBody']['content']['application/json'];

export const boardImagesApi = api.injectEndpoints({
  endpoints: (build) => ({
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
        { type: 'BoardImage' },
        { type: 'Board', id: arg.board_id }
      ],
    }),

    removeImageFromBoard: build.mutation<void, RemoveImageFromBoardArg>({
      query: ({ board_id, image_name }) => ({
        url: `board_images/`,
        method: 'DELETE',
        body: { board_id, image_name },
      }),
      invalidatesTags: (result, error, arg) => [
        { type: 'BoardImage' },
        { type: 'Board', id: arg.board_id }
      ],
    }),
  }),
});

export const {
  useAddImageToBoardMutation,
  useRemoveImageFromBoardMutation
} = boardImagesApi;
