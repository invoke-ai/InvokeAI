import { createAppAsyncThunk } from 'app/store/storeUtils';
import { $client } from '../client';
import { components } from '../schema';

type Arg = { board_id: string };

type GetAllBoardImagesForBoardResult =
  components['schemas']['GetAllBoardImagesForBoardResult'];

type GetImageUrlsThunkConfig = {
  rejectValue: {
    arg: Arg;
    error: unknown;
  };
};
/**
 * Thunk to get image URLs
 */
export const boardImageNamesReceived = createAppAsyncThunk<
  GetAllBoardImagesForBoardResult,
  Arg,
  GetImageUrlsThunkConfig
>('thunkApi/boardImageNamesReceived', async (arg, { rejectWithValue }) => {
  const { get } = $client.get();
  const { data, error, response } = await get(
    '/api/v1/board_images/{board_id}',
    {
      params: {
        path: {
          board_id: arg.board_id,
        },
      },
    }
  );

  if (error) {
    return rejectWithValue({ arg, error });
  }

  return data;
});
