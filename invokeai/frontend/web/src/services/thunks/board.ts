import { createAppAsyncThunk } from '../../app/store/storeUtils';
import { BoardsService } from '../api';

/**
 * `BoardsService.listBoards()` thunk
 */
export const receivedBoards = createAppAsyncThunk(
  'api/receivedBoards',
  async (_, { getState }) => {
    const response = await BoardsService.listBoards({});
    return response;
  }
);

type BoardCreatedArg = Parameters<(typeof BoardsService)['createBoard']>[0];

export const boardCreated = createAppAsyncThunk(
  'api/boardCreated',
  async (arg: BoardCreatedArg) => {
    const response = await BoardsService.createBoard(arg);
    return response;
  }
);
