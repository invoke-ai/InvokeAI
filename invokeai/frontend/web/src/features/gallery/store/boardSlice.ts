import {
  PayloadAction,
  Update,
  createEntityAdapter,
  createSlice,
} from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { BoardDTO } from 'services/api';
import { dateComparator } from 'common/util/dateComparator';
import { receivedBoards } from '../../../services/thunks/board';

export const boardsAdapter = createEntityAdapter<BoardDTO>({
  selectId: (board) => board.board_id,
  sortComparer: (a, b) => dateComparator(b.updated_at, a.updated_at),
});

type AdditionalBoardsState = {
  offset: number;
  limit: number;
  total: number;
  isLoading: boolean;
};

export const initialBoardsState =
  boardsAdapter.getInitialState<AdditionalBoardsState>({
    offset: 0,
    limit: 0,
    total: 0,
    isLoading: false,
  });

export type BoardsState = typeof initialBoardsState;

const boardsSlice = createSlice({
  name: 'boards',
  initialState: initialBoardsState,
  reducers: {
    boardUpserted: (state, action: PayloadAction<BoardDTO>) => {
      boardsAdapter.upsertOne(state, action.payload);
    },
    boardUpdatedOne: (state, action: PayloadAction<Update<BoardDTO>>) => {
      boardsAdapter.updateOne(state, action.payload);
    },
    boardRemoved: (state, action: PayloadAction<string>) => {
      boardsAdapter.removeOne(state, action.payload);
    },
  },
  extraReducers: (builder) => {
    builder.addCase(receivedBoards.pending, (state) => {
      state.isLoading = true;
    });
    builder.addCase(receivedBoards.rejected, (state) => {
      state.isLoading = false;
    });
    builder.addCase(receivedBoards.fulfilled, (state, action) => {
      state.isLoading = false;
      const { items, offset, limit, total } = action.payload;
      state.offset = offset;
      state.limit = limit;
      state.total = total;
      boardsAdapter.upsertMany(state, items);
    });
  },
});

export const {
  selectAll: selectBoardsAll,
  selectById: selectBoardsById,
  selectEntities: selectBoardsEntities,
  selectIds: selectBoardsIds,
  selectTotal: selectBoardsTotal,
} = boardsAdapter.getSelectors<RootState>((state) => state.boards);

export const { boardUpserted, boardUpdatedOne, boardRemoved } =
  boardsSlice.actions;

export default boardsSlice.reducer;
