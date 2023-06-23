import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { api } from 'services/api';

type BoardsState = {
  searchText: string;
  selectedBoardId?: string;
  updateBoardModalOpen: boolean;
};

export const initialBoardsState: BoardsState = {
  updateBoardModalOpen: false,
  searchText: '',
};

const boardsSlice = createSlice({
  name: 'boards',
  initialState: initialBoardsState,
  reducers: {
    boardIdSelected: (state, action: PayloadAction<string | undefined>) => {
      state.selectedBoardId = action.payload;
    },
    setBoardSearchText: (state, action: PayloadAction<string>) => {
      state.searchText = action.payload;
    },
    setUpdateBoardModalOpen: (state, action: PayloadAction<boolean>) => {
      state.updateBoardModalOpen = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      api.endpoints.deleteBoard.matchFulfilled,
      (state, action) => {
        if (action.meta.arg.originalArgs === state.selectedBoardId) {
          state.selectedBoardId = undefined;
        }
      }
    );
  },
});

export const { boardIdSelected, setBoardSearchText, setUpdateBoardModalOpen } =
  boardsSlice.actions;

export const boardsSelector = (state: RootState) => state.boards;

export default boardsSlice.reducer;
