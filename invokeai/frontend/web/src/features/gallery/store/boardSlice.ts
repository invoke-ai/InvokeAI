import { PayloadAction, createSlice } from '@reduxjs/toolkit';

type BoardsState = {
  searchText: string;
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
    setBoardSearchText: (state, action: PayloadAction<string>) => {
      state.searchText = action.payload;
    },
    setUpdateBoardModalOpen: (state, action: PayloadAction<boolean>) => {
      state.updateBoardModalOpen = action.payload;
    },
  },
});

export const { setBoardSearchText, setUpdateBoardModalOpen } =
  boardsSlice.actions;

export default boardsSlice.reducer;
