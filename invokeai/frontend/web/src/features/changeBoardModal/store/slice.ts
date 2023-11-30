import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api/types';
import { initialState } from './initialState';

const changeBoardModal = createSlice({
  name: 'changeBoardModal',
  initialState,
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    imagesToChangeSelected: (state, action: PayloadAction<ImageDTO[]>) => {
      state.imagesToChange = action.payload;
    },
    changeBoardReset: (state) => {
      state.imagesToChange = [];
      state.isModalOpen = false;
    },
  },
});

export const { isModalOpenChanged, imagesToChangeSelected, changeBoardReset } =
  changeBoardModal.actions;

export default changeBoardModal.reducer;
