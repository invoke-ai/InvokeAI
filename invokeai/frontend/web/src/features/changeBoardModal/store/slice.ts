import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { ImageDTO } from 'services/api/types';

import { initialState } from './initialState';

export const changeBoardModalSlice = createSlice({
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

export const { isModalOpenChanged, imagesToChangeSelected, changeBoardReset } = changeBoardModalSlice.actions;

export const selectChangeBoardModalSlice = (state: RootState) => state.changeBoardModal;
