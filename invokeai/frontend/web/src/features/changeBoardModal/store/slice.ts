import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';

import { initialState } from './initialState';

const getInitialState = () => deepClone(initialState);

const slice = createSlice({
  name: 'changeBoardModal',
  initialState,
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    imagesToChangeSelected: (state, action: PayloadAction<string[]>) => {
      state.image_names = action.payload;
    },
    changeBoardReset: (state) => {
      state.image_names = [];
      state.isModalOpen = false;
    },
  },
});

export const { isModalOpenChanged, imagesToChangeSelected, changeBoardReset } = slice.actions;

export const selectChangeBoardModalSlice = (state: RootState) => state.changeBoardModal;

export const changeBoardModalSliceConfig: SliceConfig<typeof slice> = {
  slice,
  getInitialState,
};
