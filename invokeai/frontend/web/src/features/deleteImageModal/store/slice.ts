import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { ImageDTO } from 'services/api/types';

import { initialDeleteImageState } from './initialState';

export const deleteImageModalSlice = createSlice({
  name: 'deleteImageModal',
  initialState: initialDeleteImageState,
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    imagesToDeleteSelected: (state, action: PayloadAction<ImageDTO[]>) => {
      state.imagesToDelete = action.payload;
    },
    imageDeletionCanceled: (state) => {
      state.imagesToDelete = [];
      state.isModalOpen = false;
    },
  },
});

export const { isModalOpenChanged, imagesToDeleteSelected, imageDeletionCanceled } = deleteImageModalSlice.actions;

export const selectDeleteImageModalSlice = (state: RootState) => state.deleteImageModal;
