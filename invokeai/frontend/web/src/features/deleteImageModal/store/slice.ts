import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api/types';
import { initialDeleteImageState } from './initialState';

const deleteImageModal = createSlice({
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

export const {
  isModalOpenChanged,
  imagesToDeleteSelected,
  imageDeletionCanceled,
} = deleteImageModal.actions;

export default deleteImageModal.reducer;
