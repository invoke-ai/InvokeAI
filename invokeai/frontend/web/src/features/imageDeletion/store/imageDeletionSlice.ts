import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api/types';

type DeleteImageState = {
  imageToDelete: ImageDTO | null;
  isModalOpen: boolean;
};

export const initialDeleteImageState: DeleteImageState = {
  imageToDelete: null,
  isModalOpen: false,
};

const imageDeletion = createSlice({
  name: 'imageDeletion',
  initialState: initialDeleteImageState,
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    imageToDeleteSelected: (state, action: PayloadAction<ImageDTO>) => {
      state.imageToDelete = action.payload;
    },
    imageToDeleteCleared: (state) => {
      state.imageToDelete = null;
      state.isModalOpen = false;
    },
  },
});

export const {
  isModalOpenChanged,
  imageToDeleteSelected,
  imageToDeleteCleared,
} = imageDeletion.actions;

export default imageDeletion.reducer;
