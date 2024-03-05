import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { ImageDTO } from 'services/api/types';

import { initialCropperState } from './initialState';

export const cropperSlice = createSlice({
  name: 'cropper',
  initialState: initialCropperState,
  reducers: {
    setImageToCrop: (state, action: PayloadAction<ImageDTO>) => {
      state.imageToCrop = action.payload;
    },
    isCropperModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isCropperModalOpen = action.payload;
    },
  },
});

export const { isCropperModalOpenChanged, setImageToCrop } = cropperSlice.actions;
