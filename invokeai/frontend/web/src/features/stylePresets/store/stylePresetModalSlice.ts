import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';

import type { PrefilledFormData, StylePresetModalState } from './types';

const initialState: StylePresetModalState = {
  isModalOpen: false,
  updatingStylePresetId: null,
  prefilledFormData: null,
};

export const stylePresetModalSlice = createSlice({
  name: 'stylePresetModal',
  initialState: initialState,
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    updatingStylePresetIdChanged: (state, action: PayloadAction<string | null>) => {
      state.updatingStylePresetId = action.payload;
    },
    prefilledFormDataChanged: (state, action: PayloadAction<PrefilledFormData | null>) => {
      state.prefilledFormData = action.payload;
    },
  },
});

export const { isModalOpenChanged, updatingStylePresetIdChanged, prefilledFormDataChanged } =
  stylePresetModalSlice.actions;
