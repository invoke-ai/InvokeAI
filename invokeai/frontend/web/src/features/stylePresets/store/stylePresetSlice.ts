import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';

import type { StylePresetState } from './types';

const initialState: StylePresetState = {
  isMenuOpen: false,
  activeStylePreset: null,
  searchTerm: '',
  viewMode: false,
};

export const stylePresetSlice = createSlice({
  name: 'stylePreset',
  initialState: initialState,
  reducers: {
    isMenuOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isMenuOpen = action.payload;
    },
    activeStylePresetChanged: (state, action: PayloadAction<StylePresetRecordWithImage | null>) => {
      state.activeStylePreset = action.payload;
    },
    searchTermChanged: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
    },
    viewModeChanged: (state, action: PayloadAction<boolean>) => {
      state.viewMode = action.payload;
    },
  },
});

export const { isMenuOpenChanged, activeStylePresetChanged, searchTermChanged, viewModeChanged } =
  stylePresetSlice.actions;
