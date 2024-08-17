import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig } from 'app/store/store';

import type { StylePresetState } from './types';

const initialState: StylePresetState = {
  activeStylePresetId: null,
  searchTerm: '',
  viewMode: false,
};

export const stylePresetSlice = createSlice({
  name: 'stylePreset',
  initialState: initialState,
  reducers: {
    activeStylePresetIdChanged: (state, action: PayloadAction<string | null>) => {
      state.activeStylePresetId = action.payload;
    },
    searchTermChanged: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
    },
    viewModeChanged: (state, action: PayloadAction<boolean>) => {
      state.viewMode = action.payload;
    },
  },
});

export const { activeStylePresetIdChanged, searchTermChanged, viewModeChanged } = stylePresetSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateStylePresetState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const stylePresetPersistConfig: PersistConfig<StylePresetState> = {
  name: stylePresetSlice.name,
  initialState,
  migrate: migrateStylePresetState,
  persistDenylist: [],
};
