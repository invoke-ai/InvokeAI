import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig } from 'app/store/store';
import { stylePresetsApi } from 'services/api/endpoints/stylePresets';

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
  extraReducers(builder) {
    builder.addMatcher(stylePresetsApi.endpoints.deleteStylePreset.matchFulfilled, (state, action) => {
      if (state.activeStylePresetId === null) {
        return;
      }
      const deletedId = action.meta.arg.originalArgs;
      if (state.activeStylePresetId === deletedId) {
        state.activeStylePresetId = null;
      }
    });
    builder.addMatcher(stylePresetsApi.endpoints.listStylePresets.matchFulfilled, (state, action) => {
      if (state.activeStylePresetId === null) {
        return;
      }
      const ids = action.payload.map((preset) => preset.id);
      if (!ids.includes(state.activeStylePresetId)) {
        state.activeStylePresetId = null;
      }
    });
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
