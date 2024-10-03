import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { newSessionRequested } from 'features/controlLayers/store/actions';
import { atom } from 'nanostores';
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
    builder.addMatcher(newSessionRequested, () => {
      return deepClone(initialState);
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

const selectStylePresetSlice = (state: RootState) => state.stylePreset;
const createStylePresetSelector = <T>(selector: Selector<StylePresetState, T>) =>
  createSelector(selectStylePresetSlice, selector);

export const selectStylePresetActivePresetId = createStylePresetSelector(
  (stylePreset) => stylePreset.activeStylePresetId
);
export const selectStylePresetViewMode = createStylePresetSelector((stylePreset) => stylePreset.viewMode);
export const selectStylePresetSearchTerm = createStylePresetSelector((stylePreset) => stylePreset.searchTerm);

/**
 * Tracks whether or not the style preset menu is open.
 */
export const $isStylePresetsMenuOpen = atom<boolean>(false);
