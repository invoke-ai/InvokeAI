import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { atom } from 'nanostores';
import { stylePresetsApi } from 'services/api/endpoints/stylePresets';

import type { StylePresetState } from './types';

const getInitialState = (): StylePresetState => ({
  activeStylePresetId: null,
  searchTerm: '',
  viewMode: false,
  showPromptPreviews: false,
});

export const slice = createSlice({
  name: 'stylePreset',
  initialState: getInitialState(),
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
    showPromptPreviewsChanged: (state, action: PayloadAction<boolean>) => {
      state.showPromptPreviews = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addCase(paramsReset, () => {
      return getInitialState();
    });
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

export const { activeStylePresetIdChanged, searchTermChanged, viewModeChanged, showPromptPreviewsChanged } =
  slice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const stylePresetSliceConfig: SliceConfig<typeof slice> = {
  slice,
  getInitialState,
  persistConfig: {
    migrate,
  },
};

export const selectStylePresetSlice = (state: RootState) => state.stylePreset;
const createStylePresetSelector = <T>(selector: Selector<StylePresetState, T>) =>
  createSelector(selectStylePresetSlice, selector);

export const selectStylePresetActivePresetId = createStylePresetSelector(
  (stylePreset) => stylePreset.activeStylePresetId
);
export const selectStylePresetViewMode = createStylePresetSelector((stylePreset) => stylePreset.viewMode);
export const selectStylePresetSearchTerm = createStylePresetSelector((stylePreset) => stylePreset.searchTerm);
export const selectShowPromptPreviews = createStylePresetSelector((stylePreset) => stylePreset.showPromptPreviews);

/**
 * Tracks whether or not the style preset menu is open.
 */
export const $isStylePresetsMenuOpen = atom<boolean>(false);
