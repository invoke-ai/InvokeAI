import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { paramsReset } from 'features/controlLayers/store/paramsSlice';
import { atom } from 'nanostores';
import { stylePresetsApi } from 'services/api/endpoints/stylePresets';
import { assert } from 'tsafe';
import z from 'zod';

const zStylePresetState = z.object({
  activeStylePresetId: z.string().nullable(),
  searchTerm: z.string(),
  viewMode: z.boolean(),
  showPromptPreviews: z.boolean(),
});

type StylePresetState = z.infer<typeof zStylePresetState>;

const getInitialState = (): StylePresetState => ({
  activeStylePresetId: null,
  searchTerm: '',
  viewMode: false,
  showPromptPreviews: false,
});

const slice = createSlice({
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

export const stylePresetSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zStylePresetState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      return zStylePresetState.parse(state);
    },
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
