import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { initialImageChanged } from 'features/parameters/store/generationSlice';

import type { InvokeTabName } from './tabMap';
import type { UIState } from './uiTypes';

export const initialUIState: UIState = {
  _version: 1,
  activeTab: 'txt2img',
  shouldShowImageDetails: false,
  shouldShowExistingModelsInSearch: false,
  shouldHidePreview: false,
  shouldShowProgressInViewer: true,
  panels: {},
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState: initialUIState,
  reducers: {
    setActiveTab: (state, action: PayloadAction<InvokeTabName>) => {
      state.activeTab = action.payload;
    },
    setShouldShowImageDetails: (state, action: PayloadAction<boolean>) => {
      state.shouldShowImageDetails = action.payload;
    },
    setShouldHidePreview: (state, action: PayloadAction<boolean>) => {
      state.shouldHidePreview = action.payload;
    },
    setShouldShowExistingModelsInSearch: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldShowExistingModelsInSearch = action.payload;
    },
    setShouldShowProgressInViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowProgressInViewer = action.payload;
    },
    panelsChanged: (
      state,
      action: PayloadAction<{ name: string; value: string }>
    ) => {
      state.panels[action.payload.name] = action.payload.value;
    },
  },
  extraReducers(builder) {
    builder.addCase(initialImageChanged, (state) => {
      state.activeTab = 'img2img';
    });
  },
});

export const {
  setActiveTab,
  setShouldShowImageDetails,
  setShouldShowExistingModelsInSearch,
  setShouldHidePreview,
  setShouldShowProgressInViewer,
  panelsChanged,
} = uiSlice.actions;

export default uiSlice.reducer;

export const selectUiSlice = (state: RootState) => state.ui;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateUIState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};
