import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import type { ImageDTO } from 'services/api/types';

import type { InvokeTabName } from './tabMap';
import type { UIState } from './uiTypes';

export const initialUIState: UIState = {
  _version: 1,
  activeTab: 'txt2img',
  shouldShowImageDetails: false,
  shouldShowProgressInViewer: true,
  showShowcase: null,
  panels: {},
  accordions: {},
  expanders: {},
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
    setShouldShowProgressInViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowProgressInViewer = action.payload;
    },
    setShouldShowShowcase: (state, action: PayloadAction<ImageDTO | null>) => {
      state.showShowcase = action.payload;
    },
    panelsChanged: (state, action: PayloadAction<{ name: string; value: string }>) => {
      state.panels[action.payload.name] = action.payload.value;
    },
    accordionStateChanged: (state, action: PayloadAction<{ id: string; isOpen: boolean }>) => {
      const { id, isOpen } = action.payload;
      state.accordions[id] = isOpen;
    },
    expanderStateChanged: (state, action: PayloadAction<{ id: string; isOpen: boolean }>) => {
      const { id, isOpen } = action.payload;
      state.expanders[id] = isOpen;
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
  setShouldShowProgressInViewer,
  setShouldShowShowcase,
  panelsChanged,
  accordionStateChanged,
  expanderStateChanged,
} = uiSlice.actions;

export const selectUiSlice = (state: RootState) => state.ui;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateUIState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const uiPersistConfig: PersistConfig<UIState> = {
  name: uiSlice.name,
  initialState: initialUIState,
  migrate: migrateUIState,
  persistDenylist: ['shouldShowImageDetails'],
};
