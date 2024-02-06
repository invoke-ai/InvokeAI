import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { galleryImageClicked } from 'features/gallery/store/gallerySlice';
import { initialImageChanged } from 'features/parameters/store/generationSlice';

import type { InvokeTabName } from './tabMap';
import type { UIState, ViewerMode } from './uiTypes';

export const initialUIState: UIState = {
  _version: 1,
  activeTab: 'txt2img',
  shouldShowImageDetails: false,
  shouldShowProgressInViewer: true,
  panels: {},
  accordions: {},
  expanders: {},
  viewerMode: 'image',
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
    viewerModeChanged: (state, action: PayloadAction<ViewerMode>) => {
      state.viewerMode = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addCase(initialImageChanged, (state) => {
      state.activeTab = 'img2img';
    });

    builder.addCase(galleryImageClicked, (state) => {
      // When a gallery image is clicked and we are in progress mode, switch to image mode.
      // This is not the same as the gallery _selection_ being changed.
      if (state.viewerMode === 'progress') {
        state.viewerMode = 'image';
      }
    });
  },
});

export const {
  setActiveTab,
  setShouldShowImageDetails,
  setShouldShowProgressInViewer,
  panelsChanged,
  accordionStateChanged,
  expanderStateChanged,
  viewerModeChanged,
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
