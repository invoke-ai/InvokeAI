import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { assert } from 'tsafe';

import { getInitialUIState, type UIState, zUIState } from './uiTypes';

const slice = createSlice({
  name: 'ui',
  initialState: getInitialUIState(),
  reducers: {
    setActiveTab: (state, action: PayloadAction<UIState['activeTab']>) => {
      state.activeTab = action.payload;
    },
    setShouldShowItemDetails: (state, action: PayloadAction<UIState['shouldShowItemDetails']>) => {
      state.shouldShowItemDetails = action.payload;
    },
    setShouldShowProgressInViewer: (state, action: PayloadAction<UIState['shouldShowProgressInViewer']>) => {
      state.shouldShowProgressInViewer = action.payload;
    },
    setShouldUsePagedGalleryView: (state, action: PayloadAction<UIState['shouldUsePagedGalleryView']>) => {
      state.shouldUsePagedGalleryView = action.payload;
    },
    accordionStateChanged: (
      state,
      action: PayloadAction<{
        id: keyof UIState['accordions'];
        isOpen: UIState['accordions'][keyof UIState['accordions']];
      }>
    ) => {
      const { id, isOpen } = action.payload;
      state.accordions[id] = isOpen;
    },
    expanderStateChanged: (
      state,
      action: PayloadAction<{
        id: keyof UIState['expanders'];
        isOpen: UIState['expanders'][keyof UIState['expanders']];
      }>
    ) => {
      const { id, isOpen } = action.payload;
      state.expanders[id] = isOpen;
    },
    textAreaSizesStateChanged: (
      state,
      action: PayloadAction<{
        id: keyof UIState['textAreaSizes'];
        size: UIState['textAreaSizes'][keyof UIState['textAreaSizes']];
      }>
    ) => {
      const { id, size } = action.payload;
      state.textAreaSizes[id] = size;
    },
    dockviewStorageKeyChanged: (
      state,
      action: PayloadAction<{
        id: keyof UIState['panels'];
        state: UIState['panels'][keyof UIState['panels']] | undefined;
      }>
    ) => {
      const { id, state: panelState } = action.payload;
      if (panelState) {
        state.panels[id] = panelState;
      } else {
        delete state.panels[id];
      }
    },
    shouldShowNotificationChanged: (state, action: PayloadAction<UIState['shouldShowNotificationV2']>) => {
      state.shouldShowNotificationV2 = action.payload;
    },
    pickerCompactViewStateChanged: (state, action: PayloadAction<{ pickerId: string; isCompact: boolean }>) => {
      state.pickerCompactViewStates[action.payload.pickerId] = action.payload.isCompact;
    },
  },
});

export const {
  setActiveTab,
  setShouldShowItemDetails,
  setShouldShowProgressInViewer,
  setShouldUsePagedGalleryView,
  accordionStateChanged,
  expanderStateChanged,
  shouldShowNotificationChanged,
  textAreaSizesStateChanged,
  dockviewStorageKeyChanged,
  pickerCompactViewStateChanged,
} = slice.actions;

export const selectUiSlice = (state: RootState) => state.ui;

export const uiSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zUIState,
  getInitialState: getInitialUIState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      if (state._version === 1) {
        state.activeTab = 'generation';
        state._version = 2;
      }
      if (state._version === 2) {
        state.activeTab = 'canvas';
        state._version = 3;
      }
      if (state._version === 3) {
        state.panels = {};
        state._version = 4;
      }
      if (state._version === 4) {
        state.shouldUsePagedGalleryView = false;
        state._version = 5;
      }
      return zUIState.parse(state);
    },
    persistDenylist: ['shouldShowItemDetails'],
  },
};
