import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { deepClone } from 'common/util/deepClone';

import { INITIAL_STATE, type UIState } from './uiTypes';

const getInitialState = (): UIState => deepClone(INITIAL_STATE);

const slice = createSlice({
  name: 'ui',
  initialState: getInitialState(),
  reducers: {
    setActiveTab: (state, action: PayloadAction<UIState['activeTab']>) => {
      state.activeTab = action.payload;
    },
    setShouldShowImageDetails: (state, action: PayloadAction<UIState['shouldShowImageDetails']>) => {
      state.shouldShowImageDetails = action.payload;
    },
    setShouldShowProgressInViewer: (state, action: PayloadAction<UIState['shouldShowProgressInViewer']>) => {
      state.shouldShowProgressInViewer = action.payload;
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
  setShouldShowImageDetails,
  setShouldShowProgressInViewer,
  accordionStateChanged,
  expanderStateChanged,
  shouldShowNotificationChanged,
  textAreaSizesStateChanged,
  dockviewStorageKeyChanged,
  pickerCompactViewStateChanged,
} = slice.actions;

export const selectUiSlice = (state: RootState) => state.ui;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
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
  return state;
};

export const uiSliceConfig: SliceConfig<typeof slice> = {
  slice,
  getInitialState,
  persistConfig: {
    migrate,
    persistDenylist: ['shouldShowImageDetails'],
  },
};
