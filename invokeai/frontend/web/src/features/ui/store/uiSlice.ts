import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';

import type { DockviewPanelState, GridviewPanelState, UIState } from './uiTypes';
import { getInitialUIState } from './uiTypes';

export const uiSlice = createSlice({
  name: 'ui',
  initialState: getInitialUIState(),
  reducers: {
    setActiveTab: (state, action: PayloadAction<UIState['activeTab']>) => {
      state.activeTab = action.payload;
    },
    activeTabCanvasRightPanelChanged: (state, action: PayloadAction<UIState['activeTabCanvasRightPanel']>) => {
      state.activeTabCanvasRightPanel = action.payload;
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
    gridviewPanelStateChanged: (
      state,
      action: PayloadAction<{
        id: string;
        panelState: GridviewPanelState;
      }>
    ) => {
      const { id, panelState } = action.payload;
      state.gridviewPanelStates[id] = panelState;
    },
    dockviewPanelStateChanged: (
      state,
      action: PayloadAction<{
        id: string;
        panelState: DockviewPanelState;
      }>
    ) => {
      const { id, panelState } = action.payload;
      state.dockviewPanelStates[id] = panelState;
    },
    shouldShowNotificationChanged: (state, action: PayloadAction<UIState['shouldShowNotificationV2']>) => {
      state.shouldShowNotificationV2 = action.payload;
    },
  },
});

export const {
  setActiveTab,
  activeTabCanvasRightPanelChanged,
  setShouldShowImageDetails,
  setShouldShowProgressInViewer,
  accordionStateChanged,
  expanderStateChanged,
  shouldShowNotificationChanged,
  textAreaSizesStateChanged,
  gridviewPanelStateChanged,
  dockviewPanelStateChanged,
} = uiSlice.actions;

export const selectUiSlice = (state: RootState) => state.ui;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateUIState = (state: any): any => {
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
    state.gridviewPanelStates = {};
    state.dockviewPanelStates = {};
    state._version = 4;
  }
  return state;
};

export const uiPersistConfig: PersistConfig<UIState> = {
  name: uiSlice.name,
  initialState: getInitialUIState(),
  migrate: migrateUIState,
  persistDenylist: ['shouldShowImageDetails'],
};
