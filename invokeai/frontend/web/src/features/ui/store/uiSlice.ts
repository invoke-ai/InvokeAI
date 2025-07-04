import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { atom } from 'nanostores';

import type { TabName, UIState } from './uiTypes';
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
    shouldShowNotificationChanged: (state, action: PayloadAction<UIState['shouldShowNotificationV2']>) => {
      state.shouldShowNotificationV2 = action.payload;
    },
    showGenerateTabSplashScreenChanged: (state, action: PayloadAction<UIState['showGenerateTabSplashScreen']>) => {
      state.showGenerateTabSplashScreen = action.payload;
    },
    showCanvasTabSplashScreenChanged: (state, action: PayloadAction<UIState['showCanvasTabSplashScreen']>) => {
      state.showCanvasTabSplashScreen = action.payload;
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
  showGenerateTabSplashScreenChanged,
  showCanvasTabSplashScreenChanged,
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
  return state;
};

export const uiPersistConfig: PersistConfig<UIState> = {
  name: uiSlice.name,
  initialState: getInitialUIState(),
  migrate: migrateUIState,
  persistDenylist: ['shouldShowImageDetails'],
};

const TABS_WITH_LEFT_PANEL: TabName[] = ['canvas', 'upscaling', 'workflows', 'generate'] as const;
export const LEFT_PANEL_MIN_SIZE_PX = 420;
export const $isLeftPanelOpen = atom(true);
export const selectWithLeftPanel = createSelector(selectUiSlice, (ui) => TABS_WITH_LEFT_PANEL.includes(ui.activeTab));

const TABS_WITH_RIGHT_PANEL: TabName[] = ['canvas', 'upscaling', 'workflows', 'generate'] as const;
export const RIGHT_PANEL_MIN_SIZE_PX = 420;
export const $isRightPanelOpen = atom(true);
export const selectWithRightPanel = createSelector(selectUiSlice, (ui) => TABS_WITH_RIGHT_PANEL.includes(ui.activeTab));
