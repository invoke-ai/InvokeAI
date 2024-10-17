import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { newSessionRequested } from 'features/controlLayers/store/actions';
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { atom } from 'nanostores';

import type { CanvasRightPanelTabName, TabName, UIState } from './uiTypes';

const initialUIState: UIState = {
  _version: 3,
  activeTab: 'canvas',
  activeTabCanvasRightPanel: 'gallery',
  shouldShowImageDetails: false,
  shouldShowProgressInViewer: true,
  accordions: {},
  expanders: {},
  shouldShowNotification: true,
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState: initialUIState,
  reducers: {
    setActiveTab: (state, action: PayloadAction<TabName>) => {
      state.activeTab = action.payload;
    },
    activeTabCanvasRightPanelChanged: (state, action: PayloadAction<CanvasRightPanelTabName>) => {
      state.activeTabCanvasRightPanel = action.payload;
    },
    setShouldShowImageDetails: (state, action: PayloadAction<boolean>) => {
      state.shouldShowImageDetails = action.payload;
    },
    setShouldShowProgressInViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowProgressInViewer = action.payload;
    },
    accordionStateChanged: (state, action: PayloadAction<{ id: string; isOpen: boolean }>) => {
      const { id, isOpen } = action.payload;
      state.accordions[id] = isOpen;
    },
    expanderStateChanged: (state, action: PayloadAction<{ id: string; isOpen: boolean }>) => {
      const { id, isOpen } = action.payload;
      state.expanders[id] = isOpen;
    },
    shouldShowNotificationChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowNotification = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addCase(workflowLoadRequested, (state) => {
      state.activeTab = 'workflows';
    });
    builder.addMatcher(newSessionRequested, (state) => {
      state.activeTab = 'canvas';
    });
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
  initialState: initialUIState,
  migrate: migrateUIState,
  persistDenylist: ['shouldShowImageDetails'],
};

const TABS_WITH_LEFT_PANEL: TabName[] = ['canvas', 'upscaling', 'workflows'] as const;
export const LEFT_PANEL_MIN_SIZE_PX = 400;
export const $isLeftPanelOpen = atom(true);
export const selectWithLeftPanel = createSelector(selectUiSlice, (ui) => TABS_WITH_LEFT_PANEL.includes(ui.activeTab));

const TABS_WITH_RIGHT_PANEL: TabName[] = ['canvas', 'upscaling', 'workflows'] as const;
export const RIGHT_PANEL_MIN_SIZE_PX = 390;
export const $isRightPanelOpen = atom(true);
export const selectWithRightPanel = createSelector(selectUiSlice, (ui) => TABS_WITH_RIGHT_PANEL.includes(ui.activeTab));
