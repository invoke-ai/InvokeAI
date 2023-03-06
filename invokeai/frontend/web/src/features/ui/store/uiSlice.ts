import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { InvokeTabName, tabMap } from './tabMap';
import { AddNewModelType, UIState } from './uiTypes';

const initialtabsState: UIState = {
  activeTab: 0,
  currentTheme: 'dark',
  parametersPanelScrollPosition: 0,
  shouldHoldParametersPanelOpen: false,
  shouldPinParametersPanel: true,
  shouldShowParametersPanel: true,
  shouldShowDualDisplay: true,
  shouldShowImageDetails: false,
  shouldUseCanvasBetaLayout: false,
  shouldShowExistingModelsInSearch: false,
  shouldUseSliders: false,
  addNewModelUIOption: null,
};

const initialState: UIState = initialtabsState;

export const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setActiveTab: (state, action: PayloadAction<number | InvokeTabName>) => {
      if (typeof action.payload === 'number') {
        state.activeTab = action.payload;
      } else {
        state.activeTab = tabMap.indexOf(action.payload);
      }
    },
    setCurrentTheme: (state, action: PayloadAction<string>) => {
      state.currentTheme = action.payload;
    },
    setParametersPanelScrollPosition: (
      state,
      action: PayloadAction<number>
    ) => {
      state.parametersPanelScrollPosition = action.payload;
    },
    setShouldPinParametersPanel: (state, action: PayloadAction<boolean>) => {
      state.shouldPinParametersPanel = action.payload;
    },
    setShouldShowParametersPanel: (state, action: PayloadAction<boolean>) => {
      state.shouldShowParametersPanel = action.payload;
    },
    setShouldHoldParametersPanelOpen: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldHoldParametersPanelOpen = action.payload;
    },
    setShouldShowDualDisplay: (state, action: PayloadAction<boolean>) => {
      state.shouldShowDualDisplay = action.payload;
    },
    setShouldShowImageDetails: (state, action: PayloadAction<boolean>) => {
      state.shouldShowImageDetails = action.payload;
    },
    setShouldUseCanvasBetaLayout: (state, action: PayloadAction<boolean>) => {
      state.shouldUseCanvasBetaLayout = action.payload;
    },
    setShouldShowExistingModelsInSearch: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldShowExistingModelsInSearch = action.payload;
    },
    setShouldUseSliders: (state, action: PayloadAction<boolean>) => {
      state.shouldUseSliders = action.payload;
    },
    setAddNewModelUIOption: (state, action: PayloadAction<AddNewModelType>) => {
      state.addNewModelUIOption = action.payload;
    },
  },
});

export const {
  setActiveTab,
  setCurrentTheme,
  setParametersPanelScrollPosition,
  setShouldHoldParametersPanelOpen,
  setShouldPinParametersPanel,
  setShouldShowParametersPanel,
  setShouldShowDualDisplay,
  setShouldShowImageDetails,
  setShouldUseCanvasBetaLayout,
  setShouldShowExistingModelsInSearch,
  setShouldUseSliders,
  setAddNewModelUIOption,
} = uiSlice.actions;

export default uiSlice.reducer;
