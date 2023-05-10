import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { setActiveTabReducer } from './extraReducers';
import { InvokeTabName, tabMap } from './tabMap';
import { AddNewModelType, Coordinates, Rect, UIState } from './uiTypes';

const initialUIState: UIState = {
  activeTab: 0,
  currentTheme: 'dark',
  parametersPanelScrollPosition: 0,
  shouldPinParametersPanel: true,
  shouldShowParametersPanel: true,
  shouldShowImageDetails: false,
  shouldUseCanvasBetaLayout: false,
  shouldShowExistingModelsInSearch: false,
  shouldUseSliders: false,
  addNewModelUIOption: null,
  shouldPinGallery: true,
  shouldShowGallery: true,
  shouldHidePreview: false,
  openLinearAccordionItems: [],
  openGenerateAccordionItems: [],
  openUnifiedCanvasAccordionItems: [],
  floatingProgressImageRect: { x: 0, y: 0, width: 0, height: 0 },
  shouldShowProgressImages: false,
  shouldAutoShowProgressImages: false,
};

const initialState: UIState = initialUIState;

export const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setActiveTab: (state, action: PayloadAction<number | InvokeTabName>) => {
      setActiveTabReducer(state, action.payload);
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
      state.shouldShowParametersPanel = true;
    },
    setShouldShowParametersPanel: (state, action: PayloadAction<boolean>) => {
      state.shouldShowParametersPanel = action.payload;
    },
    setShouldShowImageDetails: (state, action: PayloadAction<boolean>) => {
      state.shouldShowImageDetails = action.payload;
    },
    setShouldUseCanvasBetaLayout: (state, action: PayloadAction<boolean>) => {
      state.shouldUseCanvasBetaLayout = action.payload;
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
    setShouldUseSliders: (state, action: PayloadAction<boolean>) => {
      state.shouldUseSliders = action.payload;
    },
    setAddNewModelUIOption: (state, action: PayloadAction<AddNewModelType>) => {
      state.addNewModelUIOption = action.payload;
    },
    setShouldPinGallery: (state, action: PayloadAction<boolean>) => {
      state.shouldPinGallery = action.payload;
    },
    setShouldShowGallery: (state, action: PayloadAction<boolean>) => {
      state.shouldShowGallery = action.payload;
    },
    togglePinGalleryPanel: (state) => {
      state.shouldPinGallery = !state.shouldPinGallery;
    },
    togglePinParametersPanel: (state) => {
      state.shouldPinParametersPanel = !state.shouldPinParametersPanel;
    },
    toggleParametersPanel: (state) => {
      state.shouldShowParametersPanel = !state.shouldShowParametersPanel;
    },
    toggleGalleryPanel: (state) => {
      state.shouldShowGallery = !state.shouldShowGallery;
    },
    togglePanels: (state) => {
      if (state.shouldShowGallery || state.shouldShowParametersPanel) {
        state.shouldShowGallery = false;
        state.shouldShowParametersPanel = false;
      } else {
        state.shouldShowGallery = true;
        state.shouldShowParametersPanel = true;
      }
    },
    openAccordionItemsChanged: (state, action: PayloadAction<number[]>) => {
      if (tabMap[state.activeTab] === 'generate') {
        state.openGenerateAccordionItems = action.payload;
      }

      if (tabMap[state.activeTab] === 'unifiedCanvas') {
        state.openUnifiedCanvasAccordionItems = action.payload;
      }
    },
    floatingProgressImageMoved: (state, action: PayloadAction<Coordinates>) => {
      state.floatingProgressImageRect = {
        ...state.floatingProgressImageRect,
        ...action.payload,
      };
    },
    floatingProgressImageResized: (
      state,
      action: PayloadAction<Partial<Rect>>
    ) => {
      state.floatingProgressImageRect = {
        ...state.floatingProgressImageRect,
        ...action.payload,
      };
    },
    setShouldShowProgressImages: (state, action: PayloadAction<boolean>) => {
      state.shouldShowProgressImages = action.payload;
    },
    setShouldAutoShowProgressImages: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldAutoShowProgressImages = action.payload;
    },
  },
});

export const {
  setActiveTab,
  setCurrentTheme,
  setParametersPanelScrollPosition,
  setShouldPinParametersPanel,
  setShouldShowParametersPanel,
  setShouldShowImageDetails,
  setShouldUseCanvasBetaLayout,
  setShouldShowExistingModelsInSearch,
  setShouldUseSliders,
  setAddNewModelUIOption,
  setShouldHidePreview,
  setShouldPinGallery,
  setShouldShowGallery,
  togglePanels,
  togglePinGalleryPanel,
  togglePinParametersPanel,
  toggleParametersPanel,
  toggleGalleryPanel,
  openAccordionItemsChanged,
  floatingProgressImageMoved,
  floatingProgressImageResized,
  setShouldShowProgressImages,
  setShouldAutoShowProgressImages,
} = uiSlice.actions;

export default uiSlice.reducer;
