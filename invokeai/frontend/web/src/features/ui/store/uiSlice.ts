import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { setActiveTabReducer } from './extraReducers';
import { InvokeTabName, tabMap } from './tabMap';
import { AddNewModelType, Coordinates, Rect, UIState } from './uiTypes';
import { initialImageSelected } from 'features/parameters/store/actions';
import { initialImageChanged } from 'features/parameters/store/generationSlice';

export const initialUIState: UIState = {
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
  textTabAccordionState: [],
  imageTabAccordionState: [],
  canvasTabAccordionState: [],
  floatingProgressImageRect: { x: 0, y: 0, width: 0, height: 0 },
  shouldShowProgressImages: false,
  shouldShowProgressInViewer: false,
  shouldShowImageParameters: false,
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState: initialUIState,
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
      if (!state.shouldPinGallery) {
        state.shouldShowGallery = true;
      }
    },
    togglePinParametersPanel: (state) => {
      state.shouldPinParametersPanel = !state.shouldPinParametersPanel;
      if (!state.shouldPinParametersPanel) {
        state.shouldShowParametersPanel = true;
      }
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
      if (tabMap[state.activeTab] === 'txt2img') {
        state.textTabAccordionState = action.payload;
      }

      if (tabMap[state.activeTab] === 'img2img') {
        state.imageTabAccordionState = action.payload;
      }

      if (tabMap[state.activeTab] === 'unifiedCanvas') {
        state.canvasTabAccordionState = action.payload;
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
    setShouldShowProgressInViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowProgressInViewer = action.payload;
    },
    shouldShowImageParametersChanged: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldShowImageParameters = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addCase(initialImageChanged, (state) => {
      setActiveTabReducer(state, 'img2img');
    });
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
  setShouldShowProgressInViewer,
  shouldShowImageParametersChanged,
} = uiSlice.actions;

export default uiSlice.reducer;
