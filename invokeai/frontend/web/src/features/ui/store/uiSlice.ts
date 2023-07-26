import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { SchedulerParam } from 'features/parameters/types/parameterSchemas';
import { setActiveTabReducer } from './extraReducers';
import { InvokeTabName } from './tabMap';
import { UIState } from './uiTypes';

export const initialUIState: UIState = {
  activeTab: 0,
  shouldPinParametersPanel: true,
  shouldShowParametersPanel: true,
  shouldShowImageDetails: false,
  shouldUseCanvasBetaLayout: false,
  shouldShowExistingModelsInSearch: false,
  shouldUseSliders: false,
  shouldPinGallery: true,
  shouldShowGallery: true,
  shouldHidePreview: false,
  shouldShowProgressInViewer: true,
  shouldShowEmbeddingPicker: false,
  favoriteSchedulers: [],
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState: initialUIState,
  reducers: {
    setActiveTab: (state, action: PayloadAction<InvokeTabName>) => {
      setActiveTabReducer(state, action.payload);
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
    setShouldShowProgressInViewer: (state, action: PayloadAction<boolean>) => {
      state.shouldShowProgressInViewer = action.payload;
    },
    favoriteSchedulersChanged: (
      state,
      action: PayloadAction<SchedulerParam[]>
    ) => {
      state.favoriteSchedulers = action.payload;
    },
    toggleEmbeddingPicker: (state) => {
      state.shouldShowEmbeddingPicker = !state.shouldShowEmbeddingPicker;
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
  setShouldPinParametersPanel,
  setShouldShowParametersPanel,
  setShouldShowImageDetails,
  setShouldUseCanvasBetaLayout,
  setShouldShowExistingModelsInSearch,
  setShouldUseSliders,
  setShouldHidePreview,
  setShouldShowGallery,
  togglePanels,
  togglePinGalleryPanel,
  togglePinParametersPanel,
  toggleParametersPanel,
  toggleGalleryPanel,
  setShouldShowProgressInViewer,
  favoriteSchedulersChanged,
  toggleEmbeddingPicker,
} = uiSlice.actions;

export default uiSlice.reducer;
