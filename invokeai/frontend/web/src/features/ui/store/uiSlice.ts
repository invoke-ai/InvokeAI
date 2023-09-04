import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { initialImageChanged } from 'features/parameters/store/generationSlice';
import { SchedulerParam } from 'features/parameters/types/parameterSchemas';
import { setActiveTabReducer } from './extraReducers';
import { InvokeTabName } from './tabMap';
import { UIState } from './uiTypes';

export const initialUIState: UIState = {
  activeTab: 0,
  shouldShowImageDetails: false,
  shouldUseCanvasBetaLayout: false,
  shouldShowExistingModelsInSearch: false,
  shouldUseSliders: false,
  shouldHidePreview: false,
  shouldShowProgressInViewer: true,
  shouldShowEmbeddingPicker: false,
  shouldAutoChangeDimensions: false,
  favoriteSchedulers: [],
  globalContextMenuCloseTrigger: 0,
  panels: {},
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState: initialUIState,
  reducers: {
    setActiveTab: (state, action: PayloadAction<InvokeTabName>) => {
      setActiveTabReducer(state, action.payload);
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
    setShouldAutoChangeDimensions: (state, action: PayloadAction<boolean>) => {
      state.shouldAutoChangeDimensions = action.payload;
    },
    contextMenusClosed: (state) => {
      state.globalContextMenuCloseTrigger += 1;
    },
    panelsChanged: (
      state,
      action: PayloadAction<{ name: string; value: string }>
    ) => {
      state.panels[action.payload.name] = action.payload.value;
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
  setShouldShowImageDetails,
  setShouldUseCanvasBetaLayout,
  setShouldShowExistingModelsInSearch,
  setShouldUseSliders,
  setShouldHidePreview,
  setShouldShowProgressInViewer,
  favoriteSchedulersChanged,
  toggleEmbeddingPicker,
  setShouldAutoChangeDimensions,
  contextMenusClosed,
  panelsChanged,
} = uiSlice.actions;

export default uiSlice.reducer;
