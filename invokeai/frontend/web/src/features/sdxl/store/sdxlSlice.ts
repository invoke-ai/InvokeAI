import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import {
  NegativeStylePromptSDXLParam,
  PositiveStylePromptSDXLParam,
  SDXLRefinerModelParam,
  SchedulerParam,
} from 'features/parameters/types/parameterSchemas';

type SDXLInitialState = {
  positiveStylePrompt: PositiveStylePromptSDXLParam;
  negativeStylePrompt: NegativeStylePromptSDXLParam;
  shouldConcatSDXLStylePrompt: boolean;
  shouldUseSDXLRefiner: boolean;
  sdxlImg2ImgDenoisingStrength: number;
  refinerModel: SDXLRefinerModelParam | null;
  refinerSteps: number;
  refinerCFGScale: number;
  refinerScheduler: SchedulerParam;
  refinerPositiveAestheticScore: number;
  refinerNegativeAestheticScore: number;
  refinerStart: number;
};

const sdxlInitialState: SDXLInitialState = {
  positiveStylePrompt: '',
  negativeStylePrompt: '',
  shouldConcatSDXLStylePrompt: true,
  shouldUseSDXLRefiner: false,
  sdxlImg2ImgDenoisingStrength: 0.7,
  refinerModel: null,
  refinerSteps: 20,
  refinerCFGScale: 7.5,
  refinerScheduler: 'euler',
  refinerPositiveAestheticScore: 6,
  refinerNegativeAestheticScore: 2.5,
  refinerStart: 0.8,
};

const sdxlSlice = createSlice({
  name: 'sdxl',
  initialState: sdxlInitialState,
  reducers: {
    setPositiveStylePromptSDXL: (state, action: PayloadAction<string>) => {
      state.positiveStylePrompt = action.payload;
    },
    setNegativeStylePromptSDXL: (state, action: PayloadAction<string>) => {
      state.negativeStylePrompt = action.payload;
    },
    setShouldConcatSDXLStylePrompt: (state, action: PayloadAction<boolean>) => {
      state.shouldConcatSDXLStylePrompt = action.payload;
    },
    setShouldUseSDXLRefiner: (state, action: PayloadAction<boolean>) => {
      state.shouldUseSDXLRefiner = action.payload;
    },
    setSDXLImg2ImgDenoisingStrength: (state, action: PayloadAction<number>) => {
      state.sdxlImg2ImgDenoisingStrength = action.payload;
    },
    refinerModelChanged: (
      state,
      action: PayloadAction<SDXLRefinerModelParam | null>
    ) => {
      state.refinerModel = action.payload;
    },
    setRefinerSteps: (state, action: PayloadAction<number>) => {
      state.refinerSteps = action.payload;
    },
    setRefinerCFGScale: (state, action: PayloadAction<number>) => {
      state.refinerCFGScale = action.payload;
    },
    setRefinerScheduler: (state, action: PayloadAction<SchedulerParam>) => {
      state.refinerScheduler = action.payload;
    },
    setRefinerPositiveAestheticScore: (
      state,
      action: PayloadAction<number>
    ) => {
      state.refinerPositiveAestheticScore = action.payload;
    },
    setRefinerNegativeAestheticScore: (
      state,
      action: PayloadAction<number>
    ) => {
      state.refinerNegativeAestheticScore = action.payload;
    },
    setRefinerStart: (state, action: PayloadAction<number>) => {
      state.refinerStart = action.payload;
    },
  },
});

export const {
  setPositiveStylePromptSDXL,
  setNegativeStylePromptSDXL,
  setShouldConcatSDXLStylePrompt,
  setShouldUseSDXLRefiner,
  setSDXLImg2ImgDenoisingStrength,
  refinerModelChanged,
  setRefinerSteps,
  setRefinerCFGScale,
  setRefinerScheduler,
  setRefinerPositiveAestheticScore,
  setRefinerNegativeAestheticScore,
  setRefinerStart,
} = sdxlSlice.actions;

export default sdxlSlice.reducer;
