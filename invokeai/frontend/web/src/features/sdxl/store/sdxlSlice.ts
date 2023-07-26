import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import {
  MainModelParam,
  NegativeStylePromptSDXLParam,
  PositiveStylePromptSDXLParam,
  SchedulerParam,
} from 'features/parameters/types/parameterSchemas';
import { MainModelField } from 'services/api/types';

type SDXLInitialState = {
  positiveStylePrompt: PositiveStylePromptSDXLParam;
  negativeStylePrompt: NegativeStylePromptSDXLParam;
  shouldConcatSDXLStylePrompt: boolean;
  shouldUseSDXLRefiner: boolean;
  sdxlImg2ImgDenoisingStrength: number;
  refinerModel: MainModelField | null;
  refinerSteps: number;
  refinerCFGScale: number;
  refinerScheduler: SchedulerParam;
  refinerAestheticScore: number;
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
  refinerAestheticScore: 6,
  refinerStart: 0.7,
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
      action: PayloadAction<MainModelParam | null>
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
    setRefinerAestheticScore: (state, action: PayloadAction<number>) => {
      state.refinerAestheticScore = action.payload;
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
  setRefinerAestheticScore,
  setRefinerStart,
} = sdxlSlice.actions;

export default sdxlSlice.reducer;
