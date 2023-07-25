import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import {
  NegativeStylePromptSDXLParam,
  PositiveStylePromptSDXLParam,
} from 'features/parameters/types/parameterSchemas';

type SDXLInitialState = {
  shouldUseSDXLRefiner: boolean;
  positiveStylePrompt: PositiveStylePromptSDXLParam;
  negativeStylePrompt: NegativeStylePromptSDXLParam;
};

const sdxlInitialState: SDXLInitialState = {
  shouldUseSDXLRefiner: false,
  positiveStylePrompt: '',
  negativeStylePrompt: '',
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
    setShouldUseSDXLRefiner: (state, action: PayloadAction<boolean>) => {
      state.shouldUseSDXLRefiner = action.payload;
    },
  },
});

export const {
  setPositiveStylePromptSDXL,
  setNegativeStylePromptSDXL,
  setShouldUseSDXLRefiner,
} = sdxlSlice.actions;

export default sdxlSlice.reducer;
