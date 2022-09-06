import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { testLogs } from '../../app/testingData';

export interface SDState {
  prompt: string;
  imagesToGenerate: number;
  steps: number;
  cfgScale: number;
  height: number;
  width: number;
  sampler: string;
  seed: number;
  img2imgStrength: number;
  gfpganStrength: number;
  upscalingLevel: number;
  upscalingStrength: number;
}

const initialDreamMenuState = {
  prompt: 'Cyborg pickle shooting lasers',
  imagesToGenerate: 1,
  steps: 5,
  cfgScale: 7.5,
  height: 512,
  width: 512,
  sampler: 'k_lms',
  seed: -1,
  img2imgStrength: 0.75,
  gfpganStrength: 0.8,
  upscalingLevel: 0,
  upscalingStrength: 0.75,
};

const initialState: SDState = initialDreamMenuState;

export const sdSlice = createSlice({
  name: 'sd',
  initialState,
  reducers: {
    setPrompt: (state, action: PayloadAction<string>) => {
      state.prompt = action.payload;
    },
    setImagesToGenerate: (state, action: PayloadAction<number>) => {
      state.imagesToGenerate = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      state.steps = action.payload;
    },
    setCfgScale: (state, action: PayloadAction<number>) => {
      state.cfgScale = action.payload;
    },
    setHeight: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    setWidth: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    setSampler: (state, action: PayloadAction<string>) => {
      state.sampler = action.payload;
    },
    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
    },
    setGfpganStrength: (state, action: PayloadAction<number>) => {
      state.gfpganStrength = action.payload;
    },
    setUpscalingLevel: (state, action: PayloadAction<number>) => {
      state.upscalingLevel = action.payload;
    },
    setUpscalingStrength: (state, action: PayloadAction<number>) => {
      state.upscalingStrength = action.payload;
    },
    resetSeed: (state) => {
      state.seed = -1;
    },
    resetForm: (state) => {
      return {
        ...state,
        ...initialDreamMenuState,
      };
    },
  },
});

export const {
  setPrompt,
  setImagesToGenerate,
  setSteps,
  setCfgScale,
  setHeight,
  setWidth,
  setSampler,
  setSeed,
  setImg2imgStrength,
  setGfpganStrength,
  setUpscalingLevel,
  setUpscalingStrength,
  resetSeed,
  resetForm,
} = sdSlice.actions;

export default sdSlice.reducer;
